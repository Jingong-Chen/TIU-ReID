import json
import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda" if (cfg.MODEL.DEVICE != "cpu" and torch.cuda.is_available()) else "cpu"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    print('start training', flush=True)
    _LOCAL_PROCESS_GROUP = None
    if device == "cuda":
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    use_amp = device == "cuda"
    scaler = amp.GradScaler(enabled=use_amp)
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device, non_blocking=(device == "cuda"))
            target = vid.to(device, non_blocking=(device == "cuda"))
            target_cam = target_cam.to(device, non_blocking=(device == "cuda"))
            target_view = target_view.to(device, non_blocking=(device == "cuda"))
            with amp.autocast(enabled=use_amp):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if device == "cuda":
                torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                msg = "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                    epoch, (n_iter + 1), len(train_loader),
                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]
                )
                logger.info(msg)
                print(msg, flush=True)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            epoch_summary = {
                "epoch": epoch,
                "loss": float(loss_meter.avg),
                "acc": float(acc_meter.avg),
                "time_per_batch": float(time_per_batch),
                "speed": float(train_loader.batch_size / time_per_batch),
            }
            logger.info(
                "Epoch {} done. Loss: {:.4f}, Acc: {:.4f}, Time/batch: {:.3f}s, Speed: {:.1f} samples/s"
                .format(
                    epoch,
                    epoch_summary["loss"],
                    epoch_summary["acc"],
                    epoch_summary["time_per_batch"],
                    epoch_summary["speed"],
                )
            )
            print(
                "Epoch {} done. Loss: {:.4f}, Acc: {:.4f}, Time/batch: {:.3f}s, Speed: {:.1f} samples/s"
                .format(
                    epoch,
                    epoch_summary["loss"],
                    epoch_summary["acc"],
                    epoch_summary["time_per_batch"],
                    epoch_summary["speed"],
                ),
                flush=True,
            )
            try:
                if cfg.OUTPUT_DIR:
                    with open(os.path.join(cfg.OUTPUT_DIR, "epoch_log.jsonl"), "a") as f:
                        f.write(json.dumps(epoch_summary) + "\n")
            except Exception:
                pass

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device, non_blocking=(device == "cuda"))
                            camids = camids.to(device, non_blocking=(device == "cuda"))
                            target_view = target_view.to(device, non_blocking=(device == "cuda"))
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device, non_blocking=(device == "cuda"))
                        camids = camids.to(device, non_blocking=(device == "cuda"))
                        target_view = target_view.to(device, non_blocking=(device == "cuda"))
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda" if (cfg.MODEL.DEVICE != "cpu" and torch.cuda.is_available()) else "cpu"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device == "cuda":
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device, non_blocking=(device == "cuda"))
            camids = camids.to(device, non_blocking=(device == "cuda"))
            target_view = target_view.to(device, non_blocking=(device == "cuda"))
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


