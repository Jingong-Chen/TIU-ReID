import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, tensor: torch.Tensor):
        if not torch.is_tensor(tensor):
            return tensor
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            tensor = F.interpolate(tensor, size=self.size, mode="bilinear", align_corners=False)
            return tensor.squeeze(0)
        return tensor


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor: torch.Tensor):
        if not torch.is_tensor(tensor):
            return tensor
        if random.random() < self.p:
            return torch.flip(tensor, dims=[2])
        return tensor


class Pad:
    def __init__(self, padding):
        self.padding = int(padding)

    def __call__(self, tensor: torch.Tensor):
        if not torch.is_tensor(tensor):
            return tensor
        if self.padding <= 0:
            return tensor
        return F.pad(tensor, (self.padding, self.padding, self.padding, self.padding), value=0.0)


class RandomCrop:
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, tensor: torch.Tensor):
        if not torch.is_tensor(tensor):
            return tensor
        th, tw = self.size[0], self.size[1]
        _, h, w = tensor.shape
        if h < th or w < tw:
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)
            _, h, w = tensor.shape
        if h == th and w == tw:
            return tensor
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return tensor[:, y1:y1 + th, x1:x1 + tw]


class ToTensor:
    def __call__(self, img):
        if torch.is_tensor(img):
            return img
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor):
        if not torch.is_tensor(tensor):
            return tensor
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)
        return (tensor - mean) / std


def make_dataloader(cfg):
    erasing_device = "cpu"
    train_transforms = Compose([
            ToTensor(),
            Resize(cfg.INPUT.SIZE_TRAIN),
            RandomHorizontalFlip(p=cfg.INPUT.PROB),
            Pad(cfg.INPUT.PADDING),
            RandomCrop(cfg.INPUT.SIZE_TRAIN),
            Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device=erasing_device),
        ])

    val_transforms = Compose([
        ToTensor(),
        Resize(cfg.INPUT.SIZE_TEST),
        Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn, pin_memory=True
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn, pin_memory=True
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn, pin_memory=True
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
