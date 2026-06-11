# output/

Weights, splits, probes, and results live here (git-ignored).

Download `tiu_reid_weights_market1501.tar.gz` from the Releases page and extract at the repo root:

```bash
tar xzf tiu_reid_weights_market1501.tar.gz
```

This provides:

```
output/
├── transreid/market_teacher_r50/                # TransReID teacher (transformer_120.pth)
└── removable/removable_market1501_id2_seed0_retfix_F/
    ├── base_ckpt.pth                            # fine-tuned backbone
    ├── target_module.pth                        # removable Target Memory Adapter
    └── discriminator.pth
```

The ImageNet ViT-B/16 pretrain (only needed to re-train the teacher) is downloaded automatically by `scripts/train_teacher_transreid.sh` into `output/pretrained/`.
