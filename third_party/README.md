# third_party/

## TransReID (vendored)

`TransReID/` is a vendored copy of [damo-cv/TransReID](https://github.com/damo-cv/TransReID)
(MIT License, Copyright (c) 2021 heshuting555), used as the ReID backbone and teacher.

We vendor it (instead of asking you to clone upstream) because the pipeline depends on
**small local modifications** to the following files:

- `datasets/make_dataloader.py`
- `loss/make_loss.py`
- `model/backbones/vit_pytorch.py`
- `processor/processor.py`

The changes are compatibility fixes (newer PyTorch / `torch._six`, dataloader and loss
plumbing for the unlearning pipeline) and do not alter the TransReID architecture.

The original MIT license is kept at `TransReID/LICENSE`. If you use the backbone, please
also cite:

```bibtex
@inproceedings{he2021transreid,
  title     = {TransReID: Transformer-based Object Re-Identification},
  author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {15013--15022},
  year      = {2021}
}
```
