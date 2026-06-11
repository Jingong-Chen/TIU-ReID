# tools/

Optional diagnostic utilities (not required to reproduce the paper):

- **`debug_grl_grad.py`** — sanity-checks the Gradient Reversal Layer: verifies that the
  discriminator loss back-propagates non-zero, sign-reversed gradients into the backbone.

  ```bash
  PYTHONPATH=$(pwd) python tools/debug_grl_grad.py --grl_lambda 1.0
  ```

- **`eval_probe_auc.py`** — post-hoc separability probe: freezes the removal-deployment
  embedding z, trains a fresh MLP classifier (target vs. non-target) from scratch, and
  reports its validation ROC-AUC ("ProbeAUC"). Unlike the in-training discriminator AUC
  (which is trained to separate and therefore tends to ~1.0), ProbeAUC ≈ 0.5 is stronger
  evidence that the embedding itself no longer separates the target.

  ```bash
  PYTHONPATH=$(pwd) python tools/eval_probe_auc.py \
    --run_dir output/removable/<run> \
    --cfg third_party/TransReID/configs/Market/vit_transreid_stride.yml \
    --forget_id 2
  ```
