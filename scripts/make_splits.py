from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from unlearning_reid.datasets.common import group_by_pid, read_train_items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="market1501", choices=["market1501", "dukemtmc-reid"])
    ap.add_argument("--data_root", default=None, help="defaults to $REID_DATA_DIR if set")
    ap.add_argument("--out_dir", default=None, help="defaults to $REID_OUTPUT_DIR/splits/<dataset>/seed<seed>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forget_ratio", type=float, default=0.2)
    ap.add_argument(
        "--forget_ids",
        default=None,
        help="Comma-separated explicit forget IDs (overrides forget_ratio).",
    )
    ap.add_argument(
        "--forget_ids_txt",
        default=None,
        help="Path to a file with one forget ID per line (overrides forget_ratio).",
    )
    ap.add_argument(
        "--attack_ids_txt",
        default=None,
        help="Optional file with attack IDs (must be subset of forget IDs).",
    )
    ap.add_argument("--attack_ratio_within_forget", type=float, default=0.5)
    ap.add_argument("--min_imgs_per_id", type=int, default=4)
    args = ap.parse_args()

    data_root = Path(args.data_root) if args.data_root else Path(os.environ["REID_DATA_DIR"])
    out_base = (
        Path(args.out_dir)
        if args.out_dir
        else Path(os.environ["REID_OUTPUT_DIR"]) / "splits" / args.dataset / f"seed{args.seed}"
    )
    out_base.mkdir(parents=True, exist_ok=True)

    items = read_train_items(args.dataset, data_root)
    by_pid = group_by_pid(items)

    pids = [pid for pid, lst in by_pid.items() if len(lst) >= args.min_imgs_per_id]
    pids = sorted(pids)
    if len(pids) < 50:
        raise RuntimeError(
            f"Too few eligible train IDs: {len(pids)}. Reduce min_imgs_per_id or verify dataset."
        )

    def _parse_ids(text: str) -> list[int]:
        return [int(x) for x in text.split(",") if x.strip()]

    explicit_forget: list[int] = []
    if args.forget_ids:
        explicit_forget += _parse_ids(args.forget_ids)
    if args.forget_ids_txt:
        explicit_forget += [
            int(x)
            for x in Path(args.forget_ids_txt).read_text().splitlines()
            if x.strip()
        ]
    explicit_forget = sorted(set(explicit_forget))

    if explicit_forget:
        missing = [pid for pid in explicit_forget if pid not in pids]
        if missing:
            raise RuntimeError(f"Forget IDs not found or too few images: {missing}")
        forget_ids = explicit_forget
        retain_ids = sorted([pid for pid in pids if pid not in set(forget_ids)])
    else:
        rnd = random.Random(args.seed)
        rnd.shuffle(pids)
        n_forget = max(1, int(round(len(pids) * args.forget_ratio)))
        forget_ids = sorted(pids[:n_forget])
        retain_ids = sorted(pids[n_forget:])

    if args.attack_ids_txt:
        attack_ids = [
            int(x)
            for x in Path(args.attack_ids_txt).read_text().splitlines()
            if x.strip()
        ]
        attack_ids = sorted(set(attack_ids))
        if not set(attack_ids).issubset(set(forget_ids)):
            raise RuntimeError("attack_ids_txt must be a subset of forget_ids.")
    else:
        if len(forget_ids) == 1:
            attack_ids = forget_ids[:]
        else:
            rnd = random.Random(args.seed)
            n_attack = max(1, int(round(len(forget_ids) * args.attack_ratio_within_forget)))
            attack_ids = sorted(rnd.sample(forget_ids, n_attack))

    (out_base / "retain_ids.txt").write_text("\n".join(map(str, retain_ids)) + "\n")
    (out_base / "forget_ids.txt").write_text("\n".join(map(str, forget_ids)) + "\n")
    (out_base / "attack_ids.txt").write_text("\n".join(map(str, attack_ids)) + "\n")

    meta = {
        "dataset": args.dataset,
        "seed": args.seed,
        "forget_ratio": args.forget_ratio,
        "attack_ratio_within_forget": args.attack_ratio_within_forget,
        "min_imgs_per_id": args.min_imgs_per_id,
        "n_ids_total": len(pids),
        "n_retain": len(retain_ids),
        "n_forget": len(forget_ids),
        "n_attack": len(attack_ids),
        "forget_ids_source": "explicit" if explicit_forget else "ratio",
        "forget_ids": forget_ids,
        "attack_ids": attack_ids,
    }
    (out_base / "split_config.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("[OK] Wrote splits to:", out_base)


if __name__ == "__main__":
    main()


