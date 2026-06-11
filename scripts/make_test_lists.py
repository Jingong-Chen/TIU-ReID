from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from unlearning_reid.datasets.common import read_test_items


def dump_list(items, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for it in items:
            f.write(json.dumps({"path": it.path, "pid": it.pid, "camid": it.camid}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="market1501", choices=["market1501", "dukemtmc-reid"])
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root) if args.data_root else Path(os.environ["REID_DATA_DIR"])
    out_dir = Path(args.out_dir) if args.out_dir else Path(os.environ["REID_OUTPUT_DIR"]) / "test_lists" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    query, gallery = read_test_items(args.dataset, data_root)
    dump_list(query, out_dir / "test_query.jsonl")
    dump_list(gallery, out_dir / "test_gallery.jsonl")

    meta = {
        "dataset": args.dataset,
        "data_root": str(data_root),
        "counts": {"query": len(query), "gallery": len(gallery)},
        "out_dir": str(out_dir),
    }
    (out_dir / "test_list_config.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("[OK] wrote test lists ->", out_dir)


if __name__ == "__main__":
    main()

