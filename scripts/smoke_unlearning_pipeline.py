from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    repo = Path(os.environ.get("REPO", Path(__file__).resolve().parents[1]))
    os.environ["PYTHONPATH"] = str(repo) + ":" + os.environ.get("PYTHONPATH", "")

    run(
        [
            "python",
            str(repo / "scripts/make_splits.py"),
            "--dataset",
            "market1501",
            "--seed",
            "0",
            "--forget_ratio",
            "0.1",
        ]
    )
    split_dir = Path(os.environ["REID_OUTPUT_DIR"]) / "splits" / "market1501" / "seed0"
    run(
        [
            "python",
            str(repo / "scripts/make_probe_sets.py"),
            "--dataset",
            "market1501",
            "--seed",
            "0",
            "--split_dir",
            str(split_dir),
        ]
    )
    print("[OK] smoke split+probe done:", split_dir)


if __name__ == "__main__":
    main()


