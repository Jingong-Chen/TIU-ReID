import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env_name != "reid_unlearning":
        print(f"[ERR] Not in conda env reid_unlearning (current: {env_name})")
        sys.exit(2)

    project_root = Path(__file__).resolve().parents[1]
    data_dir = Path(os.environ.get("REID_DATA_DIR", project_root / "data"))
    out_dir = Path(os.environ.get("REID_OUTPUT_DIR", project_root / "output"))
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = out_dir / "runs" / "smoke_test"
    log_dir = run_dir / "logs"
    meta_dir = run_dir / "meta"
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"smoke_{stamp}.log"
    meta_path = meta_dir / f"meta_{stamp}.json"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("[OK] smoke test log\\n")
        f.write(f"python={sys.executable}\\n")
        f.write(f"data_dir={data_dir}\\n")
        f.write(f"out_dir={out_dir}\\n")

    meta = {
        "timestamp_utc": stamp,
        "python": sys.version,
        "executable": sys.executable,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "conda_env": env_name,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] wrote log: {log_path}")
    print(f"[OK] wrote meta: {meta_path}")


if __name__ == "__main__":
    main()


