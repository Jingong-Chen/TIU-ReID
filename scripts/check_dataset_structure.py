import os
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def err(msg: str) -> None:
    print(f"[ERR] {msg}")
    sys.exit(2)


def check_market1501(p: Path) -> None:
    # 期望结构：market1501/Market-1501-v15.09.15/{bounding_box_train, bounding_box_test, query}
    base = p / "market1501" / "Market-1501-v15.09.15"
    need = ["bounding_box_train", "bounding_box_test", "query"]
    if not base.exists():
        warn(f"Market1501 not found at {base}")
        return
    for d in need:
        if not (base / d).exists():
            err(f"Market1501 missing: {base/d}")
    ok("Market1501 structure looks good.")


def check_duke(p: Path) -> None:
    # 期望结构：dukemtmc-reid/DukeMTMC-reID/{bounding_box_train, bounding_box_test, query}
    base = p / "dukemtmc-reid" / "DukeMTMC-reID"
    need = ["bounding_box_train", "bounding_box_test", "query"]
    if not base.exists():
        warn(f"DukeMTMC-reID not found at {base}")
        return
    for d in need:
        if not (base / d).exists():
            err(f"DukeMTMC-reID missing: {base/d}")
    ok("DukeMTMC-reID structure looks good.")


def check_msmt17(p: Path) -> None:
    # MSMT17 常见结构会因版本略不同；此处只检查是否存在占位目录
    base = p / "msmt17"
    if not base.exists():
        warn(f"MSMT17 placeholder not found at {base}")
        return
    ok("MSMT17 placeholder exists (manual download likely required).")


if __name__ == "__main__":
    root.mkdir(parents=True, exist_ok=True)
    check_market1501(root)
    check_duke(root)
    check_msmt17(root)


