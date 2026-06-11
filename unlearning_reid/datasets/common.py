from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Item:
    path: str
    pid: int
    camid: int


_market_pat = re.compile(r"^([-\d]+)_c(\d)")
_duke_pat = re.compile(r"^([-\d]+)_c(\d)")


def parse_market_filename(name: str) -> Tuple[int, int]:
    # e.g., 0002_c1s1_000451_03.jpg -> pid=2 camid=1
    m = _market_pat.match(name)
    if not m:
        raise ValueError(f"Bad Market filename: {name}")
    pid = int(m.group(1))
    cam = int(m.group(2))
    return pid, cam


def parse_duke_filename(name: str) -> Tuple[int, int]:
    # e.g., 0002_c2_f0044158.jpg -> pid=2 camid=2
    m = _duke_pat.match(name)
    if not m:
        raise ValueError(f"Bad Duke filename: {name}")
    pid = int(m.group(1))
    cam = int(m.group(2))
    return pid, cam


def list_images(dir_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in dir_path.glob("*") if p.suffix.lower() in exts])


def read_train_items(dataset: str, data_root: Path) -> List[Item]:
    ds = dataset.lower()
    if ds == "market1501":
        base = data_root / "market1501" / "Market-1501-v15.09.15" / "bounding_box_train"
        parse = parse_market_filename
    elif ds == "dukemtmc-reid":
        base = data_root / "dukemtmc-reid" / "DukeMTMC-reID" / "bounding_box_train"
        parse = parse_duke_filename
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    items: List[Item] = []
    for p in list_images(base):
        pid, cam = parse(p.name)
        if pid == -1:  # junk
            continue
        items.append(Item(str(p), pid, cam))
    if not items:
        raise RuntimeError(f"No training images found under {base}")
    return items


def group_by_pid(items: List[Item]) -> Dict[int, List[Item]]:
    mp: Dict[int, List[Item]] = {}
    for it in items:
        mp.setdefault(it.pid, []).append(it)
    return mp


def _read_split_items(base: Path, parse) -> List[Item]:
    items: List[Item] = []
    for p in list_images(base):
        pid, cam = parse(p.name)
        if pid == -1:
            continue
        items.append(Item(str(p), pid, cam))
    if not items:
        raise RuntimeError(f"No images found under {base}")
    return items


def read_test_items(dataset: str, data_root: Path) -> Tuple[List[Item], List[Item]]:
    ds = dataset.lower()
    if ds == "market1501":
        base = data_root / "market1501" / "Market-1501-v15.09.15"
        q_dir = base / "query"
        g_dir = base / "bounding_box_test"
        parse = parse_market_filename
    elif ds == "dukemtmc-reid":
        base = data_root / "dukemtmc-reid" / "DukeMTMC-reID"
        q_dir = base / "query"
        g_dir = base / "bounding_box_test"
        parse = parse_duke_filename
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    query = _read_split_items(q_dir, parse)
    gallery = _read_split_items(g_dir, parse)
    return query, gallery


