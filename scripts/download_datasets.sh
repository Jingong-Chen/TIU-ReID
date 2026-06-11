#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REID_DATA_DIR:-$ROOT_DIR/data}"
mkdir -p "$DATA_DIR"

log(){ echo "[$(date +%H:%M:%S)] $*"; }

need_cmd(){
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERR: missing command: $1"
    return 1
  fi
  return 0
}

# prefer aria2c when available
dl_http(){
  local url="$1"; local out="$2"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 16 -s 16 -k 1M -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$out" "$url"
  else
    log "ERR: need aria2c/curl/wget"
    return 2
  fi
}

extract_any(){
  local archive="$1"; local dst="$2"
  mkdir -p "$dst"
  if command -v 7z >/dev/null 2>&1; then
    7z x -y "-o$dst" "$archive" >/dev/null
  else
    # fallback: python zipfile (zip only)
    python - <<PY
import sys, zipfile
from pathlib import Path
arc=Path(sys.argv[1]); dst=Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(arc,'r') as z: z.extractall(dst)
print("[OK] extracted via python zipfile:", arc)
PY
  fi
}

# move folder if exists anywhere under a root
find_dir(){
  local root="$1"; local name="$2"
  python - <<PY
import os, sys
root=sys.argv[1]; name=sys.argv[2]
for dirpath, dirnames, _ in os.walk(root):
    if name in dirnames:
        print(os.path.join(dirpath, name))
        raise SystemExit(0)
raise SystemExit(1)
PY
}

ensure_market_layout(){
  local base="$DATA_DIR/market1501"
  local target="$base/Market-1501-v15.09.15"
  mkdir -p "$base"
  if [[ -d "$target/bounding_box_train" && -d "$target/bounding_box_test" && -d "$target/query" ]]; then
    log "OK: Market1501 already prepared: $target"
    return 0
  fi

  local found=""
  if found="$(find_dir "$base" "Market-1501-v15.09.15" 2>/dev/null || true)"; then
    if [[ -n "$found" ]]; then
      rm -rf "$target"
      mv "$found" "$target"
    fi
  fi

  if [[ -d "$base/bounding_box_train" && -d "$base/bounding_box_test" && -d "$base/query" ]]; then
    rm -rf "$target"
    mkdir -p "$target"
    mv "$base/bounding_box_train" "$target/"
    mv "$base/bounding_box_test" "$target/"
    mv "$base/query" "$target/"
  fi

  if [[ -d "$target/bounding_box_train" && -d "$target/bounding_box_test" && -d "$target/query" ]]; then
    log "OK: Market1501 layout fixed: $target"
    return 0
  fi

  log "WARN: Market1501 layout not ready yet."
  return 1
}

ensure_duke_layout(){
  local base="$DATA_DIR/dukemtmc-reid"
  local target="$base/DukeMTMC-reID"
  mkdir -p "$base"
  if [[ -d "$target/bounding_box_train" && -d "$target/bounding_box_test" && -d "$target/query" ]]; then
    log "OK: DukeMTMC-reID already prepared: $target"
    return 0
  fi

  local found=""
  if found="$(find_dir "$base" "DukeMTMC-reID" 2>/dev/null || true)"; then
    if [[ -n "$found" ]]; then
      rm -rf "$target"
      mv "$found" "$target"
    fi
  fi

  if [[ -d "$base/bounding_box_train" && -d "$base/bounding_box_test" && -d "$base/query" ]]; then
    rm -rf "$target"
    mkdir -p "$target"
    mv "$base/bounding_box_train" "$target/"
    mv "$base/bounding_box_test" "$target/"
    mv "$base/query" "$target/"
  fi

  if [[ -d "$target/bounding_box_train" && -d "$target/bounding_box_test" && -d "$target/query" ]]; then
    log "OK: DukeMTMC-reID layout fixed: $target"
    return 0
  fi

  log "WARN: DukeMTMC-reID layout not ready yet."
  return 1
}

ensure_msmt_layout(){
  local base="$DATA_DIR/msmt17"
  mkdir -p "$base"

  if [[ -d "$base/MSMT17_V1/train" || -d "$base/MSMT17_V2/train" ]]; then
    log "OK: MSMT17 already present."
    return 0
  fi

  if [[ -n "${MSMT17_ZIP:-}" && -f "${MSMT17_ZIP:-}" ]]; then
    log "INFO: Extract MSMT17 from MSMT17_ZIP=$MSMT17_ZIP"
    extract_any "$MSMT17_ZIP" "$base/_extract"
  elif [[ -n "${MSMT17_URL:-}" ]]; then
    log "INFO: Download MSMT17 from MSMT17_URL (must be obtained per agreement)"
    mkdir -p "$base/_dl"
    dl_http "$MSMT17_URL" "$base/_dl/MSMT17.zip"
    extract_any "$base/_dl/MSMT17.zip" "$base/_extract"
  else
    cat > "$base/README_MANUAL_DOWNLOAD.txt" <<'TXT'
MSMT17 access usually requires signing the official agreement and requesting the download link.
After you obtain the official link or zip:
  Option A: set MSMT17_URL and rerun scripts/download_datasets.sh
  Option B: set MSMT17_ZIP=/path/to/MSMT17.zip and rerun scripts/download_datasets.sh
Expected structure after extraction:
  data/msmt17/MSMT17_V1/ or data/msmt17/MSMT17_V2/
TXT
    log "OK: MSMT17 placeholder created (manual agreement required)."
    return 0
  fi

  local v1=""
  v1="$(find_dir "$base/_extract" "MSMT17_V1" 2>/dev/null || true)"
  if [[ -n "$v1" && -d "$v1" ]]; then
    rm -rf "$base/MSMT17_V1"
    mv "$v1" "$base/MSMT17_V1"
    log "OK: MSMT17_V1 placed."
    return 0
  fi
  local v2=""
  v2="$(find_dir "$base/_extract" "MSMT17_V2" 2>/dev/null || true)"
  if [[ -n "$v2" && -d "$v2" ]]; then
    rm -rf "$base/MSMT17_V2"
    mv "$v2" "$base/MSMT17_V2"
    log "OK: MSMT17_V2 placed."
    return 0
  fi

  log "WARN: MSMT17 extracted but V1/V2 folder not found. Please inspect data/msmt17/_extract."
  return 1
}

# ---------- Download strategies ----------
try_kaggle(){
  local slug="$1"; local outdir="$2"
  if ! command -v kaggle >/dev/null 2>&1; then return 1; fi

  if [[ ! -f "$HOME/.kaggle/kaggle.json" && (-z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}") ]]; then
    return 1
  fi

  mkdir -p "$outdir/_kaggle"
  log "INFO: Kaggle download: $slug"
  kaggle datasets download -d "$slug" -p "$outdir/_kaggle" --unzip >/dev/null
  return 0
}

try_gdown(){
  local url="$1"; local out="$2"
  if ! command -v gdown >/dev/null 2>&1; then return 1; fi
  mkdir -p "$(dirname "$out")"
  log "INFO: gdown: $url"
  gdown --fuzzy "$url" -O "$out" >/dev/null
  return 0
}

try_torrent(){
  local torrent_url="$1"; local outdir="$2"
  if ! command -v aria2c >/dev/null 2>&1; then return 1; fi
  mkdir -p "$outdir/_torrent"
  log "INFO: aria2c torrent: $torrent_url"
  aria2c --seed-time=0 --bt-stop-timeout=60 -d "$outdir/_torrent" "$torrent_url" >/dev/null
  return 0
}

# ---------- Market-1501 ----------
download_market1501(){
  local base="$DATA_DIR/market1501"
  mkdir -p "$base"

  if ensure_market_layout; then return 0; fi

  log "STEP: Market-1501 download attempts"

  try_kaggle "pengcw1/market-1501" "$base" || true

  try_torrent "https://academictorrents.com/download/3ea1f8ae1d3155addff586a96006d122587663ee.torrent" "$base" || true
  if compgen -G "$base/_torrent/*.zip" >/dev/null; then
    for z in "$base"/_torrent/*.zip; do
      log "INFO: extract $z"
      extract_any "$z" "$base/_extract"
    done
  fi

  try_gdown "https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view" "$base/Market-1501-v15.09.15.zip" || true
  if [[ -f "$base/Market-1501-v15.09.15.zip" ]]; then
    extract_any "$base/Market-1501-v15.09.15.zip" "$base/_extract"
  fi

  if [[ -d "$base/_extract" ]]; then
    if [[ -d "$base/_extract/Market-1501-v15.09.15" ]]; then
      rm -rf "$base/Market-1501-v15.09.15"
      mv "$base/_extract/Market-1501-v15.09.15" "$base/Market-1501-v15.09.15"
    fi
    mdir="$(find_dir "$base/_extract" "Market-1501-v15.09.15" 2>/dev/null || true)"
    if [[ -n "$mdir" ]]; then
      rm -rf "$base/Market-1501-v15.09.15"
      mv "$mdir" "$base/Market-1501-v15.09.15"
    fi
  fi
  if [[ -d "$base/_kaggle/Market-1501-v15.09.15" && ! -d "$base/Market-1501-v15.09.15" ]]; then
    mv "$base/_kaggle/Market-1501-v15.09.15" "$base/Market-1501-v15.09.15"
  fi

  ensure_market_layout && return 0

  log "WARN: Market-1501 not prepared. Manual fallback:"
  log "  Put Market-1501-v15.09.15.zip under data/market1501/ and rerun."
  return 1
}

# ---------- DukeMTMC-reID ----------
download_duke(){
  local base="$DATA_DIR/dukemtmc-reid"
  mkdir -p "$base"

  if ensure_duke_layout; then return 0; fi

  log "STEP: DukeMTMC-reID download attempts"

  try_kaggle "igorkrashenyi/dukemtmc-reid" "$base" || true

  dl_http "https://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip" "$base/DukeMTMC-reID.zip" || true
  if [[ -f "$base/DukeMTMC-reID.zip" ]]; then
    extract_any "$base/DukeMTMC-reID.zip" "$base/_extract"
  fi

  try_gdown "https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view?usp=drive_open" "$base/DukeMTMC-reID.zip" || true
  if [[ -f "$base/DukeMTMC-reID.zip" ]]; then
    extract_any "$base/DukeMTMC-reID.zip" "$base/_extract"
  fi

  try_torrent "https://academictorrents.com/download/00099d85f6d8e8134b47b301b64349f469303990.torrent" "$base" || true
  if compgen -G "$base/_torrent/*.zip" >/dev/null; then
    for z in "$base"/_torrent/*.zip; do
      log "INFO: extract $z"
      extract_any "$z" "$base/_extract"
    done
  fi

  if [[ -d "$base/_extract" ]]; then
    if [[ -d "$base/_extract/DukeMTMC-reID" ]]; then
      rm -rf "$base/DukeMTMC-reID"
      mv "$base/_extract/DukeMTMC-reID" "$base/DukeMTMC-reID"
    fi
    ddir="$(find_dir "$base/_extract" "DukeMTMC-reID" 2>/dev/null || true)"
    if [[ -n "$ddir" ]]; then
      rm -rf "$base/DukeMTMC-reID"
      mv "$ddir" "$base/DukeMTMC-reID"
    fi
  fi
  if [[ -d "$base/_kaggle/bounding_box_train" && -d "$base/_kaggle/bounding_box_test" && -d "$base/_kaggle/query" && ! -d "$base/DukeMTMC-reID" ]]; then
    mkdir -p "$base/DukeMTMC-reID"
    mv "$base/_kaggle/bounding_box_train" "$base/DukeMTMC-reID/"
    mv "$base/_kaggle/bounding_box_test" "$base/DukeMTMC-reID/"
    mv "$base/_kaggle/query" "$base/DukeMTMC-reID/"
    for f in "$base"/_kaggle/*.txt "$base"/_kaggle/*.md; do
      if [[ -f "$f" ]]; then
        mv "$f" "$base/DukeMTMC-reID/"
      fi
    done
  fi

  ensure_duke_layout && return 0

  log "WARN: DukeMTMC-reID not prepared. Manual fallback:"
  log "  Put DukeMTMC-reID.zip under data/dukemtmc-reid/ and rerun."
  return 1
}

main(){
  need_cmd python || exit 2
  need_cmd 7z || log "WARN: 7z missing, will use python zipfile for zip only."

  local market_ok=0
  local duke_ok=0
  download_market1501 && market_ok=1 || market_ok=0
  download_duke && duke_ok=1 || duke_ok=0
  ensure_msmt_layout || true

  log "DONE: dataset preparation finished."
  log "Expected:"
  log "  data/market1501/Market-1501-v15.09.15/{bounding_box_train,bounding_box_test,query}"
  log "  data/dukemtmc-reid/DukeMTMC-reID/{bounding_box_train,bounding_box_test,query}"
  log "  data/msmt17/MSMT17_V1|MSMT17_V2/..."

  if [[ "$market_ok" -eq 0 || "$duke_ok" -eq 0 ]]; then
    log "WARN: Some datasets are missing. See logs above for manual fallback."
  fi
}

main "$@"


