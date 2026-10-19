#!/bin/bash
# bench/hpn/hpn_local.sh REPO OUT -- HPN's hand run on the lab's A2000 (bench/hpn/HPN-DESIGN.md): inside the gpu-dev container, with the
# card's exclusive claim already held. It stages the files tc1_run.sh's local-box mode needs (README "TC3 hand run") from a CLEAN checkout
# (the receipts cite its commit), records the free GPU memory at the start (condition c), and runs TC1_FAMILIES=qwen3hpn with TC1_LOCAL_BOX=1.
# Knobs: HPN_DEPTHS (default "2 4 8 16"), HPN_UNS_BACKEND (default native_torch), HPN_SNAPSHOT, HPN_PYTHON, GNF4_SHA.
set -euo pipefail
REPO=${1:?usage: hpn_local.sh REPO OUT}
OUT=${2:?usage: hpn_local.sh REPO OUT}
E4B_SHA=$(git -C "$REPO" rev-parse HEAD)
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo "refusing: $REPO has uncommitted changes (the receipts cite $E4B_SHA)"; exit 78; }
mkdir -p "$OUT"
find "$REPO/bench/tc1" -maxdepth 1 -type f -exec cp {} "$OUT"/ \;
cp "$REPO/bench/tp4/tp4_alpaca.py" "$REPO/bench/flagship-matrix/drivers/n9_datasets.py" "$REPO/bench/flagship-matrix/ds_manifest.json" "$OUT"/
for f in hp1_lib.py hp1_eval.py hp1_caltokens.py hp1_trunc.py hp1_reduce.py; do cp "$REPO/bench/hp1/$f" "$OUT"/; done
cp "$REPO/bench/hpn/hpn_read.py" "$REPO/bench/hpn/hpn_reduce.py" "$OUT"/
mkdir -p "$OUT/hpn" && cp "$REPO/bench/hpn/HPN-DESIGN.md" "$OUT/hpn/"   # the note the receipts cite as their prereg
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
echo "HPN free GPU memory at start: ${FREE:-unknown} MiB (e4b $E4B_SHA)"
cd "$OUT"
TC1_LOCAL_BOX=1 TC1_LOCAL_OUT="$OUT" TC1_LOCAL_SNAPSHOT="${HPN_SNAPSHOT:-/models/Qwen3-30B-A3B}" \
  TC1_LOCAL_PYTHON="${HPN_PYTHON:-/workspace/venv-k3rel/bin/python}" TC1_BOX=A TC1_FAMILIES=qwen3hpn TC1_GPU_CLASS="RTX A2000" \
  HPN_FREE_MIB="${FREE:-}" HPN_UNS_BACKEND="${HPN_UNS_BACKEND:-native_torch}" \
  E4B_SHA="$E4B_SHA" GNF4_SHA="${GNF4_SHA:-724ccc454f006c1a46836e434e997f31f293747f}" \
  bash tc1_run.sh > outer.log 2>&1
echo "HPN done: $OUT/hpn/result.json ($(grep -m1 '^HPN=' summary.txt 2>/dev/null || echo 'no HPN line'))"
