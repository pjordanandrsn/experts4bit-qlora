#!/bin/bash
# One-time setup, run UNCAPPED. Nothing here is part of the claim: it downloads
# the checkpoint, bakes the arena, and warms the dataset cache. Doing it now
# means the capped runs need no network and no 13.84 GB transient, so a failure
# under a cap is attributable to the thing being measured rather than to a
# download that happened to land at the wrong moment.
set -uo pipefail
V=/work/venv
echo "=============== PREP $(date -u) ==============="

# The original run created these by hand on the host before the container
# started, so the script never needed them and the omission was invisible until
# review. On a fresh mount -- which is exactly what the Reproducing section tells
# you to use -- the bake writes into a directory that does not exist and fails.
mkdir -p /work/arena /work/hf || exit 1

if [ ! -x $V/bin/python ]; then
  # --system-site-packages keeps the image's CUDA-matched torch. A fresh venv
  # would pull a generic wheel and change what is being measured.
  python -m venv --system-site-packages $V || exit 1
fi
$V/bin/python -c "import torch;print('torch:',torch.__version__,'cuda:',torch.cuda.is_available())" || exit 1

TV=$($V/bin/python -c "import torch;print(torch.__version__.split('+')[0])")
echo "torch==$TV" > /work/constraints.txt
echo "--- installing (torch pinned to $TV) ---"
$V/bin/pip install -q --no-cache-dir -c /work/constraints.txt \
  grouped-nf4-gemm "experts4bit-qlora[train]" 2>&1 | tail -15
echo "pip rc=$?"

$V/bin/python - <<'PY'
import importlib.metadata as md, torch
print("torch AFTER:", torch.__version__, "cuda_avail:", torch.cuda.is_available())
try: print("device:", torch.cuda.get_device_name(0))
except Exception as e: print("device: ERROR", e)
for p in ("grouped-nf4-gemm","experts4bit-qlora","transformers","datasets","peft","bitsandbytes","accelerate"):
    try: print(f"  {p} == {md.version(p)}")
    except Exception as e: print(f"  {p} == MISSING ({e.__class__.__name__})")
import experts4bit_qlora as E
for n in ("load_moe_4bit_streaming","enable_nvme_train_residency","enable_fast_train"):
    assert hasattr(E,n), f"e4b missing {n}"
print("  e4b entrypoints OK")
import nvme_bake_nf4, nvme_arena
print("  gnf4 nvme modules OK")
PY
[ $? -ne 0 ] && { echo "PREP FAILED: import check"; exit 1; }

MODEL=${E4B_MODEL:-allenai/OLMoE-1B-7B-0924}
ARENA=/work/arena/olmoe-nf4.arena

echo "--- snapshot + dataset warm ---"
SNAP=$($V/bin/python - <<PY
from huggingface_hub import snapshot_download
from datasets import load_dataset
load_dataset("tatsu-lab/alpaca", split="train")
print(snapshot_download("$MODEL"))
PY
)
RC=$?; SNAP=$(echo "$SNAP" | tail -1)
[ $RC -ne 0 ] && { echo "PREP FAILED: snapshot/dataset rc=$RC"; exit 1; }
echo "snapshot: $SNAP"
du -sh "$SNAP" 2>/dev/null

if [ -s "$ARENA" ]; then
  echo "--- arena already baked ---"
else
  echo "--- baking arena -> $ARENA ---"
  $V/bin/python -m nvme_bake_nf4 --snapshot "$SNAP" --out "$ARENA" 2>&1 | tail -15
  [ ${PIPESTATUS[0]} -ne 0 ] && { echo "PREP FAILED: bake"; exit 1; }
fi
ls -l /work/arena/
echo "arena_bytes=$(stat -c %s "$ARENA" 2>/dev/null)"
echo "=============== PREP OK ==============="
