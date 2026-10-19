#!/bin/bash
# P147's runner on the lab's RTX A2000 (bench/p147/PREREG-p147.md), inside the gpu-dev container; p147_drive.sh stages
# this directory and starts it under the lane's gpu:a2000 claim. No timing is recorded or reported.
#   1. the start floor: >= 10,000 MiB free on the card and no other compute process, else NOT_RUN (75), nothing else;
#   2. the staged closure against staged-p147.sha256 (a mismatch is VOID, 9);
#   3. the pinned environment: torch 2.8.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2, e4b at E4B_SHA,
#      grouped-nf4-gemm v0.47.1 (38b30888), checked by a tripwire (9);
#   4. p147_box.py (shards, windows, the arms; 13 = a shard off eba92302, 12 = an instrument check failed), then
#      p147_reduce.py, which writes verdict_p147.json whatever the box's exit;
#   5. P147_DONE carries the box's exit code.
set -uo pipefail
W=$(cd "$(dirname "$0")" && pwd); cd "$W" || exit 9
MODEL=${P147_MODEL:-/models/Mixtral-8x7B-Instruct-v0.1}
GNF4_SHA=38b308887688ea5bda7dd449bd3cc3dc1426b995
FREE_MIB_MIN=10000
E4B_SHA=${E4B_SHA:?the registration merge commit}
[[ $E4B_SHA =~ ^[0-9a-f]{40}$ ]] || { echo "refusing: E4B_SHA is not a full commit"; exit 78; }
say(){ echo "[$(date -u +%FT%TZ)] p147: $*" | tee -a summary.txt; }
done_(){ echo "$1" > P147_DONE; exit "$1"; }
: > summary.txt
# 1. the start floor (the card is shared)
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')
if [ "${free:-0}" -lt $FREE_MIB_MIN ] || [ "$apps" -ne 0 ]; then
  say "NOT_RUN: the card is busy (free ${free:-?} MiB < $FREE_MIB_MIN or $apps compute process(es)); nothing ran"
  echo "card busy" > NOT_RUN; done_ 75
fi
say "start floor: ${free} MiB free, $apps compute processes"
# 2. the staged closure
sha256sum -c staged-p147.sha256 > closure.log 2>&1 || { tail -3 closure.log; say "VOID: the staged closure does not match"; done_ 9; }
# 3. the environment
export HF_HOME=${P147_HF_HOME:-$W/../hf} HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
VENV=${P147_VENV:-$W/../venv}
if [ ! -x "$VENV/bin/python" ]; then python3 -m venv "$VENV" || { say "VOID: venv"; done_ 9; }; fi
PY="$VENV/bin/python"
{ "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
  "$PY" -m pip install -q "transformers==5.17.0" "bitsandbytes==0.50.2" datasets safetensors
  "$PY" -m pip install -q "experts4bit-qlora[train] @ git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA"
  # the venv is shared across run ids: a same-version VCS install is a no-op, so both pins are forced (the tripwire checks)
  "$PY" -m pip install -q --force-reinstall --no-deps "grouped-nf4-gemm @ git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA"
  "$PY" -m pip install -q --force-reinstall --no-deps "experts4bit-qlora @ git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA"
} > install.log 2>&1 || { tail -5 install.log; say "VOID: install"; done_ 9; }
"$PY" - <<PYT > versions.txt 2>&1 || { cat versions.txt; say "VOID: tripwire"; done_ 9; }
import importlib.metadata as md, torch, transformers, bitsandbytes
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert transformers.__version__ == "5.17.0" and bitsandbytes.__version__ == "0.50.2"
assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA RTX A2000 12GB", torch.cuda.get_device_name(0)
e4b = md.distribution("experts4bit-qlora").read_text("direct_url.json") or ""
gnf4 = md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or ""
assert "$E4B_SHA" in e4b and "$GNF4_SHA" in gnf4, (e4b, gnf4)
import experts4bit_qlora, nf4_grouped   # grouped-nf4-gemm's modules are nf4_grouped / nf4_qlora
print("torch", torch.__version__, "transformers", transformers.__version__, "bitsandbytes", bitsandbytes.__version__)
print("e4b", "$E4B_SHA", "gnf4", md.version("grouped-nf4-gemm"), "$GNF4_SHA")
PYT
cat versions.txt | tee -a summary.txt
for s in p147_fakequant.py p147_reduce.py p147_box.py; do
  "$PY" "$s" --self-test >> summary.txt 2>&1 || { tail -2 summary.txt; say "VOID: $s self-test"; done_ 9; }
done
# 4. the box, then the reduce
SK=""; [ "${P147_SHAKEDOWN:-0}" = 1 ] && SK=--shakedown        # a dry id: no real window is scored
"$PY" p147_box.py --model "$MODEL" --lock model_files.json --ref quality_off_mixtral.json --out out $SK > box.log 2>&1; rc=$?
grep -a "^P147_" box.log | cut -c1-400 | tee -a summary.txt
say "box exit $rc"
[ -n "$SK" ] && done_ "$rc"
"$PY" p147_reduce.py --dir out --ref quality_off_mixtral.json --out out/verdict_p147.json 2>&1 | tail -1 | tee -a summary.txt
done_ "$rc"
