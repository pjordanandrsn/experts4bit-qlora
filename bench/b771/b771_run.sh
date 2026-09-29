#!/bin/bash
# bench/b771/b771_run.sh -- lane B771, BOX side (bench/b771/PREREG-b771.md; e4b#771). Derived from bench/b771/b771_run.sh
# (preflight, install, tripwire, fetch, bake, the arm loop): the same NF4 Qwen3-30B-A3B, prompts and trace, with
# two grouped-nf4-gemm cuts in one process tree. Under the OLD cut (GNF4_SHA, the fused fp8 append as shipped): the
# byte stage (b771_bytes.py) and arms A_old (eager) and P_old (the bucket step, eager). Then the NEW cut
# (GNF4_NEW_SHA, the append with an IEEE-rounded quotient) is force-installed over it: the byte stage again and arms
# P_new and A_new. Every arm runs the device grouping. The harness is P81's step_decomp.py, unchanged.
# Started detached by b771_drive.sh with the run's nonce; B771_RUN_NONCE first, then B771_EXIT_CODE.<nonce> +
# TP_DONE.<nonce> on every exit and B771_SUCCESS.<nonce> only when the reducer ran.
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# B771_MODEL B771_REVISION B771_GPU_CLASS B771_MIN_DISK_GB B771_SEG B771_SKIP_INSTALL B771_REHEARSAL.
set -uo pipefail
W=/root/b771; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] b771: $*"; }
NONCE=${B771_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/B771_RUN_NONCE.tmp && mv $W/B771_RUN_NONCE.tmp $W/B771_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > B771_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > B771_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in B771_RUN_ID B771_DEADLINE_EPOCH B771_INSTANCE_ID E4B_SHA GNF4_SHA GNF4_NEW_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA GNF4_NEW_SHA; do case "${!v}" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac; done
[ ${#E4B_SHA} -eq 40 ] && [ ${#GNF4_SHA} -eq 40 ] && [ ${#GNF4_NEW_SHA} -eq 40 ] || { say "refusing: a pin is not a 40-char sha"; finish 78; }
# ---- registered defaults (PREREG-b771 "Instrument"); a rehearsal overrides them and says so
D_MODEL=Qwen/Qwen3-30B-A3B; D_REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
MID=${B771_MODEL:-$D_MODEL}; REV=${B771_REVISION:-$D_REV}; GPU_CLASS=${B771_GPU_CLASS:-5090}; MIN_DISK_GB=${B771_MIN_DISK_GB:-130}
SEG=${B771_SEG:-32}; SKIP_INSTALL=${B771_SKIP_INSTALL:-0}; REHEARSAL=${B771_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false E4B_MODEL_ID=$MID
# every lever this lane does not read stays at its shipped default (NF4 only; nothing int4, no folds, no fused router)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_LMHEAD_INT4_CALIB E4B_SERVE_DENSE_INT4_CALIB \
      E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 E4B_FUSE_ROUTER_EPI E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST
: > summary.txt; echo "$B771_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS model=$MID rev=$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB seg=$SEG skip_install=$SKIP_INSTALL" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$MID" != "$D_MODEL" ] || [ "$REV" != "$D_REV" ] || [ "$GPU_CLASS" != 5090 ] || [ "$SEG" != 32 ] \
   || [ "$SKIP_INSTALL" != 0 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in b771_reduce.py b771_bytes.py step_decomp.py k8_bake.py calib.json staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/b771/staged.sha256"; finish 9; }
python $W/b771_reduce.py --self-test | tee -a summary.txt || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- refusals before anything is fetched: the card class and the disk the working set needs (host-limited: 13)
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the bf16 checkpoint + the NF4 arena; overlay is not machine disk)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
# ---- deadline guard (P54's): never start a step that cannot finish 10 min before teardown
arm_alarm(){ local left=$(( B771_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 600 ] && left=600; echo "$left"; }
# ---- install: e4b pinned + P37's toolchain pins (image python); gnf4 at the registered cut
export DEBIAN_FRONTEND=noninteractive
if [ "$SKIP_INSTALL" = 0 ]; then
  say "install e4b @$E4B_SHA (image python; P37 pins) + gnf4 @$GNF4_SHA"
  perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
    "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
  perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
else
  say "install SKIPPED (B771_SKIP_INSTALL=1): the importable packages are whatever the environment provides -- recorded below"
fi
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, sys, inspect, importlib.metadata as md
sys.path.insert(0, "/root/b771")
from experts4bit_qlora.engines.paged_runner import PagedModelRunner, bucket_for  # noqa: F401
assert hasattr(PagedModelRunner, "enable_decode_graphs"), "e4b cut lacks enable_decode_graphs (#757)"
from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
assert "scratch_slots" in inspect.signature(Fp8PagedKV.__init__).parameters, "e4b cut lacks Fp8PagedKV(scratch_slots=)"
from experts4bit_qlora.engines import hot_residency as hr
assert hr.DEVICE_GROUPING[0] is False and hr.FORCE_SINGLETON_GROUPS[0] is False, "grouping flags are not at their shipped defaults"
import fp8_kv
assert hasattr(fp8_kv, "fp8_kv_append_bt1"), "gnf4 cut lacks the fused batch KV append the graphs need"
import step_decomp
assert hasattr(step_decomp, "_dynb_plan") and hasattr(step_decomp, "_dynb_stage"), "staged step_decomp lacks the P80 stage"
assert "--dynb-grouping" in open("/root/b771/step_decomp.py").read(), "staged step_decomp is not P81's (no --dynb-grouping)"
import fp8_kv as _fk
assert not hasattr(_fk, "_e4m3_group"), "the OLD grouped-nf4-gemm cut already has the fix -- wrong GNF4_SHA"
import experts4bit_qlora as e, torch, triton, transformers
gnf4 = md.version("grouped-nf4-gemm")
open("/root/b771/versions.txt", "w").write(f"e4b {e.__version__} @{os.environ['E4B_SHA']} (from {os.path.dirname(e.__file__)})\ngnf4 {gnf4} @{os.environ['GNF4_SHA']}\n"
                                          f"torch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n"
                                          f"cc {torch.cuda.get_device_capability()}\n")
print("tripwire OK: e4b", e.__version__, "gnf4", gnf4, "torch", torch.__version__, "triton", triton.__version__)
PYT
cat versions.txt | tee -a summary.txt
if [ "${B771_PROVE:-0}" = 1 ]; then
  echo "PROVE -- the proving run: no model, no arm" | tee -a summary.txt
  say "PROVE: HF CDN egress probe (50 MB range, 20 s cap; recorded, not a refusal)"
  python - <<'PYE' 2>&1 | tail -1 | tee -a summary.txt forensics.txt
import time, urllib.request
t = time.time()
try:
    r = urllib.request.urlopen(urllib.request.Request("https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors",
                                                      headers={"Range": "bytes=0-52428799"}), timeout=20)
    n = len(r.read())
    print(f"PROVE hf_cdn_mbps={n / (time.time() - t) / 1e6:.1f} bytes={n}")
except Exception as e:
    print(f"PROVE hf_cdn_probe_failed {type(e).__name__}: {str(e)[:120]}")
PYE
  : > PROVED; finish 0
fi
# ---- fetch (pinned) and bake (P39's k8_bake.py), as P70
say "fetch $MID @ $REV"
perl -e "alarm $(arm_alarm); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
bake_fail(){ say "BAKE FAIL ($1)"; python -c "import json; r=json.load(open('$W/work/bake.json')); print('BAKE_ERR', r.get('status'), r.get('step')); print(r.get('err')); print(r.get('tb'))" 2>/dev/null | tee -a logs/bake.log | tail -n 25 | cut -c1-400; finish 12; }
QA=$W/work/nf4.arena
say "bake NF4 arena"; mkdir -p $W/work
K8_MODEL="$MID" K8_WORK="$W/work" perl -e "alarm $(arm_alarm); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; bake_fail rc; }
[ -e "$QA" ] || bake_fail "no arena"
grep -aE "arena|BAKE" logs/bake.log | tail -1 | tee -a summary.txt
# ---- the byte stage, the old-cut arms, the swap, the byte stage again, the new-cut arms
bytes_stage(){ local TAG=$1
  say "byte stage ($TAG)"
  perl -e "alarm $(arm_alarm); exec @ARGV" python $W/b771_bytes.py --out $W/bytes_$TAG.json > logs/bytes_$TAG.log 2>&1
  { echo -n "bytes $TAG rc=$? "; grep -a "B771_BYTES\|Error" logs/bytes_$TAG.log | tail -1 | cut -c1-300; echo; } | tee -a summary.txt; }
arm(){ local ARM=$1 MODE=$2 TAG=$3
  say "arm $ARM (mode=$MODE gnf4=$TAG seg=$SEG)"
  perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json \
      --placement-override all-vram --amort off --batch 16 --prompt-len 512 --no-fuse-qkv \
      --dynb-mode "$MODE" --dynb-grouping device --dynb-arm "$ARM" --dynb-seg "$SEG" --out $W/b771_$ARM.json > logs/arm_$ARM.log 2>&1
  local rc=$?
  if [ ! -s $W/b771_$ARM.json ]; then
    python -c "import json,sys; json.dump({'arm':'$ARM','mode':'$MODE','status':'error','rc':$rc,'tail':sys.argv[1]}, open('$W/b771_$ARM.json','w'))" "$(tail -3 logs/arm_$ARM.log | tr '\n' ' ' | cut -c1-600)"
  fi
  # provenance stamped by the runner, not measured: which cut this arm ran under (the reducer checks it)
  python - "$W/b771_$ARM.json" "$TAG" <<'PYS'
import json, sys, importlib.metadata as md
import fp8_kv
p, tag = sys.argv[1], sys.argv[2]
r = json.load(open(p))
r["gnf4_tag"], r["gnf4_version"], r["gnf4_has_e4m3_group"] = tag, md.version("grouped-nf4-gemm"), hasattr(fp8_kv, "_e4m3_group")
json.dump(r, open(p, "w"), indent=1)
PYS
  { echo -n "arm $ARM rc=$rc "; grep -a "P81_DYNB\|TRACE MISMATCH\|Error" logs/arm_$ARM.log | tail -1 | cut -c1-400; echo; } | tee -a summary.txt
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; }
bytes_stage old
arm A_old eager old
arm P_old padded old
say "swap grouped-nf4-gemm to the new cut @$GNF4_NEW_SHA"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_NEW_SHA" > logs/pip_gnf4_new.log 2>&1 || { tail -3 logs/pip_gnf4_new.log; say "PIP FAIL (gnf4 new)"; finish 9; }
python -c "import fp8_kv, importlib.metadata as md; assert hasattr(fp8_kv, '_e4m3_group'), 'the new cut lacks the fix'; print('gnf4 new', md.version('grouped-nf4-gemm'))" 2>&1 | tee -a summary.txt versions.txt | tail -1
python -c "import fp8_kv; assert hasattr(fp8_kv, '_e4m3_group')" 2>/dev/null || { say "SWAP TRIPWIRE FAIL"; finish 9; }
bytes_stage new
arm P_new padded new
arm A_new eager new
python $W/b771_reduce.py --dir $W --out $W/verdict.json 2>&1 | tee -a summary.txt
[ -s $W/verdict.json ] || { say "REDUCER wrote no verdict"; finish 22; }
finish 0
