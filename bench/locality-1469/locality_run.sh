#!/bin/bash
# bench/locality-1469/locality_run.sh -- #1469 item 1 (the expert-locality census), BOX side. Started detached by
# locality_drive.sh with the run's nonce; LOC_RUN_NONCE first, then LOC_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every
# exit and LOC_SUCCESS.<nonce> only when the census ran. Derived from bench/k34/k34_run.sh (refusals, nonce, markers).
#
# One rented RTX A2000 (sm_86: e4b's NF4 host-residency path, not the fp8 paged runner). Expert ids are correctness-class
# data; no timing from this box is quoted. Steps:
#   install   experts4bit-qlora at E4B_SHA (the census branch) + datasets; grouped-nf4-gemm cloned at GNF4_SHA and
#             installed over the released one (its bench/cold-engine/routing-trace/capture_routing.py is reused)
#   tripwire  the loader, the pipelined residency engine and the expert profile import; versions.txt
#   fetch     Qwen/Qwen3-30B-A3B at its pinned revision (bf16, ~61 GB; quantized to NF4 on load)
#   self-test locality_capture.py and locality_summary.py (rc 21)
#   census    calibrate (profile -> hot sets) then census (decode x 4 prompt kinds, prefill windows) then summaries (rc 34)
# Host refusals name the machine (15 class, 13 disk, 16 RAM, 18 cuda unusable, 10 dud); lane failures are 9 install,
# 11 fetch, 21 self-test, 34 census.
# Knobs (any off its default marks a REHEARSAL): LOC_GPU_CLASS LOC_MIN_DISK_GB LOC_STEPS LOC_REHEARSAL.
set -uo pipefail
W=/root/loc; mkdir -p $W/logs $W/out; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] loc: $*"; }
NONCE=${LOC_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/LOC_RUN_NONCE.tmp && mv $W/LOC_RUN_NONCE.tmp $W/LOC_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > LOC_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > LOC_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in LOC_RUN_ID LOC_DEADLINE_EPOCH LOC_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA; do
  case "${!v}" in *[!0-9a-f]*) say "refusing: $v is not hex"; finish 78;; esac
  val=${!v}; [ ${#val} = 40 ] || { say "refusing: $v must be a 40-char sha"; finish 78; }
done
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
GPU_CLASS=${LOC_GPU_CLASS:-A2000}; MIN_DISK_GB=${LOC_MIN_DISK_GB:-150}; MIN_RAM_GB=64; STEPS=${LOC_STEPS:-512}
REHEARSAL=${LOC_REHEARSAL:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$W/hf
unset E4B_EXPERT_PROFILE E4B_RESIDENCY E4B_HOT_PROFILE E4B_HOT_PER_LAYER E4B_K_SLOTS E4B_FUSE_ROUTER_EPI TRITON_INTERPRET
: > summary.txt; echo "$LOC_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB steps=$STEPS" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != A2000 ] || [ "$MIN_DISK_GB" != 150 ] || [ "$STEPS" != 512 ]; then
  echo "REHEARSAL -- NOT the census: a knob is off its default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
[ -s $W/staged.sha256 ] || { say "STAGE MISSING: staged.sha256"; finish 9; }
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/locality-1469/staged.sha256"; finish 9; }
# A GPU the image's torch cannot use is the registered host floor (18, names the machine); a torch that will not import
# is the image's fault (10). As K34's runner.
CUDA_PROBE=$(python - <<'PYC' 2>/dev/null | tail -1
import sys
try:
    import torch
except Exception as e:
    print(f"no-torch {type(e).__name__}"); sys.exit(0)
try:
    ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
except Exception:
    ok = False
print("ok" if ok else f"no-cuda torch {torch.__version__}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  ok) ;;
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE}) -- registered host floor"
            echo "refused: cuda unusable (${CUDA_PROBE})" > REFUSAL; echo "BOX_REFUSED cuda=unusable" >> summary.txt; finish 18;;
  *) say "DUD BOX (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the census rents the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(free -g | awk '/Mem:/{print $2}')
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GB RAM < $MIN_RAM_GB (15 GB of experts are pinned in host RAM)"; echo "refused: ram ${RAM_GB:-?} GB" > REFUSAL; finish 16; }
arm_alarm(){ local cap=$1 left=$(( LOC_DEADLINE_EPOCH - $(date +%s) - 300 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 300 ] && left=300; echo "$left"; }
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input "$@" >> $log 2>&1; }
say "install experts4bit-qlora @$E4B_SHA + datasets, then grouped-nf4-gemm @$GNF4_SHA (as SC1's runner)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" datasets \
  || { tail -3 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4 > logs/clone.log 2>&1 \
  && git -C $W/gnf4 checkout -q $GNF4_SHA || { say "CLONE FAIL"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps $W/gnf4 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
python - <<'PYT' > versions.txt 2> logs/tripwire.log || { cat logs/tripwire.log | tail -3; say "TRIPWIRE FAIL"; finish 9; }
import importlib.metadata as md
import torch
import transformers
from experts4bit_qlora.engines.expert_profile import attach, flush, hot_sets_from_profile  # noqa: F401
from experts4bit_qlora.engines.pipelined import enable_pipelined_residency  # noqa: F401
from experts4bit_qlora.loader import load_moe_4bit_streaming  # noqa: F401
assert torch.cuda.is_available(), "the install replaced the image's CUDA torch"
for d in ("experts4bit-qlora", "grouped-nf4-gemm"):
    print(d, md.version(d))
print("torch", torch.__version__, "transformers", transformers.__version__, "cc", torch.cuda.get_device_capability())
PYT
cat versions.txt | tee -a summary.txt
say "fetch $MODEL @ $REV"
perl -e "alarm $(arm_alarm 3600); exec @ARGV" python -c "from huggingface_hub import snapshot_download; open('$W/SNAP', 'w').write(snapshot_download('$MODEL', revision='$REV'))" \
  > logs/fetch.log 2>&1 && [ -s $W/SNAP ] || { tail -3 logs/fetch.log; say "FETCH FAIL"; finish 11; }
SNAP=$(cat $W/SNAP); echo "FETCH $SNAP" | tee -a summary.txt
python $W/locality_capture.py --self-test | tee -a summary.txt && python $W/locality_summary.py --self-test | tee -a summary.txt \
  || { say "SELF-TEST FAILED"; finish 21; }
say "calibrate (alarm $(arm_alarm 3600))"
perl -e "alarm $(arm_alarm 3600); exec @ARGV" python $W/locality_capture.py --phase calibrate --revision $REV --model-path "$SNAP" \
  --out-dir $W/out --hot-profile $W/out/calibration_profile.jsonl > logs/calibrate.log 2>&1 \
  || { tail -3 logs/calibrate.log; say "CALIBRATE FAILED"; finish 34; }
grep -a "^LOCALITY_CALIBRATE" logs/calibrate.log | tee -a summary.txt
say "census, $STEPS steps x 4 prompt kinds (alarm $(arm_alarm 9000))"
perl -e "alarm $(arm_alarm 9000); exec @ARGV" python $W/locality_capture.py --phase census --revision $REV --model-path "$SNAP" \
  --out-dir $W/out --trace-dir $W/gnf4/bench/cold-engine/routing-trace --hot-profile $W/out/calibration_profile.jsonl \
  --steps $STEPS > logs/census.log 2>&1 || { tail -3 logs/census.log; say "CENSUS FAILED"; finish 34; }
grep -a "^LOCALITY_CAPTURE" logs/census.log | tee -a summary.txt
python $W/locality_summary.py --npz $W/out/decode.npz --manifest $W/out/manifest.json --phase decode --out $W/out/summary_decode.json \
  > logs/summary.log 2>&1 && python $W/locality_summary.py --npz $W/out/prefill.npz --phase prefill --out $W/out/summary_prefill.json \
  >> logs/summary.log 2>&1 || { tail -3 logs/summary.log; say "SUMMARY FAILED"; finish 34; }
grep -a "^LOCALITY_SUMMARY" logs/summary.log | tee -a summary.txt
finish 0
