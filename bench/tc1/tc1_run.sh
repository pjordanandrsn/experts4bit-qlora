#!/bin/bash
# bench/tc1/tc1_run.sh -- lane TC1, BOX side (bench/tc1/TC1-PREREG.md). Started detached by tc1_drive.sh with the run's
# nonce; writes TC1_RUN_NONCE first, then TC1_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and TC1_SUCCESS.<nonce>
# only when the plan completed (every arm reached a row and the reducer ran).
#
# PROVENANCE: copied from bench/tp4/tc1_run.sh @ e4b main 10ce711d (the tp4 tree is never edited). Kept: the nonce handshake,
# the box refusals (GPU class, overlay disk), the deadline-derived alarms, the venv installs + tripwires, the pinned Alpaca
# fixture, fetch (unpinned + the proof main == pin, e4b#404), tokenise, the `arm` wrapper (one process, one JSON, one alarm,
# the arm told its own alarm for the #548 watchdog), stubw, free_family. Removed: tp4's other boxes and plans (the anchor pair,
# the P43 diagnosis, the P45 profile, the P46 LoRA-path and the P56/P67 opt-ins), the clinical dataset. Added: `draw2`
# (the second draw of an arm: the same invocation, a fresh process, tag <tag>_d2), `todo_arm` (a not_run row for an arm this
# cut does not implement), and `tc1_family`, which runs TC1-PREREG's arm set IN ITS REGISTERED ORDER with the matched flags
# (`--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED`) on the matched set and the native flags on the native rows.
#
#   TC1_BOX=A  qwen3 (Qwen3-30B-A3B @ the pin) -- one RTX 5090, the only registered box
#
# Frameworks: e4b (GitHub main @ E4B_SHA + grouped-nf4-gemm @ GNF4_SHA, venv-e4b: transformers 5.18.0 / bitsandbytes 0.50.2 /
# peft 0.21.2), plain HF+PEFT+bnb (venv-e4b), Unsloth at the REGISTERED versions (unsloth 2026.9.14 + unsloth_zoo 2026.9.9) in
# TWO venvs -- venv-unsloth-t28 = tp4's install on the image's torch 2.8.0+cu128, venv-unsloth = unsloth[cu130-torch2121]
# (torch 2.12.1+cu130, the route its installer names for Blackwell; UPSTREAM-NOTES "Unsloth") -- and axolotl 0.20.0 in
# venv-axolotl (uv, CPython 3.12, torch 2.14.1+cu130: torch >= 2.13 Linux wheels exist only under cu130). Every cu130 venv
# is gated on the host driver >= 580 (checked BEFORE any install); below it the arms that need one are `refused` rows that
# name the driver -- never a silent fallback to the t28 venv.
set -uo pipefail
LANE=tc1; W=/root/$LANE; mkdir -p $W/logs $W/adapters $W/data; cd $W || exit 9
say(){ echo "[$(date -u +%FT%TZ)] tc1/box${TC1_BOX:-?}: $*"; }
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/TC1_RUN_NONCE.tmp && mv $W/TC1_RUN_NONCE.tmp $W/TC1_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > TC1_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > TC1_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in TC1_BOX TC1_RUN_ID TC1_DEADLINE_EPOCH TC1_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$TC1_BOX" in A) ;; *) say "refusing: TC1_BOX must be A (one RTX 5090; TC1-PREREG 'Budget and stop rules')"; finish 78;; esac
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# The pre-registration written into EVERY receipt and stub. Overridable because a draw that adds arms is governed by the
# document that REGISTERED those arms and authorised its spend -- and tc1_arm.py refuses a run without --prereg precisely so
# a receipt can never cite a pre-registration the run did not pass.
PREREG=${TC1_PREREG:-tc1/TC1-PREREG.md}
export TC1_INSTANCE_ID
# ---------------------------------------------------------------- the registered fixture (TC1-PREREG "Fixture" = tp4's field recipe): N 20, eval at 0 and N on 8 held-out rows
STEPS=${TC1_STEPS:-20}; SEQ=${TC1_SEQ:-2048}; MB=${TC1_MB:-2}; ACCUM=${TC1_ACCUM:-4}; R=${TC1_R:-16}; ALPHA=${TC1_ALPHA:-16}
LR=${TC1_LR:-2e-4}; WD=${TC1_WD:-0.001}; WARMUP=${TC1_WARMUP:-5}; SCHED=${TC1_SCHED:-linear}; OPTIM=${TC1_OPTIM:-adamw_8bit}; SEED=${TC1_SEED:-3407}
EVAL_EVERY=${TC1_EVAL_EVERY:-20}; EVAL_N=${TC1_EVAL_N:-8}; AUTOCAST=${TC1_AUTOCAST:-0}; TEMPLATE=alpaca
MATCHED_SEED=${TC1_MATCHED_SEED:-3407}            # --lora-init matched:<seed> on every matched arm (TC1-PREREG "Arms"); its own knob, distinct from the fixture seed
DS_ALPACA_SHA=${TC1_DS_ALPACA_SHA:-5324987afa4042556953026289e8dbdbe8a936b32832ed9e603b9192b706a2fb}   # tp4_alpaca.py output, registered
TF_VER=${TC1_TRANSFORMERS_VER:-5.18.0}; BNB_VER=${TC1_BNB_VER:-0.50.2}; PEFT_VER=${TC1_PEFT_VER:-0.21.2}   # TC1-PREREG "Environments"
SKIP=${TC1_SKIP:-}; PIN_FALLBACK=${TC1_PIN_FALLBACK:-0}; GPU_CLASS=${TC1_GPU_CLASS:-5090}
case "$TC1_BOX" in
  A) FAMILIES=${TC1_FAMILIES:-"qwen3"};;          # the judged family; TC1_FAMILIES=qwen3native for the labelled / native-best box (phase 3 I)
esac
: > summary.txt; echo "$TC1_INSTANCE_ID" > INSTANCE_ID
echo "FIXTURE field: template=$TEMPLATE steps=$STEPS seq=$SEQ micro_batch=$MB accum=$ACCUM r=$R alpha=$ALPHA lr=$LR wd=$WD warmup=$WARMUP sched=$SCHED optim=$OPTIM seed=$SEED eval_every=$EVAL_EVERY eval_n=$EVAL_N autocast=$AUTOCAST matched_seed=$MATCHED_SEED" | tee -a summary.txt
echo "BOX $TC1_BOX families: $FAMILIES; e4b $E4B_SHA gnf4 $GNF4_SHA; run $TC1_RUN_ID instance $TC1_INSTANCE_ID deadline $TC1_DEADLINE_EPOCH" | tee -a summary.txt
# ---------------------------------------------------------------- staged pieces, box class, forensics
for f in tc1_arm.py tc1_reduce.py tp4_alpaca.py; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "BOX REFUSED: gpu '$GPU_NAME' is not the registered class ($GPU_CLASS)"; echo "BOX_REFUSED gpu=$GPU_NAME" >> summary.txt; finish 12;; esac
# K: the class label the arm records (TC1_BOX_CLASS): "RTX <n>" for a numeric class, the class string itself otherwise (H100 NVL / SXM / PCIE pass the substring check above)
case "$GPU_CLASS" in [0-9]*) BOX_CLASS="RTX $GPU_CLASS";; *) BOX_CLASS="$GPU_CLASS";; esac
# F [F19]: every arm runs with OMP_NUM_THREADS = the box's PHYSICAL core count (recorded in the receipt's arm_facts)
PHYS=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' '); case "$PHYS" in ''|0|*[!0-9]*) PHYS=$(nproc);; esac
echo "OMP_NUM_THREADS=$PHYS (physical cores) box_class=$BOX_CLASS" | tee -a summary.txt
# P48 run 1 (2026-09-19) drew a host whose container overlay was 32 GB and died ENOSPC mid-fetch; box B fetches ~120 GB of
# checkpoints. The launcher orders machine disk, the instance overlay is what the box gets: refuse here, before any fetch.
MIN_DISK_GB=${TC1_MIN_DISK_GB:-200}; FREE_GB=$(df -BG --output=avail /root 2>/dev/null | tail -1 | tr -dc 0-9)
if [ "${FREE_GB:-0}" -lt "$MIN_DISK_GB" ]; then say "BOX REFUSED: ${FREE_GB:-?} GB free on /root < ${MIN_DISK_GB} GB (instance overlay too small for the checkpoints -- host-limited)"; echo "BOX_REFUSED disk=${FREE_GB:-?}GB" >> summary.txt; finish 13; fi
# cu130 wheels (torch 2.12.1 / 2.14.1) need an NVIDIA driver >= 580: checked here, before any install; recorded in summary.txt
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' '); DRIVER_MAJOR=${DRIVER%%.*}
CU130_OK=1; case "$DRIVER_MAJOR" in ''|*[!0-9]*) CU130_OK=0;; *) [ "$DRIVER_MAJOR" -ge 580 ] || CU130_OK=0;; esac
CU130_REASON="cu130 wheels need driver >= 580; host has ${DRIVER:-unknown}"
# TC1-PREREG amendment 1 (2026-10-01): both TC1 family tokens need the cu130 venvs (the comparator and axolotl), so a host
# below the driver floor cannot produce the lane's readings. It is a REGISTERED HOST FLOOR -- rent.py's lane-refusal class
# 18 (P86's driver refusal, machine 37958 at 575.57 on tc1-5090-3) -- so the box refuses here, before any install or
# fetch, and the receipt names the machine for exclusion on the next draw instead of running four hours of refused rows.
if [ "$CU130_OK" != 1 ]; then say "REFUSED: $CU130_REASON (registered host floor; TC1-PREREG amendment 1)"; echo "refused: driver ${DRIVER:-unknown} < 580" > REFUSAL; echo "BOX_REFUSED driver=${DRIVER:-unknown} floor=580" | tee -a summary.txt; finish 18; fi
[ "$CU130_OK" = 1 ] && say "driver $DRIVER: cu130 venvs (venv-unsloth, venv-axolotl) will be built" || say "driver ${DRIVER:-unknown}: $CU130_REASON -- venv-unsloth (cu130) and venv-axolotl are NOT built; their arms are refused rows"
echo "DRIVER $DRIVER cu130_ok=$CU130_OK" | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,power.limit,clocks.max.sm --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^CPU\(s\)" | tee -a forensics.txt; grep MemTotal /proc/meminfo | tee -a forensics.txt; cat /sys/fs/cgroup/memory.max 2>/dev/null | sed "s/^/cgroup memory.max /" | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt
python3 - "$TC1_BOX" "$TC1_RUN_ID" "$TC1_INSTANCE_ID" "$GPU_NAME" <<'PYB' > box.json
import json, os, subprocess, sys
box, run_id, iid, gpu = sys.argv[1:5]
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e: return f"ERR {e}"
print(json.dumps({"box": box, "run_id": run_id, "instance_id": iid, "gpu": gpu, "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
                  "cpu": sh("lscpu | grep 'Model name' | cut -d: -f2 | xargs"), "nproc": os.cpu_count(), "mem_total_kb": sh("grep MemTotal /proc/meminfo | awk '{print $2}'"),
                  "cgroup_memory_max": sh("cat /sys/fs/cgroup/memory.max 2>/dev/null"), "disk_root": sh("df -h /root | tail -1"), "hostname": sh("hostname"),
                  "registered_gpu_class": os.environ.get("TC1_GPU_CLASS", "5090"), "prereg": "tc1/TC1-PREREG.md"}, indent=1))
PYB
# ---------------------------------------------------------------- deadline-derived alarms (p39's rule): what is LEFT minus a fetch margin, never a literal that outlives the rental
left(){ echo $(( TC1_DEADLINE_EPOCH - $(date +%s) )); }
alarm_for(){ local want=$1 l; l=$(( $(left) - 900 )); [ "$l" -lt "$want" ] && want=$l; [ "$want" -lt 300 ] && want=300; echo "$want"; }
can_run(){ local need=$1 name=$2; [ $(( $(left) - 900 )) -ge "$need" ] && return 0; say "STOP-DEADLINE: $name needs ${need}s, $(left)s left -- skipped (host-limited)"; echo "SKIPPED $name host-limited deadline" >> summary.txt; return 1; }
# ---------------------------------------------------------------- installs
export DEBIAN_FRONTEND=noninteractive
PY_E4B=$W/venv-e4b/bin/python; PY_UNS=$W/venv-unsloth/bin/python; PY_UNS_T28=$W/venv-unsloth-t28/bin/python; PY_AX=$W/venv-axolotl/bin/python
say "venv-e4b (system torch): e4b @$E4B_SHA + gnf4 @$GNF4_SHA + transformers==$TF_VER bitsandbytes==$BNB_VER peft==$PEFT_VER"
python -m venv --system-site-packages $W/venv-e4b || { say "VENV FAIL (e4b)"; finish 9; }
perl -e 'alarm 2400; exec @ARGV' $PY_E4B -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "transformers==$TF_VER" "bitsandbytes==$BNB_VER" "peft==$PEFT_VER" accelerate safetensors "huggingface_hub>=0.23" sentencepiece tiktoken > logs/pip_e4b.log 2>&1
rc=$?; echo "pip(e4b) rc=$rc"; [ $rc -ne 0 ] && { tail -6 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
E4B_SHA="$E4B_SHA" GNF4_SHA="$GNF4_SHA" TF_VER="$TF_VER" $PY_E4B - <<'PYT' > logs/tripwire_e4b.log 2>&1 || { tail -5 logs/tripwire_e4b.log; say "TRIPWIRE FAIL (e4b)"; finish 9; }
import importlib.metadata as md, inspect, json, os
import experts4bit_qlora as e, torch, triton, transformers, bitsandbytes, peft
from experts4bit_qlora import enable_fast_train, enable_batched_train, ExpertsLoRA, load_moe_4bit_streaming, verify_moe_4bit, disable_fast_train, disable_batched_train
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit, detect_attention_projections
from experts4bit_qlora.train import save_adapter
from nf4_qlora import fused_grouped_lora
assert "dgrad_kernel" in inspect.signature(fused_grouped_lora).parameters
def commit(dist):
    d = md.distribution(dist)
    du = d.read_text("direct_url.json")
    return (json.loads(du).get("vcs_info") or {}).get("commit_id") if du else None
ce, cg = commit("experts4bit-qlora"), commit("grouped-nf4-gemm")
assert ce == os.environ["E4B_SHA"], f"e4b installed from {ce}, wanted {os.environ['E4B_SHA']}"
assert cg == os.environ["GNF4_SHA"], f"gnf4 installed from {cg}, wanted {os.environ['GNF4_SHA']}"
assert transformers.__version__ == os.environ["TF_VER"], transformers.__version__
assert torch.cuda.is_available(), "no CUDA in venv-e4b"
print("tc1 tripwire OK (e4b):", e.__version__, "@", ce[:12], "gnf4", md.version("grouped-nf4-gemm"), "@", cg[:12], "torch", torch.__version__, "triton", triton.__version__,
      "transformers", transformers.__version__, "bnb", bitsandbytes.__version__, "peft", peft.__version__)
open("/root/tc1/versions.txt", "a").write(f"e4b {e.__version__} @{ce} (GitHub main)\ngnf4 {md.version('grouped-nf4-gemm')} @{cg} (GitHub main)\ntorch(e4b/hf) {torch.__version__}\ntriton(e4b/hf) {triton.__version__}\n"
                                          f"transformers(e4b/hf) {transformers.__version__}\nbitsandbytes(e4b/hf) {bitsandbytes.__version__}\npeft(hf) {peft.__version__}\n")
PYT
tail -1 logs/tripwire_e4b.log
# Unsloth: the REGISTERED versions (TC1-PREREG "Environments": unsloth 2026.9.14 + unsloth_zoo 2026.9.9), NO transformers/bnb/peft
# pins from us (P38 amendment 1); torchao removed on the ScalingType tripwire (P38 amendment 2). TWO venvs (phase 2):
#   venv-unsloth-t28  unsloth[cu128-torch280]  -- tp4's install on the image's torch 2.8.0+cu128 (the field-image row, `ckpt_unsloth_t28`)
#   venv-unsloth      unsloth[cu130-torch2121] -- torch 2.12.1+cu130 (needs driver >= 580; UPSTREAM-NOTES "Unsloth": _auto_install.py:44),
#                     where torch._grouped_mm runs on sm_120 and the grouped_mm backend can engage (moe_utils.py:374-378)
UNS_VER=${TC1_UNSLOTH_VERSION:-2026.9.14}; ZOO_VER=${TC1_UNSLOTH_ZOO_VERSION:-2026.9.9}
UNS_T28_OK=1
say "venv-unsloth-t28: unsloth[cu128-torch280]==$UNS_VER unsloth_zoo==$ZOO_VER (the image's torch 2.8.0+cu128)"
python -m venv $W/venv-unsloth-t28 && perl -e 'alarm 2700; exec @ARGV' $PY_UNS_T28 -m pip install -q --no-input --no-cache-dir \
  "unsloth[cu128-torch280]==$UNS_VER" ${ZOO_VER:+"unsloth_zoo==$ZOO_VER"} datasets safetensors "huggingface_hub>=0.23" > logs/pip_unsloth_t28.log 2>&1
rc=$?; echo "pip(unsloth-t28) rc=$rc"; [ $rc -ne 0 ] && { tail -6 logs/pip_unsloth_t28.log; echo "PIP FAIL (unsloth-t28) -- its rows = install_failed"; UNS_T28_OK=0; }
UNS_OK=0
if [ "$CU130_OK" = 1 ]; then
  UNS_OK=1; say "venv-unsloth: unsloth[cu130-torch2121]==$UNS_VER unsloth_zoo==$ZOO_VER (torch 2.12.1+cu130)"
  python -m venv $W/venv-unsloth && perl -e 'alarm 2700; exec @ARGV' $PY_UNS -m pip install -q --no-input --no-cache-dir \
    "unsloth[cu130-torch2121]==$UNS_VER" ${ZOO_VER:+"unsloth_zoo==$ZOO_VER"} datasets safetensors "huggingface_hub>=0.23" > logs/pip_unsloth.log 2>&1
  rc=$?; echo "pip(unsloth) rc=$rc"; [ $rc -ne 0 ] && { tail -6 logs/pip_unsloth.log; echo "PIP FAIL (unsloth cu130) -- its rows = install_failed"; UNS_OK=0; }
else
  say "venv-unsloth (cu130) SKIPPED: $CU130_REASON"
fi
cat > $W/tripwire_unsloth.py <<'PYU'
import importlib.metadata as md, os, torch, transformers, bitsandbytes, peft
import unsloth, unsloth_zoo
from unsloth import FastLanguageModel
from unsloth_zoo.temporary_patches.common import is_transformers_v5_moe_quantization_available
from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import forward_moe_backend_bnb4bit, _is_bnb4bit_param, _moe_uses_bnb4bit_expert_weights
from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend, _should_use_separated_lora
assert is_transformers_v5_moe_quantization_available(), "the transformers-v5 4-bit MoE path is NOT available in this environment: Unsloth would not train 4-bit MoE here"
assert torch.cuda.is_available(), "no CUDA in venv-unsloth"
tri = None
try:
    import triton; tri = triton.__version__
except Exception: pass
tao = None
try:
    tao = md.version("torchao")
except Exception: pass
fm = hasattr(unsloth, "FastModel")
tag = os.environ.get("TC1_VENV_TAG", "unsloth")          # phase 2: one tripwire per Unsloth venv, lines suffixed with the venv
print(f"tc1 tripwire OK ({tag}):", unsloth.__version__, "zoo", unsloth_zoo.__version__, "torch", torch.__version__, "triton", tri, "transformers", transformers.__version__,
      "bnb", bitsandbytes.__version__, "peft", peft.__version__, "torchao", tao, "moe_backend", select_moe_backend(), "separated_lora", _should_use_separated_lora(), "FastModel", fm)
open("/root/tc1/versions.txt", "a").write(f"unsloth({tag}) {unsloth.__version__}\nunsloth_zoo({tag}) {unsloth_zoo.__version__}\ntorch({tag}) {torch.__version__}\ntriton({tag}) {tri}\n"
                                          f"transformers({tag}) {transformers.__version__}\nbitsandbytes({tag}) {bitsandbytes.__version__}\npeft({tag}) {peft.__version__}\ntorchao({tag}) {tao}\nmoe_backend({tag}) {select_moe_backend()}\n")
PYU
# tripwire_unsloth VENV_TAG PY OK_VAR: the import tripwire for one Unsloth venv (torchao removed once on ScalingType, P38 amendment 2)
tripwire_unsloth(){ local TAG=$1 PY=$2 ok=1
  TC1_VENV_TAG=$TAG $PY $W/tripwire_unsloth.py > logs/tripwire_$TAG.log 2>&1; local trc=$?
  if [ $trc -ne 0 ] && grep -q "ScalingType" logs/tripwire_$TAG.log; then
    say "tripwire($TAG) failed on ScalingType -> removing torchao (P38 amendment 2)"; cp logs/tripwire_$TAG.log logs/tripwire_$TAG.attempt1.log
    $PY -m pip uninstall -y -q torchao > logs/pip_${TAG}_torchao_removed.log 2>&1
    echo "AMENDMENT-CLASS (P38 amendment 2): torchao removed from $TAG after the tripwire failed on ScalingType" | tee -a summary.txt
    TC1_VENV_TAG=$TAG $PY $W/tripwire_unsloth.py > logs/tripwire_$TAG.log 2>&1; trc=$?
  fi
  [ $trc -ne 0 ] && { echo "TRIPWIRE FAIL ($TAG) -- other arms still run; its Unsloth rows = install_failed"; tail -5 logs/tripwire_$TAG.log; ok=0; }
  tail -1 logs/tripwire_$TAG.log; return $(( 1 - ok )); }
[ "$UNS_T28_OK" = 1 ] && { tripwire_unsloth unsloth-t28 $PY_UNS_T28 || UNS_T28_OK=0; }
[ "$UNS_OK" = 1 ] && { tripwire_unsloth unsloth $PY_UNS || UNS_OK=0; }
# J [F7]: e4b + grouped-nf4-gemm at the same pins into venv-unsloth (torch 2.12.1+cu130) for the `fused_attn4_m_t212` row; a failed install
# or tripwire is a row (install_failed), the cu130 gate a refused row
T212_OK=0; T212_REASON=""
if [ "$UNS_OK" = 1 ]; then
  say "venv-unsloth + e4b @$E4B_SHA + gnf4 @$GNF4_SHA (the t212 row)"
  perl -e 'alarm 2400; exec @ARGV' $PY_UNS -m pip install -q --no-input --prefer-binary \
    "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_e4b_t212.log 2>&1
  rc=$?; echo "pip(e4b-t212) rc=$rc"
  if [ $rc -ne 0 ]; then tail -6 logs/pip_e4b_t212.log; T212_REASON="e4b/gnf4 install into venv-unsloth failed rc=$rc (logs/pip_e4b_t212.log): $(tail -3 logs/pip_e4b_t212.log | tr '\n' ' ' | cut -c1-300)"
  else
    E4B_SHA="$E4B_SHA" GNF4_SHA="$GNF4_SHA" $PY_UNS - <<'PYT2' > logs/tripwire_e4b_t212.log 2>&1; trc=$?
import importlib.metadata as md, inspect, json, os
import experts4bit_qlora as e, torch
from experts4bit_qlora import enable_fast_train, load_moe_4bit_streaming
from nf4_qlora import fused_grouped_lora, LORA_PATH_STATS
assert "dgrad_kernel" in inspect.signature(fused_grouped_lora).parameters
def commit(dist):
    d = md.distribution(dist); du = d.read_text("direct_url.json")
    return (json.loads(du).get("vcs_info") or {}).get("commit_id") if du else None
assert commit("experts4bit-qlora") == os.environ["E4B_SHA"] and commit("grouped-nf4-gemm") == os.environ["GNF4_SHA"], (commit("experts4bit-qlora"), commit("grouped-nf4-gemm"))
assert torch.cuda.is_available()
print("tc1 tripwire OK (e4b-t212):", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
open("/root/tc1/versions.txt", "a").write(f"e4b(t212) {e.__version__} @{commit('experts4bit-qlora')}\ngnf4(t212) {md.version('grouped-nf4-gemm')} @{commit('grouped-nf4-gemm')}\ntorch(e4b-t212) {torch.__version__}\n")
PYT2
    if [ $trc -ne 0 ]; then tail -4 logs/tripwire_e4b_t212.log; T212_REASON="e4b-t212 tripwire failed (logs/tripwire_e4b_t212.log): $(tail -2 logs/tripwire_e4b_t212.log | tr '\n' ' ' | cut -c1-300)"; else T212_OK=1; tail -1 logs/tripwire_e4b_t212.log; fi
  fi
else
  T212_REASON="venv-unsloth (cu130) unavailable: ${CU130_REASON}"
fi
# ---------------------------------------------------------------- axolotl 0.20.0 (phase 2, axolotl-arm-spec.md): its own venv, uv, CPython 3.12, torch cu130
# Pins (wheel METADATA): python >= 3.12, torch >= 2.13.0 <= 2.14.0, transformers == 5.17.0, peft == 0.21.0, bitsandbytes == 0.50.2.
# torch >= 2.13 Linux cp312 wheels exist only under /whl/cu130 (verified 2026-10-01 by the coordinator), so the driver gate above applies.
AX_VER=${TC1_AXOLOTL_VERSION:-0.20.0}; AX_OK=0; AX_REASON=""
if [ "$CU130_OK" != 1 ]; then
  AX_REASON="$CU130_REASON"; say "venv-axolotl SKIPPED: $AX_REASON -- its arms are refused rows"
else
  say "venv-axolotl: uv venv --python 3.12 + axolotl==$AX_VER --extra-index-url https://download.pytorch.org/whl/cu130 (alarm 2700 s)"
  python -m pip install -q --no-input uv > logs/pip_uv.log 2>&1 && python -m uv venv --python 3.12 $W/venv-axolotl > logs/uv_venv_axolotl.log 2>&1 \
    && perl -e 'alarm 2700; exec @ARGV' python -m uv pip install --python $PY_AX "axolotl==$AX_VER" --extra-index-url https://download.pytorch.org/whl/cu130 > logs/pip_axolotl.log 2>&1
  rc=$?; echo "pip(axolotl) rc=$rc"
  if [ $rc -ne 0 ]; then tail -6 logs/pip_axolotl.log; AX_REASON="axolotl venv install failed rc=$rc (logs/pip_axolotl.log): $(tail -3 logs/pip_axolotl.log 2>/dev/null | tr '\n' ' ' | cut -c1-300)"; echo "PIP FAIL (axolotl) -- its rows = install_failed"
  else
    $PY_AX - <<'PYA' > logs/tripwire_axolotl.log 2>&1; trc=$?
import importlib.metadata as md, torch, transformers, peft, bitsandbytes
import axolotl
from axolotl.cli.config import load_cfg
from axolotl.loaders import ModelLoader, load_tokenizer
from axolotl.monkeypatch.moe_quant import get_moe_quantized_count, patch_moe_quantization_on_load
from axolotl.integrations.base import PluginManager
from bitsandbytes.nn.parametrize import replace_parameter_4bit
assert torch.cuda.is_available(), "no CUDA in venv-axolotl"
print("tc1 tripwire OK (axolotl):", md.version("axolotl"), "torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "bnb", bitsandbytes.__version__, "python", __import__("sys").version.split()[0])
open("/root/tc1/versions.txt", "a").write(f"axolotl {md.version('axolotl')}\ntorch(axolotl) {torch.__version__}\ntransformers(axolotl) {transformers.__version__}\npeft(axolotl) {peft.__version__}\nbitsandbytes(axolotl) {bitsandbytes.__version__}\npython(axolotl) {__import__('sys').version.split()[0]}\n")
PYA
    if [ $trc -ne 0 ]; then tail -5 logs/tripwire_axolotl.log; AX_REASON="axolotl tripwire failed (logs/tripwire_axolotl.log): $(tail -2 logs/tripwire_axolotl.log | tr '\n' ' ' | cut -c1-300)"; echo "TRIPWIRE FAIL (axolotl) -- its rows = install_failed"; else AX_OK=1; tail -1 logs/tripwire_axolotl.log; fi
  fi
fi
# ---------------------------------------------------------------- the fixed texts: Alpaca (field recipe) + clinical (the anchor pair only)
say "dataset alpaca (tp4_alpaca.py: unsloth/alpaca-cleaned @ pinned revision, seed 3407, 1200/48)"
(cd $W/data && perl -e 'alarm 900; exec @ARGV' $PY_E4B $W/tp4_alpaca.py --out $W/data/ds_alpaca.json > $W/logs/dataset_alpaca.log 2>&1) || { tail -3 logs/dataset_alpaca.log; say "DATASET FAIL (alpaca)"; finish 13; }
tail -1 logs/dataset_alpaca.log
GOT=$(sha256sum $W/data/ds_alpaca.json | awk '{print $1}'); [ "$GOT" = "$DS_ALPACA_SHA" ] || { say "DATASET MISMATCH alpaca: $GOT != $DS_ALPACA_SHA"; finish 13; }
echo "DATASET alpaca sha=$DS_ALPACA_SHA" | tee -a summary.txt
# ---------------------------------------------------------------- helpers
vram_start(){ ( while :; do echo "$(date -u +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)"; sleep 1; done ) > $W/vram_$1.txt 2>/dev/null & echo $!; }
vram_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
skip(){ case " $SKIP " in *" $1 "*) return 0;; *) return 1;; esac; }
status_of(){ $PY_E4B -c "import json,sys; print(json.load(open(sys.argv[1])).get('status','missing'))" "$W/${1}_${2}_${3}.json" 2>/dev/null || echo missing; }
# stubw FAM FW TAG ARM STATUS REASON [extra-json]: a row for an attempt that never reached the harness
stubw(){ $PY_E4B - "$W" "$STEPS" "$SEQ" "$ACCUM" "$MB" "$PREREG" "$@" <<'PYS'
import json, os, sys
W, steps, seq, accum, mb, prereg, fam, fw, tag, arm, status, reason = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], *sys.argv[7:13]
extra = json.loads(sys.argv[13]) if len(sys.argv) > 13 else {}
rec = {"framework": fw, "fam": fam, "arm": arm, "tag": tag, "status": status, "reason": reason[:800], "steps": steps, "seq": seq, "accum": accum, "micro_batch": mb,
       "written_by": "tc1_run.sh", "prereg": prereg, **extra}
p = os.path.join(W, f"{fam}_{fw}_{tag}.json")
json.dump(rec, open(p, "w"), indent=1)
print(f"STUB {status.upper()} {fam}/{fw}/{tag}: {reason[:160]}")
PYS
  echo "$1/$2/$3 STUB $5: $6" | cut -c1-300 >> summary.txt; }
# fetch FAM MID REV ALARM: the snapshot UNPINNED (the loader resolves `main`, e4b#404), then the PROOF main == pin; refs/main written from the pin only then
fetch(){ local FAM=$1 MID=$2 REV=$3 AL=$4; say "fetch $FAM ($MID, unpinned; pin $REV)"
  perl -e "alarm $(alarm_for $AL); exec @ARGV" $PY_E4B - "$MID" <<'PYF' > logs/fetch_$FAM.log 2>&1
import os, sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "merges.txt", "vocab.json", "*.tiktoken", "*.jinja"], max_workers=4)
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}")
print("STAGED", os.path.basename(os.path.realpath(p)))
PYF
  local rc=$?; tail -2 logs/fetch_$FAM.log | head -1
  [ $rc -ne 0 ] && { echo "$FAM: FETCH FAILED rc=$rc" | tee -a summary.txt; FETCH_REASON="fetch failed rc=$rc (alarm $AL s; logs/fetch_$FAM.log)"; return 1; }
  local GOT; GOT=$(grep -a "^STAGED " logs/fetch_$FAM.log | tail -1 | awk '{print $2}')
  local RDIR=/root/.cache/huggingface/hub/models--${MID//\//--}
  if [ "$GOT" = "$REV" ]; then
    mkdir -p $RDIR/refs && printf '%s' "$REV" > $RDIR/refs/main
    echo "PIN OK $FAM staged=$GOT == pin; refs/main written from the pin (P38 amendment 2 / e4b#404)" | tee -a summary.txt
  elif [ "$PIN_FALLBACK" = "1" ]; then
    say "PIN MISMATCH $FAM: staged $GOT != pin $REV -> fetching the pinned revision (TC1_PIN_FALLBACK=1, an amendment)"
    perl -e "alarm $(alarm_for $AL); exec @ARGV" $PY_E4B -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MID', revision='$REV', allow_patterns=['*.safetensors', '*.json', 'tokenizer*', '*.model', '*.txt', 'merges.txt', 'vocab.json', '*.tiktoken', '*.jinja'], max_workers=4))" > logs/fetch_${FAM}_pinned.log 2>&1 \
      || { echo "$FAM: PINNED FETCH FAILED" | tee -a summary.txt; FETCH_REASON="main moved past the pin ($GOT != $REV) and the pinned fetch failed"; return 1; }
    mkdir -p $RDIR/refs && printf '%s' "$REV" > $RDIR/refs/main
    echo "AMENDMENT (TC1_PIN_FALLBACK): $FAM main=$GOT != pin=$REV; pinned snapshot fetched; refs/main := pin (cache pointer only)" | tee -a summary.txt
  else
    echo "PIN MISMATCH $FAM: staged=$GOT != pin=$REV -- main moved past the pin; the family is ABORTED with load_fault stubs (not coerced)" | tee -a summary.txt
    FETCH_REASON="staged snapshot $GOT != pinned revision $REV (main moved past the pin); family aborted, not coerced (set TC1_PIN_FALLBACK=1 as an amendment)"
    return 2
  fi
  df -h /root | tail -1; return 0; }
# expect_of FAM E4BTAG: the family's e4b trainable count (the primary receipt of the same recipe) for --expect-trainable
expect_of(){ $PY_E4B - "$W" "$1" "$2" <<'PYE' 2>/dev/null
import json, os, sys
W, fam, tag = sys.argv[1], sys.argv[2], sys.argv[3]
for t in (tag, "reference_attn4_m"):
    p = os.path.join(W, f"{fam}_e4b_{t}.json")
    if os.path.exists(p):
        r = json.load(open(p))
        if r.get("status") == "ok" and r.get("trainable_params"):
            print(r["trainable_params"]); break
PYE
}
# arm FAM FW TAG ARM ALARM MID REV OFFLOAD RECIPE(field|anchor|mb1) TOK TOK_SHA [extra args...]: one process, one JSON, one alarm
arm(){ local FAM=$1 FW=$2 TAG=$3 ARM=$4 AL=$5 MID=$6 REV=$7 OFF=$8 RECIPE=$9 TOK=${10} TOK_SHA=${11}; shift 11
  { skip $FAM || skip $FAM/$FW/$TAG; } && { say "skip $FAM/$FW/$TAG"; stubw $FAM $FW $TAG $ARM not_run "skipped by TC1_SKIP"; return 0; }
  # phase 2: the interpreter per framework; UNS_VENV=t28 (a prefix assignment on the call) selects tp4's torch-2.8 venv for an
  # Unsloth arm, else the cu130 venv. A cu130 venv the driver gate refused -> `refused` rows naming the driver; a venv that
  # did not install/import -> `install_failed` rows. Never a silent fallback to the other venv.
  local PY=$PY_E4B
  if [ "$FW" = e4b ] && [ "${E4B_VENV:-}" = t212 ]; then PY=$PY_UNS          # J: e4b on torch 2.12.1+cu130
    [ "$CU130_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM refused "$CU130_REASON" '{"venv": "venv-unsloth (cu130-torch2121) + e4b"}'; return 0; }
    [ "$T212_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM install_failed "$T212_REASON"; return 0; }
  elif [ "$FW" = hf ] && [ "${HF_VENV:-}" = t214 ]; then PY=$PY_AX              # J: the HF arm on the axolotl venv's torch 2.14
    [ "$CU130_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM refused "$CU130_REASON" '{"venv": "venv-axolotl (torch cu130)"}'; return 0; }
    [ "$AX_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM install_failed "$AX_REASON"; return 0; }
  elif [ "$FW" = unsloth ]; then
    if [ "${UNS_VENV:-cu130}" = t28 ]; then PY=$PY_UNS_T28
      [ "$UNS_T28_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM install_failed "venv-unsloth-t28 did not install/import (logs/pip_unsloth_t28.log, logs/tripwire_unsloth-t28.log)"; return 0; }
    else PY=$PY_UNS
      [ "$CU130_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM refused "$CU130_REASON" '{"venv": "venv-unsloth (cu130-torch2121)"}'; return 0; }
      [ "$UNS_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM install_failed "venv-unsloth (cu130) did not install/import (logs/pip_unsloth.log, logs/tripwire_unsloth.log)"; return 0; }
    fi
  elif [ "$FW" = axolotl ]; then PY=$PY_AX
    [ "$CU130_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM refused "$CU130_REASON" '{"venv": "venv-axolotl (torch cu130)"}'; return 0; }
    [ "$AX_OK" = 1 ] || { stubw $FAM $FW $TAG $ARM install_failed "$AX_REASON"; return 0; }
  fi
  local s=$STEPS q=$SEQ m=$MB ac=$ACCUM r=$R al=$ALPHA lr=$LR wd=$WD wu=$WARMUP sc=$SCHED op=$OPTIM sd=$SEED en=$EVAL_N ee=$EVAL_EVERY ex_tag=fused_attn4_m
  case "$RECIPE" in
    mb1)    m=1; ac=$(( MB * ACCUM )); ex_tag=fused_attn4_m_mb1;;
  esac
  local EXP EXPARG=""; [ "$FW" != e4b ] && { EXP=$(expect_of $FAM $ex_tag); [ -n "$EXP" ] && EXPARG="--expect-trainable $EXP"; }
  local A; A=$(alarm_for $AL)
  say "arm $FAM/$FW/$TAG (arm=$ARM recipe=$RECIPE steps=$s seq=$q mb=$m accum=$ac r=$r lr=$lr optim=$op sched=$sc offload=$OFF alarm=$A expect_trainable=${EXP:-none} $*)"
  local sp; sp=$(vram_start ${FAM}_${FW}_$TAG)
  # e4b#548: the arm is told the alarm it is running under, so it can refuse ITSELF while still inside an over-budget
  # prologue phase (status phase_alarm, exit 16) instead of leaving SIGALRM to kill a process that cannot write a stub.
  # P56: the batched arm is a PARITY arm, so it must run the batched arithmetic on every
  # call. `enable_batched_train` falls back to the reference forward above a pad-waste
  # ratio, and the tp1 bundle records exactly that producing VOID rows on OLMoE, Qwen3 and
  # Gemma-4 -- an arm that fell back is measuring the reference against itself. The guard
  # is a SPEED guard; raising it trades peak memory for engagement and never numerics, and
  # `batched_fallback_stats` puts the limit in force on the receipt.
  #
  # It goes through `env`, and that is NOT cosmetic. An unquoted expansion spliced into
  # the assignment prefix -- `A=1 $ARM_ENV B=2 cmd` -- makes the shell stop treating the
  # words after it as assignments, because it decides which words are assignments BEFORE
  # expanding: the next `VAR=value` becomes the COMMAND. Run p56-gemma4-ladder-1 lost all
  # four e4b arms to rc=127 `TC1_BOX_CLASS=RTX 5090: command not found` this way, and it
  # fails whether the variable is empty or set. As an argument to `env` the expansion is
  # an ordinary word and an empty one simply vanishes.
  local ARM_ENV=""; [ "$ARM" = batched ] && ARM_ENV="E4B_BATCHED_PAD_WASTE_LIMIT=${TC1_BATCHED_PAD_WASTE_LIMIT:-64}"
  env $ARM_ENV HF_HUB_OFFLINE=1 UNSLOTH_ENABLE_LOGGING=1 OMP_NUM_THREADS=$PHYS TC1_BOX_CLASS="$BOX_CLASS" TC1_ARM_ALARM_S=$A perl -e "alarm $A; exec @ARGV" $PY -u $W/tc1_arm.py --framework $FW --arm $ARM --tag $TAG --fam $FAM --model "$MID" --revision $REV \
      --steps $s --seq $q --micro-batch $m --accum $ac --autocast $AUTOCAST --lr $lr --r $r --alpha $al --seed $sd --offload $OFF \
      --optim $op --weight-decay $wd --lr-schedule $sc --warmup-steps $wu \
      --tokens $TOK --tokens-sha $TOK_SHA --eval-every $ee --eval-n $en --unsloth-loader FastLanguageModel $EXPARG \
      --prereg $PREREG --out $W --adapter-dir $W/adapters "$@" > logs/run_${FAM}_${FW}_$TAG.log 2>&1
  local rc=$?; vram_stop $sp
  if [ $rc -eq 142 ] && [ ! -s $W/${FAM}_${FW}_$TAG.json ]; then stubw $FAM $FW $TAG $ARM alarm "arm alarm $A s (SIGALRM; the process could not write its own stub)"; fi
  grep -aE "^CELL |^LOAD OK|^PROLOGUE |^PHASE ALARM|^ENGAGE|^STUB|^LOADER FALLBACK|Enabling LoRA on MoE|MoE bnb4bit|Error|error:" logs/run_${FAM}_${FW}_$TAG.log | tail -3 | cut -c1-300 | sed "s/^/    /"
  # e4b#548: the phase table goes into summary.txt beside the CELL line, so the box's own summary answers "where did the prologue go"
  grep -aE "^PROLOGUE |^PHASE ALARM" logs/run_${FAM}_${FW}_$TAG.log | tail -1 | cut -c1-600 | sed "s|^|$FAM/$FW/$TAG |" >> summary.txt
  { echo -n "$FAM/$FW/$TAG rc=$rc "; grep -aE "^CELL " logs/run_${FAM}_${FW}_$TAG.log | tail -1 | cut -c1-400; echo; } >> summary.txt
  $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader
  rm -rf $W/adapters/* 2>/dev/null; }
free_family(){ rm -rf /root/.cache/huggingface/hub/models--$2; say "freed $1 (disk: $(df -h /root | tail -1 | awk '{print $4}') free)"; }
tokenise(){ local FAM=$1 MID=$2 REV=$3 TEMPLATE_=$4 SEQ_=$5 DATA=$6 DATA_SHA=$7 TOK=$8 EVAL_N_=${9:-$EVAL_N}
  say "tokenise $FAM ($TEMPLATE_, seq $SEQ_) -> $(basename $TOK)"
  HF_HUB_OFFLINE=1 $PY_E4B $W/tc1_arm.py --prepare --fam $FAM --model "$MID" --revision $REV --data $DATA --data-sha $DATA_SHA --seq $SEQ_ --eval-n $EVAL_N_ --template $TEMPLATE_ --tokens $TOK > logs/prepare_${FAM}_$TEMPLATE_.log 2>&1 || return 1
  tail -1 logs/prepare_${FAM}_$TEMPLATE_.log; return 0; }
tok_sha(){ $PY_E4B -c "import json; print(json.load(open('$1'))['sha256'])"; }
# ---------------------------------------------------------------- the TC1 arm set (TC1-PREREG "Arms, in this order")
dmon_start(){ ( nvidia-smi dmon -s ut -d 1 -o T > $W/logs/dmon_$1.txt 2>/dev/null ) & echo $!; }
dmon_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
# draw2 FAM FW TAG ARM ...: the SECOND draw of an arm -- the same invocation, the same everything, a fresh process, tag <TAG>_d2.
# The reducer reads the pair as one arm with two draws (median over both; STABILITY = |d1-d2|/mean <= 5 % e4b / 10 % others).
draw2(){ local FAM=$1 FW=$2 TAG=$3; shift 3; arm "$FAM" "$FW" "${TAG}_d2" "$@"; }
# todo_arm FAM FW TAG ARM: an arm this harness cut does not implement yet -- a not_run row with the reason, never a silent omission (unused since phase 2; kept for the next cut).
todo_arm(){ stubw "$1" "$2" "$3" "$4" not_run "arm not yet implemented (TC1 follow-up)"; }
UT7="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"     # the notebooks' seven targets (TC1-PREREG arm 2)
PROFILE_STEPS=${TC1_PROFILE_STEPS:-3}; PROFILE_WARM=${TC1_PROFILE_WARM:-3}   # arm 10: P45's instrument, 3 warm + 3 profiled
# tc1_fetch_tokenise FAM MID REV FETCH_AL STUBLIST: fetch + tokenise, or stub every arm in STUBLIST (fw:tag:arm words); sets TOK / TS
tc1_prepare(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 ALL=$5
  stub_all(){ local st=$1 why=$2 t fw tag arm; for t in $ALL; do IFS=: read -r fw tag arm <<< "$t"; stubw $FAM $fw $tag $arm $st "$why"; done; }
  if skip $FAM; then say "skip family $FAM (TC1_SKIP)"; stub_all not_run "family skipped by TC1_SKIP"; return 1; fi
  FETCH_REASON=""; fetch $FAM $MID $REV $FAL; local frc=$?
  if [ $frc -ne 0 ]; then local st=not_run; [ $frc -eq 2 ] && st=load_fault; stub_all $st "$FETCH_REASON"; free_family $FAM ${MID//\//--}; return 1; fi
  TOK=$W/tokens_$FAM.json
  if ! tokenise $FAM "$MID" $REV alpaca $SEQ $W/data/ds_alpaca.json $DS_ALPACA_SHA $TOK; then
    tail -3 logs/prepare_${FAM}_alpaca.log; echo "$FAM: TOKENS FAIL" | tee -a summary.txt
    local why; why="tokenise failed (logs/prepare_${FAM}_alpaca.log): $(tail -1 logs/prepare_${FAM}_alpaca.log | cut -c1-200)"
    stub_all harness_error "$why"; free_family $FAM ${MID//\//--}; return 1
  fi
  TS=$(tok_sha $TOK); echo "TOKENS $FAM alpaca sha=$TS" | tee -a summary.txt; return 0; }
# tc1_family FAM MID REV FETCH_AL E4B_AL UNS_AL HF_AL AX_AL REF_AL PROF_AL -- the JUDGED family (phase 3 I [F4]), one process per arm, in this order:
#   1 e4b/fused_attn4_m  2 unsloth/ckpt_unsloth_m  3 e4b/reference_attn4_m (THIRD: the e4b-side control sits beside the pair it controls)
#   4 e4b/fused_attn4_m_d2  5 unsloth/ckpt_unsloth_m_d2  6 hf/hf_peft_m  7 axolotl/ckpt_axolotl_m
#   8 e4b/fused_attn4_m_prof  9 unsloth/ckpt_unsloth_prof (3 warm + 3 profiled, dmon beside each)  then the _mb1 pair when a primary matched arm OOMed.
# The labelled / native-best rows run under the `qwen3native` token (tc1_native_family) on their own box.
tc1_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6 HAL=$7 AAL=$8 RAL=$9 PAL=${TC1_PROF_ALARM:-${10}}
  local ALL="e4b:fused_attn4_m:fused unsloth:ckpt_unsloth_m:unsloth e4b:reference_attn4_m:reference e4b:fused_attn4_m_d2:fused unsloth:ckpt_unsloth_m_d2:unsloth hf:hf_peft_m:hf axolotl:ckpt_axolotl_m:axolotl e4b:fused_attn4_m_prof:fused unsloth:ckpt_unsloth_prof:unsloth"
  say "===== family $FAM ($MID @ $REV; matched seed $MATCHED_SEED; alarms e4b $EAL unsloth $UAL hf $HAL axolotl $AAL reference $RAL prof $PAL)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"      # the matched set
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"                   # the notebooks' recipe (tp4's arm); double-quant OFF is the arm's default
  local PROF="--log-every 1 --microbatch-timing 1 --profile-steps $PROFILE_STEPS --profile-warm $PROFILE_WARM"
  can_run 600 $FAM/e4b/fused_m     && arm   $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m       && arm   $FAM unsloth ckpt_unsloth_m unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 900 $FAM/e4b/reference_m && arm   $FAM e4b reference_attn4_m reference $RAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/e4b/fused_m_d2  && draw2 $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/m_d2    && draw2 $FAM unsloth ckpt_unsloth_m unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
  can_run 600 $FAM/hf/m            && arm   $FAM hf hf_peft_m hf $HAL "$MID" $REV 0 field $TOK $TS $MATCH
  can_run 600 $FAM/axolotl/m       && arm   $FAM axolotl ckpt_axolotl_m axolotl $AAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json $MATCH
  local dp; dp=$(dmon_start ${FAM}_e4b_fused_attn4_m_prof)
  can_run 600 $FAM/e4b/prof        && arm   $FAM e4b fused_attn4_m_prof fused $PAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH $PROF
  dmon_stop $dp
  dp=$(dmon_start ${FAM}_unsloth_ckpt_unsloth_prof)
  can_run 600 $FAM/unsloth/prof    && arm   $FAM unsloth ckpt_unsloth_prof unsloth $PAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH $PROF
  dmon_stop $dp
  for f in $W/logs/dmon_*.txt; do [ -s "$f" ] && echo "DMON $(basename $f) $(wc -l < $f) samples" >> summary.txt; done
  # the secondary pair (TC1-PREREG "Arms"): micro-batch 1 x accum 8 -- same tokens per step -- for any framework whose primary matched arm OOMed, as tp4
  local se su sh; se=$(status_of $FAM e4b fused_attn4_m); su=$(status_of $FAM unsloth ckpt_unsloth_m); sh=$(status_of $FAM hf hf_peft_m)
  if [ "$se" = oom ] || [ "$su" = oom ] || [ "$sh" = oom ]; then
    echo "SECONDARY $FAM: a primary matched arm OOMed (e4b=$se unsloth=$su hf=$sh) -> mb1 pair" | tee -a summary.txt
    can_run 600 $FAM/e4b/fused_m_mb1 && arm $FAM e4b fused_attn4_m_mb1 fused $EAL "$MID" $REV 0 mb1 $TOK $TS --attn-4bit 1 $MATCH
    can_run 600 $FAM/unsloth/m_mb1   && arm $FAM unsloth ckpt_unsloth_m_mb1 unsloth $UAL "$MID" $REV 0 mb1 $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH
    [ "$sh" = oom ] && can_run 600 $FAM/hf/m_mb1 && arm $FAM hf hf_peft_m_mb1 hf $HAL "$MID" $REV 0 mb1 $TOK $TS $MATCH
  fi
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# tc1_native_family FAM MID REV FETCH_AL E4B_AL UNS_AL HF_AL AX_AL -- the LABELLED / native-best rows on their own box (phase 3 I/J), each a position
# within this box against the e4b fused_m it runs first:
#   e4b/fused_attn4_m (once)  unsloth/ckpt_unsloth_best  unsloth/ckpt_unsloth_t28  unsloth/ckpt_unsloth_triton  e4b/fused_attn4_shipped
#   e4b/fused_attn4_m_nodgrad (--dgrad 0: enable_fast_train's default)  e4b/fused_attn4_m_t212 (e4b + gnf4 on torch 2.12.1+cu130)
#   axolotl/ckpt_axolotl_best  hf/hf_peft_m_mb1_t214 (the HF arm on the axolotl venv's torch 2.14, experts_implementation=grouped_mm; only when
#   the judged family's hf_peft_m on THIS box OOMed -- otherwise a not_run row saying the gate could not be read)
tc1_native_family(){ local FAM=$1 MID=$2 REV=$3 FAL=$4 EAL=$5 UAL=$6 HAL=$7 AAL=$8
  local ALL="e4b:fused_attn4_m:fused unsloth:ckpt_unsloth_best:unsloth unsloth:ckpt_unsloth_t28:unsloth unsloth:ckpt_unsloth_triton:unsloth e4b:fused_attn4_shipped:fused e4b:fused_attn4_m_nodgrad:fused e4b:fused_attn4_m_t212:fused axolotl:ckpt_axolotl_best:axolotl hf:hf_peft_m_mb1_t214:hf"
  say "===== NATIVE family $FAM ($MID @ $REV; the labelled rows, each against this box's own e4b fused_m)"
  local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0
  local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"
  local NATIVE="--adapter-dtype native --lora-init native"
  local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"
  can_run 600 $FAM/e4b/fused_m     && arm   $FAM e4b fused_attn4_m fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/unsloth/best    && arm   $FAM unsloth ckpt_unsloth_best unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend grouped_mm --unsloth-speed-tilt 1 --adapter-dtype fp32 --lora-init native
  can_run 600 $FAM/unsloth/t28     && UNS_VENV=t28 arm $FAM unsloth ckpt_unsloth_t28 unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend default $MATCH
  can_run 600 $FAM/unsloth/triton  && arm   $FAM unsloth ckpt_unsloth_triton unsloth $UAL "$MID" $REV 0 field $TOK $TS $UNS --unsloth-moe-backend unsloth_triton $MATCH
  can_run 600 $FAM/e4b/shipped     && arm   $FAM e4b fused_attn4_shipped fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $NATIVE
  can_run 600 $FAM/e4b/nodgrad     && arm   $FAM e4b fused_attn4_m_nodgrad fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 --dgrad 0 $MATCH
  can_run 600 $FAM/e4b/t212        && E4B_VENV=t212 arm $FAM e4b fused_attn4_m_t212 fused $EAL "$MID" $REV 0 field $TOK $TS --attn-4bit 1 $MATCH
  can_run 600 $FAM/axolotl/best    && arm   $FAM axolotl ckpt_axolotl_best axolotl $AAL "$MID" $REV 0 field $TOK $TS --axolotl-dataset $W/data/ds_alpaca.json --axolotl-best 1 --adapter-dtype fp32 --lora-init native
  local sh; sh=$(status_of qwen3 hf hf_peft_m)
  if [ "$sh" = oom ]; then
    can_run 600 $FAM/hf/mb1_t214   && HF_VENV=t214 arm $FAM hf hf_peft_m_mb1_t214 hf $HAL "$MID" $REV 0 mb1 $TOK $TS --hf-experts-implementation grouped_mm $MATCH
  else
    stubw $FAM hf hf_peft_m_mb1_t214 hf not_run "runs only when the judged family's hf/hf_peft_m on this box OOMed; its status here is '$sh'"
  fi
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}; }
# ---------------------------------------------------------------- the plan (TC1-PREREG "Model", "Alarms": e4b 3600, Unsloth 3600, HF 1800, axolotl 2700, reference 5400, profiled 2400; fetch as tp4's qwen3)
#      FAM    MID                 REV                                       FETCH E4B  UNS  HF   AX   REF  PROF
for FAM in $FAMILIES; do case "$FAM" in
  qwen3)       tc1_family        qwen3       Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600 1800 2700 5400 2400;;
  qwen3native) tc1_native_family qwen3native Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600 1800 2700;;
  *) say "unknown family token $FAM"; echo "UNKNOWN $FAM" >> summary.txt;;
esac; done
# ---------------------------------------------------------------- reduce, summarise, mark
say "reduce"
$PY_E4B $W/tc1_reduce.py $W --md $W/RESULTS-tc1.md --steps $STEPS > RESULTS.txt 2>&1; tail -40 RESULTS.txt
echo "----- summary.txt -----"; cat summary.txt; echo "----- versions.txt -----"; cat versions.txt
[ "$UNS_OK" = 1 ] || echo "NO cu130 UNSLOTH COMPARATOR on this box (venv-unsloth: ${CU130_OK:-?} driver gate, install ok=$UNS_OK): its rows are refused/install_failed" | tee -a summary.txt
[ "$UNS_T28_OK" = 1 ] || echo "NO torch-2.8 UNSLOTH ROW on this box (venv-unsloth-t28 did not install/import)" | tee -a summary.txt
[ "$AX_OK" = 1 ] || echo "NO AXOLOTL on this box: $AX_REASON" | tee -a summary.txt
finish 0
