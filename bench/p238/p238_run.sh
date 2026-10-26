#!/bin/bash
# P238's box runner (bench/p238/PREREG-p238.md): grouped-nf4-gemm 0.51.0's fused NF4 training GEMM (through experts4bit-qlora 0.58.1)
# against unsloth-zoo 2026.10.8's NF4-fused grouped GEMM, on Qwen3.6-35B-A3B, one RTX 5090. P236's runner (bench/p236/p236_run.sh) by
# named edits for the shape (the refusals, the guard fit, STARTED, finish), with TC1 amendment 80's venv-unsloth recipe
# (bench/tc1/tc1_run.sh: unsloth[cu130-torch2121], e4b and grouped-nf4-gemm from PyPI, the release proofs, the single-thread and card
# floors) for the environment, and this lane's pins (PEFT 0.21.2, bitsandbytes 0.50.2), its op processes (p238_op.py) and its 40 step
# loads (tc1_arm.py with the observe-only hook p238_hook.py), in the fixed order of p238_schedule.py. No instance id and no host name
# are written here. This script never creates compute.
#
# Exit codes (the PREREG's table; every one is a code the driver passes through):
#   0 a READ; 27 VOID; 13 disk, 15 card / cc / SMs / VRAM, 16 host RAM, 18 CUDA unusable / driver / single-thread / card clock
#   (host refusals, before STARTED); 41 the guard fit (before STARTED); 9 staging / install / pins / release proof, 10 torch does not
#   import, 78 run id, frozen knobs or an ambient knob, 130 a signal. This runner NEVER exits 13/14/17/18 except for the genuine
#   host fault its line names: the launcher excludes the machine on those.
set -uo pipefail
W=/root/p238; mkdir -p $W/logs $W/loads $W/ops $W/data; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p238: $*"; }
NONCE=${P238_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P238_RUN_NONCE.tmp && mv $W/P238_RUN_NONCE.tmp $W/P238_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P238_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P238_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P238_RUN_ID P238_BOX P238_DEADLINE_EPOCH E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
# ---- the run id and the box it runs (the driver derives BOX from the id; a mismatch is refused, never trusted): a whole reading
# runs every process (all); box i of a K-box split runs its whole (draw, point) blocks (K-i, p238_schedule.blocks); the shakedown
case "$P238_RUN_ID" in
  p238-shake-1|p238-shake-r[1-5]) WANT_BOX=shake;;
  p238-5090-[12]|p238-5090-r[1-3]) WANT_BOX=all;;
  p238-5090-[12]-2-[12]|p238-5090-[12]-4-[1-4]|p238-5090-[12]-8-[1-8]) WANT_BOX=${P238_RUN_ID#p238-5090-[12]-};;
  *) say "unregistered run ID: $P238_RUN_ID"; finish 78;;
esac
[ "$P238_BOX" = "$WANT_BOX" ] || { say "refusing: box $P238_BOX is not $P238_RUN_ID's ($WANT_BOX)"; finish 78; }
# ---- the pins: constants, refused if overridden
E4B_VERSION=0.58.1; GNF4_VERSION=0.51.0; UNSLOTH_VERSION=2026.10.8; ZOO_VERSION=2026.10.8; PEFT_VERSION=0.21.2; BNB_VERSION=0.50.2
NVML_VERSION=13.615.71              # nvidia-ml-py: the energy counter (p238_op.py) and card_clock's reason bits
TORCH_WANT=2.12.1+cu130
MODEL=Qwen/Qwen3.6-35B-A3B; REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0
CLOCK_FLOOR=2700; MIN_MOPS=15; GPU_CLASS=5090; MIN_RAM_GB=64; MIN_DISK_GB=200; MIN_DRIVER=580
STEPS=60; SEQ=2048; EVAL_N=48
LOAD_S=${P238_LOAD_S_REGISTERED:-900}   # the per-load wall the guard fit uses: the shakedown's measured wall, registered before the reading
case "$LOAD_S" in ''|*[!0-9]*|0*) say "refusing: P238_LOAD_S_REGISTERED '$LOAD_S' is not whole seconds (no leading 0: \$(( )) reads octal)"; finish 78;; esac
[ "$LOAD_S" -ge 300 ] || { say "refusing: P238_LOAD_S_REGISTERED $LOAD_S s is under the 300 s floor (a guard fit that passes trivially)"; finish 78; }
INSTALL_S=2700; FETCH_S=2400; OP_TIMING_S=1800; RESERVE=900
[ -z "${P238_MIN_DISK_GB:-}${P238_MIN_RAM_GB:-}${P238_STEPS:-}${P238_GPU_CLASS:-}" ] || { say 'reading knobs are frozen'; finish 78; }
# ---- ambient knobs (both frameworks alike): every knob an arm depends on is SET by this runner, explicitly; any other UNSLOTH_*,
# E4B_*, GNF4_* or NF4_QLORA_* key already in the box's environment is refused before STARTED (an e4b knob such as
# E4B_TRAIN_DETERMINISTIC can move e4b's step time without moving its route, as an ambient Unsloth knob can move Unsloth's)
for k in $(env | grep -oE '^(UNSLOTH|E4B|GNF4|NF4_QLORA)_[A-Za-z0-9_]+' | sort -u); do
  case "$k" in UNSLOTH_MOE_NF4_AUTO_MAX_ROWS|UNSLOTH_MOE_NF4_AUTOTUNE|UNSLOTH_MOE_GROUPED_TRITON|UNSLOTH_MOE_FUSED_ROUTER|UNSLOTH_ENABLE_LOGGING) ;;
    E4B_SHA) ;;                     # the launch commit the driver hands in (p238_drive.sh), required above
    *) say "refusing: an ambient $k in the box's environment"; finish 78;; esac
done
# ---- end ambient knobs
export UNSLOTH_MOE_NF4_AUTO_MAX_ROWS=128 UNSLOTH_MOE_NF4_AUTOTUNE=0 UNSLOTH_MOE_GROUPED_TRITON=auto UNSLOTH_MOE_FUSED_ROUTER=0 UNSLOTH_ENABLE_LOGGING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_TELEMETRY=1
: > summary.txt; echo "$P238_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_VERSION gnf4=$GNF4_VERSION unsloth=$UNSLOTH_VERSION zoo=$ZOO_VERSION peft=$PEFT_VERSION bnb=$BNB_VERSION torch=$TORCH_WANT box=$P238_BOX load_s=$LOAD_S" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
STAGED="p238_run.sh p238_op.py p238_hook.py p238_bytes.py p238_reduce.py p238_schedule.py staged-p238.sha256 unsloth_moe_counter.py
        card_clock.py tc1_arm.py f15v2.py release_proof.py tp4_alpaca.py"
for f in $STAGED; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p238.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p238/staged-p238.sha256"; finish 9; }
# ---- refusals before anything is installed: CUDA, the driver, the card, the disk, the host RAM, the single thread, the card clock, the fit
CUDA_PROBE=$(python3 - <<'PYC' 2>/dev/null | tail -1
import sys
try:
    import torch
except Exception as e:
    print(f"no-torch {type(e).__name__}"); sys.exit(0)
try:
    ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
except Exception:
    ok = False
if not ok:
    print(f"no-cuda torch {torch.__version__}"); sys.exit(0)
cc, p = torch.cuda.get_device_capability(0), torch.cuda.get_device_properties(0)
print(f"ok torch={torch.__version__} cc={cc[0]}.{cc[1]} sms={p.multi_processor_count}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  ok*) ;;
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE})"; echo "refused: cuda unusable" > REFUSAL; finish 18;;
  *) say "IMAGE FAULT: torch does not import (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' '); DRIVER_MAJOR=${DRIVER%%.*}
case "$DRIVER_MAJOR" in ''|*[!0-9]*) DRIVER_MAJOR=0;; esac
[ "$DRIVER_MAJOR" -ge "$MIN_DRIVER" ] || { say "REFUSED: driver ${DRIVER:-unknown} < $MIN_DRIVER (the cu130 wheels need it)"; echo "refused: driver ${DRIVER:-unknown}" > REFUSAL; finish 18; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap,power.limit,clocks.max.sm --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
# the image's own env, information only (never a fault; the reducer copies it as image_env): the version prefixes, never the whole
# env, which on a rented box carries the instance's key
env | grep -E '^(PYTORCH_|TORCH_|CUDA_|NV_|NVIDIA_)' | sort > $W/env_image.txt || true
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
SMS=${CUDA_PROBE##*sms=}; case "$CUDA_PROBE" in *" cc=12.0 sms="*) ;; *) SMS=0;; esac
[ "${SMS:-0}" -ge 160 ] 2>/dev/null || { say "REFUSED: $CUDA_PROBE -- the lane registers sm_120 with >= 160 SMs"; echo "refused: config $CUDA_PROBE" > REFUSAL; finish 15; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge 30000 ] || { say "REFUSED: VRAM ${VRAM_MIB:-?} MiB below the 30000 MiB floor"; echo "refused: vram ${VRAM_MIB:-?}" > REFUSAL; finish 15; }
# the card facts every row carries (p238_reduce.py's check_card): the name, the compute capability, the SM count
CC=${CUDA_PROBE##*cc=}; CC=${CC%% *}
printf '{"name": "%s", "cap": [%s, %s], "sm_count": %s, "vram_mib": %s, "driver": "%s"}\n' "$GPU_NAME" "${CC%%.*}" "${CC##*.}" "$SMS" "$VRAM_MIB" "$DRIVER" > $W/box.json
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
# TC1 amendment 77's single-thread floor (the launcher's cpu_loop_mops loop, the median of 5 runs of >= 2 s)
MOPS=$(python3 - <<'PYM' 2>/dev/null
import statistics, time
def loop():
    x = 0
    for i in range(1_000_000):
        x = (x + i * 7) & 0xFFFF
    return x
def once(min_s=2.0):
    t0, n = time.perf_counter(), 0
    while True:
        loop()
        n += 1
        dt = time.perf_counter() - t0
        if dt >= min_s:
            return n * 1_000_000 / dt / 1e6
print("%.2f" % statistics.median(once() for _ in range(5)))
PYM
)
echo "SINGLE_THREAD ${MOPS:-unreadable} Mops (floor $MIN_MOPS)" | tee -a summary.txt
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "${MOPS:-0}" "$MIN_MOPS" 2>/dev/null \
  || { say "REFUSED: single-thread speed ${MOPS:-unreadable} Mops under the floor ($MIN_MOPS)"; echo "refused: single thread ${MOPS:-unreadable}" > REFUSAL; finish 18; }
# TC1 amendment 83's loaded card clock (clock mode, named): a 20 s bf16 burn, busy SM median >= 2700 MHz, no slowdown bit on any sample,
# Reliability on < 95 % of busy samples; SwPowerCap recorded only
perl -e 'alarm 120; exec @ARGV' python3 $W/card_clock.py burn --floor $CLOCK_FLOOR --seconds 20 --mode clock --out $W/card_clock_burn.json > logs/card_clock_burn.log 2>&1; crc=$?
{ echo -n "CARD_CLOCK rc=$crc "; tail -1 logs/card_clock_burn.log | cut -c1-240; } | tee -a summary.txt
case "$crc" in 0) ;; 18) say "REFUSED: the card clock (TC1 amendment 83, clock mode)"; echo "refused: card clock" > REFUSAL; finish 18;; *) say "CARD CLOCK PROBE FAULT rc=$crc"; finish 9;; esac
# the guard fit: installs + the snapshot fetch + every process this box runs x the registered per-load wall + the op timing, x 1.25
case "$P238_BOX" in
  shake) N_PROC=7;;                       # the shakedown: draw 1's two op processes + one load per arm at 2k
  *) N_PROC=$(python3 $W/p238_schedule.py --box "$P238_BOX" | wc -l);;
esac
FIT=$((INSTALL_S + FETCH_S + N_PROC * LOAD_S + OP_TIMING_S + RESERVE)); FIT_NEED=$((FIT * 125 / 100)); LEFT=$((P238_DEADLINE_EPOCH - $(date +%s)))
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x 125%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT processes=$N_PROC fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
# ---- the start: every pre-install check has passed; from here the box counts toward the box-started cap
: > $W/STARTED; echo "STARTED $(date -u +%FT%TZ)" | tee -a summary.txt
left(){ echo $(( P238_DEADLINE_EPOCH - $(date +%s) )); }
alarm_for(){ local want=$1 l; l=$(( $(left) - RESERVE )); [ "$l" -lt "$want" ] && want=$l; [ "$want" -lt 300 ] && want=300; echo "$want"; }
# ---- one venv for both frameworks (TC1 amendment 80's venv-unsloth), every distribution pinned; --no-compile under the fresh prefix rule
export DEBIAN_FRONTEND=noninteractive
PY=$W/venv-unsloth/bin/python
python3 -m venv $W/venv-unsloth && $PY -m pip install -q --no-input -U pip > logs/pip_upgrade.log 2>&1 || { say "VENV FAIL"; finish 9; }
say "venv-unsloth: unsloth[cu130-torch2121]==$UNSLOTH_VERSION unsloth_zoo==$ZOO_VERSION peft==$PEFT_VERSION bitsandbytes==$BNB_VERSION"
perl -e 'alarm 2700; exec @ARGV' $PY -m pip install -q --no-input --no-compile --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 \
  "unsloth[cu130-torch2121]==$UNSLOTH_VERSION" "unsloth_zoo==$ZOO_VERSION" "peft==$PEFT_VERSION" "bitsandbytes==$BNB_VERSION" \
  "nvidia-ml-py==$NVML_VERSION" datasets safetensors "huggingface_hub>=0.23" > logs/pip_unsloth.log 2>&1 \
  || { tail -6 logs/pip_unsloth.log; say "PIP FAIL (venv-unsloth)"; finish 9; }
say "venv-unsloth + experts4bit-qlora==$E4B_VERSION + grouped-nf4-gemm==$GNF4_VERSION from PyPI"
perl -e 'alarm 2400; exec @ARGV' $PY -m pip install -q --no-input --no-compile --prefer-binary --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 --only-binary experts4bit-qlora,grouped-nf4-gemm \
  "experts4bit-qlora==$E4B_VERSION" "grouped-nf4-gemm==$GNF4_VERSION" "peft==$PEFT_VERSION" "bitsandbytes==$BNB_VERSION" > logs/pip_e4b.log 2>&1 \
  || { tail -6 logs/pip_e4b.log; say "PIP FAIL (e4b / gnf4)"; finish 9; }
TF_VERSION=$($PY -c "import importlib.metadata as m; print(m.version('transformers'))" 2>/dev/null)
echo "TRANSFORMERS resolved $TF_VERSION (unsloth's cap <= 5.17.0)" | tee -a summary.txt
WANT="experts4bit-qlora=$E4B_VERSION grouped-nf4-gemm=$GNF4_VERSION unsloth=$UNSLOTH_VERSION unsloth_zoo=$ZOO_VERSION peft=$PEFT_VERSION bitsandbytes=$BNB_VERSION" \
WANT_TORCH=$TORCH_WANT $PY - <<'PYT' > logs/tripwire.log 2>&1 || { tail -5 logs/tripwire.log; say "TRIPWIRE FAIL"; finish 9; }
import importlib.metadata as md, os
import torch
assert torch.__version__ == os.environ["WANT_TORCH"], torch.__version__
for pair in os.environ["WANT"].split():
    dist, want = pair.split("=")
    d = md.distribution(dist)
    assert d.version == want, (dist, d.version, want)
    assert d.read_text("direct_url.json") is None, (dist, "not an index install")
import unsloth, unsloth_zoo, peft, bitsandbytes                     # noqa: F401
from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import _moe_uses_bnb4bit_expert_weights   # noqa: F401 -- the counter's hand-off rule
from unsloth_zoo.temporary_patches.moe_nf4_grouped import maybe_nf4_grouped_mm, nf4_auto_fused, NF4_FUSED_STATS   # noqa: F401 -- #1935
import experts4bit_qlora                                            # noqa: F401
from experts4bit_qlora.engines.fast import enable_fast_train, FAST_TRAIN_STATS   # noqa: F401
import nf4_qlora, nf4_route                                         # noqa: F401
import pynvml                                                       # noqa: F401
assert torch.cuda.is_available()
open("/root/p238/versions.txt", "a").write("".join(f"{d} {md.version(d)}\n" for d in
    ("experts4bit-qlora", "grouped-nf4-gemm", "unsloth", "unsloth_zoo", "peft", "bitsandbytes", "transformers", "torch", "nvidia-ml-py")))
print("tripwire OK")
PYT
cat versions.txt | tee -a summary.txt
# the lane's own self-tests, in the arms' environment
$PY $W/p238_reduce.py --selftest | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 9; }
# ---- the data: TC1's registered Alpaca text, then one packed token file (>= 60 x 4 rows of 2048) and one field file (micro-batch 2)
DS_ALPACA_SHA=5324987afa4042556953026289e8dbdbe8a936b32832ed9e603b9192b706a2fb
(cd $W/data && perl -e 'alarm 900; exec @ARGV' $PY $W/tp4_alpaca.py --out $W/data/ds_alpaca.json > $W/logs/dataset_alpaca.log 2>&1) || { say "DATASET FAIL"; finish 9; }
[ "$(sha256sum $W/data/ds_alpaca.json | awk '{print $1}')" = "$DS_ALPACA_SHA" ] || { say "DATASET MISMATCH"; finish 9; }
# ---- one snapshot download, shared by both frameworks
AL=$(alarm_for $FETCH_S); say "snapshot $MODEL @$REVISION (alarm $AL)"
perl -e "alarm $AL; exec @ARGV" $PY -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REVISION'))" > logs/snapshot.log 2>&1 \
  || { tail -3 logs/snapshot.log; say "SNAPSHOT FAIL"; finish 9; }
export HF_HUB_OFFLINE=1
TC1_FLAGS="--fam qwen3_5 --model $MODEL --revision $REVISION --r 8 --alpha 16 --lora-init matched:3407 --adapter-dtype fp32 --seq $SEQ
           --grad-ckpt unsloth --optim adamw_8bit --steps $STEPS --accum 1 --eval-n $EVAL_N --eval-every $STEPS --profile-steps 3 --profile-warm 20
           --template alpaca"
for kind in packed field; do
  if [ $kind = packed ]; then PK="--pack 1 --pack-src $W/data/alpaca_data_cleaned.json --pack-min-rows $(( STEPS * 4 ))"; else PK=""; fi
  $PY $W/tc1_arm.py --prepare $TC1_FLAGS --data $W/data/ds_alpaca.json --data-sha $DS_ALPACA_SHA --tokens $W/data/tokens_$kind.json $PK \
    > logs/prepare_$kind.log 2>&1 || { tail -3 logs/prepare_$kind.log; say "PREPARE FAIL ($kind)"; finish 9; }
  sha256sum $W/data/tokens_$kind.json | tee -a summary.txt
done
# ---- the binding release proof, in the arms' own import environment, the last step before the first process
PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/p238-pycache.XXXXXX); export PYTHONPYCACHEPREFIX RELEASE_PROOF_FRESH_PYCACHE=$PYTHONPYCACHEPREFIX
(cd $W && RELEASE_PROOF_NETWORK=1 PYTHONPATH= perl -e 'alarm 1200; exec @ARGV' $PY $W/release_proof.py \
   "experts4bit-qlora==$E4B_VERSION" "grouped-nf4-gemm==$GNF4_VERSION" "unsloth==$UNSLOTH_VERSION" "unsloth_zoo==$ZOO_VERSION" \
   "peft==$PEFT_VERSION" "bitsandbytes==$BNB_VERSION" "transformers==$TF_VERSION") > $W/release_proof.json 2> logs/release_proof.err
rp=$?; { echo -n "RELEASE_PROOF rc=$rp "; head -c 300 logs/release_proof.err; echo; } | tee -a summary.txt
[ "$rp" = 0 ] || { say "RELEASE PROOF FAILED"; finish 9; }
env | grep -E '^(UNSLOTH_|E4B_|GNF4_|NF4_QLORA_)|^PYTORCH_CUDA_ALLOC_CONF=' | sort > $W/env_box.txt
# the hook's pin: the sha256 the staged pin file names for p238_hook.py (tc1_arm.py's load_arm_hook refuses any other bytes, exit 10)
HOOK_SHA=$(awk '$2 == "p238_hook.py" || $2 == "*p238_hook.py" {print $1}' $W/staged-p238.sha256)
[ ${#HOOK_SHA} -eq 64 ] || { say "STAGE: staged-p238.sha256 names no p238_hook.py"; finish 9; }
# one clock sample every 0.25 s in card_clock.py's gpuclk format ("EPOCH sm, mem, temp, reasons"), per load
clock_log(){ while :; do echo "$(date +%s) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits -i 0 2>/dev/null)"; sleep 0.25; done; }
# ---- the processes, in the schedule's fixed order
arm_env(){ case "$1" in
  E_F) echo "GNF4_TRAIN_GEMM=fused E4B_ABSMAX_DQ=1";;   E_A) echo "GNF4_TRAIN_GEMM=auto E4B_ABSMAX_DQ=1";;
  U0) echo "UNSLOTH_MOE_NF4_FUSED=0";;  U1) echo "UNSLOTH_MOE_NF4_FUSED=1";;  UA) echo "UNSLOTH_MOE_NF4_FUSED=auto";; esac; }
arm_flags(){ case "$1" in
  E_F|E_A) echo "--framework e4b --arm fused --attn-4bit 1 --absmax-dq 1";;
  U0|U1|UA) echo "--framework unsloth --arm unsloth --unsloth-loader FastModel --unsloth-double-quant default --unsloth-targets q_proj,k_proj,v_proj,o_proj --unsloth-target-parameters mlp.experts.gate_up_proj,mlp.experts.down_proj";; esac; }
point_flags(){ case "$1" in
  2k) echo "--micro-batch 1 --tokens $W/data/tokens_packed.json";;  4k) echo "--micro-batch 2 --tokens $W/data/tokens_packed.json";;
  8k) echo "--micro-batch 4 --tokens $W/data/tokens_packed.json";;  field) echo "--micro-batch 2 --tokens $W/data/tokens_field.json";; esac; }
BOX_ARGS="--box $P238_BOX"; [ "$P238_BOX" = shake ] && BOX_ARGS="--draw 1"
python3 $W/p238_schedule.py $BOX_ARGS > $W/schedule.txt
while read -r idx kind draw point who pos; do
  if [ "$P238_BOX" = shake ]; then case "$kind:$point" in op:*|step:2k) ;; *) continue;; esac; fi
  AL=$(alarm_for $((LOAD_S * 2)))
  if [ "$kind" = op ]; then
    say "[$idx] op $who draw $draw (alarm $AL)"
    (cd $W && PYTHONPATH=$W perl -e "alarm $AL; exec @ARGV" $PY $W/p238_op.py --framework $who --draw $draw --out $W/ops/op_${idx}_$who.json \
       $( [ "$P238_BOX" = shake ] && echo "--rows 64" )) > logs/op_$idx.log 2>&1
    echo "PROC $idx op $who draw=$draw rc=$?" | tee -a summary.txt
  else
    say "[$idx] step $who $point draw $draw pos $pos (alarm $AL)"
    clock_log > $W/loads/gpuclk_$idx.txt 2>/dev/null &
    SMI=$!
    nvidia-smi --query-gpu=timestamp,power.draw.instant --format=csv,noheader,nounits -lms 250 > $W/loads/power_$idx.txt 2>/dev/null &
    PWR=$!
    # the hook seam (tc1_arm's TC1_ARM_HOOK, its own reviewed PR): the observe-only P238 hook, its sha256 in the staged pin
    (cd $W && env $(arm_env $who) TC1_ARM_HOOK=p238_hook:make TC1_ARM_HOOK_SHA256=$HOOK_SHA PYTHONPATH=$W perl -e "alarm $AL; exec @ARGV" \
       $PY $W/tc1_arm.py $TC1_FLAGS $(arm_flags $who) $(point_flags $point) --tag p238_${idx}_${who}_${point} --out $W/loads/$idx \
       --adapter-dir $W/loads/$idx/adapters --prereg bench/p238/PREREG-p238.md) > logs/load_$idx.log 2>&1
    rc=$?; kill $SMI $PWR 2>/dev/null; wait $SMI $PWR 2>/dev/null
    echo "PROC $idx step $who $point draw=$draw pos=$pos rc=$rc" | tee -a summary.txt
  fi
done < $W/schedule.txt
# ---- the rule
# a box of a split (K-i) reduces its own faults only (PART_OK, never read alone; the controller merges the K boxes), the
# shakedown likewise (SHAKE_OK, never read); a one-box reading must cover the whole schedule
case "$P238_BOX" in all) MODE=read;; shake) MODE=shake;; *) MODE=part;; esac
say "reduce"; $PY $W/p238_reduce.py --work $W --mode $MODE --out $W/verdict_p238.json 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p238.json" ] || { say "REDUCER FAILED: no reading"; finish 27; }
V=$($PY -c "import json; print(json.load(open('$W/verdict_p238.json'))['verdict'])")
[ "$P238_BOX" = shake ] && { say "shakedown: $V (never read)"; finish 0; }
case "$V:$P238_BOX" in READ:all|PART_OK:[248]-[1-8]) finish 0;; *) say "P238 $V (box $P238_BOX) -- no reading"; finish 27;; esac
