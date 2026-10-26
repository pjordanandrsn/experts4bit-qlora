#!/bin/bash
# P237's box runner (bench/p237/PREREG-p237.md): experts4bit-qlora 0.58.1's CPU expert offload with layer-major accumulation (E)
# against Unsloth 2026.10.8's offload_layers ("auto" UA, tuned UT) on Qwen3-30B-A3B, one RTX 4090, at a 12 GiB cap and the full card.
# P236's runner (bench/p236/p236_run.sh) by named edits for the shape (the nonce, the refusals, the guard fit, STARTED, finish), with
# TC1 amendment 80's venv-unsloth recipe (bench/tc1/tc1_run.sh's qwen3pos80 family: unsloth[cu130-torch2121], e4b and
# grouped-nf4-gemm from PyPI, the release proof, the card clock) for the environment, amendment 80's two arms (fused_attn4_m_fd and
# ckpt_unsloth_m_fv, unchanged) plus the harness extension's options (e4b#2146: --vram-cap-gib, --unsloth-offload-layers,
# --layer-major) and the observe-only hook p237_arm.py, in the fixed order of p237_schedule.py. No instance id and no host name are
# written here. This script never creates compute.
#
# Exit codes (the PREREG's table; every one is a code the driver passes through):
#   0 a READ (or a finished shakedown); 27 VOID; 13 disk, 14 egress (a MEASURED download under the floor), 18 CUDA unusable / driver /
#   CPU class / card clock (the launcher's host codes: each is written only on a line stating its host-floor reason); 15 card class /
#   cc / SMs / VRAM, 16 host RAM (host facts the launcher does not exclude on); 41 the guard fit (before STARTED); 9 staging / install
#   / pins / release proof / a probe that measured nothing, 10 torch does not import, 29 the dataset (TC1 amendment 82: never 13,
#   and never 21, the driver's own start failure), 78 run id, frozen knobs or an ambient knob, 130 a signal. No box code is one the
#   driver exits on its own (p237_drive.sh, bench/common/drive_tail.sh), except 78, a config refusal on both sides.
set -uo pipefail
W=/root/p237; mkdir -p $W/logs $W/loads $W/data; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p237: $*"; }
NONCE=${P237_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P237_RUN_NONCE.tmp && mv $W/P237_RUN_NONCE.tmp $W/P237_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P237_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P237_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P237_RUN_ID P237_MODE P237_DEADLINE_EPOCH E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
# ---- the run id and the mode it runs (the driver derives MODE from the id; a mismatch is refused, never trusted)
case "$P237_RUN_ID" in
  p237-shake-1|p237-shake-r[12]) WANT_MODE=shake;;
  p237-4090-1|p237-4090-r[12]) WANT_MODE=read;;
  *) say "unregistered run ID: $P237_RUN_ID"; finish 78;;
esac
[ "$P237_MODE" = "$WANT_MODE" ] || { say "refusing: mode $P237_MODE is not $P237_RUN_ID's ($WANT_MODE)"; finish 78; }
# ---- the pins: constants, refused if overridden
E4B_VERSION=0.58.1; GNF4_VERSION=0.51.0; UNSLOTH_VERSION=2026.10.8; ZOO_VERSION=2026.10.8
TORCH_WANT=2.12.1+cu130
MODEL=Qwen/Qwen3-30B-A3B; REVISION=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
CLOCK_FLOOR=2500; GPU_CLASS=4090; MIN_SMS=128; MIN_VRAM_MIB=23000; MIN_RAM_GIB=128; MIN_DISK_GB=200; MIN_DRIVER=580
STEPS=60; PRIMER_STEPS=10; PROBE_STEPS=8; SHAKE_STEPS=24; CAP_12G=12
LOAD_S=${P237_LOAD_S_REGISTERED:-1500}  # the per-process wall the guard fit uses: the shakedown's measured wall, registered before the reading
case "$LOAD_S" in ''|*[!0-9]*|0*) say "refusing: P237_LOAD_S_REGISTERED '$LOAD_S' is not whole seconds (no leading 0: \$(( )) reads octal)"; finish 78;; esac
[ "$LOAD_S" -ge 300 ] || { say "refusing: P237_LOAD_S_REGISTERED $LOAD_S s is under the 300 s floor (a guard fit that passes trivially)"; finish 78; }
INSTALL_S=2700; FETCH_S=3600; RESERVE=900
[ -z "${P237_MIN_DISK_GB:-}${P237_MIN_RAM_GIB:-}${P237_STEPS:-}${P237_GPU_CLASS:-}${P237_CLOCK_FLOOR:-}" ] || { say 'reading knobs are frozen'; finish 78; }
# ---- ambient knobs (both frameworks alike): every knob an arm depends on is SET by this runner, explicitly; any other UNSLOTH_*,
# E4B_*, GNF4_*, NF4_QLORA_* or TC1_* key already in the box's environment is refused before STARTED (an e4b knob such as
# E4B_TRAIN_DETERMINISTIC can move e4b's step time as an ambient Unsloth knob can move Unsloth's; a TC1_* key changes the harness).
# The per-process keys (E4B_OFFLOAD_STATS, TC1_STEP_MEM, TC1_ARM_HOOK, TC1_ARM_HOOK_SHA256) are set on each process's own line.
# p237_reduce.py's ENV_PREFIXES / ENV_EXACT / ENV_BOX are this list; tests/test_p237.py binds the two.
for k in $(env | grep -oE '^(UNSLOTH|E4B|GNF4|NF4_QLORA|TC1)_[A-Za-z0-9_]+' | sort -u); do
  case "$k" in UNSLOTH_MOE_GROUPED_TRITON|UNSLOTH_MOE_NF4_FUSED|UNSLOTH_MOE_NF4_AUTO_MAX_ROWS|UNSLOTH_MOE_NF4_AUTOTUNE|UNSLOTH_MOE_FUSED_ROUTER|UNSLOTH_ENABLE_LOGGING) ;;
    GNF4_TRAIN_GEMM) ;;
    E4B_SHA) ;;                     # the launch commit the driver hands in (p237_drive.sh), required above
    *) say "refusing: an ambient $k in the box's environment"; finish 78;; esac
done
# ---- end ambient knobs
export UNSLOTH_MOE_GROUPED_TRITON=auto UNSLOTH_MOE_NF4_FUSED=0 UNSLOTH_MOE_NF4_AUTO_MAX_ROWS=128 UNSLOTH_MOE_NF4_AUTOTUNE=0 UNSLOTH_MOE_FUSED_ROUTER=0 UNSLOTH_ENABLE_LOGGING=1
export GNF4_TRAIN_GEMM=auto
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_TELEMETRY=1
: > summary.txt; echo "$P237_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_VERSION gnf4=$GNF4_VERSION unsloth=$UNSLOTH_VERSION zoo=$ZOO_VERSION torch=$TORCH_WANT mode=$P237_MODE load_s=$LOAD_S" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
STAGED="p237_run.sh p237_arm.py p237_host.py p237_egress_probe.py p237_reduce.py p237_schedule.py staged-p237.sha256
        unsloth_moe_counter.py card_clock.py tc1_arm.py f15v2.py release_proof.py tp4_alpaca.py"
for f in $STAGED; do [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-p237.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p237/staged-p237.sha256"; finish 9; }
# ---- refusals before anything is installed: CUDA, the driver, the card, the disk, the host RAM, the CPU class, the card clock, egress, the fit
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
print(f"ok torch={torch.__version__} cc={cc[0]}.{cc[1]} sms={p.multi_processor_count} total={p.total_memory}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  ok*) ;;
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE})"; echo "BOX_REFUSED host floor: cuda unusable" | tee -a summary.txt > REFUSAL; finish 18;;
  *) say "IMAGE FAULT: torch does not import (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' '); DRIVER_MAJOR=${DRIVER%%.*}
case "$DRIVER_MAJOR" in ''|*[!0-9]*) DRIVER_MAJOR=0;; esac
[ "$DRIVER_MAJOR" -ge "$MIN_DRIVER" ] || { say "REFUSED: driver ${DRIVER:-unknown} < $MIN_DRIVER"; echo "BOX_REFUSED host floor: driver ${DRIVER:-unknown} (the cu130 wheels need $MIN_DRIVER)" | tee -a summary.txt > REFUSAL; finish 18; }
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,power.limit,clocks.max.sm,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader | tee forensics.txt
PCIE_GEN=$(nvidia-smi --query-gpu=pcie.link.gen.max --format=csv,noheader,nounits | head -1 | tr -dc 0-9); PCIE_WIDTH=$(nvidia-smi --query-gpu=pcie.link.width.max --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
env | grep -E '^(PYTORCH_|TORCH_|CUDA_|NV_|NVIDIA_)' | sort > $W/env_image.txt || true   # the image's own environment: information, never a fault
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
SMS=${CUDA_PROBE##*sms=}; SMS=${SMS%% *}; case "$CUDA_PROBE" in *" cc=8.9 sms="*) ;; *) SMS=0;; esac
[ "${SMS:-0}" -ge $MIN_SMS ] 2>/dev/null || { say "REFUSED: $CUDA_PROBE -- the lane registers sm_89 with >= $MIN_SMS SMs"; echo "refused: config $CUDA_PROBE" > REFUSAL; finish 15; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge $MIN_VRAM_MIB ] || { say "REFUSED: VRAM ${VRAM_MIB:-?} MiB below the $MIN_VRAM_MIB MiB floor"; echo "refused: vram ${VRAM_MIB:-?}" > REFUSAL; finish 15; }
TOTAL_BYTES=${CUDA_PROBE##*total=}
CARD_GIB=$(python3 $W/p237_host.py cap_gib "$TOTAL_BYTES") || { say "unreadable card total: $CUDA_PROBE"; finish 9; }
CPU_NAME=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)
CPU_CLASS=$(python3 $W/p237_host.py cpu "$CPU_NAME"); ccr=$?
CC=${CUDA_PROBE##*cc=}; CC=${CC%% *}
printf '{"name": "%s", "cap": [%s, %s], "sm_count": %s, "vram_mib": %s, "total_bytes": %s, "card_gib": %s, "driver": "%s", "cpu": "%s", "cpu_class": "%s", "pcie_gen_max": "%s", "pcie_width_max": "%s"}\n' \
  "$GPU_NAME" "${CC%%.*}" "${CC##*.}" "$SMS" "$VRAM_MIB" "$TOTAL_BYTES" "$CARD_GIB" "$DRIVER" "$CPU_NAME" "$CPU_CLASS" "${PCIE_GEN:-}" "${PCIE_WIDTH:-}" > $W/box.json
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "BOX_REFUSED disk=${FREE_GB:-?}GB (host floor $MIN_DISK_GB GB)" | tee -a summary.txt > REFUSAL; finish 13; }
RAM_GIB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GIB:-0}" -ge "$MIN_RAM_GIB" ] || { say "REFUSED: ${RAM_GIB:-?} GiB host RAM < ${MIN_RAM_GIB} GiB"; echo "refused: ram ${RAM_GIB:-?} GiB" > REFUSAL; finish 16; }
[ "$ccr" = 0 ] || { say "REFUSED: CPU '$CPU_NAME' is not a registered fast class"; echo "BOX_REFUSED host floor: cpu class ($CPU_NAME)" | tee -a summary.txt > REFUSAL; finish 18; }
echo "HOST cpu_class=$CPU_CLASS ram=${RAM_GIB}GiB disk=${FREE_GB}GB card_gib=$CARD_GIB" | tee -a summary.txt
# TC1 amendment 80's loaded card clock, with the 4090's own floor: a 20 s bf16 burn, busy SM median >= 2500 MHz, no hard reason on any
# sample, SwPowerCap clear on >= 95 %
perl -e 'alarm 120; exec @ARGV' python3 $W/card_clock.py burn --floor $CLOCK_FLOOR --seconds 20 --min-clean 0.95 --out $W/card_clock_burn.json > logs/card_clock_burn.log 2>&1; crc=$?
{ echo -n "CARD_CLOCK rc=$crc "; tail -1 logs/card_clock_burn.log | cut -c1-240; } | tee -a summary.txt
case "$crc" in 0) ;; 18) say "REFUSED: the card clock is under its floor"; echo "BOX_REFUSED host floor: card clock (busy median under $CLOCK_FLOOR MHz or a reason bit)" | tee -a summary.txt > REFUSAL; finish 18;; *) say "CARD CLOCK PROBE FAULT rc=$crc"; finish 9;; esac
# the download floor (40 MB/s): only a MEASURED slow read is the egress code; no transfer at all names nothing (9)
perl -e 'alarm 120; exec @ARGV' python3 $W/p237_egress_probe.py "$MODEL" "$REVISION" > logs/egress_probe.log 2>&1; erc=$?
tail -1 logs/egress_probe.log | tee -a summary.txt
if [ "$erc" = 4 ]; then echo "BOX_REFUSED egress: $(tail -1 logs/egress_probe.log | cut -c1-200)" | tee -a summary.txt > REFUSAL; finish 14
elif [ "$erc" != 0 ]; then say "EGRESS PROBE ERROR rc=$erc (not a host refusal)"; finish 9; fi
# the guard fit (p237_host.py fit: one function for the runner, the reducer and the PREREG's lines): (installs + the snapshot fetch +
# every process this box runs x the registered per-process wall + reserve) x 1.25, against the cap min(the per-run wall-clock override's 18 h, the
# USD 15 run-worst); a fit over the cap is never trimmed: the lane stops and reports
read -r N_PROC FIT_NEED CAP_S PMAX < <(python3 $W/p237_host.py fit "$P237_MODE" "$LOAD_S") && [ -n "${PMAX:-}" ] || { say "FIT UNREAD"; finish 9; }
LEFT=$((P237_DEADLINE_EPOCH - $(date +%s)))
[ "$FIT_NEED" -le "$CAP_S" ] || { say "REFUSED: the fit ${FIT_NEED}s is over the cap ${CAP_S}s (policy wall clock / USD 15): the lane stops"; echo "refused: fit ${FIT_NEED}s over the cap ${CAP_S}s" > REFUSAL; finish 41; }
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s, ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT processes=$N_PROC need=${FIT_NEED}s cap=${CAP_S}s left=${LEFT}s probes_max=$PMAX" | tee -a summary.txt
# ---- the start: every pre-install check has passed; from here the box counts toward the box-started cap
: > $W/STARTED; echo "STARTED $(date -u +%FT%TZ)" | tee -a summary.txt
left(){ echo $(( P237_DEADLINE_EPOCH - $(date +%s) )); }
alarm_for(){ local want=$1 l; l=$(( $(left) - RESERVE )); [ "$l" -lt "$want" ] && want=$l; [ "$want" -lt 300 ] && want=300; echo "$want"; }
# ---- one venv for both frameworks (TC1 amendment 80's venv-unsloth), every distribution pinned; --no-compile under the fresh prefix rule
export DEBIAN_FRONTEND=noninteractive
PY=$W/venv-unsloth/bin/python
python3 -m venv $W/venv-unsloth && $PY -m pip install -q --no-input -U pip > logs/pip_upgrade.log 2>&1 || { say "VENV FAIL"; finish 9; }
say "venv-unsloth: unsloth[cu130-torch2121]==$UNSLOTH_VERSION unsloth_zoo==$ZOO_VERSION"
perl -e 'alarm 2700; exec @ARGV' $PY -m pip install -q --no-input --no-compile --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 \
  "unsloth[cu130-torch2121]==$UNSLOTH_VERSION" "unsloth_zoo==$ZOO_VERSION" nvidia-ml-py datasets safetensors "huggingface_hub>=0.23" > logs/pip_unsloth.log 2>&1 \
  || { tail -6 logs/pip_unsloth.log; say "PIP FAIL (venv-unsloth)"; finish 9; }
say "venv-unsloth + experts4bit-qlora==$E4B_VERSION + grouped-nf4-gemm==$GNF4_VERSION from PyPI"
perl -e 'alarm 2400; exec @ARGV' $PY -m pip install -q --no-input --no-compile --prefer-binary --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 --only-binary experts4bit-qlora,grouped-nf4-gemm \
  "experts4bit-qlora==$E4B_VERSION" "grouped-nf4-gemm==$GNF4_VERSION" > logs/pip_e4b.log 2>&1 \
  || { tail -6 logs/pip_e4b.log; say "PIP FAIL (e4b / gnf4)"; finish 9; }
TF_VERSION=$($PY -c "import importlib.metadata as m; print(m.version('transformers'))" 2>/dev/null)
echo "TRANSFORMERS resolved $TF_VERSION (unsloth's cap <= 5.17.0)" | tee -a summary.txt
WANT="experts4bit-qlora=$E4B_VERSION grouped-nf4-gemm=$GNF4_VERSION unsloth=$UNSLOTH_VERSION unsloth_zoo=$ZOO_VERSION" \
WANT_TORCH=$TORCH_WANT $PY - <<'PYT' > logs/tripwire.log 2>&1 || { tail -5 logs/tripwire.log; say "TRIPWIRE FAIL"; finish 9; }
import importlib.metadata as md, os
import torch
assert torch.__version__ == os.environ["WANT_TORCH"], torch.__version__
assert callable(getattr(torch.cuda, "get_per_process_memory_fraction", None)), "no get_per_process_memory_fraction: the cap cannot be read back"
for pair in os.environ["WANT"].split():
    dist, want = pair.split("=")
    d = md.distribution(dist)
    assert d.version == want, (dist, d.version, want)
    assert d.read_text("direct_url.json") is None, (dist, "not an index install")
import unsloth, unsloth_zoo, peft, bitsandbytes                     # noqa: F401
from unsloth_zoo import block_swap                                  # noqa: F401 -- the hook wraps _Block.prefetch / evict
assert callable(block_swap._Block.prefetch) and callable(block_swap._Block.evict)
from unsloth_zoo.temporary_patches.moe_nf4_grouped import NF4_FUSED_STATS   # noqa: F401 -- check_unfused reads it
import experts4bit_qlora                                            # noqa: F401
from experts4bit_qlora.engines import offload, layer_major          # noqa: F401 -- the hook wraps record_copy / layer_major_step
assert callable(offload._OffloadStats.record_copy) and callable(layer_major.layer_major_step)
import nf4_qlora, nf4_route                                         # noqa: F401
import pynvml                                                       # noqa: F401
assert torch.cuda.is_available()
open("/root/p237/versions.txt", "a").write("".join(f"{d} {md.version(d)}\n" for d in
    ("experts4bit-qlora", "grouped-nf4-gemm", "unsloth", "unsloth_zoo", "peft", "bitsandbytes", "transformers", "torch", "nvidia-ml-py")))
print("tripwire OK")
PYT
cat versions.txt | tee -a summary.txt
# the lane's own self-tests, in the arms' environment
$PY $W/p237_reduce.py --selftest | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 9; }
# ---- the data: TC1's registered Alpaca text (a dataset refusal is lane code 21, TC1 amendment 82)
DS_ALPACA_SHA=5324987afa4042556953026289e8dbdbe8a936b32832ed9e603b9192b706a2fb
(cd $W/data && perl -e 'alarm 900; exec @ARGV' $PY $W/tp4_alpaca.py --out $W/data/ds_alpaca.json > $W/logs/dataset_alpaca.log 2>&1) || { say "DATASET FAIL (alpaca)"; finish 29; }
[ "$(sha256sum $W/data/ds_alpaca.json | awk '{print $1}')" = "$DS_ALPACA_SHA" ] || { say "DATASET MISMATCH alpaca"; finish 29; }
# ---- one snapshot download, shared by both frameworks
AL=$(alarm_for $FETCH_S); say "snapshot $MODEL @$REVISION (alarm $AL)"
perl -e "alarm $AL; exec @ARGV" $PY -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REVISION'))" > logs/snapshot.log 2>&1 \
  || { tail -3 logs/snapshot.log; say "SNAPSHOT FAIL"; finish 9; }
export HF_HUB_OFFLINE=1
TOK=$W/data/tokens_qwen3.json
$PY $W/tc1_arm.py --prepare --fam qwen3 --model "$MODEL" --revision $REVISION --data $W/data/ds_alpaca.json --data-sha $DS_ALPACA_SHA \
  --seq 2048 --eval-n 8 --template alpaca --tokens $TOK > logs/prepare.log 2>&1 || { tail -3 logs/prepare.log; say "PREPARE FAIL"; finish 9; }
TOK_SHA=$($PY -c "import json; print(json.load(open('$TOK'))['sha256'])")
echo "TOKENS sha=$TOK_SHA" | tee -a summary.txt
# ---- the binding release proof, in the arms' own import environment, the last step before the first process
PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/p237-pycache.XXXXXX); export PYTHONPYCACHEPREFIX RELEASE_PROOF_FRESH_PYCACHE=$PYTHONPYCACHEPREFIX
(cd $W && RELEASE_PROOF_NETWORK=1 PYTHONPATH= perl -e 'alarm 1200; exec @ARGV' $PY $W/release_proof.py \
   "experts4bit-qlora==$E4B_VERSION" "grouped-nf4-gemm==$GNF4_VERSION" "unsloth==$UNSLOTH_VERSION" "unsloth_zoo==$ZOO_VERSION" \
   "transformers==$TF_VERSION") > $W/release_proof.json 2> logs/release_proof.err
rp=$?; { echo -n "RELEASE_PROOF rc=$rp "; head -c 300 logs/release_proof.err; echo; } | tee -a summary.txt
[ "$rp" = 0 ] || { say "RELEASE PROOF FAILED"; finish 9; }
env | grep -E '^(UNSLOTH_|E4B_|GNF4_|NF4_QLORA_|TC1_)|^PYTORCH_CUDA_ALLOC_CONF=' | sort > $W/env_box.txt
# the hook's pin: the sha256 the staged pin file names for p237_arm.py (tc1_arm.py's load_arm_hook refuses any other bytes, exit 10)
HOOK_SHA=$(awk '$2 == "p237_arm.py" || $2 == "*p237_arm.py" {print $1}' $W/staged-p237.sha256)
[ ${#HOOK_SHA} -eq 64 ] || { say "STAGE: staged-p237.sha256 names no p237_arm.py"; finish 9; }
# per process: one clock sample every 0.25 s in card_clock.py's gpuclk format ("EPOCH sm, mem, temp, reasons"), the card's used memory,
# and nvidia-smi dmon's PCIe rx/tx (reported beside the software byte counts, never ruling)
clock_log(){ while :; do echo "$(date +%s) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits -i 0 2>/dev/null)"; sleep 0.25; done; }
cache_count(){ local n=0 d; for d in /root/.triton/cache /tmp/torchinductor_root /root/.cache/unsloth_compiled_cache $W/unsloth_compiled_cache; do
  [ -d "$d" ] && n=$((n + $(find "$d" -type f 2>/dev/null | wc -l))); done; echo $n; }
# ---- the arms: amendment 80's two arms (tc1_run.sh's tc1_pos80_family), unchanged, plus the extension's options
FIELD="--fam qwen3 --model $MODEL --revision $REVISION --seq 2048 --micro-batch 2 --accum 4 --autocast 0 --lr 2e-4 --r 16 --alpha 16 --seed 3407
       --optim adamw_8bit --weight-decay 0.001 --lr-schedule linear --warmup-steps 5 --tokens $TOK --tokens-sha $TOK_SHA --eval-n 8
       --unsloth-loader FastLanguageModel --adapter-dtype fp32 --lora-init matched:3407 --prereg bench/p237/PREREG-p237.md"
UT7="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
arm_flags(){ case "$1" in
  E) echo "--framework e4b --arm fused --offload 1 --attn-4bit 1 --layer-major 1";;
  UA|UT) echo "--framework unsloth --arm unsloth --offload 0 --grad-ckpt unsloth --unsloth-targets $UT7 --unsloth-moe-backend grouped_mm${EXPECT:+ --expect-trainable $EXPECT}";; esac; }
cap_flags(){ case "$1:$2" in 12g:*) echo "--vram-cap-gib $CAP_12G";; full:E) echo "";; full:*) echo "--vram-cap-gib $CARD_GIB";; esac; }
cap_of(){ case "$1" in 12g) echo $CAP_12G;; full) echo $CARD_GIB;; esac; }
EXPECT=""
# run_proc INDEX KIND BUDGET ARM DRAW STEPS [OFFLOAD]: one tc1_arm process; the receipt lands in loads/INDEX/qwen3_<fw>_<tag>.json
run_proc(){ local idx=$1 kind=$2 budget=$3 arm=$4 draw=$5 s=$6 off=${7:-} tag ee pp env_e AL c0 c1 rc
  tag=p237_${idx}_${kind}_${budget}_${arm}_d${draw}${off:+_n$off}
  ee=$s; [ "$s" = "$STEPS" ] && ee=20; pp="--profile-steps 0"; [ "$s" -ge 24 ] && pp="--profile-steps 3 --profile-warm 21"   # loop steps 21..23: past the post-eval cold step 20
  env_e=""; [ "$arm" = E ] && env_e="E4B_OFFLOAD_STATS=1"
  AL=$(alarm_for $((LOAD_S * 2))); mkdir -p $W/loads/$idx
  say "[$idx] $kind $budget $arm draw $draw steps $s${off:+ offload_layers $off} (alarm $AL)"
  c0=$(cache_count)
  clock_log > $W/loads/gpuclk_$idx.txt 2>/dev/null & local CLK=$!
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > $W/loads/vram_$idx.txt 2>/dev/null & local VR=$!
  nvidia-smi dmon -s ut -d 1 > $W/loads/dmon_$idx.txt 2>/dev/null & local DM=$!
  # the hook seam (tc1_arm's TC1_ARM_HOOK, e4b#2146): the observe-only P237 hook, its sha256 in the staged pin
  (cd $W && env $env_e TC1_STEP_MEM=1 TC1_ARM_HOOK=p237_arm:make TC1_ARM_HOOK_SHA256=$HOOK_SHA PYTHONPATH=$W UNSLOTH_ENABLE_LOGGING=1 \
     perl -e "alarm $AL; exec @ARGV" $PY -u $W/tc1_arm.py $FIELD $(arm_flags $arm) $(cap_flags $budget $arm) ${off:+--unsloth-offload-layers $off} \
     --steps $s --eval-every $ee $pp --tag $tag --out $W/loads/$idx --adapter-dir $W/loads/$idx/adapters) > logs/load_$idx.log 2>&1
  rc=$?; kill $CLK $VR $DM 2>/dev/null; wait $CLK $VR $DM 2>/dev/null
  c1=$(cache_count)
  RECEIPT=$(ls $W/loads/$idx/qwen3_*_$tag.json 2>/dev/null | head -1)
  echo "PROC $idx $kind $budget $arm draw=$draw steps=$s${off:+ n=$off} rc=$rc cache_files ${c0}->${c1} receipt=$(basename "${RECEIPT:-none}")" | tee -a summary.txt
  grep -aE "Unsloth: .*(Mega-cache|offload_layers|block_swap|pinned)" logs/load_$idx.log | tail -4 | cut -c1-240 | sed "s/^/    [$idx] /" | tee -a summary.txt; }
# ---- the processes, in the schedule's fixed order (probes expand here, once UA's first draw at that budget has recorded N_auto)
SCHED_ARGS=""; [ "$P237_MODE" = shake ] && SCHED_ARGS="--shake"
python3 $W/p237_schedule.py $SCHED_ARGS > $W/schedule.txt
declare -A NAUTO NT
while read -r idx kind budget arm draw; do
  case "$P237_MODE:$kind" in shake:draw) S=$SHAKE_STEPS;; *:primer) S=$PRIMER_STEPS;; *:draw) S=$STEPS;; *:probes) S=$PROBE_STEPS;; esac
  if [ "$kind" = probes ]; then
    n=${NAUTO[$budget]:-unread}
    case "$n" in unread) echo "PROBES $budget skipped: UA's N_auto is unread" | tee -a summary.txt; NT[$budget]=unread; continue;;
                 0) echo "PROBES $budget skipped: N_auto 0 (auto swaps nothing; UT has nothing to tune)" | tee -a summary.txt; NT[$budget]=none; continue;; esac
    outs=""; k=1
    while [ $k -le "$PMAX" ] && [ $((n - k)) -ge 1 ]; do
      run_proc "$idx.$k" probe "$budget" UT 0 $S $((n - k))
      o=$(python3 $W/p237_host.py probe "$RECEIPT" "$(cap_of $budget)"); echo "PROBE $budget n=$((n - k)) $o" | tee -a summary.txt
      outs="$outs ${o%% *}"; [ "${o%% *}" = oom ] && break; [ "${o%% *}" = fault ] && break
      k=$((k + 1))
    done
    NT[$budget]=$(python3 $W/p237_host.py tune "$n" $outs)
    echo "TUNED $budget N_auto=$n N_t=${NT[$budget]}$( [ "${NT[$budget]}" = "$n" ] && echo ' (tuned equals auto)')" | tee -a summary.txt
    continue
  fi
  off=""
  case "$arm" in
    UA) off=auto;;
    UT) case "${NT[$budget]:-unread}" in unread|none) echo "PROC $idx $kind $budget UT draw=$draw skipped: N_t ${NT[$budget]:-unread}" | tee -a summary.txt; continue;; esac
        off=${NT[$budget]};;
  esac
  run_proc $idx $kind $budget $arm $draw $S $off
  if [ "$arm" = E ] && [ -z "$EXPECT" ] && [ -n "$RECEIPT" ]; then   # the trainable count Unsloth's arms must match (TC1's --expect-trainable, T6)
    EXPECT=$($PY -c "import json; print(json.load(open('$RECEIPT')).get('trainable_params') or '')" 2>/dev/null)
    echo "EXPECT_TRAINABLE ${EXPECT:-unread} (from $idx)" | tee -a summary.txt
  fi
  if [ "$arm" = UA ] && [ "$kind" = draw ] && [ "$draw" = 1 ]; then
    NAUTO[$budget]=$(python3 $W/p237_host.py nauto "${RECEIPT:-/nonexistent}" logs/load_$idx.log)
    echo "N_AUTO $budget ${NAUTO[$budget]}" | tee -a summary.txt
  fi
done < $W/schedule.txt
# ---- the rule
say "reduce"; $PY $W/p237_reduce.py --work $W --mode $P237_MODE --out $W/verdict_p237.json 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$W/verdict_p237.json" ] || { say "REDUCER FAILED: no reading"; finish 27; }
V=$($PY -c "import json; print(json.load(open('$W/verdict_p237.json'))['verdict'])")
[ "$P237_MODE" = shake ] && { say "shakedown: $V (never read)"; finish 0; }
case "$V" in READ) finish 0;; *) say "P237 $V -- no reading"; finish 27;; esac
