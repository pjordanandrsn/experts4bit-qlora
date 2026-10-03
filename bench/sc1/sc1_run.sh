#!/bin/bash
# bench/sc1/sc1_run.sh -- lane SC1, BOX side (SC1-PREREG.md; experts4bit-qlora#846). Started detached by
# sc1_drive.sh with the run's nonce; SC1_RUN_NONCE first, then SC1_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# SC1_SUCCESS.<nonce> only on a clean lane.
#
# One script, three boxes (SC1_BOX=A|B|C), each its OWN draw with its OWN anchors, every ratio within its box:
#   A  AMD host, pytorch 2.8 cu12.9 image: e4b NF4 control / LICENSED pack (built + gated HERE) / RTN (labelled) / the
#      scheduler-slope arms / controls / energy; vLLM 0.30.0 (gptq graph, fp8 KV, SAMEPROMPT, the nodetok pair); vLLM
#      quality; the e4b prefill-shaped rows.
#   B  same image: the anchors re-measured (e4b RTN window + sched, vLLM gptq, two draws), the bf16 upstream oracle on both
#      windows, llama.cpp b11327 / ExLlamaV3 1.5.3 (cu128) / LMDeploy 0.18.0 at B=1/16 + their quality, TTFT on every
#      engine, energy at B=1.
#   C  CUDA 13 image + python 3.12 (e4b on torch 2.8 cu128 wheels in its own venv): the anchors, SGLang 0.5.20 matched +
#      native + quality, ExLlamaV3's cu13 wheel as a labelled native row, TTFT; SGLang is ratioed only against C's anchors.
# Shape: P88's bake / licensed pack build / K8 rows / premise test; P58's nonce handshake, can_run / arm_alarm deadline
# rules, prompt dump (now sc1_prompts.py around step_decomp._k8_window), summary lines, samplers; the comparator drivers'
# contracts as their reports define them (bench/sc1/{vllm,sglang,llamacpp,exl3,lmdeploy}/). Every arm writes one receipt
# JSON + one log + one summary.txt line under a per-arm nvidia-smi / host sampler (sc1_sampler.sh). The e4b scheduler
# arms (sc1_e4b_sched.py) target the FIXED serve_paged `_apply_fusions` (fix/paged-fusions 2719171: E4B_PAGED_FUSE_QKV=1
# together with the fold flags is the registered fused-with-folds set; the census reports the fold counts fuse_qkv captured).
#
# Refusals BEFORE anything is installed (every one leaves a REFUSAL file): card class (rc 15 -- the launcher never sends
# another class, so a wrong card is a launcher fault, not a host to exclude), disk < 320 GB (rc 13), driver < 580 (rc 18;
# the CUDA 13 wheels vLLM 0.30.0 / SGLang 0.5.20 ship), box A host not AuthenticAMD (rc 18; the calibration build is
# CPU-bound, P87/P88). rc 13 / 18 are the launcher's machine-exclusion codes BY DESIGN; nothing else here uses 13/14/17/18.
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# SC1_GPU_CLASS SC1_MIN_DISK_GB SC1_MIN_DRIVER SC1_CPU_VENDOR SC1_CALIB_NSEQ SC1_REHEARSAL SC1_QUIESCE_S SC1_VLLM_WHEEL.
# SC1_PROVE=1 is the PROVING RENTAL (one per box image; guards per Amendments A1/A2): pre-flight, every install + tripwire
# for the box, the e4b paged engine end to end on Granite (fetch @ P94's GR_REV, NF4 bake, sc1_e4b_sched.py --smoke at B=1
# and 16 with graphs, census printed), and on box C SGLang's Marlin MoE JIT + /health against SC1_PROVE_SGLANG_MODEL=<repo@rev>
# (REQUIRED on box C; since A7 the lane's own 17 GB GPTQ checkpoint) -- no bf16 Qwen3 fetch in the proof. PROVED is written only when every one of those RAN and passed (A2).
set -uo pipefail
W=/root/sc1; mkdir -p $W/logs $W/samples; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] sc1: $*"; }
NONCE=${SC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/SC1_RUN_NONCE.tmp && mv $W/SC1_RUN_NONCE.tmp $W/SC1_RUN_NONCE
export SC1_SAMPLES_DIR=$W/samples
finish(){ local rc=$1
  bash $W/sc1_sampler.sh stop-all 2>/dev/null
  type sglang_server_stop >/dev/null 2>&1 && sglang_server_stop 2>/dev/null
  type llamacpp_server_stop >/dev/null 2>&1 && llamacpp_server_stop 2>/dev/null
  printf '%s\n' "$rc" > SC1_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > SC1_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in SC1_RUN_ID SC1_DEADLINE_EPOCH SC1_INSTANCE_ID SC1_BOX E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
BOX=$SC1_BOX; case "$BOX" in A|B|C|D) ;; *) say "refusing: SC1_BOX must be A, B, C or D (SC1b)"; finish 78;; esac
# ---- registered constants (v3 "Fixture"); E4B_SHA is the launch commit the driver derived from its checkout
GNF4_SHA=34da93d6fe8d2a401b7001705658ce00b2b18213   # grouped-nf4-gemm v0.34.1 -- the COMMIT the tag points to (`git rev-parse v0.34.1^{commit}`; the tag OBJECT is e7ae8e2e)
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
GPTQ_MID=Qwen/Qwen3-30B-A3B-GPTQ-Int4; GPTQ_REV=9b534e4318b7ebc3c961a839f13eb18b1833f441
GGUF_REPO=unsloth/Qwen3-30B-A3B-GGUF; GGUF_REV=d5b1d57bd0b504ac62ae6c725904e96ef228dc74
GGUF_Q4KM=Qwen3-30B-A3B-Q4_K_M.gguf; GGUF_IQ4XS=Qwen3-30B-A3B-IQ4_XS.gguf
EXL3_MID=turboderp/Qwen3-30B-A3B-exl3; EXL3_REV=0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef           # branch 4.0bpw (head_bits 6)
GR=ibm-granite/granite-3.1-3b-a800m-instruct; GR_REV=a02780686e08a03fe0d2679a293b5c74a90efa89       # P94's pin (the proof's model)
LLAMACPP_COMMIT=552f18f912a32ea86edf82e2b76431cb7131538d                                            # b11327
# A10: LMDeploy 0.18.0 TurboMind W4A16 cannot run on sm_120 -- P5b's stated alternative, UNSUPPORTED. The host dispatch
# runs the SM80 GEMM kernels on an sm_120 device (src/turbomind/kernels/gemm/arch.h:53), but each kernel body is compiled
# only where Kernel::Arch::is_compatible(__CUDA_ARCH__) holds (gemm_universal.h:174-178) and Sm80 is Arch<800, 900>
# (arch.h:25): at 1200 every 4-bit GEMM is an empty kernel. sc1b-5090-1: all 11 LMDeploy runs aborted (rc 134) on their
# first real request, 0.55-0.62 s in whatever the prompt length (512 to 4096 tokens); vLLM ran the same checkpoint. Box B no longer installs it.
LMD_UNSUPPORTED="LMDeploy 0.18.0 TurboMind W4A16 on sm_120: its SM80 GEMM fallback compiles to empty kernels (A10; sc1b-5090-1 aborted every run)"
GPU_CLASS=${SC1_GPU_CLASS:-5090}; MIN_DISK_GB=${SC1_MIN_DISK_GB:-320}; MIN_DRIVER=${SC1_MIN_DRIVER:-580}; CPU_VENDOR=${SC1_CPU_VENDOR:-AuthenticAMD}
NSEQ=${SC1_CALIB_NSEQ:-128}; REHEARSAL=${SC1_REHEARSAL:-0}; QUIESCE_S=${SC1_QUIESCE_S:-900}; PROVE=${SC1_PROVE:-0}
MIN_RAM_GB=98                                                                                       # the bf16 oracle spills to host RAM (box B); recorded, the oracle is host-limited below it
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false E4B_MODEL_ID=$MID
export PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24 E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64
export DEBIAN_FRONTEND=noninteractive
# every lever is unset; each arm names its own switches, INCLUDING the four route knobs (never inherited)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_LMHEAD_INT4_CALIB E4B_SERVE_DENSE_INT4_CALIB \
      E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 E4B_FUSE_ROUTER_EPI E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_INT4_ARTIFACT_DIR E4B_INT4_EXPECTED_FINGERPRINT E4B_INT4_DUMP_ARTIFACT_DIR E4B_INT4_ASSIGNMENT \
      E4B_CALIB_NSEQ E4B_CALIB_SOURCE E4B_CALIB_LAYERS_PER_PASS E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_NF4_GROUPED_SMALLM \
      E4B_MXFP4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_INT4_KEEP_NF4 E4B_PAGED_FUSE_QKV E4B_INT4_PREFILL
# A13: the int4 store's prefill route every SC1 box ran. #937 made `auto` (K19 wherever it can run) the default afterwards.
# Exported once, after the scrub, so every e4b process reads it; FOLDS / SPEEDENV / ROUTEENV stay byte-identical to the
# lanes that pin them to SC1's (P100, P102). No arm sets it.
export E4B_INT4_PREFILL=loop
# SC1b: the paged prefill attention route every SC1 box ran (#960 adds the knob; its default flips to flash afterwards);
# exported like A13's pin, after the scrub, before the tripwire and every arm
unset E4B_PAGED_PREFILL_ATTN; export E4B_PAGED_PREFILL_ATTN=math
: > summary.txt; echo "$SC1_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS box=$BOX e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MID rev=$REV gptq=$GPTQ_REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_driver=$MIN_DRIVER cpu_vendor=$CPU_VENDOR calib_nseq=$NSEQ quiesce_s=$QUIESCE_S prove=$PROVE" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 320 ] || [ "$MIN_DRIVER" != 580 ] || [ "$NSEQ" != 128 ] || { [ "$BOX" = A ] && [ "$CPU_VENDOR" != AuthenticAMD ]; }; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte (the comparator dirs are pinned file by file; a missing dir for the box is an UNSUPPORTED row, not a stop)
for f in sc1_e4b_sched.py sc1_prompts.py sc1_sampler.sh step_decomp.py k8_bake.py calib.json hook/usercustomize.py test_k19_row_exact_gpu.py staged.sha256 vllm/install.sh vllm/sc1_vllm_arm.py; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/sc1/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: a dud box, the card class, the disk, the driver, box A's host vendor
nvidia-smi -L > forensics.txt 2>&1 || { say "DUD BOX (nvidia-smi)"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader | tee -a forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID" | tee -a forensics.txt; echo "nproc $(nproc)" | tee -a forensics.txt
echo "cgroup cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)  memory.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)" | tee -a forensics.txt
echo "loadavg $(cat /proc/loadavg)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
{ nvcc --version 2>/dev/null | tail -1; grep PRETTY_NAME /etc/os-release; python3 --version 2>&1; } | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (bf16 checkpoint, NF4 arena, comparator checkpoints, venvs)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' '); DRV_MAJOR=${DRV%%.*}
[ -n "$DRV_MAJOR" ] && [ "$DRV_MAJOR" -ge "$MIN_DRIVER" ] 2>/dev/null || { say "REFUSED: driver '${DRV:-unreadable}' < R${MIN_DRIVER} (the CUDA 13 wheels of vLLM 0.30.0 / SGLang 0.5.20)"; echo "refused: driver ${DRV:-unreadable}" > REFUSAL; finish 18; }
HOST_VENDOR=$(lscpu | awk -F: '/^Vendor ID/{gsub(/ /, "", $2); print $2; exit}'); echo "cpu_vendor $HOST_VENDOR" | tee -a forensics.txt
if [ "$BOX" = A ]; then [ "$HOST_VENDOR" = "$CPU_VENDOR" ] || { say "REFUSED: box A host CPU vendor is '${HOST_VENDOR:-unknown}', the lane registers $CPU_VENDOR (the calibration build is CPU-bound; P87/P88)"; echo "refused: cpu vendor ${HOST_VENDOR:-unknown}" > REFUSAL; finish 18; }; fi
RAM_GB=$(free -g | awk '/Mem:/{print $2}'); echo "ram_gb ${RAM_GB:-?} (oracle floor $MIN_RAM_GB)" | tee -a forensics.txt
CPU_MODEL=$(lscpu | awk -F: '/Model name/{sub(/^ +/, "", $2); print $2; exit}' | tr -d '"')
printf '{"SC1_BOX": "%s", "run_id": "%s", "instance_id": "%s", "nonce": "%s", "e4b_sha": "%s", "gnf4_sha": "%s", "prove": %s, "host": {"gpu": "%s", "driver": "%s", "cpu_vendor": "%s", "cpu_model": "%s", "nproc": %s, "cpu_max": "%s", "ram_gb": %s, "free_disk_gb": %s}}\n' \
  "$BOX" "$SC1_RUN_ID" "$SC1_INSTANCE_ID" "$NONCE" "$E4B_SHA" "$GNF4_SHA" "$PROVE" "$GPU_NAME" "$DRV" "${HOST_VENDOR:-unknown}" "${CPU_MODEL:-unknown}" "$(nproc)" "$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | tr ' ' /)" "${RAM_GB:-0}" "${FREE_GB:-0}" > box.json   # the reducer reads SC1_BOX + host facts from here
# ---- deadline guard + per-arm alarm (P54's): never start an arm that cannot finish 10 min before teardown
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$SC1_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((SC1_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( SC1_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
host_snapshot(){ echo "nproc=$(nproc) cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | tr ' ' /) load=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' /)"; }
phase(){ say "phase $1: $2"; echo "PHASE $1 $2 at=$(date -u +%FT%TZ) $(host_snapshot)" >> summary.txt; }
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
declare -A OK; have(){ [ "${OK[$1]:-0}" = 1 ]; }
# ---- quiescence gate (v3 Phase 0 / "Validity"): no install / build / JIT process and loadavg < nproc/2, waited for (<= SC1_QUIESCE_S) and RECORDED
quiesce(){ local TAG=$1 t0 np half busy load1 ok=no waited; t0=$(date +%s); np=$(nproc); half=$(( np / 2 )); [ "$half" -lt 1 ] && half=1
  while :; do
    busy=$(pgrep -f -- 'pip install|cmake|ninja|nvcc|git clone' 2>/dev/null | wc -l | tr -d ' '); load1=$(cut -d' ' -f1 /proc/loadavg)   # NOT `pgrep -c` with an echo-0 fallback: on no match procps prints 0 AND exits 1, giving two zeros (A2)
    if [ "$busy" = 0 ] && awk -v l="$load1" -v h="$half" 'BEGIN{exit !(l < h)}'; then ok=yes; break; fi
    waited=$(( $(date +%s) - t0 )); [ "$waited" -ge "$QUIESCE_S" ] && break; sleep 15
  done
  waited=$(( $(date +%s) - t0 ))
  echo "QUIESCE $TAG quiesced=$ok waited=${waited}s busy_procs=$busy load1=$load1 nproc=$np cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | tr ' ' /) at=$(date -u +%FT%TZ)" | tee -a summary.txt
  printf '{"tag": "%s", "quiesced": %s, "waited_s": %s, "busy_procs": %s, "load1": %s, "nproc": %s, "cpu_max": "%s", "at": "%s"}\n' \
    "$TAG" "$([ "$ok" = yes ] && echo true || echo false)" "$waited" "$busy" "$load1" "$np" "$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | tr ' ' /)" "$(date -u +%FT%TZ)" > quiesce_$TAG.json; }
# ---- samplers + stubs + summary lines
sampler_start(){ bash $W/sc1_sampler.sh start "${SAMPLE_AS:-$1}"; }
sampler_stop(){ bash $W/sc1_sampler.sh stop "${SAMPLE_AS:-$1}"; }
# stub OUT ENGINE ARM BATCH STATUS REASON [LOG] -- a row that was not measured still gets a receipt: {engine, arm, batch, status, reason, log_tail}
stub(){ "$BASEPY" - "$1" "$2" "$3" "$4" "$5" "$6" "${7:-}" <<'PY'
import json, os, sys
out, engine, arm, b, status, reason, log = sys.argv[1:8]
tail = "".join(open(log, errors="replace").readlines()[-40:])[-4000:] if log and os.path.exists(log) else ""
json.dump({"engine": engine, "arm": arm, "batch": (int(b) if b.isdigit() else b), "status": status, "reason": reason,
           "log_tail": tail, "written_by": "sc1_run.sh"}, open(out, "w"), indent=1)
PY
}
unsupported(){ local ENGINE=$1 REASON=$2 LOG=${3:-}; OK[$ENGINE]=0
  stub $W/unsupported_$ENGINE.json "$ENGINE" install "-" unsupported "$REASON" "$LOG"
  echo "UNSUPPORTED $ENGINE: $REASON" | tee -a summary.txt; }
line(){ echo "$*" >> summary.txt; }
# ---- the python that runs e4b: a venv WITH --system-site-packages (a plain venv sets site.ENABLE_USER_SITE=False and the
# P42 hook -- usercustomize via PYTHONPATH -- silently never loads; measured 2026-10-01). Box C builds it on python 3.12
# with torch 2.8 cu128 wheels; boxes A/B inherit the image's torch 2.8 cu12.9.
case "$BOX" in A|B) BASEPY=python;; C|D) BASEPY=python3;; esac
ensure_tools(){ local need=0 t; for t in git cmake curl; do command -v $t >/dev/null 2>&1 || need=1; done
  "$BASEPY" -c "import venv, ensurepip" 2>/dev/null || need=1
  [ "$BOX" = C ] && { command -v python >/dev/null 2>&1 || need=1; }
  [ "$need" = 1 ] || return 0
  say "apt: python3-venv / python-is-python3 / git / cmake / curl / build-essential (missing on this image)"
  perl -e 'alarm 900; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq --no-install-recommends python3 python3-dev python3-venv python3-pip python-is-python3 git cmake ninja-build curl ca-certificates build-essential' > logs/apt.log 2>&1 \
    || { tail -3 logs/apt.log; say "APT FAIL"; finish 9; }; }
ensure_tools
command -v "$BASEPY" >/dev/null 2>&1 || { say "no base python ($BASEPY)"; finish 9; }
# one pip install with ONE retry after 20 s (P88: a git clone can fail transiently)
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" "$PY" -m pip install -q --no-input "$@" > "$log" 2>&1 && return 0
  say "pip failed ($(tail -1 "$log" | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" "$PY" -m pip install -q --no-input "$@" >> "$log" 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (venv-e4b --system-site-packages; P58 pins transformers 5.16.1 / bitsandbytes 0.50.1)"
"$BASEPY" -m venv --system-site-packages $W/venv-e4b > logs/venv_e4b.log 2>&1 || { tail -3 logs/venv_e4b.log; say "VENV FAIL (e4b)"; finish 9; }
PY=$W/venv-e4b/bin/python
if [ "$BOX" = C ] || [ "$BOX" = D ]; then pipx logs/pip_torch.log 1800 "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128 || { tail -3 logs/pip_torch.log; say "PIP FAIL (torch cu128)"; finish 9; }; fi
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" pytest || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA "$PY" - <<'PYT' || { say "TRIPWIRE FAIL (e4b)"; finish 9; }
import site, os, json, re, inspect, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (plain venv?) -- the P42 hook needs --system-site-packages"
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
from int4_smallm import gemm_int4_b32_grouped_smallm  # noqa: F401  (K19 is installed)
_p = inspect.signature(gemm_int4_b32_grouped_smallm).parameters
assert (_p["block_n"].default, _p["kc"].default) == (32, 256), "gnf4 lacks K20's default plan (BLOCK_N 32 / KC 256)"
from experts4bit_qlora.engines import hot_residency as hr
src = inspect.getsource(hr)
for knob in ("E4B_INT4_GROUPED_SMALLM", "E4B_INT4_LEAN_GLUE", "E4B_MXFP4_GROUPED_SMALLM", "E4B_NF4_GROUPED_SMALLM"):   # A6
    m = re.search(rf'environ\.get\("{knob}",\s*"([^"]*)"\)', src)
    assert m, f"{knob} is not read from the environment (K19/K23/K21/K25): ROUTEENV could not pin it"
    print(f"ROUTE_DEFAULT {knob}={m.group(1)} (every e4b arm pins it via ROUTEENV)", flush=True)
# A13: the int4 store's prefill route is read from the environment, and the box's exported `loop` resolves to it (the
# registered route; #937 made `auto` -- K19 wherever it can run -- the default after every box had run)
m = re.search(r'environ\.get\("E4B_INT4_PREFILL",\s*"([^"]*)"\)', src)
assert m, "E4B_INT4_PREFILL is not read from the environment: the box's export could not pin the prefill route"
assert os.environ.get("E4B_INT4_PREFILL") == "loop", "the box did not export E4B_INT4_PREFILL=loop"
assert hr._int4_prefill_mode_env() == "loop", "E4B_INT4_PREFILL=loop does not resolve to the loop route"
# SC1b: the paged prefill attention route is read from the environment, and the box's exported `math` is what it holds
import experts4bit_qlora.engines.paged_attention as pa
assert re.search(r'environ\.get\("E4B_PAGED_PREFILL_ATTN"', inspect.getsource(pa)), "E4B_PAGED_PREFILL_ATTN is not read from the environment"
assert os.environ.get("E4B_PAGED_PREFILL_ATTN") == "math", "the box did not export E4B_PAGED_PREFILL_ATTN=math"
assert pa._prefill_attn_mode_env() == "math", "E4B_PAGED_PREFILL_ATTN=math does not resolve to the math route"
print(f"ROUTE_DEFAULT E4B_INT4_PREFILL={m.group(1)} (the box exports loop to every e4b process, A13)", flush=True)
assert hasattr(hr, "_collapsed_grouping"), "e4b lacks the T == 1 extension (#804): K8 would read the GEMV"
from experts4bit_qlora.engines.int4_attn import Int4Linear, _smallm_kernels
assert getattr(Int4Linear, "SMALLM_ROWS_MAX", None) == 16 and hasattr(Int4Linear, "fuse"), "e4b cut lacks the K16 route or Int4Linear.fuse"
_smallm_kernels()
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated as f
for k in ("artifact_dir", "expected_fingerprint", "dump_artifact_dir"):
    assert k in inspect.signature(f).parameters, f"e4b cut lacks the #405 knob {k!r}"
from experts4bit_qlora.serve_paged import build_engine, PagedServeConfig, EngineParts  # noqa: F401  (the sched arms' construction)
from experts4bit_qlora.engines.scheduler import ContinuousScheduler
assert "stop_ids" in inspect.signature(ContinuousScheduler.add_request).parameters, "scheduler lacks #848's stop set"
from experts4bit_qlora.engines.paged_runner import PagedModelRunner
assert hasattr(PagedModelRunner, "enable_decode_graphs"), "runner lacks bucketed decode graphs (#511)"
import int4_b32
assert int4_b32.gemv_fused_reduce_default() is False, "the fused reduce is defaulted on (the registered rows read the two-launch default)"
import usercustomize  # noqa: F401
assert usercustomize.__file__.startswith("/root/sc1/hook/"), usercustomize.__file__
import experts4bit_qlora as e, torch, triton, transformers
assert torch.cuda.is_available(), "torch sees no CUDA device in venv-e4b"
open("/root/sc1/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch(e4b) {torch.__version__} cuda {torch.version.cuda}\ntriton(e4b) {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK (e4b):", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__, "K19/K23 defaults auto, K25 0, serve_paged + stop_ids present")
PYT
cat versions.txt | tee -a summary.txt
# ---- comparator installs for the box, each under its alarm; a failure is an UNSUPPORTED row and the lane continues
install_vllm(){ say "install vllm (bench/sc1/vllm/install.sh)"; SC1_PIP_LOG=$W/logs/pip_vllm.log bash $W/vllm/install.sh 2700 $W/versions.txt $W/venv-vllm > logs/install_vllm.log 2>&1
  local rc=$?; if [ $rc -eq 0 ]; then OK[vllm]=1; tail -1 logs/install_vllm.log | tee -a summary.txt; else unsupported vllm "install.sh rc=$rc" logs/install_vllm.log; fi; }
install_sglang(){ [ -s $W/sglang/install.sh ] || { unsupported sglang "driver not staged"; return; }
  say "install sglang (bench/sc1/sglang/install.sh)"; bash $W/sglang/install.sh 2700 $W/versions.txt $W > logs/install_sglang.log 2>&1
  local rc=$?; if [ $rc -eq 0 ]; then OK[sglang]=1; . $W/sglang/server.sh; export SGLANG_WORK=$W; tail -1 logs/install_sglang.log | tee -a summary.txt
  else unsupported sglang "install.sh rc=$rc: $(tail -1 logs/install_sglang.log | cut -c1-160)" logs/install_sglang.log; fi; }
install_llamacpp(){ [ -s $W/llamacpp/llamacpp_box.sh ] || { unsupported llamacpp "driver not staged"; return; }
  say "build llama.cpp @$LLAMACPP_COMMIT (CUDA 120a-real) + the NLL harness"
  LLAMACPP_PY=$PY perl -e "alarm 2700; exec @ARGV" bash -c ". $W/llamacpp/llamacpp_box.sh && llamacpp_build $W/llama.cpp $LLAMACPP_COMMIT" > logs/install_llamacpp.log 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -x $W/llama.cpp/build/bin/llama-server ] && [ -x $W/llama.cpp/build/bin/nll_teacher_forced ]; then
    LLAMACPP_SRC_DIR=$W/llama.cpp LLAMACPP_BIN=$W/llama.cpp/build/bin; export LLAMACPP_SRC_DIR LLAMACPP_BIN LD_LIBRARY_PATH="$LLAMACPP_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    . $W/llamacpp/llamacpp_box.sh; OK[llamacpp]=1; echo "llamacpp $(cat $W/llama.cpp/build/llama-server.version.txt 2>/dev/null | head -1)" | tee -a versions.txt summary.txt
  else unsupported llamacpp "build rc=$rc" logs/install_llamacpp.log; fi; }
install_exl3(){ local FLAVOR=$1; [ -s $W/exl3/install.sh ] || { unsupported exl3 "driver not staged"; return; }
  say "install exllamav3 1.5.3 ($FLAVOR)"; SC1_PYTHON=$BASEPY bash $W/exl3/install.sh $W/venv-exl3 2700 $FLAVOR > logs/install_exl3.log 2>&1
  local rc=$?; if [ $rc -eq 0 ]; then OK[exl3]=1; grep -a SC1_TRIPWIRE logs/install_exl3.log | tail -1 | cut -c1-300 >> versions.txt; echo "exl3 install OK ($FLAVOR)" | tee -a summary.txt
  else unsupported exl3 "install.sh rc=$rc ($FLAVOR)" logs/install_exl3.log; fi; }
install_lmdeploy(){ [ -s $W/lmdeploy/install.sh ] || { unsupported lmdeploy "driver not staged"; return; }
  say "install lmdeploy 0.18.0"; SC1_PYTHON=$BASEPY bash $W/lmdeploy/install.sh $W/venv-lmdeploy 2700 > logs/install_lmdeploy.log 2>&1
  local rc=$?; if [ $rc -eq 0 ]; then OK[lmdeploy]=1; grep -a SC1_TRIPWIRE logs/install_lmdeploy.log | tail -1 | cut -c1-300 >> versions.txt; echo "lmdeploy install OK" | tee -a summary.txt
  else unsupported lmdeploy "install.sh rc=$rc" logs/install_lmdeploy.log; fi; }
case "$BOX" in
  A) install_vllm ;;
  B) install_vllm; install_llamacpp; install_exl3 cu128; unsupported lmdeploy "$LMD_UNSUPPORTED" ;;
  C) install_vllm; install_exl3 cu132; [ "$PROVE" = 1 ] && install_sglang ;;   # the real lane installs SGLang AFTER the anchors (v3's order)
  D) . $W/sc1b_box_d.sh; install_vllm; install_sglang; install_llamacpp; install_nsys ;;   # SC1b's census box (bench/sc1b)
esac
# ---- environments (P88, byte for byte) + SC1's explicit route knobs (v3 "Fixture": never inherited)
FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"
LICENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_EXP_INT4_CALIB=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4=0 E4B_CALIB_SOURCE=c4 E4B_CALIB_NSEQ=$NSEQ $FOLDS"
NF4ENV="E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0"
GR_ENV="E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"   # P94's Granite env (the unfused set)
ROUTEENV="E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto"   # pinned main's defaults as of 2026-10-02 (P94: NF4 stays 0)
K8ARGS="--placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source wikitext"
PACKENV=""; QA=""; GA=""
# ---- fetches (pinned revisions; the HF cache path of each snapshot lands in fetch_<name>.path)
fetch(){ local NAME=$1 M=$2 R=$3 AL=$4; say "fetch $M @ $R"
  perl -e "alarm $(arm_alarm $AL); exec @ARGV" "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$M', revision='$R', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch_$NAME.log 2>&1 \
    || { tail -2 logs/fetch_$NAME.log; say "DL FAIL ($NAME)"; line "FETCH $NAME FAILED"; return 11; }
  tail -1 logs/fetch_$NAME.log > fetch_$NAME.path; line "FETCH $NAME $M@$R $(cat fetch_$NAME.path)"; }
fetch_ggufs(){ have llamacpp || return 0; local f; mkdir -p $W/gguf
  for f in $GGUF_Q4KM $GGUF_IQ4XS; do say "fetch $GGUF_REPO/$f @ $GGUF_REV"
    LLAMACPP_PY=$PY perl -e "alarm $(arm_alarm 3600); exec @ARGV" bash -c ". $W/llamacpp/llamacpp_box.sh && llamacpp_fetch $GGUF_REPO $f $W/gguf $GGUF_REV" > logs/fetch_gguf_${f%.gguf}.log 2>&1 \
      || { tail -2 logs/fetch_gguf_${f%.gguf}.log; say "DL FAIL ($f)"; line "FETCH gguf $f FAILED"; [ "$f" = "$GGUF_Q4KM" ] && OK[llamacpp]=0; continue; }
    line "FETCH gguf $f $(cat $W/gguf/$f.sha256 2>/dev/null | cut -c1-16)"; done; }
# ---- bake (P39's k8_bake.py): the NF4 arena the window arms AND the sched arms load; every box bakes (the bake reads the bf16 checkpoint)
bake(){ local TAG=$1 M=$2 AL=$3; say "bake NF4 arena ($TAG)"; mkdir -p $W/work_$TAG
  K8_MODEL="$M" K8_WORK="$W/work_$TAG" perl -e "alarm $(arm_alarm $AL); exec @ARGV" "$PY" $W/k8_bake.py > logs/bake_$TAG.log 2>&1 \
    || { tail -3 logs/bake_$TAG.log; "$PY" -c "import json; r=json.load(open('$W/work_$TAG/bake.json')); print('BAKE_ERR', r.get('status'), r.get('step'), r.get('err'))" 2>/dev/null | tee -a logs/bake_$TAG.log | tail -3 | cut -c1-400; say "BAKE FAIL ($TAG)"; return 12; }
  [ -e "$W/work_$TAG/nf4.arena" ] || { say "BAKE FAIL ($TAG): no arena"; return 12; }
  line "BAKE $TAG $(grep -a '"bake_s"\|BAKE' logs/bake_$TAG.log | tail -1 | cut -c1-120)"; }
prompts(){ say "prompt dump: sc1_prompts.py (step_decomp._k8_window) -> prompts_b{1,16,1_4096,16_same}.json + k8_window_{wikitext,c4val1}.json"
  SC1_HARNESS_DIR=$W perl -e "alarm $(arm_alarm 1200); exec @ARGV" "$PY" $W/sc1_prompts.py --model "$MID" --rev "$REV" --out $W > logs/prompts.log 2>&1 || { tail -5 logs/prompts.log; say "PROMPT DUMP FAIL"; return 19; }
  grep -aE "^PROMPTS|^WINDOW" logs/prompts.log | tee -a summary.txt; }
# ============================================================================ arm kinds (one function per kind)
# Receipt names are the reducer's (bench/sc1/sc1_reduce.py "RECEIPT CONVENTIONS"): timed arms <engine>_<arm>_b<B>_r<n>.json with
# logs/run_<same>.log (engine e4bsched for the scheduler arms); k8_<name>.json; oracle_<src>.json; nll_<engine>[_<variant>]_<mode>_<src>.json;
# ttft_<engine>_<len>.json; energy_<tag>.{json,csv}. smoke_<tag>.json, e4b_k19census_b1.json and the <engine>_energy_* receipts carry no
# _r<n>, so the reducer's arm scan ignores them. One per-arm sampler (sc1_sampler.sh) under samples/<stem>.*.
stem_of(){ case "$2" in ttft_*|nll_*|k8_*|smoke_*|oracle_*) echo "$2";; *) echo "$1_$2";; esac; }
status_of_rc(){ [ "$1" -eq 142 ] && echo alarm || echo "${2:-harness_error}"; }
# e4b_window TAG B STACKENV FUSEFLAG CENSUS(1|"") [EXTRA ENV...]  -- P88's speed(): the graph-replay window (the kernel ceiling); receipt e4b_<TAG>.json
e4b_window(){ local TAG=$1 B=$2 STACK=$3 FUSE=$4 CEN=$5; shift 5; local S=e4b_$TAG AL; AL=$(arm_alarm 1800)
  say "arm $S (window B=$B fuse=$FUSE census=${CEN:-no} alarm=$AL)"; sampler_start $S
  { echo "SC1 arm=$S box=$BOX e4b=$E4B_SHA gnf4=$GNF4_SHA at=$(date -u +%FT%TZ) $(host_snapshot)"; echo "ENV: $ROUTEENV $STACK $*"; } > logs/run_$S.log
  env $ROUTEENV $STACK "$@" perl -e "alarm $AL; exec @ARGV" "$PY" $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch $B --prompt-len 512 --gen-tokens 128 --b1d-loop graph --b1d-timed $FUSE ${CEN:+--replay-profile-out $W/logs/census_$S.txt} --out $W/$S.json >> logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json e4b "$TAG" "$B" "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  grep -aE "B1D_TIMED|BV3_|INT4EXP|ATTNINT4|fused q/k/v|REFUSED|Error" logs/run_$S.log | tail -4 | cut -c1-240 | sed "s/^/    /"
  line "$S B=$B rc=$rc $(grep -aE 'B1D_TIMED|BV3_' logs/run_$S.log | tail -1 | cut -c1-240)"
  "$PY" -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; return $rc; }
# e4b_k8 NAME STACKENV SRC [EXTRA ARGS...]  -- P88's k8(): K8ARGS byte for byte, then --ppl-source SRC (argparse: the last wins) and the row's extras;
# receipt k8_<NAME>.json (the eager prefill-shaped rows pass NAME=nll_e4b_prefill_<src> and get that name, the reducer's)
e4b_k8(){ local NAME=$1 STACK=$2 SRC=$3; shift 3; local S AL; S=$(stem_of k8 "$NAME"); AL=$(arm_alarm 2400)
  say "arm $S ($SRC $* alarm=$AL)"; sampler_start $S
  { echo "SC1 arm=$S box=$BOX at=$(date -u +%FT%TZ) $(host_snapshot)"; echo "ENV: $ROUTEENV $STACK"; } > logs/run_$S.log
  env $ROUTEENV $STACK perl -e "alarm $AL; exec @ARGV" "$PY" $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json $K8ARGS --ppl-source $SRC "$@" --out $W/$S.json >> logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json e4b "$S" 1 "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S src=$SRC rc=$rc $(grep -aE 'K8_PPL|REFUSED|Error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }
# k8_census -- the K8 `=1` rows' engagement evidence (v3 Phase B: "the receipt's census must show both engaged"). step_decomp's eager K8 loop has
# no kernel census of its own, so the SAME stack (LICENV + the pack by fingerprint + E4B_INT4_GROUPED_SMALLM=1 E4B_INT4_LEAN_GLUE=auto,
# --no-fuse-qkv as the K8 rows) is censused at T == 1 in a B=1 graph window and the table is placed under the names the reducer reads
# (logs/census_k8_lic_1_<src>.txt); the receipt e4b_k19census_b1.json (no _r<n>: not a timed row) and the K8CENSUS line say so.
k8_census(){ e4b_window k19census_b1 1 "$LICENV $PACKENV E4B_INT4_GROUPED_SMALLM=1 E4B_INT4_LEAN_GLUE=auto" --no-fuse-qkv 1; local rc=$?
  for SRC in wikitext c4val1; do cp logs/census_e4b_k19census_b1.txt logs/census_k8_lic_1_$SRC.txt 2>/dev/null; done
  line "K8CENSUS lic_1: logs/census_e4b_k19census_b1.txt -> logs/census_k8_lic_1_{wikitext,c4val1}.txt (the same stack at T == 1 in a graph window; the K8 rows run eager)"; return $rc; }
# K8 build watchdog (P85 amendment 1): a host that cannot finish the first calibration chunk in 1500 s is host-limited
first_chunk_watchdog(){ local pid=$1 log=$2 t0; t0=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    grep -qa "INT4EXP calibrated experts" "$log" 2>/dev/null && { say "build: calibration chunk 1 in $(( $(date +%s) - t0 ))s"; return 0; }
    if [ $(( $(date +%s) - t0 )) -ge 1500 ]; then say "HOST-LIMITED: no calibration chunk in 1500s -- killing the build"; line "HOSTLIMITED build no calibration chunk in 1500s"
      kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null; return 30; fi
    sleep 20
  done; return 0; }
fp_of(){ "$PY" -c "import json; print(json.load(open('$1/manifest.json'))['pack_fingerprint'])" 2>/dev/null; }
verifies(){ "$PY" -c "from experts4bit_qlora.engines.pack_manifest import verify_artifact as v; v('$1', expected_model_revision='$REV')" > logs/verify_$(basename $1).log 2>&1; }
# e4b_build -- P88's `k8 build watch`: the licensed pack built ONCE (its K8 reading is the k8_build_wikitext row), dumped as a hash-pinned
# artifact, verified; PACK <fp> on summary.txt; every later lic arm loads it by fingerprint
e4b_build(){ local AL; AL=$(arm_alarm 5400); rm -rf $W/artifact; say "build the licensed pack (K8 build row, wikitext; alarm=$AL; first-chunk watchdog 1500 s)"; sampler_start k8_build
  { echo "SC1 arm=k8_build box=$BOX at=$(date -u +%FT%TZ) $(host_snapshot)"; } > logs/run_k8_build.log
  env $ROUTEENV $LICENV E4B_CALIB_LAYERS_PER_PASS=10 E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact perl -e "alarm $AL; exec @ARGV" "$PY" $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json $K8ARGS \
    --out $W/k8_build_wikitext.json >> logs/run_k8_build.log 2>&1 &
  local pid=$! brc=0 wrc=0; first_chunk_watchdog "$pid" logs/run_k8_build.log || brc=$?
  wait "$pid" 2>/dev/null || wrc=$?; [ "$brc" = 0 ] && brc=$wrc; sampler_stop k8_build
  line "k8_build rc=$brc $(grep -aE 'K8_PPL|REFUSED|Error' logs/run_k8_build.log | tail -1 | cut -c1-200)"
  [ "$brc" = 30 ] && finish 30
  if verifies $W/artifact && FP=$(fp_of $W/artifact) && [ -n "$FP" ]; then
    mkdir -p $W/manifests && cp $W/artifact/manifest.json $W/manifests/experts.json
    PACKENV="E4B_INT4_ARTIFACT_DIR=$W/artifact E4B_INT4_EXPECTED_FINGERPRINT=$FP"; echo "PACK $FP" | tee -a summary.txt
  else say "the build left no expert pack that verifies"; line "BUILD FAILED (experts)"; rec 20; fi
  return $brc; }
# lic_ready STEM ENGINE B -- a lic arm without the fingerprint would rebuild the calibration: stub it instead
lic_ready(){ [ -n "$PACKENV" ] || { stub $W/$1.json "$2" "$1" "${3:-1}" harness_error "no licensed pack (the build failed)"; line "$1 SKIPPED no licensed pack"; return 1; }; }
# premise -- P88's: K19's rows are bit-equal alone (T == 1) and inside a B=16 step on THIS card; rc 25 if not (v3 Phase 0 puts it after the build)
premise(){ can_run 900 premise || return 0; say "premise: tests/test_k19_row_exact_gpu.py on this card"
  (cd $W && PYTHONPATH= perl -e 'alarm 900; exec @ARGV' "$PY" -m pytest test_k19_row_exact_gpu.py -q -p no:cacheprovider) > logs/rowexact.log 2>&1
  local rc=$?; { echo -n "PREMISE row-exact rc=$rc: "; tail -1 logs/rowexact.log; } | tee -a summary.txt
  "$PY" -c "import json; json.dump({'rc': $rc, 'last': open('$W/logs/rowexact.log').read().strip().splitlines()[-1:]}, open('$W/rowexact.json', 'w'))"
  [ "$rc" = 0 ] || { say "PREMISE FAILED: K19's rows are not row-count invariant on this card (or the test errored)"; echo "premise failed rc=$rc" > REFUSAL; finish 25; }; }
# oracle SRC -- the bf16 reference: step_decomp --ppl-oracle upstream (transformers bf16, device_map auto over card + host RAM, eager, prefill-shaped); receipt oracle_<src>.json
oracle(){ local SRC=$1 S=oracle_$1; local AL; AL=$(arm_alarm 3600)
  [ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { stub $W/$S.json oracle "$S" 1 not_run "host-limited: ${RAM_GB:-?} GB RAM < $MIN_RAM_GB (the 61 GB bf16 checkpoint spills to host RAM)"; line "$S SKIPPED host-limited RAM"; return 0; }
  say "arm $S (alarm=$AL)"; sampler_start $S
  { echo "SC1 arm=$S box=$BOX at=$(date -u +%FT%TZ) $(host_snapshot)"; } > logs/run_$S.log
  perl -e "alarm $AL; exec @ARGV" "$PY" $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json --placement-override all-vram --amort off --batch 1 --prompt-len 512 \
      --ppl-steps 2048 --ppl-oracle upstream --ppl-source $SRC --out $W/$S.json >> logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json oracle "$S" 1 "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S rc=$rc $(grep -aE 'K8_PPL|UPSTREAM|REFUSED|Error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }
# sched_env B MAX_TOKENS_PER_SEQ FUSE [MID REV ARENA] -- the serve_paged engine env; buckets = the registered list up to B (vLLM's capture list is derived from B the same way)
sched_env(){ local B=$1 MTS=$2 FUSE=$3 M=${4:-$MID} R=${5:-$REV} ARENA=${6:-$QA} BK
  case "$B" in 1) BK=1;; 2) BK=1,2;; 4) BK=1,2,4;; 8) BK=1,2,4,8;; *) BK=1,2,4,8,16;; esac
  echo "E4B_PAGED_MODEL=$M E4B_PAGED_REVISION=$R E4B_PAGED_ARENA=$ARENA E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_PLACEMENT=all-vram E4B_PAGED_MAX_SEQS=$B E4B_PAGED_MAX_TOKENS_PER_SEQ=$MTS E4B_PAGED_CHUNK_TOKENS=512 E4B_PAGED_GRAPHS=1 E4B_PAGED_BUCKETS=$BK E4B_PAGED_FUSE_QKV=$FUSE E4B_PAGED_TORCH_THREADS=8"; }
# sched_arm TAG B STACKENV MAX_TOKENS_PER_SEQ FUSE PROMPTS [MODE FLAGS...] -- sc1_e4b_sched.py: e4b through ContinuousScheduler.step() (the ratio axis); receipt e4bsched_<TAG>.json (ttft_* bare)
# PYTHONPATH= (A4): serve_paged.build_engine applies the int4 levers itself; the P42 hook on PYTHONPATH would apply them a second time.
sched_arm(){ local TAG=$1 B=$2 STACK=$3 MTS=$4 FUSE=$5 PROMPTS=$6; shift 6; local S AL; S=$(stem_of e4bsched "$TAG"); AL=$(arm_alarm 1800)
  say "arm $S (sched B=$B fuse_qkv=$FUSE max_tokens_per_seq=$MTS $* alarm=$AL)"; sampler_start $S
  { echo "SC1 arm=$S box=$BOX e4b=$E4B_SHA gnf4=$GNF4_SHA at=$(date -u +%FT%TZ) $(host_snapshot)"; echo "ENV: $ROUTEENV $STACK $(sched_env $B $MTS $FUSE)"; } > logs/run_$S.log
  env PYTHONPATH= $ROUTEENV $STACK $(sched_env $B $MTS $FUSE) E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA SC1_ARM=$TAG SC1_BATCH=$B SC1_PROMPTS=$PROMPTS SC1_OUT=$W/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID \
    perl -e "alarm $AL; exec @ARGV" "$PY" $W/sc1_e4b_sched.py "$@" >> logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json e4bsched "$TAG" "$B" "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  grep -aE "SC1SCHED|SC1_ENERGY|INT4EXP|ATTNINT4|fusions|DECODE_GRAPH|REFUSED|Error" logs/run_$S.log | tail -4 | cut -c1-240 | sed "s/^/    /"
  line "$S B=$B rc=$rc $(grep -a 'SC1SCHED ' logs/run_$S.log | tail -1 | cut -c1-240)"; return $rc; }
# sched_smoke TAG B STACKENV FUSE [MID REV ARENA] -- build_engine + 16 tokens with graphs; exit 0 only if every set lever reports a nonzero count; receipt smoke_<TAG>.json
sched_smoke(){ local TAG=$1 B=$2 STACK=$3 FUSE=$4 M=${5:-$MID} R=${6:-$REV} ARENA=${7:-$QA}; local S=smoke_$TAG AL; AL=$(arm_alarm 1200)
  say "smoke $S (build_engine B=$B fuse_qkv=$FUSE model=$M alarm=$AL)"
  { echo "SC1 smoke=$S box=$BOX at=$(date -u +%FT%TZ) $(host_snapshot)"; echo "ENV: $ROUTEENV $STACK $(sched_env $B 2048 $FUSE "$M" "$R" "$ARENA")"; } > logs/run_$S.log
  env PYTHONPATH= $ROUTEENV $STACK $(sched_env $B 2048 $FUSE "$M" "$R" "$ARENA") E4B_MODEL_ID=$M SC1_ARM=$S SC1_BATCH=$B SC1_OUT=$W/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID \
    perl -e "alarm $AL; exec @ARGV" "$PY" $W/sc1_e4b_sched.py --smoke >> logs/run_$S.log 2>&1
  local rc=$?
  [ -s $W/$S.json ] || stub $W/$S.json e4bsched "$S" "$B" "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S B=$B rc=$rc $(grep -a 'SC1SCHED_CENSUS' logs/run_$S.log | tail -1 | cut -c1-300)"; return $rc; }
# vllm_arm TAG ARM B MODEL REV [EXTRA ENV...] -- bench/sc1/vllm/sc1_vllm_arm.py (p37 extended); receipt vllm_<TAG>.json; SC1_LOG distinct from the runner's redirect
vllm_arm(){ local TAG=$1 ARM=$2 B=$3 MODEL=$4 MREV=$5; shift 5; local S; S=$(stem_of vllm "$TAG")
  have vllm || { stub $W/$S.json vllm "$ARM" "$B" unsupported "vllm not installed on this box" $W/logs/install_vllm.log; line "$S SKIPPED unsupported"; return 0; }
  local AL; AL=$(arm_alarm 1800); say "arm $S (arm=$ARM B=$B $MODEL $* alarm=$AL)"; sampler_start $S
  { echo "SC1 arm=$S box=$BOX at=$(date -u +%FT%TZ) $(host_snapshot)"; } > logs/run_$S.log
  env SC1_ARM=$ARM SC1_BATCH=$B SC1_PROMPTS=$W/prompts_b$B.json SC1_MODEL=$MODEL SC1_REV=$MREV SC1_OUT=$W/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID SC1_LOG=$W/logs/run_$S.engine.log VLLM_LOGGING_LEVEL=INFO "$@" \
    perl -e "alarm $AL; exec @ARGV" $W/venv-vllm/bin/python $W/vllm/sc1_vllm_arm.py >> logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json vllm "$ARM" "$B" "$(status_of_rc $rc load_fault)" "rc=$rc" logs/run_$S.log
  grep -aE "SC1VLLM|MARLIN|Marlin|Graph capturing finished|Actual usage|Engine core|Error" logs/run_$S.log | tail -4 | cut -c1-240 | sed "s/^/    /"
  line "$S B=$B rc=$rc $(grep -a 'SC1VLLM' logs/run_$S.log | tail -1 | cut -c1-300)"; return $rc; }
# vllm_ttft LEN -- sc1_vllm_ttft.py (graph config, max_num_seqs=1, max_model_len >= prompt+8); receipt ttft_vllm_<len>.json
vllm_ttft(){ local L=$1 S=ttft_vllm_$1 PF=$W/prompts_b1.json; [ "$L" = 4096 ] && PF=$W/prompts_b1_4096.json
  have vllm || { stub $W/$S.json vllm ttft 1 unsupported "vllm not installed"; line "$S SKIPPED unsupported"; return 0; }
  local AL; AL=$(arm_alarm 1800); say "arm $S ($PF alarm=$AL)"; sampler_start $S
  env SC1_ARM=graph_r1 SC1_PROMPTS=$PF SC1_MODEL=$GPTQ_MID SC1_REV=$GPTQ_REV SC1_OUT=$W/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID SC1_LOG=$W/logs/run_$S.engine.log VLLM_LOGGING_LEVEL=INFO \
    perl -e "alarm $AL; exec @ARGV" $W/venv-vllm/bin/python $W/vllm/sc1_vllm_ttft.py > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json vllm ttft 1 "$(status_of_rc $rc load_fault)" "rc=$rc" logs/run_$S.log
  line "$S rc=$rc $(grep -aE 'SC1VLLM|ttft' logs/run_$S.log | tail -1 | cut -c1-240)"; return $rc; }
# vllm_nll MODE SRC [KV] -- the K8-window scorer: prefill (prompt_logprobs=0, one call) | served (2048 max_tokens=1 requests, prefix caching ON);
# receipt nll_vllm_<mode>_<src>.json, nll_vllm_fp8kv_<mode>_<src>.json with KV=fp8 (the quality hazard v3 says is SCORED; P8)
vllm_nll(){ local MODE=$1 SRC=$2 KV=${3:-} S; S=nll_vllm${3:+_fp8kv}_${MODE}_$SRC
  have vllm || { stub $W/$S.json vllm "nll_$MODE" 1 unsupported "vllm not installed"; line "$S SKIPPED unsupported"; return 0; }
  local AL; AL=$(arm_alarm 2400); say "arm $S (nll $MODE $SRC ${KV:+kv=$KV }alarm=$AL)"; sampler_start $S
  env SC1_WINDOW=$W/k8_window_$SRC.json SC1_MODE=$MODE SC1_MODEL=$GPTQ_MID SC1_REV=$GPTQ_REV SC1_OUT=$W/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID SC1_LOG=$W/logs/run_$S.engine.log ${KV:+SC1_KV=$KV} VLLM_LOGGING_LEVEL=INFO \
    perl -e "alarm $AL; exec @ARGV" $W/venv-vllm/bin/python $W/vllm/sc1_vllm_nll.py > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json vllm "nll_$MODE" 1 "$(status_of_rc $rc load_fault)" "rc=$rc" logs/run_$S.log
  line "$S mode=$MODE src=$SRC rc=$rc $(grep -aE 'mean_nll|ppl|VOID|Error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }
# sglang_up MODE / sglang_down -- bench/sc1/sglang/server.sh (sourced by install_sglang); the first start in a fresh work dir is the Marlin MoE JIT
SGL_MODE=""; SGL_LOG=""; SGL_DEAD=""
# A10: every server transition writes a summary line BEFORE it starts (the heartbeat shows only the last summary line, so a
# stall inside a transition was unlocatable on sc1c-5090-1), and a stop that leaves a process stuck in the driver ends SGLang
# on this box: later starts are refused at once rather than piled onto GPU memory a stuck process still holds.
sglang_up(){ local MODE=$1; have sglang || return 1; SGL_LOG=$W/logs/sglang_server_$MODE.log; say "sglang server up (mode=$MODE)"
  if [ -n "$SGL_DEAD" ]; then line "sglang server mode=$MODE NOT STARTED (a previous stop left pid $SGL_DEAD stuck)"; return 46; fi
  line "sglang server mode=$MODE starting"
  sglang_server_start "$GPTQ_MID" "$GPTQ_REV" 30000 "$SGL_LOG" "$MODE" > logs/sglang_start_$MODE.log 2>&1; local rc=$?
  grep -a SGLANG_ENGAGEMENT logs/sglang_start_$MODE.log | tail -2 | cut -c1-300 | sed "s/^/    /"
  if [ $rc -ne 0 ]; then line "sglang server mode=$MODE FAILED rc=$rc $(tail -1 logs/sglang_start_$MODE.log | cut -c1-160)"; SGL_MODE=""; return $rc; fi
  SGL_MODE=$MODE; line "sglang server mode=$MODE up $(grep -a 'SGLANG_ENGAGEMENT {' logs/sglang_start_$MODE.log | tail -1 | cut -c1-200)"; }
sglang_down(){ [ -n "$SGL_MODE" ] || return 0
  line "sglang server mode=$SGL_MODE stopping"; sglang_server_stop > logs/sglang_stop_$SGL_MODE.log 2>&1
  if [ -n "${SGLANG_STOP_STUCK:-}" ]; then SGL_DEAD=$SGLANG_STOP_STUCK; line "sglang server mode=$SGL_MODE STOP STUCK pid $SGL_DEAD (later sglang starts refused)"
  else line "sglang server mode=$SGL_MODE stopped"; fi
  SGL_MODE=""; }
# sglang_arm TAG ARM B [EXTRA ARGS...] -- sc1_sglang_arm.py against the running server; receipt sglang_<TAG>.json (ttft_* bare)
sglang_arm(){ local TAG=$1 ARM=$2 B=$3; shift 3; local S; S=$(stem_of sglang "$TAG")
  [ -n "$SGL_MODE" ] || { stub $W/$S.json sglang "$ARM" "$B" "$(have sglang && echo harness_error || echo unsupported)" "no sglang server (install or start failed)" $W/logs/install_sglang.log; line "$S SKIPPED no server"; return 0; }
  local AL; AL=$(arm_alarm 1800); say "arm $S (arm=$ARM B=$B mode=$SGL_MODE $* alarm=$AL)"; sampler_start $S
  env SC1_INSTANCE_ID=$SC1_INSTANCE_ID perl -e "alarm $AL; exec @ARGV" $W/venv-sglang/bin/python $W/sglang/sc1_sglang_arm.py --arm $ARM --batch $B --prompts $W/prompts_b$B.json --port 30000 --server-mode $SGL_MODE \
      --server-info $SGL_LOG.server_info.json --engagement $SGL_LOG.engagement.json --tripwire $W/sglang_tripwire.json --instance-id "$SC1_INSTANCE_ID" --out $W/$S.json "$@" > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json sglang "$ARM" "$B" "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S B=$B rc=$rc $(grep -a 'SC1SGLANG' logs/run_$S.log | tail -1 | cut -c1-300)"; return $rc; }
# sglang_nll MODE SRC -- prefill (return_logprob, logprob_start_len 0) | served (generation position, token_ids_logprob, radix ON); receipt nll_sglang_<mode>_<src>.json
sglang_nll(){ local MODE=$1 SRC=$2 S=nll_sglang_$1_$2
  [ -n "$SGL_MODE" ] || { stub $W/$S.json sglang "nll_$MODE" 1 "$(have sglang && echo harness_error || echo unsupported)" "no sglang server" $W/logs/install_sglang.log; line "$S SKIPPED no server"; return 0; }
  local AL; AL=$(arm_alarm 2400); say "arm $S (nll $MODE $SRC alarm=$AL)"; sampler_start $S
  perl -e "alarm $AL; exec @ARGV" $W/venv-sglang/bin/python $W/sglang/sc1_sglang_nll.py --window $W/k8_window_$SRC.json --mode $MODE --port 30000 \
      --server-info $SGL_LOG.server_info.json --engagement $SGL_LOG.engagement.json --out $W/$S.json > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json sglang "nll_$MODE" 1 "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S mode=$MODE src=$SRC rc=$rc $(grep -aE 'mean_nll|ppl|VOID|Error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }
# llamacpp_up GGUF NP [CTX] / llamacpp_down -- bench/sc1/llamacpp/llamacpp_box.sh (REFUSES unless 49/49 layers on the GPU, FA on, n_slots = np)
LL_UP=0
llamacpp_up(){ local GGUF=$1 NP=$2 CTX=${3:-}; have llamacpp || return 1; local NAME; NAME=$(basename "$GGUF" .gguf)_np$NP${CTX:+_c$CTX}
  say "llama-server up ($NAME)"; llamacpp_server_start "$GGUF" "$NP" 8080 "$W/logs/llamacpp_server_$NAME.log" $CTX > logs/llamacpp_start_$NAME.log 2>&1; local rc=$?
  if [ $rc -ne 0 ]; then line "llamacpp server $NAME FAILED rc=$rc $(tail -1 logs/llamacpp_start_$NAME.log | cut -c1-160)"; LL_UP=0; return $rc; fi
  LL_UP=1; line "llamacpp server $NAME up"; }
llamacpp_down(){ [ "$LL_UP" = 1 ] && llamacpp_server_stop; LL_UP=0; }
# llamacpp_arm TAG B [EXTRA ARGS...] -- sc1_llamacpp_arm.py against the running server; receipt llamacpp_<TAG>.json
llamacpp_arm(){ local TAG=$1 B=$2; shift 2; local S; S=$(stem_of llamacpp "$TAG")
  [ "$LL_UP" = 1 ] || { stub $W/$S.json llamacpp "$TAG" "$B" "$(have llamacpp && echo harness_error || echo unsupported)" "no llama-server (build, fetch or start failed)" $W/logs/install_llamacpp.log; line "$S SKIPPED no server"; return 0; }
  local AL; AL=$(arm_alarm 1800); say "arm $S (B=$B $* alarm=$AL)"; sampler_start $S
  perl -e "alarm $AL; exec @ARGV" "$PY" $W/llamacpp/sc1_llamacpp_arm.py --server http://127.0.0.1:8080 --prompts $W/prompts_b$B.json --batch $B --arm $TAG --out $W/$S.json \
      --server-version "$(head -1 $LLAMACPP_SRC_DIR/build/llama-server.version.txt 2>/dev/null)" --server-log "$LLAMACPP_SERVER_LOG" "$@" > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json llamacpp "$TAG" "$B" "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S B=$B rc=$rc $(grep -a 'SC1LLAMACPP' logs/run_$S.log | tail -1 | cut -c1-300)"; return $rc; }
# llamacpp_ttft LEN -- the arm driver's --ttft (stream, n_predict 8, wall to the first token) against an np=1 server; receipt ttft_llamacpp_<len>.json
llamacpp_ttft(){ local L=$1 PF=$W/prompts_b1.json; [ "$L" = 4096 ] && PF=$W/prompts_b1_4096.json
  llamacpp_arm ttft_llamacpp_$L 1 --ttft --prompt-len 0 --prompts $PF; }
# llamacpp_nll GGUF MODE SRC -- nll_teacher_forced (libllama, the server's runtime: FA on, f16 KV, full offload); the server must be DOWN (two 18 GB copies do not fit); receipt nll_llamacpp_<mode>_<src>.json
llamacpp_nll(){ local GGUF=$1 MODE=$2 SRC=$3 S=nll_llamacpp_$2_$3
  have llamacpp && [ -s "$GGUF" ] || { stub $W/$S.json llamacpp "nll_$MODE" 1 unsupported "no llama.cpp build or GGUF"; line "$S SKIPPED"; return 0; }
  local AL; AL=$(arm_alarm 2400); say "arm $S (nll $MODE $SRC alarm=$AL)"; sampler_start $S
  perl -e "alarm $AL; exec @ARGV" $LLAMACPP_BIN/nll_teacher_forced --model "$GGUF" --tokens $W/k8_window_$SRC.json --out $W/$S.json --prompt-len 512 --steps 2048 --mode $MODE \
      --n-gpu-layers 99 --flash-attn on --type-kv f16 > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json llamacpp "nll_$MODE" 1 "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S mode=$MODE src=$SRC rc=$rc $(grep -aE 'mean_nll|ppl|REFUSE|error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }
# exl3_arm TAG B [--ttft] / exl3_nll MODE SRC -- bench/sc1/exl3 (in-process; SC1_MODEL_DIR = the pre-fetched snapshot); receipts exl3_<TAG>.json / nll_exl3_<mode>_<src>.json
EXL3_DIR=""; GPTQ_DIR=""; EXL3_TTFT_PROMPTS=""; LMD_TTFT_PROMPTS=""
exl3_arm(){ local TAG=$1 B=$2; shift 2; local S PROMPTS=$W/prompts_b$B.json; S=$(stem_of exl3 "$TAG"); case " $* " in *" --ttft "*) PROMPTS=$EXL3_TTFT_PROMPTS;; esac
  have exl3 && [ -n "$EXL3_DIR" ] || { stub $W/$S.json exllamav3 "$TAG" "$B" unsupported "exllamav3 not installed or checkpoint not fetched" $W/logs/install_exl3.log; line "$S SKIPPED unsupported"; return 0; }
  local AL; AL=$(arm_alarm 1800); say "arm $S (B=$B $* alarm=$AL)"; sampler_start $S
  env SC1_ARM=$TAG SC1_BATCH=$B SC1_PROMPTS=$PROMPTS SC1_MODEL=$EXL3_MID SC1_REV=$EXL3_REV SC1_MODEL_DIR=$EXL3_DIR SC1_OUT=$W/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID \
    perl -e "alarm $AL; exec @ARGV" $W/venv-exl3/bin/python $W/exl3/sc1_exl3_arm.py "$@" > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json exllamav3 "$TAG" "$B" "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S B=$B rc=$rc $(grep -a 'SC1EXL3' logs/run_$S.log | tail -1 | cut -c1-300)"; return $rc; }
exl3_nll(){ local MODE=$1 SRC=$2 S=nll_exl3_$1_$2
  have exl3 && [ -n "$EXL3_DIR" ] || { stub $W/$S.json exllamav3 "nll_$MODE" 1 unsupported "exllamav3 not installed"; line "$S SKIPPED"; return 0; }
  local AL; AL=$(arm_alarm 2400); say "arm $S (nll $MODE $SRC alarm=$AL)"; sampler_start $S
  env SC1_WINDOW=$W/k8_window_$SRC.json SC1_MODEL=$EXL3_MID SC1_REV=$EXL3_REV SC1_MODEL_DIR=$EXL3_DIR SC1_OUT=$W/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID \
    perl -e "alarm $AL; exec @ARGV" $W/venv-exl3/bin/python $W/exl3/sc1_exl3_nll.py --mode $MODE > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json exllamav3 "nll_$MODE" 1 "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S mode=$MODE src=$SRC rc=$rc $(grep -aE 'mean_nll|ppl|REFUSED|Error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }
# lmdeploy_arm TAG B [--ttft] / lmdeploy_nll MODE SRC -- bench/sc1/lmdeploy (TurboMind W4A16 on the GPTQ checkpoint); receipts lmdeploy_<TAG>.json / nll_lmdeploy_<mode>_<src>.json
lmdeploy_arm(){ local TAG=$1 B=$2; shift 2; local S PROMPTS=$W/prompts_b$B.json; S=$(stem_of lmdeploy "$TAG"); case " $* " in *" --ttft "*) PROMPTS=$LMD_TTFT_PROMPTS;; esac
  have lmdeploy && [ -n "$GPTQ_DIR" ] || { stub $W/$S.json lmdeploy-turbomind "$TAG" "$B" unsupported "${LMD_UNSUPPORTED:-lmdeploy not installed or GPTQ checkpoint not fetched}" $W/logs/install_lmdeploy.log; line "$S SKIPPED unsupported"; return 0; }
  local AL; AL=$(arm_alarm 1800); say "arm $S (B=$B $* alarm=$AL)"; sampler_start $S
  env SC1_ARM=$TAG SC1_BATCH=$B SC1_PROMPTS=$PROMPTS SC1_MODEL=$GPTQ_MID SC1_REV=$GPTQ_REV SC1_MODEL_DIR=$GPTQ_DIR SC1_OUT=$W/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID \
    perl -e "alarm $AL; exec @ARGV" $W/venv-lmdeploy/bin/python $W/lmdeploy/sc1_lmdeploy_arm.py "$@" > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json lmdeploy-turbomind "$TAG" "$B" "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S B=$B rc=$rc $(grep -a 'SC1LMD ' logs/run_$S.log | tail -1 | cut -c1-300)"; return $rc; }
lmdeploy_nll(){ local MODE=$1 SRC=$2 S=nll_lmdeploy_$1_$2
  have lmdeploy && [ -n "$GPTQ_DIR" ] || { stub $W/$S.json lmdeploy-turbomind "nll_$MODE" 1 unsupported "${LMD_UNSUPPORTED:-lmdeploy not installed}"; line "$S SKIPPED"; return 0; }
  local AL; AL=$(arm_alarm 2400); say "arm $S (nll $MODE $SRC alarm=$AL)"; sampler_start $S
  env SC1_WINDOW=$W/k8_window_$SRC.json SC1_MODEL=$GPTQ_MID SC1_REV=$GPTQ_REV SC1_MODEL_DIR=$GPTQ_DIR SC1_OUT=$W/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID \
    perl -e "alarm $AL; exec @ARGV" $W/venv-lmdeploy/bin/python $W/lmdeploy/sc1_lmdeploy_nll.py --mode $MODE > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/$S.json ] || stub $W/$S.json lmdeploy-turbomind "nll_$MODE" 1 "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S mode=$MODE src=$SRC rc=$rc $(grep -aE 'mean_nll|ppl|REFUSED|Error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }
# energy TAG ENGINE B RECEIPT CMD... -- ONE sampler (samples/energy_<TAG>.csv, copied to energy_<TAG>.csv) brackets the whole command; energy_<TAG>.json
# carries the reducer's fields {engine, batch, start_epoch, stop_epoch, tokens, interval_ms, sampler_start_epoch}: the receipt's own window where it
# has one (e4b-sched energy_window_epoch; vLLM's LONG rep at SC1_LONG=1024), else the whole arm after load_s -- labelled in `basis`
energy(){ local TAG=$1 ENGINE=$2 B=$3 RECEIPT=$4; shift 4; local t0 t1 rc; t0=$(date +%s.%N)
  SAMPLE_AS=energy_$TAG "$@"; rc=$?; t1=$(date +%s.%N)
  cp samples/energy_$TAG.csv energy_$TAG.csv 2>/dev/null
  "$PY" - "$TAG" "$ENGINE" "$B" "$RECEIPT" "$t0" "$t1" <<'PY' | tee -a summary.txt
import json, os, sys
tag, engine, b, receipt, t0, t1 = sys.argv[1:7]
b, t0, t1 = int(b), float(t0), float(t1)
rec = json.load(open(receipt)) if os.path.exists(receipt) else {}
mp = f"samples/energy_{tag}.meta.json"
meta = json.load(open(mp)) if os.path.exists(mp) else {}
start = stop = tokens = None
basis = "no receipt"
if rec.get("energy_window_epoch"):
    start, stop = rec["energy_window_epoch"]
    tokens = rec.get("energy_tokens")
    basis = "e4b-sched energy_window_epoch (1024 decoded tokens per row)"
elif isinstance(rec.get("timed_windows_epoch"), dict) and rec["timed_windows_epoch"].get("long"):
    start, stop = rec["timed_windows_epoch"]["long"][0]
    tokens = sum(len(v) for v in (rec.get("tokens") or {}).values()) or None
    basis = "vLLM timed_windows_epoch.long[0] (SC1_LONG=1024 x B decoded tokens, one rep)"
elif rec:
    start, stop, tokens = t0 + float(rec.get("load_s") or 0), t1, 640 * b
    basis = "whole arm after load_s: 4x32 + 4x128 decoded tokens per row (640 x B); the prefills' energy is INCLUDED -- an upper bound on J/token, labelled"
out = {"tag": tag, "engine": engine, "batch": b, "start_epoch": start, "stop_epoch": stop, "tokens": tokens, "interval_ms": 50,
       "sampler_start_epoch": meta.get("start_epoch"), "sampler_fields": meta.get("fields"), "arm_start_epoch": t0, "arm_end_epoch": t1,
       "receipt": os.path.basename(receipt), "receipt_status": rec.get("status", rec.get("verdict")), "basis": basis, "csv": f"energy_{tag}.csv"}
json.dump(out, open(f"energy_{tag}.json", "w"), indent=1)
print(f"ENERGY {tag} engine={engine} B={b} tokens={tokens} window={start}..{stop} basis={basis[:70]}")
PY
  return $rc; }
# needs_third A B -- two draws disagreeing by > 3 % on decode_tok_s want a third (the point is the median of three)
needs_third(){ "$PY" - "$1" "$2" <<'PY'
import json, sys
try:
    a, b = (json.load(open(p)).get("decode_tok_s") for p in sys.argv[1:3])
    sys.exit(0 if (a and b and abs(a - b) / ((a + b) / 2) > 0.03) else 1)
except Exception:
    sys.exit(1)
PY
}
reduce(){ if [ -s $W/sc1_reduce.py ]; then say "reduce"; "$PY" $W/sc1_reduce.py $W --box $BOX --out-dir $W > logs/reduce.log 2>&1; local rc=$?; tail -30 logs/reduce.log | tee -a summary.txt
    [ -s $W/verdict.json ] || { say "REDUCER wrote no verdict (rc=$rc)"; rec 22; }
  else echo "REDUCE skipped: sc1_reduce.py not staged (the receipts are complete; reduce off-box)" | tee -a summary.txt; fi; }
# ============================================================================ the PROVING RENTAL (SC1_PROVE=1): no bf16 Qwen3 fetch
if [ "$PROVE" = 1 ]; then
  echo "PROVE -- the proving rental: pre-flight passed; installs + tripwires above; the e4b paged engine end to end on Granite" | tee -a summary.txt
  case "$BOX" in A) PROVE_NEEDS="vllm";; B) PROVE_NEEDS="vllm llamacpp exl3";; C) PROVE_NEEDS="vllm exl3 sglang";; D) PROVE_NEEDS="vllm sglang llamacpp nsys";; esac   # = the install dispatch's sets
  for E in $PROVE_NEEDS; do have $E || { say "PROVE: $E did not install -- NOT PROVED"; rec 23; }; done
  quiesce prove
  if fetch granite "$GR" "$GR_REV" 900 && bake granite "$GR" 1500; then
    GA=$W/work_granite/nf4.arena
    for B in 1 16; do if can_run 600 smoke_granite_b$B; then sched_smoke granite_b$B $B "$GR_ENV" 0 "$GR" "$GR_REV" "$GA"; rec $?; else say "PROVE: smoke_granite_b$B NOT RUN (deadline) -- a skipped smoke is not a proof (A2)"; rec 23; fi; done
  else rec 12; fi
  if [ "$BOX" = B ]; then
    # A10: box B's proof starts the comparator server sc1b-5090-1 could not start: the registered llama-server start and
    # engagement checks on the lane's own Q4_K_M at np=1. Installs + tripwires alone (A2's proof) never started a server.
    # A skipped (can_run) or failed fetch or start is NOT PROVED. LMDeploy is not installed on box B (LMD_UNSUPPORTED).
    PB_STEPS=""
    if can_run 900 prove_gguf_q4km && mkdir -p $W/gguf && LLAMACPP_PY=$PY perl -e "alarm $(arm_alarm 1800); exec @ARGV" \
         bash -c ". $W/llamacpp/llamacpp_box.sh && llamacpp_fetch $GGUF_REPO $GGUF_Q4KM $W/gguf $GGUF_REV" > logs/fetch_gguf_prove.log 2>&1; then
      if can_run 300 prove_llama_server && llamacpp_up $W/gguf/$GGUF_Q4KM 1; then
        line "PROVE llama-server engaged: $(grep -a 'healthy on' logs/llamacpp_start_*np1.log | tail -1 | cut -c1-160)"; llamacpp_down; PB_STEPS="$PB_STEPS llama_server"
      else say "PROVE: llama-server did not start or was refused -- NOT PROVED (A10)"; rec 23; fi
    else say "PROVE: the Q4_K_M fetch did not run or failed -- NOT PROVED (A10)"; rec 23; fi
  fi
  if [ "$BOX" = C ]; then
    if [ -z "${SC1_PROVE_SGLANG_MODEL:-}" ]; then say "PROVE: box C needs SC1_PROVE_SGLANG_MODEL=<repo@rev> -- NOT PROVED (A2)"; rec 23
    elif ! have sglang; then rec 23
    elif ! can_run 900 prove_sglang_jit; then say "PROVE: the sglang JIT + /health NOT RUN (deadline) -- NOT PROVED (A2)"; rec 23
    else   # <repo@rev>: a small GPTQ MoE checkpoint the proof may fetch (never the lane's 30B); the JIT + /health answer is the box-C proof's purpose
      PM=${SC1_PROVE_SGLANG_MODEL%@*}; PR=${SC1_PROVE_SGLANG_MODEL#*@}; say "PROVE: sglang Marlin MoE JIT + /health on $PM @ $PR"
      t0=$(date +%s); sglang_server_start "$PM" "$PR" 30000 "$W/logs/sglang_server_prove.log" matched > logs/sglang_start_prove.log 2>&1; rc=$?
      line "PROVE sglang jit+health rc=$rc startup_s=$(( $(date +%s) - t0 )) $(grep -a 'SGLANG_ENGAGEMENT {' logs/sglang_start_prove.log | tail -1 | cut -c1-200)"; sglang_server_stop; rec $rc
    fi
  fi
  [ "$BOX" = D ] && prove_d                                                   # SC1b: the toy + four captures that must reduce
  [ "$rc_any" = 0 ] || { say "PROVE: NOT PROVED (rc_any=$rc_any)"; finish 23; }
  echo "PROVED box=$BOX installs=[$PROVE_NEEDS] smokes=[granite_b1 granite_b16]$([ "$BOX" = C ] && echo ' sglang_jit=ran')$([ "$BOX" = B ] && echo " comparators=[${PB_STEPS# }]")$([ "$BOX" = D ] && echo ' census=[toy e4b_granite_b16_graph vllm_b1_node sglang_b1_node llamacpp_b16_graph]')" | tee -a summary.txt
  : > PROVED; finish 0
fi
# ============================================================================ the REAL lane: common Phase 0 pieces
fetch_common(){ fetch qwen3 "$MID" "$REV" 4800 || return 11                 # every box: the NF4 bake reads the bf16 checkpoint (box B also: the oracle)
  if have vllm || have lmdeploy || [ "$BOX" = C ]; then                      # the GPTQ checkpoint: vLLM (every box), LMDeploy (B), SGLang (C, installed later)
    if fetch gptq "$GPTQ_MID" "$GPTQ_REV" 1800; then GPTQ_DIR=$(cat fetch_gptq.path)
    else line "GPTQ DL FAIL: no vLLM / LMDeploy / SGLang arms"; OK[vllm]=0; OK[lmdeploy]=0; OK[sglang]=0; fi
  fi
  return 0; }
fetch_exl3(){ have exl3 || return 0; fetch exl3 "$EXL3_MID" "$EXL3_REV" 1800 && EXL3_DIR=$(cat fetch_exl3.path); }
bake_qwen3(){ bake qwen3 "$MID" 5400 || return 12; QA=$W/work_qwen3/nf4.arena; }
# ---- arm recipes (DRAW = r1|r2|r3 names the draw; the receipt is <engine>_<arm>_b<B>_<draw>.json, the reducer's vocabulary)
arm_lic_window(){ local D=$1 B=$2 CEN=""; [ "$D" = r1 ] && CEN=1; lic_ready e4b_lic_b${B}_$D e4b $B || return 0; e4b_window lic_b${B}_$D $B "$LICENV $PACKENV" --fuse-qkv "$CEN"; }
arm_lic_sched(){ local D=$1 B=$2; lic_ready e4bsched_lic_sched_b${B}_$D e4bsched $B || return 0; sched_arm lic_sched_b${B}_$D $B "$LICENV $PACKENV" 2048 1 $W/prompts_b$B.json; }
arm_rtn(){ local D=$1 B=$2 CEN=""; [ "$D" = r1 ] && CEN=1; e4b_window rtn_b${B}_$D $B "$SPEEDENV" --fuse-qkv "$CEN"; }
arm_nf4(){ local D=$1 B=$2 CEN=""; [ "$D" = r1 ] && CEN=1; e4b_window nf4_ctrl_b${B}_$D $B "$NF4ENV" --no-fuse-qkv "$CEN"; }
arm_int4_window(){ local D=$1 B=$2 CEN=""; [ "$D" = r1 ] && CEN=1; e4b_window int4_b${B}_$D $B "$SPEEDENV" --fuse-qkv "$CEN"; }        # boxes B/C: the RTN pack as the anchor (licensed by box A's |lic - rtn| <= 2 %)
arm_int4_sched(){ local D=$1 B=$2; sched_arm int4_sched_b${B}_$D $B "$SPEEDENV" 2048 1 $W/prompts_b$B.json; }
arm_degraded(){ local D=$1; lic_ready e4b_lic_degraded_b16_$D e4b 16 || return 0; e4b_window lic_degraded_b16_$D 16 "$LICENV $PACKENV" --fuse-qkv 1 E4B_INT4_GROUPED_SMALLM=0; }
arm_sched_sameprompt(){ local D=$1; lic_ready e4bsched_lic_sched_sameprompt_b16_$D e4bsched 16 || return 0; sched_arm lic_sched_sameprompt_b16_$D 16 "$LICENV $PACKENV" 2048 1 $W/prompts_b16.json --sameprompt; }
arm_vllm_gptq(){ local D=$1 B=$2; vllm_arm gptq_graph_b${B}_$D "$([ "$D" = r2 ] && echo graph_r2 || echo graph_r1)" $B "$GPTQ_MID" "$GPTQ_REV"; }
arm_vllm_fp8kv(){ local D=$1 B=$2; vllm_arm gptq_fp8kv_b${B}_$D fp8kv $B "$GPTQ_MID" "$GPTQ_REV"; }
arm_vllm_sameprompt(){ local D=$1; vllm_arm gptq_graph_sameprompt_b16_$D sameprompt 16 "$GPTQ_MID" "$GPTQ_REV"; }
arm_vllm_nodetok(){ local B=$1; vllm_arm gptq_graph_nodetok_b${B}_r1 nodetok $B "$GPTQ_MID" "$GPTQ_REV"; }   # the F7 pair: the matched vLLM arms keep detokenize=True (a comparator's loop is never trimmed to e4b's omission); this runs detokenize=False at both B
arm_llamacpp_q4km(){ local D=$1 B=$2; llamacpp_up $W/gguf/$GGUF_Q4KM $B && { llamacpp_arm q4km_b${B}_$D $B; local rc=$?; llamacpp_down; return $rc; }; llamacpp_arm q4km_b${B}_$D $B; }
arm_llamacpp_iq4xs(){ llamacpp_up $W/gguf/$GGUF_IQ4XS 1 && { llamacpp_arm iq4xs_b1_r1 1; local rc=$?; llamacpp_down; return $rc; }; llamacpp_arm iq4xs_b1_r1 1; }
arm_exl3(){ local D=$1 B=$2; exl3_arm 4bpw_b${B}_$D $B; }
arm_exl3_cu13(){ local B=$1; exl3_arm 4bpw_cu13_b${B}_r1 $B; }                                                                    # box C: the cu13 wheel, a labelled NATIVE row
arm_lmdeploy(){ local D=$1 B=$2; lmdeploy_arm w4a16_b${B}_$D $B; }
arm_sglang_matched(){ local D=$1 B=$2; sglang_arm gptq_matched_b${B}_$D gptq_default $B; }
arm_sglang_native(){ local D=$1 B=$2; sglang_arm gptq_native_b${B}_$D gptq_native $B; }
third(){ local ENGINE=$1 BASE=$2 FN=$3; shift 3; needs_third $W/${ENGINE}_${BASE}_r1.json $W/${ENGINE}_${BASE}_r2.json || return 0
  line "THIRD DRAW $ENGINE/$BASE: r1 and r2 disagree by > 3 %"; can_run 900 ${ENGINE}_${BASE}_r3 && { $FN r3 "$@"; rec $?; }; }
# TTFT recipes: 512 from prompts_b1.json, 4096 from prompts_b1_4096.json (a separate single-sequence configuration sized 4096 + 8); receipts ttft_<engine>_<len>.json
SCHED_NAME=lic; SCHED_STACK=""
ttft_sched(){ local L=$1 PF=$W/prompts_b1.json MTS=2048; [ "$L" = 4096 ] && { PF=$W/prompts_b1_4096.json; MTS=4104; }
  [ "$SCHED_NAME" = lic ] && { lic_ready ttft_e4b_$L e4bsched 1 || return 0; }
  can_run 600 ttft_e4b_$L && { sched_arm ttft_e4b_$L 1 "$SCHED_STACK" $MTS 1 $PF --ttft; rec $?; }; }
ttft_vllm(){ local L=$1; can_run 600 ttft_vllm_$L && { vllm_ttft $L; rec $?; }; }
ttft_sglang(){ local L=$1 PF=$W/prompts_b1.json; [ "$L" = 4096 ] && PF=$W/prompts_b1_4096.json; can_run 600 ttft_sglang_$L && { sglang_arm ttft_sglang_$L ttft 1 --ttft --prompts $PF; rec $?; }; }
ttft_llamacpp(){ local L=$1; can_run 600 ttft_llamacpp_$L && { llamacpp_ttft $L; rec $?; }; }
ttft_exl3(){ local L=$1; EXL3_TTFT_PROMPTS=$W/prompts_b1.json; [ "$L" = 4096 ] && EXL3_TTFT_PROMPTS=$W/prompts_b1_4096.json; can_run 600 ttft_exl3_$L && { exl3_arm ttft_exl3_$L 1 --ttft; rec $?; }; }
ttft_lmdeploy(){ local L=$1; LMD_TTFT_PROMPTS=$W/prompts_b1.json; [ "$L" = 4096 ] && LMD_TTFT_PROMPTS=$W/prompts_b1_4096.json; can_run 600 ttft_lmdeploy_$L && { lmdeploy_arm ttft_lmdeploy_$L 1 --ttft; rec $?; }; }
# energy recipes (tags = the reducer's energy_<engine>_b<B>): e4b-sched --energy (1024 tokens at B, its own epoch window); vLLM via SC1_SHORT=32 SC1_LONG=1024
# SC1_REPS=1 (the LONG rep's timed_windows_epoch); the box-B comparators' drivers expose neither a 1024-token mode nor an epoch window -> whole-arm, labelled
energy_sched(){ local B=$1; [ "$SCHED_NAME" = lic ] && { lic_ready e4bsched_energy_b$B e4bsched $B || return 0; }
  can_run 900 energy_e4b_b$B && { energy e4b_b$B e4b $B $W/e4bsched_energy_b$B.json sched_arm energy_b$B $B "$SCHED_STACK" 2048 1 $W/prompts_b$B.json --energy; rec $?; }; }
energy_vllm(){ local B=$1; can_run 900 energy_vllm_b$B && { energy vllm_b$B vllm $B $W/vllm_energy_b$B.json vllm_arm energy_b$B graph_r1 $B "$GPTQ_MID" "$GPTQ_REV" SC1_SHORT=32 SC1_LONG=1024 SC1_REPS=1; rec $?; }; }
energy_llamacpp(){ can_run 900 energy_llamacpp_b1 || return 0; local rc    # the server is brought up OUTSIDE the sampler window (its load is not decode)
  if llamacpp_up $W/gguf/$GGUF_Q4KM 1; then energy llamacpp_b1 llamacpp 1 $W/llamacpp_energy_q4km_b1.json llamacpp_arm energy_q4km_b1 1; rc=$?; llamacpp_down
  else energy llamacpp_b1 llamacpp 1 $W/llamacpp_energy_q4km_b1.json llamacpp_arm energy_q4km_b1 1; rc=$?; fi; rec $rc; }
energy_exl3(){ can_run 900 energy_exl3_b1 && { energy exl3_b1 exl3 1 $W/exl3_energy_4bpw_b1.json exl3_arm energy_4bpw_b1 1; rec $?; }; }
energy_lmdeploy(){ can_run 900 energy_lmdeploy_b1 && { energy lmdeploy_b1 lmdeploy 1 $W/lmdeploy_energy_w4a16_b1.json lmdeploy_arm energy_w4a16_b1 1; rec $?; }; }
# ============================================================================ the per-box phase lists (v3 "Arms and their order", in that order)
box_a(){ SCHED_NAME=lic; SCHED_STACK="$LICENV $PACKENV"
  phase 0 "fetch bf16 + the GPTQ checkpoint, bake, prompts, build_engine smoke (both fusion sets), quiesce, the licensed pack build, the premise"
  fetch_common || finish 11
  bake_qwen3 || finish 12; prompts || finish 19
  for B in 1 16; do   # the fused sched path first meets the GPU HERE, before the pack build spends 35 min (the proof's Granite cannot exercise fuse_qkv)
    can_run 600 smoke_qwen3_fused_b$B && { sched_smoke qwen3_fused_b$B $B "$SPEEDENV" 1; rec $?; }
    can_run 600 smoke_qwen3_folds_b$B && { sched_smoke qwen3_folds_b$B $B "$SPEEDENV" 0; rec $?; }
  done
  quiesce build
  can_run 5400 k8_build && e4b_build
  SCHED_STACK="$LICENV $PACKENV"
  premise
  phase A "e4b/lic_b16_r1, vllm/gptq_graph_b16_r1, e4b/lic_b1_r1, vllm/gptq_graph_b1_r1"
  quiesce arms
  can_run 900 e4b_lic_b16_r1 && { arm_lic_window r1 16; rec $?; }
  can_run 900 vllm_gptq_graph_b16_r1 && { arm_vllm_gptq r1 16; rec $?; }
  can_run 900 e4b_lic_b1_r1 && { arm_lic_window r1 1; rec $?; }
  can_run 900 vllm_gptq_graph_b1_r1 && { arm_vllm_gptq r1 1; rec $?; }
  phase A2 "e4bsched/lic_sched_b16_r1, e4bsched/lic_sched_b1_r1 (the ratio axis)"
  can_run 900 e4bsched_lic_sched_b16_r1 && { arm_lic_sched r1 16; rec $?; }
  can_run 900 e4bsched_lic_sched_b1_r1 && { arm_lic_sched r1 1; rec $?; }
  phase B "the licence on this box: K8 lic at auto (k8_lic_auto_*) and at =1 (k8_lic_1_*, K19 + K23 at T == 1, censused), wikitext + c4val1; the NF4 control K8 both texts"
  for SRC in wikitext c4val1; do [ -n "$PACKENV" ] && can_run 1500 k8_lic_auto_$SRC && { e4b_k8 lic_auto_$SRC "$LICENV $PACKENV E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto" $SRC; rec $?; }; done
  [ -n "$PACKENV" ] && can_run 600 k8_census && { k8_census; rec $?; }
  for SRC in wikitext c4val1; do [ -n "$PACKENV" ] && can_run 1500 k8_lic_1_$SRC && { e4b_k8 lic_1_$SRC "$LICENV $PACKENV E4B_INT4_GROUPED_SMALLM=1 E4B_INT4_LEAN_GLUE=auto" $SRC; rec $?; }; done
  for SRC in wikitext c4val1; do can_run 900 k8_nf4_$SRC && { e4b_k8 nf4_$SRC "$NF4ENV" $SRC; rec $?; }; done
  phase D-vLLM "vLLM quality both windows, prefill + served (nll_vllm_*); the e4b prefill-shaped rows (nll_e4b_prefill_*: --ppl-oracle eager on the lic env)"
  for SRC in wikitext c4val1; do can_run 1500 nll_vllm_prefill_$SRC && { vllm_nll prefill $SRC; rec $?; }; done
  for SRC in wikitext c4val1; do can_run 1500 nll_vllm_served_$SRC && { vllm_nll served $SRC; rec $?; }; done
  for SRC in wikitext c4val1; do [ -n "$PACKENV" ] && can_run 900 nll_e4b_prefill_$SRC && { e4b_k8 nll_e4b_prefill_$SRC "$LICENV $PACKENV" $SRC --ppl-oracle eager; rec $?; }; done
  phase G1 "second draws of A and A2"
  can_run 900 e4b_lic_b16_r2 && { arm_lic_window r2 16; rec $?; }
  can_run 900 vllm_gptq_graph_b16_r2 && { arm_vllm_gptq r2 16; rec $?; }
  can_run 900 e4b_lic_b1_r2 && { arm_lic_window r2 1; rec $?; }
  can_run 900 vllm_gptq_graph_b1_r2 && { arm_vllm_gptq r2 1; rec $?; }
  can_run 900 e4bsched_lic_sched_b16_r2 && { arm_lic_sched r2 16; rec $?; }
  can_run 900 e4bsched_lic_sched_b1_r2 && { arm_lic_sched r2 1; rec $?; }
  phase C "vllm/gptq_fp8kv_b{16,1}, e4b/rtn_b{16,1}, e4b/nf4_ctrl_b{16,1} (first draws; the AWQ arm is CUT from SC1 -- v4 Phase C)"
  for B in 16 1; do can_run 900 vllm_gptq_fp8kv_b${B}_r1 && { arm_vllm_fp8kv r1 $B; rec $?; }; done
  for B in 16 1; do can_run 900 e4b_rtn_b${B}_r1 && { arm_rtn r1 $B; rec $?; }; done
  for B in 16 1; do can_run 900 e4b_nf4_ctrl_b${B}_r1 && { arm_nf4 r1 $B; rec $?; }; done
  phase F "controls: vllm/gptq_graph_sameprompt_b16, e4bsched/lic_sched_sameprompt_b16, e4b/lic_degraded_b16 (E4B_INT4_GROUPED_SMALLM=0)"
  can_run 900 vllm_gptq_graph_sameprompt_b16_r1 && { arm_vllm_sameprompt r1; rec $?; }
  can_run 900 e4bsched_lic_sched_sameprompt_b16_r1 && { arm_sched_sameprompt r1; rec $?; }
  can_run 900 e4b_lic_degraded_b16_r1 && { arm_degraded r1; rec $?; }
  phase E "TTFT 512 / 4096 on e4b (sched engine, max_tokens=1) and vLLM"
  ttft_sched 512; ttft_sched 4096; ttft_vllm 512; ttft_vllm 4096
  phase EN "energy windows: e4b_b1, e4b_b16 (the sched engine), vllm_b1, vllm_b16 (1024 decoded tokens under the sampler)"
  energy_sched 1; energy_sched 16; energy_vllm 1; energy_vllm 16
  phase G2 "second draws of C and the controls; third draws where r1/r2 disagree > 3 %; then the designated droppables in order: the nodetok pair (both B), nf4_ctrl r2, the fp8-KV served quality rows"
  for B in 16 1; do can_run 900 vllm_gptq_fp8kv_b${B}_r2 && { arm_vllm_fp8kv r2 $B; rec $?; }; done
  for B in 16 1; do can_run 900 e4b_rtn_b${B}_r2 && { arm_rtn r2 $B; rec $?; }; done
  can_run 900 vllm_gptq_graph_sameprompt_b16_r2 && { arm_vllm_sameprompt r2; rec $?; }
  can_run 900 e4bsched_lic_sched_sameprompt_b16_r2 && { arm_sched_sameprompt r2; rec $?; }
  can_run 900 e4b_lic_degraded_b16_r2 && { arm_degraded r2; rec $?; }
  for B in 16 1; do third e4b lic_b$B arm_lic_window $B; third e4bsched lic_sched_b$B arm_lic_sched $B; third vllm gptq_graph_b$B arm_vllm_gptq $B
                   third vllm gptq_fp8kv_b$B arm_vllm_fp8kv $B; third e4b rtn_b$B arm_rtn $B; done
  for B in 1 16; do can_run 900 vllm_gptq_graph_nodetok_b${B}_r1 && { arm_vllm_nodetok $B; rec $?; }; done
  for B in 16 1; do can_run 900 e4b_nf4_ctrl_b${B}_r2 && { arm_nf4 r2 $B; rec $?; }; done
  for SRC in wikitext c4val1; do can_run 1500 nll_vllm_fp8kv_served_$SRC && { vllm_nll served $SRC fp8; rec $?; }; done   # P8's quality clause (v3 Environments: the fp8-KV hazard is SCORED); not in v3's phase list, so last
  reduce; }
box_b(){ SCHED_NAME=int4; SCHED_STACK="$SPEEDENV"
  phase 0 "fetches (bf16 for the bake + the oracle, GPTQ, two GGUFs, EXL3), bake, prompts; llama.cpp build + ExLlamaV3 + LMDeploy venvs done above; quiesce before the first timed arm"
  fetch_common || finish 11; fetch_ggufs; fetch_exl3
  bake_qwen3 || finish 12; prompts || finish 19
  quiesce arms
  phase AN "anchors: e4b/int4_b{16,1} window, e4bsched/int4_sched_b{16,1}, vllm/gptq_graph_b{16,1} (first draws)"
  for B in 16 1; do can_run 900 e4b_int4_b${B}_r1 && { arm_int4_window r1 $B; rec $?; }; done
  for B in 16 1; do can_run 900 e4bsched_int4_sched_b${B}_r1 && { arm_int4_sched r1 $B; rec $?; }; done
  for B in 16 1; do can_run 900 vllm_gptq_graph_b${B}_r1 && { arm_vllm_gptq r1 $B; rec $?; }; done
  phase OR "the bf16 upstream oracle, both windows (oracle_<src>; box A's whole delta_bf16 axis depends on it)"
  for SRC in wikitext c4val1; do can_run 1800 oracle_$SRC && { oracle $SRC; rec $?; }; done
  phase CMP "llamacpp/q4km_b{16,1}, exl3/4bpw_b{16,1}, lmdeploy/w4a16_b{16,1} (first draws; LMDeploy last among them); llamacpp/iq4xs_b1 (single draw, speed only)"
  for B in 16 1; do can_run 900 llamacpp_q4km_b${B}_r1 && { arm_llamacpp_q4km r1 $B; rec $?; }; done
  for B in 16 1; do can_run 900 exl3_4bpw_b${B}_r1 && { arm_exl3 r1 $B; rec $?; }; done
  for B in 16 1; do can_run 900 lmdeploy_w4a16_b${B}_r1 && { arm_lmdeploy r1 $B; rec $?; }; done
  can_run 600 llamacpp_iq4xs_b1_r1 && { arm_llamacpp_iq4xs; rec $?; }
  phase Q "quality: llama.cpp / ExLlamaV3 / LMDeploy, both windows, prefill + served shape (nll_<engine>_<mode>_<src>)"
  for SRC in wikitext c4val1; do for M in prefill decode; do can_run 1500 nll_llamacpp_${M}_$SRC && { llamacpp_nll $W/gguf/$GGUF_Q4KM $M $SRC; rec $?; }; done; done
  for SRC in wikitext c4val1; do for M in prefill decode; do can_run 1500 nll_exl3_${M}_$SRC && { exl3_nll $M $SRC; rec $?; }; done; done
  for SRC in wikitext c4val1; do for M in prefill decode_prefix; do can_run 1500 nll_lmdeploy_${M}_$SRC && { lmdeploy_nll $M $SRC; rec $?; }; done; done
  phase AN2 "anchors' second draws"
  for B in 16 1; do can_run 900 e4b_int4_b${B}_r2 && { arm_int4_window r2 $B; rec $?; }; done
  for B in 16 1; do can_run 900 e4bsched_int4_sched_b${B}_r2 && { arm_int4_sched r2 $B; rec $?; }; done
  for B in 16 1; do can_run 900 vllm_gptq_graph_b${B}_r2 && { arm_vllm_gptq r2 $B; rec $?; }; done
  phase TT "TTFT 512 / 4096 on every engine present (ttft_<engine>_<len>)"
  ttft_sched 512; ttft_sched 4096; ttft_vllm 512; ttft_vllm 4096
  if llamacpp_up $W/gguf/$GGUF_Q4KM 1 8192; then ttft_llamacpp 512; ttft_llamacpp 4096; llamacpp_down; else ttft_llamacpp 512; ttft_llamacpp 4096; fi
  ttft_exl3 512; ttft_exl3 4096; ttft_lmdeploy 512; ttft_lmdeploy 4096
  phase EN "energy windows, B=1 only: llamacpp_b1, exl3_b1, lmdeploy_b1 (whole-arm, labelled)"
  energy_llamacpp; energy_exl3; energy_lmdeploy
  phase CMP2 "second draws of the six comparator arms; third draws where r1/r2 disagree > 3 %"
  for B in 16 1; do can_run 900 llamacpp_q4km_b${B}_r2 && { arm_llamacpp_q4km r2 $B; rec $?; }; done
  for B in 16 1; do can_run 900 exl3_4bpw_b${B}_r2 && { arm_exl3 r2 $B; rec $?; }; done
  for B in 16 1; do can_run 900 lmdeploy_w4a16_b${B}_r2 && { arm_lmdeploy r2 $B; rec $?; }; done
  for B in 16 1; do third e4b int4_b$B arm_int4_window $B; third e4bsched int4_sched_b$B arm_int4_sched $B; third vllm gptq_graph_b$B arm_vllm_gptq $B
                   third llamacpp q4km_b$B arm_llamacpp_q4km $B; third exl3 4bpw_b$B arm_exl3 $B; third lmdeploy w4a16_b$B arm_lmdeploy $B; done
  reduce; }
box_c(){ SCHED_NAME=int4; SCHED_STACK="$SPEEDENV"
  phase 0 "fetches (bf16 for the bake -- the oracle is box B's; GPTQ; EXL3), bake, prompts; vLLM + ExLlamaV3-cu13 venvs done above; quiesce before the first timed arm"
  fetch_common || finish 11; fetch_exl3
  bake_qwen3 || finish 12; prompts || finish 19
  quiesce arms
  phase AN "anchors as box B: e4b/int4_b{16,1} window, e4bsched/int4_sched_b{16,1}, vllm/gptq_graph_b{16,1}"
  for B in 16 1; do can_run 900 e4b_int4_b${B}_r1 && { arm_int4_window r1 $B; rec $?; }; done
  for B in 16 1; do can_run 900 e4bsched_int4_sched_b${B}_r1 && { arm_int4_sched r1 $B; rec $?; }; done
  for B in 16 1; do can_run 900 vllm_gptq_graph_b${B}_r1 && { arm_vllm_gptq r1 $B; rec $?; }; done
  phase SG0 "SGLang install + first JIT (the matched server's start compiles the Marlin MoE kernel), then quiescence"
  can_run 2700 sglang_install && install_sglang
  sglang_up matched; quiesce sglang
  phase SG "sglang/gptq_matched_b{16,1}, sglang/gptq_native_b{16,1} (first draws)"
  for B in 16 1; do can_run 900 sglang_gptq_matched_b${B}_r1 && { arm_sglang_matched r1 $B; rec $?; }; done
  sglang_down; sglang_up native
  for B in 16 1; do can_run 900 sglang_gptq_native_b${B}_r1 && { arm_sglang_native r1 $B; rec $?; }; done
  sglang_down
  phase SGQ "SGLang quality both windows: prefill (return_logprob, logprob_start_len 0) and served (generation position, token_ids_logprob, radix ON, cached_tokens predicate)"
  sglang_up quality
  for SRC in wikitext c4val1; do can_run 1500 nll_sglang_prefill_$SRC && { sglang_nll prefill $SRC; rec $?; }; done
  for SRC in wikitext c4val1; do can_run 1500 nll_sglang_served_$SRC && { sglang_nll served $SRC; rec $?; }; done
  sglang_down
  phase EX "exl3/4bpw_cu13_b{16,1} (the cu13 wheel: a labelled native row)"
  for B in 16 1; do can_run 900 exl3_4bpw_cu13_b${B}_r1 && { arm_exl3_cu13 $B; rec $?; }; done
  phase TT "TTFT 512 / 4096: e4b-sched, vLLM, SGLang (ttft_matched), ExLlamaV3"
  ttft_sched 512; ttft_sched 4096; ttft_vllm 512; ttft_vllm 4096
  if sglang_up ttft_matched; then ttft_sglang 512; ttft_sglang 4096; sglang_down; else ttft_sglang 512; ttft_sglang 4096; fi
  ttft_exl3 512; ttft_exl3 4096
  phase SG2 "second draws of the SGLang arms; third draws where r1/r2 disagree > 3 %; the anchors' second draws if the clock allows (droppable)"
  sglang_up matched
  for B in 16 1; do can_run 900 sglang_gptq_matched_b${B}_r2 && { arm_sglang_matched r2 $B; rec $?; }; done
  for B in 16 1; do third sglang gptq_matched_b$B arm_sglang_matched $B; done
  sglang_down; sglang_up native
  for B in 16 1; do can_run 900 sglang_gptq_native_b${B}_r2 && { arm_sglang_native r2 $B; rec $?; }; done
  for B in 16 1; do third sglang gptq_native_b$B arm_sglang_native $B; done
  sglang_down
  for B in 16 1; do can_run 900 e4b_int4_b${B}_r2 && { arm_int4_window r2 $B; rec $?; }; done
  for B in 16 1; do can_run 900 e4bsched_int4_sched_b${B}_r2 && { arm_int4_sched r2 $B; rec $?; }; done
  for B in 16 1; do can_run 900 vllm_gptq_graph_b${B}_r2 && { arm_vllm_gptq r2 $B; rec $?; }; done
  reduce; }
case "$BOX" in A) box_a;; B) box_b;; C) box_c;; D) box_d;; esac
say "----- summary -----"; cat summary.txt; echo "----- versions -----"; cat versions.txt
finish "$rc_any"
