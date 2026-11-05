#!/bin/bash
# Lane PGX, BOX side (bench/pgx/PREREG-pgx.md, Amendment 1): THE SHAKEDOWN pgx-shake-1 on one RTX 4090. Never read. It covers
# the registered shakedown paragraph ("Run ids, order and the stop"): the three engine venvs and e4b; the fetch of the three
# pinned checkpoints; the NF4 bake; grouped-nf4-gemm's bench/calibrate.py writing E-hyb's calib.json on this host; the
# measured bytes per expert and K table; the card-clock burn; then ONE lifetime per arm at 8 GB (S4, S8, V8, E-hyb, in draw
# 1's order) with P1 and P2 at c1, each counter shim on a real record (an untimed graphs-off instrument lifetime for S4 and
# S8, the pager's probes for V8, the placement proof and step trace for E-hyb), and each lifetime's wall, peak host RSS and
# peak VRAM, from which the reading's guard and RAM floor are fit. If any arm cannot serve at its budget: STOP (rc 27).
#   order    the frozen knobs; the staged files; the refusals (CUDA, the torch image, the card, the disk, the RAM, the driver,
#            the download probe) and the guard fit; STARTED; installs + tripwires; self-tests; fetch; bytes and K; bake;
#            calibrate + the placement preview; the clock burn; the four lifetimes and two instrument lifetimes; the verdict.
# P233's runner (bench/p233/p233_run.sh) for the shape (the refusals, the guard fit, STARTED, finish) and SC2's server
# helpers (bench/sc2/sc2_box_e.sh: wait_http, wait_e4b_ready, stop_pid), with SC1's SGLang venv installer
# (bench/sc1/sglang/install.sh) at SGLang 0.5.23.
# Refusals: CUDA unusable 18 (the host floor; a torch that will not import stays 10), torch image 28, card class 15, disk 13,
# host RAM 16, driver 18, download below 40 MB/s 19, guard fit 41. Only 13 and 18 name the machine for the launcher's next draw.
# After STARTED: an install, stage or tripwire failure 9, a download failure 11, the bake 12, calibrate 33, a premise
# self-test 25, a step that cannot fit the deadline 40, the verdict STOP or SHAKE_FAULTS 27. This script never creates compute.
set -uo pipefail
W=/root/pgx; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] pgx: $*"; }
NONCE=${PGX_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/PGX_RUN_NONCE.tmp && mv $W/PGX_RUN_NONCE.tmp $W/PGX_RUN_NONCE
SRV_PID=""
finish(){ local rc=$1; [ -n "$SRV_PID" ] && stop_pid "$SRV_PID"; printf '%s\n' "$rc" > PGX_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > PGX_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in PGX_RUN_ID PGX_BOX PGX_DEADLINE_EPOCH E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
case "$PGX_RUN_ID" in
  pgx-shake-1|pgx-shake-r[1-3]) WANT_BOX=shake;;
  *) say "unregistered run ID: $PGX_RUN_ID"; finish 78;;
esac
[ "$PGX_BOX" = "$WANT_BOX" ] || { say "refusing: box $PGX_BOX is not $WANT_BOX for $PGX_RUN_ID"; finish 78; }
# ---- the pins (PREREG "The arms"; Amendment 1): every engine at its registered release, every checkpoint at its revision
TORCH_WANT=2.8.0+cu128          # the image's torch: e4b serves from the image python (P233's form)
GNF4_VERSION=0.51.0; GNF4_TAG_SHA=30adb1bfa76f95f9ce4a4c29734b8f36de907eca   # grouped-nf4-gemm v0.51.0 (calibrate.py is not packaged)
SGLANG_VERSION=0.5.23; SGLANG_TAG_SHA=297e54fd105e974cdf401e5b4d20253a22c16f90
VLLM_VERSION=0.29.0; PAGER_SHA=e9264fda36632a3904b7b1bf7ff13d85f02b3139; PAGER_VERSION=0.1.1
BF16=Qwen/Qwen3-30B-A3B;            BF16_REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
FP8=Qwen/Qwen3-30B-A3B-FP8;         FP8_REV=d206ba732169f29bb77fbf80fc2c4b81d4d30782
INT4=Qwen/Qwen3-30B-A3B-GPTQ-Int4;  INT4_REV=9b534e4318b7ebc3c961a839f13eb18b1833f441
GPU_CLASS=4090; WANT_CC=8.9; MIN_SMS=128; MIN_VRAM_MIB=24000; MIN_RAM_GIB=60; MIN_DISK_GB=250; MIN_DL_MB_S=40; MIN_DRIVER=580
INSTALL_S=1800; FETCH_S=2750; BAKE_S=500; CALIB_S=300; LIFETIME_S=900; INSTR_S=450; RESERVE=600
K_S4=67; K_S8=35; K_V8=35; EHYB_VRAM_GIB=7.451; EHYB_DRAM_GIB=16     # the registered 8 GB row (pgx_tripwire.py checks the table)
PORT_SGL=30000; PORT_VLLM=8000; PORT_E4B=8778
# ---- the knobs are frozen: no engine, kernel or lane key may arrive in the box's environment (the image's own are recorded)
for k in $(env | grep -oE '^(E4B|GNF4|VLLM|SGLANG|PGX)_[A-Za-z0-9_]+' | sort -u); do
  case "$k" in E4B_SHA|PGX_RUN_ID|PGX_BOX|PGX_DEADLINE_EPOCH|PGX_RUN_NONCE) ;; *) say "refusing: $k is set in the box's environment"; finish 78;; esac
done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
: > summary.txt; echo "$PGX_RUN_ID" > RUN_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_VERSION sglang=$SGLANG_VERSION vllm=$VLLM_VERSION pager=$PAGER_SHA torch=$TORCH_WANT K=S4:$K_S4,S8:$K_S8,V8:$K_V8 ehyb_vram_gib=$EHYB_VRAM_GIB" | tee -a summary.txt
# ---- staged pieces, byte-for-byte
for f in pgx_drive.py pgx_reduce.py pgx_residency.py pgx_bytes.py pgx_place_preview.py pgx_tripwire.py pgx_shake.py pgx_box.py \
         fit_profile.jsonl bytes_per_expert.json replay.py card_clock.py k8_bake.py sglang_install.sh hook/usercustomize.py \
         hook/sitecustomize.py hook_e4b/sitecustomize.py prompts/warm_c1.json prompts/warm_c4.json prompts/shift.json \
         prompts/manifest.json staged-pgx.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-pgx.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/pgx/staged-pgx.sha256"; finish 9; }
# pgx_residency.py loads replay.py from ../train-routing-trace (the repo's layout); the box keeps that layout
mkdir -p $W/rt/train-routing-trace $W/rt/pgx && cp $W/replay.py $W/rt/train-routing-trace/ && cp $W/pgx_residency.py $W/pgx_box.py $W/rt/pgx/
# ---- refusals before anything is installed or fetched
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
if not ok:
    print(f"no-cuda torch {torch.__version__}"); sys.exit(0)
cc, p = torch.cuda.get_device_capability(0), torch.cuda.get_device_properties(0)
print(f"ok torch={torch.__version__} cc={cc[0]}.{cc[1]} sms={p.multi_processor_count}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  ok*) ;;
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE}) -- registered host floor"
            echo "refused: cuda unusable (${CUDA_PROBE})" > REFUSAL; finish 18;;
  *) say "DUD BOX (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
case "$CUDA_PROBE" in *" torch=$TORCH_WANT "*) ;; *) say "REFUSED: the image's torch is not $TORCH_WANT ($CUDA_PROBE)"; echo "refused: torch ($CUDA_PROBE)" > REFUSAL; finish 28;; esac
# no uuid: the card's GPU UUID is never collected
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,enforced.power.limit,clocks.max.sm,temperature.gpu,clocks.sm --format=csv,noheader | sed "s/^/power.limit,enforced.power.limit,clocks.max.sm,temperature.gpu,clocks.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)|NUMA node\(s\)|^Flags" | cut -c1-2000 | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
cat /sys/fs/cgroup/memory.max 2>/dev/null | sed "s/^/cgroup memory.max /" | tee -a forensics.txt
env | grep -E '^(PYTORCH_|TORCH_|CUDA_|NV_|NVIDIA_)' | sort > env_image.txt || true
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
SMS=${CUDA_PROBE##*sms=}
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
[ "$CC" = "$WANT_CC" ] || { say "REFUSED: compute capability $CC is not $WANT_CC"; echo "refused: cc $CC" > REFUSAL; finish 15; }
[ "${SMS:-0}" -ge "$MIN_SMS" ] 2>/dev/null || { say "REFUSED: ${SMS:-?} SMs < $MIN_SMS"; echo "refused: sms ${SMS:-?}" > REFUSAL; finish 15; }
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
[ "${VRAM_MIB:-0}" -ge "$MIN_VRAM_MIB" ] || { say "REFUSED: VRAM ${VRAM_MIB:-?} MiB < $MIN_VRAM_MIB"; echo "refused: vram ${VRAM_MIB:-?}" > REFUSAL; finish 15; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (three checkpoints, the NF4 arena, three venvs)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
# the measured host RAM: min(MemTotal, the cgroup's limit), in GiB
RAM_GIB=$(python - <<'PYR'
mt = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal:"))) * 1024
try:
    cg = open("/sys/fs/cgroup/memory.max").read().strip()
    lim = int(cg) if cg.isdigit() else None
except OSError:
    lim = None
print(int(min(mt, lim) if lim else mt) // 2**30)
PYR
)
echo "HOST_RAM_GIB ${RAM_GIB:-?} (min of MemTotal and the cgroup limit)" | tee -a summary.txt
[ "${RAM_GIB:-0}" -ge "$MIN_RAM_GIB" ] || { say "REFUSED: ${RAM_GIB:-?} GiB host RAM < ${MIN_RAM_GIB} GiB"; echo "refused: ram ${RAM_GIB:-?} GiB" > REFUSAL; finish 16; }
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); DRIVER_MAJOR=${DRIVER%%.*}
[ "${DRIVER_MAJOR:-0}" -ge "$MIN_DRIVER" ] 2>/dev/null || { say "REFUSED: driver ${DRIVER:-unknown} < $MIN_DRIVER (SGLang 0.5.23's torch 2.14.1 wheels are CUDA 13)"; echo "refused: driver ${DRIVER:-unknown}" > REFUSAL; finish 18; }
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc 0-9)
echo "POWER_LIMIT_W ${PL:-unreadable} (recorded, never ruled)" | tee -a summary.txt
# the download probe (P233's form, on the largest checkpoint's first shard): below MIN_DL_MB_S, a non-2xx or an error is 19
rm -f $W/download_probe.json
DL=$(MIX=$BF16 MIX_REV=$BF16_REV FLOOR=$MIN_DL_MB_S OUT=$W/download_probe.json timeout 120 python - <<'PYD' 2>/dev/null | tail -1
import json, os, time, urllib.request
m, rev = os.environ["MIX"], os.environ["MIX_REV"]
rec = {"model": m, "revision": rev, "floor_mb_s": int(os.environ["FLOOR"]), "status": None, "bytes": 0, "seconds": None,
       "mb_s": None, "shard": None, "error": None, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
t0 = None
try:
    meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{m}/revision/{rev}", timeout=30))
    rec["shard"] = sorted(s["rfilename"] for s in meta["siblings"] if s["rfilename"].endswith(".safetensors"))[0]
    t0 = time.time()
    with urllib.request.urlopen(f"https://huggingface.co/{m}/resolve/{rev}/{rec['shard']}", timeout=60) as r:
        rec["status"] = r.status
        while True:
            dt = time.time() - t0
            if (rec["bytes"] >= 256 << 20 and dt >= 20) or dt >= 60:
                break
            b = r.read(1 << 20)
            if not b:
                break
            rec["bytes"] += len(b)
except Exception as e:  # noqa: BLE001 -- a refusal reason, recorded by name
    rec["status"] = rec["status"] or getattr(e, "code", None)
    rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
finally:
    if t0 is not None:
        rec["seconds"] = round(time.time() - t0, 2)
        if rec["seconds"] and rec["error"] is None:
            rec["mb_s"] = round(rec["bytes"] / rec["seconds"] / 1e6, 1)
    json.dump(rec, open(os.environ["OUT"], "w"), indent=1)
print(f"status={rec['status']} bytes={rec['bytes']} seconds={rec['seconds']} mb_s={rec['mb_s'] if rec['mb_s'] is not None else 'none'}")
PYD
)
[ -s $W/download_probe.json ] || printf '{"status": null, "bytes": null, "mb_s": null, "error": "the probe wrote no record"}\n' > $W/download_probe.json
echo "DOWNLOAD ${DL:-none}" | tee -a summary.txt
DL_MB_S=${DL##*mb_s=}; DL_MB_S=${DL_MB_S%%.*}
case "$DL" in status=2[0-9][0-9]\ *) ;; *) DL_MB_S=0;; esac
[ "${DL_MB_S:-0}" -ge "$MIN_DL_MB_S" ] 2>/dev/null || { say "REFUSED: download ${DL:-none} -- below ${MIN_DL_MB_S} MB/s"; echo "refused: download ${DL:-none}" > REFUSAL; finish 19; }
# ---- the pre-install fit check (bench/common/README.md, "Guard fit"): Amendment 1's shakedown fit
FIT_HEADROOM_PCT=125
INSTALL_TO_GATE_S=$((INSTALL_S + FETCH_S + BAKE_S + CALIB_S))   # the setup still ahead of the first lifetime: 5,350 s
NEED_LIFETIMES=$((4 * LIFETIME_S + 2 * INSTR_S))                # four timed lifetimes and two instrument lifetimes: 4,500 s
FIT=$((INSTALL_TO_GATE_S + NEED_LIFETIMES + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((PGX_DEADLINE_EPOCH - $(date +%s)))
# rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
# ---- the start: every pre-install check has passed; from here the box counts toward the box-started cap
: > $W/STARTED; echo "STARTED $(date -u +%FT%TZ)" | tee -a summary.txt
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + RESERVE)); left=$((PGX_DEADLINE_EPOCH - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s, only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( PGX_DEADLINE_EPOCH - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 300 ] && left=300; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null 2>&1 || { say "no git in the image: apt-get install git"; perl -e 'alarm 600; exec @ARGV' sh -c "apt-get update -qq && apt-get install -y -qq git" > logs/apt_git.log 2>&1; }
command -v git >/dev/null 2>&1 || { tail -3 logs/apt_git.log; say "NO GIT"; finish 9; }
# ---- e4b (image python, torch held at the image's): the launch commit and the pinned grouped-nf4-gemm release
echo "torch==${TORCH_WANT%%+*}" > $W/constraints.txt
say "install e4b @$E4B_SHA + grouped-nf4-gemm==$GNF4_VERSION (image python; torch held at $TORCH_WANT)"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input -c $W/constraints.txt --prefer-binary \
  "experts4bit-qlora[serve] @ git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" "grouped-nf4-gemm==$GNF4_VERSION" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" accelerate safetensors "huggingface_hub>=0.23" "fastapi==0.141.1" "uvicorn==0.54.0" \
  > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
# grouped-nf4-gemm's calibrate.py and its C bench are not packaged: the release tag's tree, by its commit
perl -e 'alarm 600; exec @ARGV' git clone -q --filter=blob:none --no-checkout https://github.com/pjordanandrsn/grouped-nf4-gemm.git $W/gnf4-src > logs/gnf4_clone.log 2>&1 \
  && git -C $W/gnf4-src checkout -q $GNF4_TAG_SHA -- bench/calibrate.py bench/hybrid_calib.c >> logs/gnf4_clone.log 2>&1 \
  || { tail -3 logs/gnf4_clone.log; say "GNF4 SOURCE FAIL"; finish 9; }
(cd $W && PYTHONPATH= python $W/pgx_tripwire.py --kit $W --box "$E4B_SHA" "$TORCH_WANT") | tee logs/tripwire_e4b.log
[ "${PIPESTATUS[0]}" = 0 ] || { say "TRIPWIRE FAIL (e4b)"; finish 9; }
python - <<PYV >> versions.txt
import importlib.metadata as md, torch, transformers, experts4bit_qlora as e
print(f"e4b {e.__version__} @$E4B_SHA\ngnf4 {md.version('grouped-nf4-gemm')} (calibrate.py @$GNF4_TAG_SHA)\ntorch(e4b) {torch.__version__}\ntransformers(e4b) {transformers.__version__}")
PYV
# ---- SGLang 0.5.23 in its own venv (SC1's installer, unchanged, at this lane's pin)
say "install sglang==$SGLANG_VERSION (bench/sc1/sglang/install.sh)"
SGLANG_PIN=$SGLANG_VERSION SGLANG_TAG_COMMIT=$SGLANG_TAG_SHA bash $W/sglang_install.sh 2700 $W/versions.txt $W > logs/sglang_install.log 2>&1 \
  || { tail -6 logs/sglang_install.log; say "SGLANG INSTALL FAIL"; finish 9; }
grep -a "SGLANG_TRIPWIRE" logs/sglang_install.log | tee -a summary.txt
. $W/sglang.env
# the routing hook must load in the venv: it is imported through hook/sitecustomize.py (a venv never runs usercustomize)
PYTHONPATH=$W/hook $W/venv-sglang/bin/python -c 'import sys; assert "usercustomize" in sys.modules, "the PGX hook did not load in venv-sglang"' \
  || { say "HOOK FAIL: the routing hook does not load in venv-sglang"; finish 9; }
PYTHONPATH= $W/venv-sglang/bin/python - <<'PYS' > logs/tripwire_sglang.log 2>&1 || { cat logs/tripwire_sglang.log; say "TRIPWIRE FAIL (sglang paged experts)"; finish 9; }
import importlib
m = importlib.import_module("sglang.srt.layers.moe.paged_experts.method")
x = importlib.import_module("sglang.srt.layers.moe.paged_experts.executor")
assert hasattr(m, "PagedExpertsMoEMethod") and hasattr(m.PagedExpertsMoEMethod, "apply"), "PagedExpertsMoEMethod.apply"
assert hasattr(x, "plan_waves"), "executor.plan_waves"
from sglang.srt.arg_groups.fields import exec_ as f
src = open(f.__file__).read()
assert "enable_paged_experts" in src and "paged_experts_num_resident" in src, "the paged-experts flags"
print("tripwire OK (sglang paged experts)")
PYS
# ---- vLLM 0.29.0 + the pager at its tag commit, in their own venv
say "install vllm==$VLLM_VERSION + vllm-expert-pager @$PAGER_SHA"
python -m venv $W/venv-vllm > logs/venv_vllm.log 2>&1 || { say "VENV FAIL (vllm)"; finish 9; }
perl -e 'alarm 2700; exec @ARGV' $W/venv-vllm/bin/python -m pip install -q --no-input --prefer-binary "vllm==$VLLM_VERSION" \
  "vllm-expert-pager @ git+https://github.com/monochromegane/vllm-expert-pager.git@$PAGER_SHA" > logs/pip_vllm.log 2>&1 \
  || { tail -6 logs/pip_vllm.log; say "PIP FAIL (vllm / pager)"; finish 9; }
WANT_V=$VLLM_VERSION WANT_P=$PAGER_VERSION WANT_PSHA=$PAGER_SHA PYTHONPATH= $W/venv-vllm/bin/python - <<'PYV' > logs/tripwire_vllm.log 2>&1 || { cat logs/tripwire_vllm.log; say "TRIPWIRE FAIL (vllm / pager)"; finish 9; }
import importlib.metadata as md, json, os
import vllm, torch
assert vllm.__version__ == os.environ["WANT_V"], vllm.__version__
assert md.version("vllm-expert-pager") == os.environ["WANT_P"], md.version("vllm-expert-pager")
d = json.loads(md.distribution("vllm-expert-pager").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_PSHA"], d
eps = [e for e in md.entry_points(group="vllm.general_plugins") if e.name == "expert_pager"]
assert eps, "the expert_pager plugin entry point is missing"
import vllm_expert_pager  # noqa: F401
open("/root/pgx/versions.txt", "a").write(f"vllm {vllm.__version__}\nvllm-expert-pager {md.version('vllm-expert-pager')} @{os.environ['WANT_PSHA']}\ntorch(vllm) {torch.__version__}\n")
print("tripwire OK (vllm / pager)", vllm.__version__, torch.__version__)
PYV
cat versions.txt | tee -a summary.txt
# ---- the kit's own self-tests, on the box's interpreter
for t in "pgx_shake.py --self-test" "rt/pgx/pgx_box.py --self-test"; do
  (cd $W && python $W/$t) > logs/selftest_$(echo "$t" | cut -d' ' -f1 | tr '/.' '__').log 2>&1 || { say "SELF-TEST FAILED: $t"; finish 25; }
done
echo "self-tests ok" | tee -a summary.txt
# ---- fetch the three checkpoints at their revisions
can_run $FETCH_S fetch || finish 40
fetch(){ local tag=$1 m=$2 rev=$3
  say "fetch $m @ $rev"
  perl -e "alarm $(step_alarm 3600); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$m', revision='$rev', allow_patterns=['*.safetensors','*.json','tokenizer*','*.txt','merges.txt','vocab.json'], max_workers=8))" > logs/fetch_$tag.log 2>&1 \
    || { tail -2 logs/fetch_$tag.log; say "DL FAIL ($tag)"; finish 11; }
  local snap; snap=$(tail -1 logs/fetch_$tag.log); [ -d "$snap" ] || { say "DL FAIL ($tag): no snapshot dir"; finish 11; }
  echo "FETCH $tag $m@$rev $snap" | tee -a summary.txt; eval "SNAP_$tag=\$snap"; }
fetch bf16 $BF16 $BF16_REV; fetch fp8 $FP8 $FP8_REV; fetch int4 $INT4 $INT4_REV
# ---- the bytes per expert and the K table, measured from the box's own snapshots; the headers must be the registered ones
python $W/pgx_bytes.py --local "bf16=$SNAP_bf16" "fp8=$SNAP_fp8" "int4=$SNAP_int4" --out $W/box_bytes.json > logs/bytes.log 2>&1 || { tail -3 logs/bytes.log; say "BYTES FAIL"; finish 9; }
K_MEASURED=$(python - <<'PYK'
import json
reg, box = json.load(open("/root/pgx/bytes_per_expert.json")), json.load(open("/root/pgx/box_bytes.json"))
for fmt in ("bf16", "fp8", "int4"):
    assert box[fmt]["header_sha256"] == reg[fmt]["header_sha256"], f"{fmt}: the box's headers are not the registered checkpoint's"
k = box["k_table"]
print(f"{k['S4']['8']} {k['S8']['8']} {k['V8']['8']}")
PYK
) || { say "BYTES FAIL: the box's checkpoint headers differ from the registered ones"; finish 9; }
read -r KM_S4 KM_S8 KM_V8 <<< "$K_MEASURED"
echo "K_MEASURED S4=$KM_S4 S8=$KM_S8 V8=$KM_V8 (registered $K_S4/$K_S8/$K_V8; a different K replaces the table's, recorded)" | tee -a summary.txt
# ---- the NF4 arena (P39's k8_bake.py, as SC1 and P233 bake it)
can_run $BAKE_S bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP_bf16" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
# ---- calibrate (E-hyb's solver refuses to guess bandwidths) and the placement preview on THIS host's blob
can_run $CALIB_S calibrate || finish 40
say "calibrate (grouped-nf4-gemm bench/calibrate.py @$GNF4_TAG_SHA)"
perl -e "alarm 1500; exec @ARGV" python $W/gnf4-src/bench/calibrate.py --out $W/calib.json --nvme-dir $W --nvme-gib 8 --tag pgx > logs/calibrate.log 2>&1 \
  || { tail -5 logs/calibrate.log; say "CALIBRATE FAIL"; finish 33; }
rm -f $W/hybrid_calib_nvme.dat
tail -4 logs/calibrate.log | tee -a summary.txt
PYTHONPATH= python $W/pgx_place_preview.py --calib $W/calib.json --profile $W/fit_profile.jsonl --out $W/place_preview.json > logs/place_preview.log 2>&1 \
  || { tail -3 logs/place_preview.log; say "PLACEMENT PREVIEW FAIL"; finish 9; }
tail -4 logs/place_preview.log | tee -a summary.txt
# ---- the card-clock burn: recorded, never ruled
perl -e 'alarm 120; exec @ARGV' python $W/card_clock.py burn --floor 0 --seconds 20 --out $W/card_clock.json > logs/card_clock.log 2>&1 || true
tail -1 logs/card_clock.log | tee -a summary.txt
# ---- server helpers (SC2's: bench/sc2/sc2_box_e.sh)
wait_http(){ local url=$1 cap=$2 pid=$3 log=$4 t0 code; t0=$(date +%s)
  while :; do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$url" 2>/dev/null || true); [ "$code" = 200 ] && return 0
    kill -0 "$pid" 2>/dev/null || { say "server exited before $url answered: $(tail -2 "$log" | tr '\n' ' ' | cut -c1-240)"; return 45; }
    [ $(( $(date +%s) - t0 )) -ge "$cap" ] && { say "server startup timeout ${cap}s at $url"; return 44; }
    sleep 2
  done; }
wait_e4b_ready(){ local url=$1 cap=$2 pid=$3 log=$4 t0 st; t0=$(date +%s)
  while :; do
    st=$(curl -s -m 5 "$url" 2>/dev/null | grep -a -o -E '"status": ?"[a-z]+"' | head -1 | grep -a -o -E '[a-z]+"$' | tr -d '"')
    case "$st" in ready|busy) return 0;; error) say "e4b engine error: $(curl -s -m 5 "$url" | cut -c1-300)"; return 45;; esac
    kill -0 "$pid" 2>/dev/null || { say "server exited before $url reported ready: $(tail -2 "$log" | tr '\n' ' ' | cut -c1-240)"; return 45; }
    [ $(( $(date +%s) - t0 )) -ge "$cap" ] && { say "e4b not ready after ${cap}s (status=${st:-none})"; return 44; }
    sleep 3
  done; }
stop_pid(){ local pid=$1 i; [ -n "$pid" ] || return 0; kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
  for i in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ] && return 0; sleep 1; done
  say "GPU still held 60 s after the stop: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"; }
# one line a second: <epoch> <vram_mib> <rss_kib of the server's process group>
sampler_start(){ local out=$1 pgid=$2
  ( while :; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc 0-9) $(ps -o rss= -g "$pgid" 2>/dev/null | awk '{s+=$1} END {print s+0}')"; sleep 1; done ) > "$out" 2>/dev/null &
  echo $!; }
sampler_stop(){ kill "$1" 2>/dev/null; wait "$1" 2>/dev/null; }
drive(){ local port=$1 set=$2 pass=$3 out=$4 tag=$5
  perl -e "alarm $(step_alarm 900); exec @ARGV" python $W/pgx_drive.py --url "http://127.0.0.1:$port" --model pgx --prompts $W/prompts/$set.json \
    --concurrency 1 --pass-id "$pass" --out "$out" > logs/drive_${tag}_$pass.log 2>&1; }
# V8's eager probe: one 1,024-token prefill with one output token (longer than any captured size, so the MoE layers run
# eagerly and the pager logs its cumulative counters); returns the server log's byte offset once the lines are flushed
probe(){ local port=$1 log=$2
  python - "$port" <<'PYP' > /dev/null 2>&1
import json, sys, urllib.request
ids = json.load(open("/root/pgx/prompts/warm_c1.json"))[0]["token_ids"]
body = {"model": "pgx", "prompt": ids, "max_tokens": 1, "temperature": 0}
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=300).read()
PYP
  sleep 3; wc -c < "$log" | tr -d ' '; }
# ---- the servers: each sets SRV_PID (its own process group) or returns non-zero
sgl_start(){ local m=$1 rev=$2 k=$3 log=$4 extra=${5:-}
  # shellcheck disable=SC2086  # extra is a flag list by design
  setsid env PYTHONPATH=${SGL_HOOK_PATH:-} ${SGL_DUMP_ENV:-} $W/venv-sglang/bin/python -m sglang.launch_server --model-path "$m" --revision "$rev" \
    --host 127.0.0.1 --port $PORT_SGL --served-model-name pgx --enable-paged-experts --paged-experts-num-resident "$k" \
    --disable-radix-cache --context-length 2048 --max-running-requests 4 --random-seed 0 --log-level info $extra > "$log" 2>&1 < /dev/null &
  SRV_PID=$!; wait_http "http://127.0.0.1:$PORT_SGL/health" 2400 $SRV_PID "$log" || return $?
  curl -fsS -m 30 "http://127.0.0.1:$PORT_SGL/server_info" -o "$log.server_info.json" 2>/dev/null || true
  WANT_K=$k python - "$log.server_info.json" <<'PYE' || return 43
import json, os, sys
i = json.load(open(sys.argv[1]))
bad = [f"{k}={i.get(k)!r}" for k, want in (("version", "0.5.23"), ("enable_paged_experts", True),
       ("paged_experts_num_resident", int(os.environ["WANT_K"])), ("disable_radix_cache", True)) if i.get(k) != want]
print("SGLANG_ENGAGEMENT " + ("ok" if not bad else "REFUSED " + " ".join(bad)))
sys.exit(1 if bad else 0)
PYE
}
vllm_start(){ local k=$1 log=$2
  setsid env PYTHONPATH= VLLM_PLUGINS=expert_pager VLLM_EXPERT_PAGER_CACHE_SLOTS=$k VLLM_EXPERT_PAGER_LOG_INTERVAL=1 VLLM_LOGGING_LEVEL=INFO \
    $W/venv-vllm/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port $PORT_VLLM --model "$FP8" --revision "$FP8_REV" \
    --tokenizer-revision "$FP8_REV" --served-model-name pgx --max-model-len 2048 --max-num-seqs 4 --no-enable-prefix-caching --seed 0 \
    > "$log" 2>&1 < /dev/null &
  SRV_PID=$!; wait_http "http://127.0.0.1:$PORT_VLLM/v1/models" 2400 $SRV_PID "$log" || return $?
  grep -aq "vllm-expert-pager" "$log" || { say "V8: the pager logged nothing at startup (plugin not engaged?)"; return 43; }; }
e4b_start(){ local log=$1 lt=$2
  setsid env PYTHONPATH=$W/hook_e4b PGX_PLACEMENT_OUT=$lt/placement.json E4B_PAGED_MODEL=$BF16 E4B_PAGED_REVISION=$BF16_REV \
    E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_PLACEMENT=solver E4B_PAGED_VRAM_GB=$EHYB_VRAM_GIB \
    E4B_PAGED_DRAM_GB=$EHYB_DRAM_GIB E4B_PAGED_PROFILE=$W/fit_profile.jsonl E4B_PAGED_STEP_TRACE=$lt/step_trace.jsonl \
    E4B_PAGED_MAX_SEQS=4 E4B_PAGED_MAX_TOKENS_PER_SEQ=2048 E4B_PAGED_SERVED_NAME=pgx E4B_HOST=127.0.0.1 E4B_PORT=$PORT_E4B \
    python -m experts4bit_qlora.serve_paged > "$log" 2>&1 < /dev/null &
  SRV_PID=$!; wait_e4b_ready "http://127.0.0.1:$PORT_E4B/health" 2400 $SRV_PID "$log" || return $?; }
# ---- one lifetime: start, P1, P2 (V8: a probe before, between and after), stop; then lifetime.json
lifetime(){ local arm=$1 lt=$W/shake/$1/8/d1 t0 rc smp port log failed="" probes=""
  mkdir -p "$lt"; log=$W/logs/server_$arm.log; t0=$(date +%s)
  say "lifetime $arm (8 GB)"
  case "$arm" in
    S4) port=$PORT_SGL; sgl_start "$INT4" "$INT4_REV" "$KM_S4" "$log" "--dtype float16";;
    S8) port=$PORT_SGL; sgl_start "$FP8" "$FP8_REV" "$KM_S8" "$log";;
    V8) port=$PORT_VLLM; vllm_start "$KM_V8" "$log";;
    E-hyb) port=$PORT_E4B; e4b_start "$log" "$lt";;
  esac
  rc=$?
  if [ "$rc" != 0 ]; then failed="rc $rc: $(grep -aiE 'error|exception|refus' "$log" | tail -1 | cut -c1-200)"; say "$arm did not start ($failed)"
  else
    smp=$(sampler_start $lt/samples.txt "$SRV_PID")
    [ "$arm" = V8 ] && probes="$(probe $port "$log")"
    drive $port warm_c1 P1 $lt/P1.json "$arm"; echo "$arm P1 rc=$? $(python -c "import json; r=json.load(open('$lt/P1.json')); print('faults', r['faults'], 'tok/s', round(r['out_tok_s'] or 0, 1))" 2>/dev/null)" | tee -a summary.txt
    [ "$arm" = V8 ] && probes="$probes $(probe $port "$log")"
    drive $port warm_c1 P2 $lt/P2.json "$arm"; echo "$arm P2 rc=$? $(python -c "import json; r=json.load(open('$lt/P2.json')); print('faults', r['faults'], 'tok/s', round(r['out_tok_s'] or 0, 1))" 2>/dev/null)" | tee -a summary.txt
    [ "$arm" = V8 ] && probes="$probes $(probe $port "$log")"
    [ "$arm" = E-hyb ] && curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $lt/health.json 2>/dev/null
    sampler_stop "$smp"
  fi
  stop_pid "$SRV_PID"; SRV_PID=""
  local wall=$(( $(date +%s) - t0 ))
  local args=(lifetime --arm "$arm" --lt "$lt" --wall-s "$wall" --samples "$lt/samples.txt")
  [ -n "$failed" ] && args+=(--startup-failed "$failed")
  case "$arm" in
    S4) args+=(--k "$K_S4" --k-measured "$KM_S4" --route-dump "$W/instr/S4");;
    S8) args+=(--k "$K_S8" --k-measured "$KM_S8" --route-dump "$W/instr/S8");;
    V8) args+=(--k "$K_V8" --k-measured "$KM_V8" --pager-log "$log")
        # three offsets or none: a probe that failed leaves the counters unread (counters_ok false), never a guessed offset
        # shellcheck disable=SC2086  # probes is three numbers by design
        set -- $probes; [ $# = 3 ] && args+=(--probes "$1" "$2" "$3");;
    E-hyb) args+=(--health "$lt/health.json" --placement "$lt/placement.json" --step-trace "$lt/step_trace.jsonl");;
  esac
  python $W/rt/pgx/pgx_box.py "${args[@]}" 2>&1 | tee -a summary.txt; }
# ---- the untimed instrument lifetime for S4 / S8: graphs off, the routing dump on, the P1 prompts once (pass I1)
instrument(){ local arm=$1 m=$2 rev=$3 k=$4 extra=${5:-} d=$W/instr/$1 log=$W/logs/server_${1}_instr.log
  mkdir -p "$d"; echo I1 > "$d/PASS"
  say "instrument lifetime $arm (graphs off, routing dump)"
  SGL_HOOK_PATH=$W/hook SGL_DUMP_ENV="PGX_ROUTE_DUMP=$d" sgl_start "$m" "$rev" "$k" "$log" \
    "--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled $extra" \
    || { say "$arm instrument lifetime did not start"; stop_pid "$SRV_PID"; SRV_PID=""; return 1; }
  drive $PORT_SGL warm_c1 I1 $d/I1.json "${arm}_instr"
  stop_pid "$SRV_PID"; SRV_PID=""
  echo "$arm instrument: $(cat $d/route-*.jsonl 2>/dev/null | wc -l | tr -d ' ') dump rows" | tee -a summary.txt; }
# ---- draw 1's order (PREREG: draw d starts at arm d of S4, S8, V8, E-hyb), the instrument lifetimes before their arm's
can_run $INSTR_S instrument_S4 && instrument S4 "$INT4" "$INT4_REV" "$KM_S4" "--dtype float16"
can_run $LIFETIME_S lifetime_S4 && lifetime S4
can_run $INSTR_S instrument_S8 && instrument S8 "$FP8" "$FP8_REV" "$KM_S8"
can_run $LIFETIME_S lifetime_S8 && lifetime S8
can_run $LIFETIME_S lifetime_V8 && lifetime V8
can_run $LIFETIME_S lifetime_E-hyb && lifetime E-hyb
# ---- the verdict: SHAKE_OK (0), or STOP / SHAKE_FAULTS (27); never a reading
say "verdict"; python $W/pgx_shake.py --work $W/shake --out $W/shake_verdict.json 2>&1 | tee -a summary.txt
rc=${PIPESTATUS[0]}
case "$rc" in 0) finish 0;; 27) finish 27;; *) say "the verdict step failed (rc $rc)"; finish 27;; esac
