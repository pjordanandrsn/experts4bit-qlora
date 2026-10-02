#!/usr/bin/env bash
# install.sh -- the vLLM venv for SC1's vLLM arms (lane SC1, experts4bit-qlora#846). P58's recipe
# (`bench/p58/p58_run.sh` vllm_install: `python -m venv` + `pip install "vllm==<ver>" "huggingface_hub>=0.23"` under a
# perl alarm, then an import tripwire) with the tripwire widened: vllm / torch / triton / flashinfer versions, the
# compute capability, the nvidia-smi driver, and a REFUSAL when the wheel's CUDA runtime needs a newer driver than the
# box has (PyPI vllm==0.30.0 is the CUDA 13.0 build -- torch 2.13.0+cu130 replaces the image's torch 2.8 INSIDE the venv,
# fine in its own venv, P58 did exactly this on the same image -- and CUDA 13 wheels need an R580+ driver:
# docs/getting_started/installation/gpu.cuda.inc.md:327 at the tag; the cu129 GitHub wheel needs R575+).
#
# usage: install.sh <alarm_seconds> <versions_out> [venv_dir=$PWD/venv-vllm]
# env:   SC1_VLLM_VERSION=0.30.0  SC1_VLLM_WHEEL=pypi|cu129  SC1_PIP_LOG=<file>  SC1_REQUIRE_CC=12.0 (optional refusal)
# rc:    0 ok | 1 venv/pip failed | 2 import tripwire failed | 18 driver below the wheel's floor (checked BEFORE pip) |
#        19 compute capability != SC1_REQUIRE_CC | 20 torch.cuda unavailable after install | 142 alarm
set -u
ALARM=${1:?alarm seconds}; VERS=${2:?versions file}; VENV=${3:-$PWD/venv-vllm}
VER=${SC1_VLLM_VERSION:-0.30.0}; WHEEL=${SC1_VLLM_WHEEL:-pypi}
PIPLOG=${SC1_PIP_LOG:-$(dirname "$VERS")/pip_vllm.log}
say(){ printf '[install-vllm %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
mkdir -p "$(dirname "$VERS")" "$(dirname "$PIPLOG")"

# 1. the driver floor, BEFORE a 2 GB download: a box that cannot run the wheel refuses in one second, with the reason.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
DRV_MAJOR=${DRV%%.*}
case "$WHEEL" in
  pypi)  FLOOR=580 ;;      # CUDA 13.0 build (torch 2.13.0+cu130)
  cu129) FLOOR=575 ;;      # vllm-<ver>+cu129 GitHub asset + torch cu129 index (CUDA 12.9 minimum driver)
  *) say "unknown SC1_VLLM_WHEEL=$WHEEL (pypi|cu129)"; exit 2 ;;
esac
if [ -z "$DRV_MAJOR" ] || ! [ "$DRV_MAJOR" -ge "$FLOOR" ] 2>/dev/null; then
  say "REFUSED (rc 18): driver '${DRV:-unreadable}' is below the R${FLOOR}+ floor for vllm==$VER ($WHEEL wheel). Options: SC1_VLLM_WHEEL=cu129 (R575+), or a box with a newer driver. Nothing was installed."
  printf 'vllm REFUSED driver=%s floor=R%s wheel=%s version=%s\n' "${DRV:-unreadable}" "$FLOOR" "$WHEEL" "$VER" >> "$VERS"
  exit 18
fi
say "driver $DRV >= R$FLOOR: installing vllm==$VER ($WHEEL) into $VENV (alarm ${ALARM}s, pip log $PIPLOG)"

# 2. venv + pip under the alarm (P58's exact shape; the perl alarm turns a hung index into rc 142)
python -m venv "$VENV" || { say "venv creation failed"; exit 1; }
case "$WHEEL" in
  pypi)
    perl -e "alarm $ALARM; exec @ARGV" "$VENV/bin/python" -m pip install -q --no-input --no-cache-dir \
      "vllm==$VER" "huggingface_hub>=0.23" > "$PIPLOG" 2>&1 ;;
  cu129)
    URL="https://github.com/vllm-project/vllm/releases/download/v$VER/vllm-$VER+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    perl -e "alarm $ALARM; exec @ARGV" "$VENV/bin/python" -m pip install -q --no-input --no-cache-dir \
      "$URL" "huggingface_hub>=0.23" --extra-index-url https://download.pytorch.org/whl/cu129 > "$PIPLOG" 2>&1 ;;
esac
rc=$?
if [ $rc -eq 142 ]; then say "ALARM: pip exceeded ${ALARM}s"; printf 'vllm ALARM pip %ss\n' "$ALARM" >> "$VERS"; exit 142; fi
if [ $rc -ne 0 ]; then say "PIP FAIL rc=$rc"; tail -6 "$PIPLOG"; printf 'vllm PIP_FAIL rc=%s\n' "$rc" >> "$VERS"; exit 1; fi

# 3. tripwire: the versions the receipts will quote, the device, and the two refusals that need the installed torch.
"$VENV/bin/python" - "$VERS" "${SC1_REQUIRE_CC:-}" "$WHEEL" <<'PY'
import subprocess, sys
vers_path, req_cc, wheel = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import vllm, torch
except Exception as e:  # noqa: BLE001
    print(f"TRIPWIRE IMPORT FAIL: {e!r}", flush=True)
    open(vers_path, "a").write(f"vllm IMPORT_FAIL {e!r}\n")
    sys.exit(2)

def _v(mod):
    try:
        m = __import__(mod)
        return str(getattr(m, "__version__", "present"))
    except Exception as e:  # noqa: BLE001
        return f"missing({type(e).__name__})"

tri, fi = _v("triton"), _v("flashinfer")
avail = bool(torch.cuda.is_available())
cap = tuple(torch.cuda.get_device_capability()) if avail else None
name = torch.cuda.get_device_name() if avail else None
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total,power.limit,clocks.max.sm",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
except Exception as e:  # noqa: BLE001
    smi = f"unreadable({type(e).__name__})"
lines = [f"vllm {vllm.__version__}", f"vllm_wheel {wheel}", f"torch(vllm) {torch.__version__}",
         f"torch.version.cuda(vllm) {torch.version.cuda}", f"triton(vllm) {tri}", f"flashinfer(vllm) {fi}",
         f"cuda_capability {cap}", f"gpu {name}", f"nvidia-smi {smi}", f"python(vllm) {sys.version.split()[0]}"]
open(vers_path, "a").write("\n".join(lines) + "\n")
print("tripwire (vllm): " + " | ".join(lines), flush=True)
if not avail:
    print("TRIPWIRE FAIL (rc 20): torch.cuda.is_available() is False after install -- the driver is too old for the "
          f"wheel's CUDA runtime ({torch.version.cuda}) or no GPU is visible", flush=True)
    sys.exit(20)
if req_cc:
    want = tuple(int(x) for x in req_cc.split("."))
    if cap != want:
        print(f"TRIPWIRE FAIL (rc 19): compute capability {cap} != required {want} (SC1_REQUIRE_CC)", flush=True)
        sys.exit(19)
print(f"tripwire OK (vllm): {vllm.__version__} torch {torch.__version__} cuda {torch.version.cuda} triton {tri} "
      f"flashinfer {fi} cc {cap} gpu {name}", flush=True)
PY
rc=$?
[ $rc -ne 0 ] && { say "TRIPWIRE rc=$rc -- NO vLLM COMPARATOR on this box"; exit $rc; }
say "vllm==$VER ready in $VENV"
exit 0
