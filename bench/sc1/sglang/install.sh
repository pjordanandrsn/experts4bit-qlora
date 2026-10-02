#!/bin/bash
# bench/sc1/sglang/install.sh -- the SGLang comparator venv of lane SC1 (bench/sc1/SC1-PREREG.md, experts4bit-qlora#846).
#
#   install.sh <alarm_s> <versions_file> [work_dir]
#
# Creates <work_dir>/venv-sglang (python -m venv, NO system-site-packages: the e4b venv's torch must never leak in),
# pip-installs sglang==0.5.20 (tag v0.5.20 = commit 94602c9c2b7cbdb8efd5c52802dac6a1c180089e) under `perl -e alarm <alarm_s>`,
# points every SGLang cache under <work_dir>/sglang-cache, then runs the TRIPWIRE: import sglang, print version / torch
# version+cuda / flashinfer / sglang-kernel / compute capability / `nvcc --version` first line, and append ONE line to
# <versions_file>. Any failure exits non-zero with a clear LAST line (`SGLANG_INSTALL FAIL: ...`) -- the lane records
# the SGLang rows as UNSUPPORTED-in-this-container from it and moves on (the registered box-C contingency).
#
# Why the extra pieces (every fact read from the v0.5.20 source, python/sglang/...):
#   * nvcc at RUNTIME: the Marlin MoE GEMM that serves GPTQ/AWQ experts is JIT-compiled on first use
#     (kernels/ops/moe/moe_wna16_marlin.py:18-33 `load_jit("moe_wna16_marlin", ...)`); the JIT resolves nvcc from
#     CUDA_HOME / CUDA_PATH, then $PATH, then /usr/local/cuda (kernels/jit/utils/arch.py:41-69 `_jit_cuda_version`)
#     and picks the sm_120f target only when that nvcc is >= 12.9, else sm_120a (arch.py:72-90 `_cuda_arch_suffix`).
#   * `ninja` binary: the build runs `ninja -f build.ninja` as a subprocess (kernels/jit/utils/compile/ninja.py:155-166),
#     so the venv installs the `ninja` wheel and server.sh puts the venv's bin on PATH.
#   * caches: SGLANG_CACHE_DIR defaults to ~/.cache/sglang (srt/environ.py:1164) but the JIT build cache has its OWN
#     env, SGLANG_JIT_CACHE_DIR, whose unset default is ~/.cache/sglang/jit and does NOT track SGLANG_CACHE_DIR
#     (environ.py:1171; kernels/jit/utils/compile/cache.py:301-303 `cache_root`) -- both are exported, under the work dir,
#     so the Marlin leaf server.sh asserts on is where we look and survives container restarts.
set -uo pipefail
ALARM=${1:?usage: install.sh <alarm_s> <versions_file> [work_dir]}
VERSIONS=${2:?usage: install.sh <alarm_s> <versions_file> [work_dir]}
W=${3:-$PWD}
SGLANG_PIN=${SGLANG_PIN:-0.5.20}
SGLANG_TAG_COMMIT=94602c9c2b7cbdb8efd5c52802dac6a1c180089e
say(){ echo "[$(date -u +%FT%TZ)] sglang-install: $*"; }
fail(){ echo "SGLANG_INSTALL FAIL: $1"; exit "${2:-10}"; }
mkdir -p "$W/logs" "$W/sglang-cache/jit" || fail "cannot create $W" 78
VENV=$W/venv-sglang
PY=${SGLANG_BASE_PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || fail "no base python ($PY)" 78
"$PY" - <<'PYT' || fail "base python < 3.10 (sglang 0.5.20 requires-python >= 3.10)" 78
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
PYT

# ---- CUDA toolkit for the JIT (presence + version printed BEFORE pip spends the alarm)
CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}
NVCC=$CUDA_HOME/bin/nvcc
[ -x "$NVCC" ] || NVCC=$(command -v nvcc 2>/dev/null || true)
[ -n "$NVCC" ] && [ -x "$NVCC" ] || fail "nvcc not found (CUDA_HOME=$CUDA_HOME; the Marlin MoE JIT needs it at runtime)" 11
NVCC_LINE=$("$NVCC" --version 2>/dev/null | grep -m1 -i "release" || true)
say "nvcc: $NVCC :: ${NVCC_LINE:-<no release line>}"
CUDA_HOME=$(cd "$(dirname "$NVCC")/.." && pwd)
export CUDA_HOME

# ---- venv + pip under the alarm
say "venv $VENV (python -m venv, no system site packages)"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV" > "$W/logs/venv_sglang.log" 2>&1 || { tail -3 "$W/logs/venv_sglang.log"; fail "python -m venv" 9; }
say "pip install sglang==$SGLANG_PIN ninja (alarm ${ALARM}s; log $W/logs/pip_sglang.log)"
# shellcheck disable=SC2086
perl -e "alarm $ALARM; exec @ARGV" "$VENV/bin/python" -m pip install -q --no-input --prefer-binary --upgrade pip wheel > "$W/logs/pip_sglang.log" 2>&1
# shellcheck disable=SC2086
perl -e "alarm $ALARM; exec @ARGV" "$VENV/bin/python" -m pip install -q --no-input --prefer-binary "sglang==$SGLANG_PIN" ninja ${SGLANG_PIP_EXTRA_ARGS:-} >> "$W/logs/pip_sglang.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then tail -6 "$W/logs/pip_sglang.log"; [ $rc -eq 142 ] && fail "pip alarm (${ALARM}s)" 142; fail "pip (rc=$rc)" 9; fi

# ---- env file every later step sources (server.sh, one_batch.sh)
cat > "$W/sglang.env" <<EOF
export CUDA_HOME=$CUDA_HOME
export PATH=$VENV/bin:$CUDA_HOME/bin:\$PATH
export SGLANG_CACHE_DIR=$W/sglang-cache
export SGLANG_JIT_CACHE_DIR=$W/sglang-cache/jit
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
EOF
# shellcheck disable=SC1090
. "$W/sglang.env"

# ---- TRIPWIRE: imports + versions + capability + nvcc, one versions line, one JSON; no model is loaded, the GPU is only queried
say "tripwire"
SGLANG_PIN=$SGLANG_PIN SGLANG_TAG_COMMIT=$SGLANG_TAG_COMMIT NVCC=$NVCC VERSIONS=$VERSIONS W=$W "$VENV/bin/python" - <<'PYT'
import importlib.metadata as md, importlib.util, json, os, re, shutil, subprocess, sys
out = {"pin": os.environ["SGLANG_PIN"], "tag_commit": os.environ["SGLANG_TAG_COMMIT"], "python": sys.version.split()[0],
       "venv": sys.prefix, "errors": [], "warnings": []}
def dist(*names):
    for n in names:
        try: return md.version(n)
        except md.PackageNotFoundError: pass
    return None
try:
    import sglang
    from sglang.version import __version__ as sgl_v
    out["sglang"] = sgl_v
    if sgl_v != out["pin"]: out["errors"].append(f"sglang version {sgl_v} != pin {out['pin']}")
except Exception as e: out["errors"].append(f"import sglang: {e!r}")
try:
    import torch
    out["torch"] = torch.__version__; out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        cc = torch.cuda.get_device_capability(0); out["compute_capability"] = f"{cc[0]}.{cc[1]}"; out["cc_tuple"] = list(cc)
        out["device_name"] = torch.cuda.get_device_name(0)
    else: out["errors"].append("torch.cuda.is_available() is False (cu130 wheel vs the host driver? record the driver)")
except Exception as e: out["errors"].append(f"import torch: {e!r}")
try:
    import flashinfer; out["flashinfer"] = getattr(flashinfer, "__version__", None) or dist("flashinfer-python", "flashinfer_python")
except Exception as e: out["errors"].append(f"import flashinfer: {e!r}")
out["sglang_kernel"] = dist("sglang-kernel", "sgl-kernel", "sgl_kernel")
try:
    import sgl_kernel  # noqa: F401  (the AOT companion wheel; Marlin repack / scalar_types live here)
except Exception as e: out["errors"].append(f"import sgl_kernel: {e!r}")
try:
    import tvm_ffi  # noqa: F401  (the JIT loader: kernels/jit/utils/compile/loader.py:208-211 `load_module`)
    out["tvm_ffi"] = dist("apache-tvm-ffi")
except Exception as e: out["errors"].append(f"import tvm_ffi (apache-tvm-ffi, the JIT loader): {e!r}")
out["ninja"] = shutil.which("ninja")
if not out["ninja"]: out["errors"].append("ninja binary not on PATH (kernels/jit/utils/compile/ninja.py:155 runs `ninja -f build.ninja`)")
for mod in ("sglang.launch_server", "sglang.benchmark.one_batch", "sglang.srt.entrypoints.http_server"):
    try:
        if importlib.util.find_spec(mod) is None: out["errors"].append(f"module {mod} missing at this version")
    except Exception as e: out["errors"].append(f"find_spec({mod}): {e!r}")
try:
    import sglang.srt.server_args  # noqa: F401  (the arg_groups pipeline: msgspec etc. must import for the server to start)
except Exception as e: out["errors"].append(f"import sglang.srt.server_args: {e!r}")
nvcc = os.environ["NVCC"]
try:
    txt = subprocess.check_output([nvcc, "--version"], text=True, timeout=60)
    out["nvcc_first_line"] = txt.strip().splitlines()[0]
    m = re.search(r"release (\d+)\.(\d+)", txt)
    out["nvcc_release"] = f"{m.group(1)}.{m.group(2)}" if m else None
    out["nvcc_release_line"] = next((l.strip() for l in txt.splitlines() if "release" in l), None)
except Exception as e: out["errors"].append(f"nvcc --version: {e!r}")
# the JIT's own target decision for this card (kernels/jit/utils/arch.py:72-90): 12.0 + nvcc >= 12.9 -> sm_120f, else sm_120a
cc_t = out.get("cc_tuple"); rel = out.get("nvcc_release")
if cc_t and rel:
    maj, mi = cc_t; r = tuple(int(x) for x in rel.split("."))
    suf = "f" if (maj == 12 and mi == 0 and r >= (12, 9)) else ("a" if maj >= 9 else "")
    out["jit_target"] = f"sm_{maj}{mi}{suf}"
    out["jit_cache_target_dir"] = f"sm{maj}{mi}{suf}"   # cache.py:210-221 `_target_tag`
    if (maj, mi) != (12, 0): out["warnings"].append(f"compute capability {maj}.{mi} is not the registered RTX 5090 (12.0)")
    tc = out.get("torch_cuda") or ""
    if tc.split(".")[0] != str(r[0]): out["warnings"].append(f"torch built for CUDA {tc} but the JIT nvcc is {rel}: the box-A container question (cu130 wheels + 12.9 toolkit)")
out["cache_dir"] = os.environ.get("SGLANG_CACHE_DIR"); out["jit_cache_dir"] = os.environ.get("SGLANG_JIT_CACHE_DIR")
json.dump(out, open(os.path.join(os.environ["W"], "sglang_tripwire.json"), "w"), indent=1)
line = (f"sglang {out.get('sglang')} (tag {out['tag_commit'][:8]}) torch {out.get('torch')} cuda {out.get('torch_cuda')} "
        f"flashinfer {out.get('flashinfer')} sglang-kernel {out.get('sglang_kernel')} tvm-ffi {out.get('tvm_ffi')} "
        f"cc {out.get('compute_capability')} ({out.get('device_name')}) nvcc '{out.get('nvcc_release_line')}' "
        f"jit_target {out.get('jit_target')} python {out['python']} ninja {'yes' if out['ninja'] else 'NO'}")
with open(os.environ["VERSIONS"], "a") as f: f.write(line + "\n")
print("SGLANG_TRIPWIRE " + line)
for w in out["warnings"]: print("SGLANG_TRIPWIRE WARN " + w)
for e in out["errors"]: print("SGLANG_TRIPWIRE ERROR " + e)
sys.exit(1 if out["errors"] else 0)
PYT
rc=$?
[ $rc -eq 0 ] || fail "tripwire (see $W/sglang_tripwire.json)" 10
say "ok: $VENV ; env $W/sglang.env ; caches $W/sglang-cache"
echo "SGLANG_INSTALL OK"
