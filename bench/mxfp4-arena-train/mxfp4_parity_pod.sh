#!/bin/bash
# Runs ON the pod. Published wheels + this branch's overlay, then the MXFP4
# parity suite on a real card.
#
# No stage is `cmd | tail || fail`: a pipeline's status is its LAST command's,
# which is how a failed bake let two arms run against an arena that did not exist
# (PREREG amendment 1). Every step checks its own rc.
set -uo pipefail
W=/root/parity
cd "$W" || exit 90

say() { echo "[$(date -u +%FT%TZ)] $*"; }
finish() {   # finish <status> <note>
  ${PY:-python3} - "$1" "$2" <<'PYEOF' || true
import json, os, sys
r = {"status": sys.argv[1], "note": sys.argv[2]}
for f, k in (("env.json", "env"), ("probe.json", "probe")):
    if os.path.exists(f):
        try:
            r[k] = json.load(open(f))
        except Exception as e:
            r[k] = f"unparsable: {e}"
r["pytest_rc"] = int(open("pytest.rc").read().strip()) if os.path.exists("pytest.rc") else None
json.dump(r, open("result.json", "w"), indent=1)
print("RESULT", json.dumps(r)[:400])
PYEOF
  touch "$W/DONE"
  say "DONE written ($1)"
  exit 0                       # always 0: outcomes go in files, never exit codes
}

say "=== interpreter discovery ==="
# `python3` is NOT necessarily the interpreter that owns torch: a RunPod pytorch
# image ships it inside a venv, and a non-interactive `ssh host cmd` never sources
# the profile that puts that venv on PATH. Assuming `python3` cost one pod
# (2.5 min, ~$0.02) and reported "torch missing in the image", which was false —
# torch was there, under a python this script had not looked for. So: probe, and
# record what was probed, so a second failure is diagnosable without a third pod.
{
  echo "PATH=$PATH"
  echo "--- candidates ---"
  for c in python python3 /venv/bin/python /opt/conda/bin/python \
           /usr/local/bin/python3 /usr/bin/python3; do
    if [ -x "$c" ] || command -v "$c" > /dev/null 2>&1; then
      echo "$c -> $("$c" -c 'import sys;print(sys.version.split()[0], sys.prefix)' 2>&1 | head -1)"
      echo "   torch: $("$c" -c 'import torch;print(torch.__version__)' 2>&1 | head -1)"
    else
      echo "$c -> absent"
    fi
  done
} > interp.log 2>&1
cat interp.log

PY=""
for c in python /venv/bin/python /opt/conda/bin/python python3 /usr/local/bin/python3; do
  command -v "$c" > /dev/null 2>&1 || [ -x "$c" ] || continue
  if "$c" -c "import torch" > /dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || finish ENV_BROKEN "no interpreter on this image imports torch (see interp.log)"
say "using interpreter: $PY ($("$PY" -c 'import sys;print(sys.prefix)'))"

say "=== environment before install ==="
"$PY" -c "import torch,json;json.dump({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'gpu':(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),'interp':'$PY'},open('env_before.json','w'))"
[ $? -eq 0 ] || finish ENV_BROKEN "torch present but unusable under $PY"
cat env_before.json; echo

say "=== install published wheels ==="
"$PY" -m pip install -q --no-cache-dir "experts4bit-qlora[fast]" grouped-nf4-gemm bitsandbytes pytest > pip.log 2>&1
PIP_RC=$?
tail -5 pip.log
[ $PIP_RC -eq 0 ] || finish PIP_FAILED "pip exit $PIP_RC"

# The image's CUDA torch must SURVIVE the install. gnf4 pins torch permissively
# on purpose (a plain >=2.8 makes pip replace every dev/nightly CUDA build), but
# "should not move" is a claim to check, not to assume: a swapped torch turns a
# GPU parity run into a CPU one that still says "passed".
"$PY" - <<'PYEOF'
import json, torch, sys
before = json.load(open("env_before.json"))
env = {"torch_before": before["torch"], "torch_after": torch.__version__,
       "cuda": torch.cuda.is_available(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
try:
    import triton; env["triton"] = triton.__version__
except Exception as e:
    env["triton"] = f"MISSING: {e}"
import importlib.metadata as m
for p in ("experts4bit-qlora", "grouped-nf4-gemm"):
    try:
        env[p] = m.version(p)
    except Exception as e:
        env[p] = f"MISSING: {e}"
json.dump(env, open("env.json", "w"), indent=1)
print(json.dumps(env, indent=1))
sys.exit(0 if (env["cuda"] and env["torch_before"] == env["torch_after"]) else 7)
PYEOF
ENV_RC=$?
[ $ENV_RC -eq 0 ] || finish ENV_BROKEN "cuda unavailable or torch moved under the install (rc $ENV_RC)"

say "=== apply overlay ==="
rm -rf ov && mkdir ov && tar xzf e4b_overlay.tgz -C ov
[ $? -eq 0 ] || finish OVERLAY_FAILED "untar"
OUT="$W" PY="$PY" bash ov/apply_overlay.sh > overlay.log 2>&1
OV_RC=$?
cat overlay.log
[ $OV_RC -eq 0 ] || finish OVERLAY_FAILED "apply_overlay exit $OV_RC"

say "=== MEASURED parity, printed as numbers ==="
# A green tick says "inside tolerance". The receipt should say how far inside.
"$PY" probe_parity.py > probe.log 2>&1
PROBE_RC=$?
cat probe.log
[ $PROBE_RC -eq 0 ] || say "probe exited $PROBE_RC (recorded; the suite below is the verdict)"

say "=== the suite: same file that gates on CPU, now with the GPU tests live ==="
mkdir -p suite && cp test_mxfp4_arena_train.py suite/
"$PY" -m pytest suite/test_mxfp4_arena_train.py -q --no-header -p no:cacheprovider -rA > pytest.log 2>&1
echo $? > pytest.rc
tail -35 pytest.log
RC=$(cat pytest.rc)

if [ "$RC" = "0" ]; then finish PASSED "gpu parity suite green"; fi
finish FAILED "pytest exit $RC"
