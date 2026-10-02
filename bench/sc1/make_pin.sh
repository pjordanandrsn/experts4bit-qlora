#!/bin/bash
# bench/sc1/make_pin.sh -- regenerate bench/sc1/staged.sha256 from the staged sources, named as the BOX sees them after
# sc1_drive.sh stages them (flat in $W; the comparator drivers under their directories; P42's hook under hook/).
# The box runs `sha256sum -c staged.sha256`; the controller and tests/test_sc1_staged_pin.py resolve each name to its
# source by the same rule (sc1_drive.sh's `case`) and compare. Run it after editing any staged file; commit the result.
#
#   make_pin.sh [out=bench/sc1/staged.sha256]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
OUT=${1:-$HERE/staged.sha256}
P39="$REPO/bench/p39"; P42="$REPO/bench/p42"; TESTS="$REPO/tests"
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
{
  echo "# Names as the BOX sees them after staging (\$W; the comparator drivers under their directories; P42's hook under hook/)."
  echo "# The box runs \`sha256sum -c\` on this file and the controller (sc1_drive.sh's case) and tests/test_sc1_staged_pin.py"
  echo "# resolve each name to its source and compare. Regenerate with bench/sc1/make_pin.sh after editing any staged file."
  echo "# sc1_reduce.py: pinned at integration (the reducer is another agent's file; SC1_PIN_REDUCER=1 make_pin.sh adds its line then;"
  echo "# until then the driver stages it by path unpinned and the box's sha256sum -c does not cover it)."
  for f in sc1_run.sh sc1_e4b_sched.py sc1_prompts.py sc1_sampler.sh; do echo "$(sha_of "$HERE/$f")  $f"; done
  [ "${SC1_PIN_REDUCER:-0}" = 1 ] && [ -s "$HERE/sc1_reduce.py" ] && echo "$(sha_of "$HERE/sc1_reduce.py")  sc1_reduce.py"
  for f in step_decomp.py k8_bake.py calib.json; do echo "$(sha_of "$P39/$f")  $f"; done
  echo "$(sha_of "$P42/hook/usercustomize.py")  hook/usercustomize.py"
  echo "$(sha_of "$TESTS/test_k19_row_exact_gpu.py")  test_k19_row_exact_gpu.py"
  for d in vllm sglang llamacpp exl3 lmdeploy; do
    [ -d "$HERE/$d" ] || continue
    (cd "$HERE" && find "$d" -type f ! -name '*.pyc' ! -path '*/__pycache__/*' ! -name '.DS_Store' | LC_ALL=C sort) | while read -r rel; do
      echo "$(sha_of "$HERE/$rel")  $rel"
    done
  done
} > "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "pinned $(grep -vc '^#' "$OUT") files -> $OUT"
