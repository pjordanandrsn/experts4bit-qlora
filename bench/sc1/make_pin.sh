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
  echo "# sc1_reduce.py is pinned whenever it is present (it is staged by path; the box's sha256sum -c covers it)."
  for f in sc1_run.sh sc1_e4b_sched.py sc1_prompts.py sc1_sampler.sh; do echo "$(sha_of "$HERE/$f")  $f"; done
  [ -s "$HERE/sc1_reduce.py" ] && echo "$(sha_of "$HERE/sc1_reduce.py")  sc1_reduce.py"
  for f in step_decomp.py k8_bake.py calib.json; do echo "$(sha_of "$P39/$f")  $f"; done
  echo "$(sha_of "$P42/hook/usercustomize.py")  hook/usercustomize.py"
  echo "$(sha_of "$TESTS/test_k19_row_exact_gpu.py")  test_k19_row_exact_gpu.py"
  for f in sc1b_census.py sc1b_e4b_census.py sc1b_vllm_census.py sc1b_serve_census.py sc1b_toy.py kernel_classes.json sc1b_box_d.sh; do echo "$(sha_of "$REPO/bench/sc1b/$f")  $f"; done   # SC1b, staged flat on every box
  for f in sc2_driver.py sc2_prompts.py sc2_reduce.py sc2_box_e.sh sc2_identity.py sc2b_box_f.sh sc2b_reduce.py sc2_trace.py sc2g_box_g.sh sc2g_reduce.py sc1g_box_i.sh sc1g_reduce.py sc1g_k8.py sc1g_gemv_check.py sc1g_attn_check.py sc1g_kl.py sc2c_box_h.sh sc2c_reduce.py sc2c_census.py sc2d_box_k.sh sc2d_reduce.py sc2e_box_l.sh sc2e_reduce.py sc2e_census.py sc2e_basis.py; do echo "$(sha_of "$REPO/bench/sc2/$f")  $f"; done   # SC2 + SC2b + SC2g
  for f in sc5_box_m.sh sc5_driver.py sc5_quality.py sc5_e4b_quality.py sc5_ref.py sc5_windows.py sc5_reduce.py sc5_record.py sc5_windows_w64.json; do
    echo "$(sha_of "$REPO/bench/sc5/$f")  $f"; done   # SC5 (box M), staged flat on every box
  for f in vllm.lock.txt sglang.lock.txt e4b-wheels.lock; do echo "$(sha_of "$REPO/bench/sc5/locks/$f")  $f"; done   # SC5's provenance locks
  echo "$(sha_of "$REPO/bench/p117/p117_box.py")  p117_box.py"   # SC5: the decode-shaped quality pass imports it
  for f in sc5_ref.json sc5_ref_chunked.json; do [ -s "$REPO/bench/sc5/ref/$f" ] && echo "$(sha_of "$REPO/bench/sc5/ref/$f")  $f"; done   # once registered
  echo "$(sha_of "$REPO/bench/p98/p98_bake.py")  p98_bake.py"   # SC2d (box K): P98's Qwen3.6 arena bake, staged flat on every box
  for d in vllm sglang llamacpp exl3 lmdeploy sc1g_ref; do
    [ -d "$HERE/$d" ] || continue
    (cd "$HERE" && find "$d" -type f ! -name '*.pyc' ! -path '*/__pycache__/*' ! -name '.DS_Store' | LC_ALL=C sort) | while read -r rel; do
      echo "$(sha_of "$HERE/$rel")  $rel"
    done
  done
} > "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "pinned $(grep -vc '^#' "$OUT") files -> $OUT"
