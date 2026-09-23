#!/bin/bash
# bench/b374/b374_run.sh -- lane B374, BOX side. The prereg and the test live in grouped-nf4-gemm
# (kernel/PREREG-b374-word-boundary-gpu.md, kernel/test_offset_boundary_words_gpu.py); only the runner is here,
# because pod-launch.sh pins exactly `adertha` and `e4b` by design. Pattern: bench/k18/k18_run.sh.
#
# No model, no bench numbers: install gnf4 at GNF4_SHA, clone the SAME sha, prove the INSTALLED module is the
# pinned cut, then run the word-boundary file twice from two work dirs (the test puts its own directory first on
# sys.path, so the dir decides which nf4_grouped it imports):
#   shipped/     the test alone                            -> the installed nf4_grouped           P1: every case PASSES
#   unpromoted/  the test + nf4_grouped with SIX promotions -> the stripped copy                    P2: every case FAILS
#                removed (4x `eid = eid.to(tl.int64)`, 2x the dot-pad `tl.load(eids_ptr + g).to(tl.int64)`)
# Each GPU case runs in its OWN pytest process in both passes: an illegal access poisons the CUDA context, and a
# later case failing on the poisoned context would read as P2 for the wrong reason.
set -uo pipefail
W=/root/b374; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] b374: $*"; }
NONCE=${B374_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/B374_RUN_NONCE.tmp && mv $W/B374_RUN_NONCE.tmp $W/B374_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > B374_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > B374_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in B374_RUN_ID B374_DEADLINE_EPOCH B374_INSTANCE_ID GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; finish 78; }
: > summary.txt; echo "$B374_INSTANCE_ID" > INSTANCE_ID
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/b374/staged.sha256"; finish 9; }

python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *5090*) ;; *) say "REFUSED: card is '$GPU', the lane registers the RTX 5090 class"; echo "refused: class $GPU" > REFUSAL; finish 15;; esac
FREE=$(python -c "import torch; f, _ = torch.cuda.mem_get_info(); print(int(f / 2**20))")
echo "free MiB at start: $FREE" | tee -a forensics.txt
[ "$FREE" -ge 17920 ] || { say "REFUSED: $FREE MiB free, the cases need >= 17.5 GiB"; echo "refused: free $FREE MiB" > REFUSAL; finish 15; }

say "install gnf4 @$GNF4_SHA"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 \
  || { tail -3 logs/pip_gnf4.log; say "PIP FAIL"; finish 9; }
say "clone the same sha for the test file"
perl -e 'alarm 900; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git src > logs/clone.log 2>&1 \
  && git -C src checkout -q "$GNF4_SHA" || { tail -3 logs/clone.log; say "CLONE FAIL"; finish 9; }
GOT=$(git -C src rev-parse HEAD)
[ "$GOT" = "$GNF4_SHA" ] || { say "TRIPWIRE: clone is $GOT, pin is $GNF4_SHA"; finish 9; }
python -c "import pytest" 2>/dev/null || perl -e 'alarm 300; exec @ARGV' python -m pip install -q --no-input pytest > logs/pip_pytest.log 2>&1 \
  || { tail -3 logs/pip_pytest.log; say "PYTEST INSTALL FAIL"; finish 9; }
T=test_offset_boundary_words_gpu.py
mkdir -p shipped unpromoted && cp "src/kernel/$T" shipped/ && cp "src/kernel/$T" unpromoted/ || { say "the test file is missing at the pin"; finish 9; }

say "strip the six eid promotions from a copy of the INSTALLED nf4_grouped"
python - <<'PYT' || { say "UNPROMOTE FAIL"; finish 9; }
import difflib, importlib.metadata, inspect, os, re
import nf4_grouped
src = inspect.getsourcefile(nf4_grouped)
assert "/root/b374/src/" not in os.path.abspath(src), f"nf4_grouped resolved to the CLONE ({src})"
text = open(src).read()
stripped, n_to = re.subn(r"^(\s*)eid = eid\.to\(tl\.int64\)\s*$", r"\1eid = eid", text, flags=re.M)
stripped, n_ld = re.subn(r"^(\s*)eid = tl\.load\(eids_ptr \+ g\)\.to\(tl\.int64\)\s*$", r"\1eid = tl.load(eids_ptr + g)",
                         stripped, flags=re.M)
assert (n_to, n_ld) == (4, 2), f"expected 4 eid.to + 2 dot-pad load promotions, removed {n_to} + {n_ld}"
open("/root/b374/unpromoted/nf4_grouped.py", "w").write(stripped)
open("/root/b374/unpromote.diff", "w").writelines(difflib.unified_diff(
    text.splitlines(True), stripped.splitlines(True), "installed/nf4_grouped.py", "unpromoted/nf4_grouped.py"))
import torch, triton
open("/root/b374/versions.txt", "w").write(
    f"gnf4 {importlib.metadata.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\n"
    f"installed nf4_grouped {src}\nremoved promotions: {n_to} eid.to + {n_ld} dot-pad loads\n"
    f"torch {torch.__version__}\ntriton {triton.__version__}\n")
print("unpromoted copy written: 6 promotions removed")
PYT
# The tripwire: import nf4_grouped the way the test does (its own dir first on sys.path), in each dir.
for d in shipped unpromoted; do
  (cd $W/$d && python -c "import os, sys; sys.path.insert(0, os.getcwd()); import nf4_grouped; print('$d resolves nf4_grouped to', nf4_grouped.__file__)") >> versions.txt 2>&1 \
    || { say "TRIPWIRE FAIL ($d)"; finish 9; }
done
grep -q "^shipped resolves nf4_grouped to .*site-packages" versions.txt || { say "TRIPWIRE: shipped/ does not resolve to the installed package"; cat versions.txt; finish 9; }
grep -q "^unpromoted resolves nf4_grouped to /root/b374/unpromoted/nf4_grouped.py" versions.txt || { say "TRIPWIRE: unpromoted/ does not resolve to the stripped copy"; cat versions.txt; finish 9; }
cat versions.txt | tee -a summary.txt

left=$(( B374_DEADLINE_EPOCH - $(date +%s) - 300 )); [ "$left" -lt 600 ] && { say "STOP: no time left"; finish 30; }
IDS=$(cd $W/shipped && python -m pytest --collect-only -q -p no:cacheprovider $T 2>/dev/null | grep -E "::test_(wide|dotpad)_")
[ "$(echo "$IDS" | grep -c .)" = 4 ] || { say "expected 4 GPU cases, collected: $IDS"; finish 9; }
run_pass(){  # $1 = shipped | unpromoted ; one pytest process per case, and the pure geometry test once
  local d=$1 id rc; : > logs/$d.log
  (cd $W/$d && perl -e 'alarm 120; exec @ARGV' python -m pytest -q -rA -p no:cacheprovider "$T::test_the_geometry_straddles_the_word_boundary" >> $W/logs/$d.log 2>&1)
  echo "geometry rc=$?" >> logs/$d.verdicts
  for id in $IDS; do
    echo "===== $id" >> logs/$d.log
    (cd $W/$d && perl -e 'alarm 600; exec @ARGV' python -m pytest -q -rA -p no:cacheprovider "$id" >> $W/logs/$d.log 2>&1); rc=$?
    echo "$id rc=$rc" >> logs/$d.verdicts
  done
}
say "P1: the shipped kernels (4 GPU cases, each in its own process)"; : > logs/shipped.verdicts; run_pass shipped
say "P2: the unpromoted copy (4 GPU cases, each in its own process)"; : > logs/unpromoted.verdicts; run_pass unpromoted
cat logs/shipped.verdicts logs/unpromoted.verdicts | tee -a summary.txt

p1=ok; p2=ok
grep -q "^geometry rc=0$" logs/shipped.verdicts || p1=fail
while read -r id rc; do [ "$rc" = "rc=0" ] || p1=fail; done < <(grep "::" logs/shipped.verdicts)
grep -qi "skipped" logs/shipped.log && p1=fail
n_fail=0; while read -r id rc; do [ "$rc" = "rc=0" ] && p2=fail || n_fail=$((n_fail+1)); done < <(grep "::" logs/unpromoted.verdicts)
n_wrap=$(grep -c "the offset wrapped" logs/unpromoted.log); n_fault=$(grep -ci "illegal memory access\|an illegal instruction" logs/unpromoted.log)
{ echo "P1 (shipped: 4/4 GPU cases pass, geometry passes, nothing skipped): $p1"
  echo "P2 (unpromoted: every GPU case fails): $p2 -- $n_fail/4 failed; misread (decoy) lines $n_wrap; fault lines $n_fault"; } | tee -a summary.txt
[ "$p1" = ok ] || finish 31
[ "$p2" = ok ] || finish 32
finish 0
