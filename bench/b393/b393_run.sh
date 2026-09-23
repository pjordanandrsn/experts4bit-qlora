#!/bin/bash
# bench/b393/b393_run.sh -- lane B393, BOX side. The prereg and the census live in grouped-nf4-gemm
# (kernel/PREREG-b393-combine-reduce-bitwise.md, kernel/b393_bitwise_census.py); only the runner is here,
# because pod-launch.sh pins exactly `adertha` and `e4b` by design. Pattern: bench/b374/b374_run.sh.
#
# No model, no bench numbers: install gnf4 at GNF4_SHA, clone the SAME sha for the census script (it is not
# in the wheel), prove the INSTALLED int4_b32 is the pinned cut and is what the census imports, then run the
# census once. The verdict is the JSON, read against the prereg; this script only says whether it ran.
set -uo pipefail
W=/root/b393; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] b393: $*"; }
NONCE=${B393_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/B393_RUN_NONCE.tmp && mv $W/B393_RUN_NONCE.tmp $W/B393_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > B393_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > B393_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in B393_RUN_ID B393_DEADLINE_EPOCH B393_INSTANCE_ID GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; finish 78; }
: > summary.txt; echo "$B393_INSTANCE_ID" > INSTANCE_ID
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/b393/staged.sha256"; finish 9; }

python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *5090*) ;; *) say "REFUSED: card is '$GPU', the lane registers the RTX 5090 class"; echo "refused: class $GPU" > REFUSAL; finish 15;; esac

say "install gnf4 @$GNF4_SHA"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 \
  || { tail -3 logs/pip_gnf4.log; say "PIP FAIL"; finish 9; }
say "clone the same sha for the census script"
perl -e 'alarm 900; exec @ARGV' git clone -q https://github.com/pjordanandrsn/grouped-nf4-gemm.git src > logs/clone.log 2>&1 \
  && git -C src checkout -q "$GNF4_SHA" || { tail -3 logs/clone.log; say "CLONE FAIL"; finish 9; }
GOT=$(git -C src rev-parse HEAD)
[ "$GOT" = "$GNF4_SHA" ] || { say "TRIPWIRE: clone is $GOT, pin is $GNF4_SHA"; finish 9; }
C=b393_bitwise_census.py
mkdir -p run && cp "src/kernel/$C" run/ || { say "the census script is missing at the pin"; finish 9; }
# The tripwire: the census puts its own directory first on sys.path, so from run/ (which holds only the
# census) int4_b32 must resolve to the installed package, never to the clone.
(cd $W/run && python -c "import os, sys; sys.path.insert(0, os.getcwd()); import int4_b32, triton, torch, importlib.metadata as m; print('int4_b32', int4_b32.__file__); print('gnf4', m.version('grouped-nf4-gemm'), '@$GNF4_SHA'); print('torch', torch.__version__); print('triton', triton.__version__)") > versions.txt 2>&1 \
  || { cat versions.txt; say "TRIPWIRE FAIL (import)"; finish 9; }
grep -q "^int4_b32 .*site-packages" versions.txt || { cat versions.txt; say "TRIPWIRE: run/ does not resolve int4_b32 to the installed package"; finish 9; }
grep -q "/root/b393/src/" versions.txt && { cat versions.txt; say "TRIPWIRE: resolved into the clone"; finish 9; }
cat versions.txt | tee -a summary.txt

left=$(( B393_DEADLINE_EPOCH - $(date +%s) - 300 )); [ "$left" -lt 300 ] && { say "STOP: no time left"; finish 30; }
say "census (414 cases)"
(cd $W/run && perl -e 'alarm 1800; exec @ARGV' python -u $C --out $W/census.json) > census.txt 2>&1; rc=$?
cat census.txt | tee -a summary.txt
[ "$rc" = 0 ] || { say "CENSUS rc=$rc"; finish $rc; }
python -c "import json; d=json.load(open('census.json')); assert len(d['cases'])==414, len(d['cases']); assert 'NOT a reading' not in d['device']['name']; print('census.json:', len(d['cases']), 'cases on', d['device']['name'])" | tee -a summary.txt \
  || { say "census.json is incomplete or not a GPU reading"; finish 9; }
finish 0
