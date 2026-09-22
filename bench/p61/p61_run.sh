#!/bin/bash
# bench/p61/p61_run.sh -- lane P61, BOX side (P61-PREREG.md). Pattern: bench/k18/k18_run.sh, minus the clone and the
# kernel contracts: P61 measures the SHIPPED gnf4 GEMV, so gnf4 is installed at GNF4_SHA and the bench
# (sweep_gemv.py) plus P60's recorded routing are staged from this repo and pinned by staged.sha256.
set -uo pipefail
W=/root/p61; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] p61: $*"; }
NONCE=${P61_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P61_RUN_NONCE.tmp && mv $W/P61_RUN_NONCE.tmp $W/P61_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P61_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P61_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P61_RUN_ID P61_DEADLINE_EPOCH P61_INSTANCE_ID GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; finish 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > summary.txt; echo "$P61_INSTANCE_ID" > INSTANCE_ID

python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *5090*) ;; *) say "REFUSED: card is '$GPU', the lane registers the RTX 5090 class"; echo "refused: class $GPU" > REFUSAL; finish 15;; esac
# Amendment 1 (P61-PREREG.md): run 1's card was power-capped at 450 W and its served arm read +11.9 % against the P0
# anchor (P60 600 W, K18 575 W). The read box must match the anchor's class: refuse below 575 W, before any install.
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
case "$PL" in ''|*[!0-9]*) say "REFUSED: unreadable power.limit '$PL'"; echo "refused: power.limit '$PL'" > REFUSAL; finish 15;; esac
[ "$PL" -ge 575 ] || { say "REFUSED: power.limit ${PL} W < 575 W (Amendment 1)"; echo "refused: power.limit ${PL} W" > REFUSAL; finish 15; }
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p61/staged.sha256"; finish 9; }

say "install gnf4 @$GNF4_SHA"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 \
  || { tail -3 logs/pip_gnf4.log; say "PIP FAIL"; finish 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import importlib.metadata, inspect, os
import int4_b32
got = os.path.abspath(inspect.getsourcefile(int4_b32))
assert "site-packages" in got, f"int4_b32 resolved to {got}, not the installed package"
for n in ("gemv_int4_b32", "_plan", "_sm_count", "quant_x_rows"):
    assert hasattr(int4_b32, n), f"installed gnf4 lacks {n}"
assert int4_b32.gemv_fused_reduce_default() is False, "the fused reduce is defaulted on: not the served GEMV"
import torch, triton
open("/root/p61/versions.txt", "w").write(
    f"gnf4 {importlib.metadata.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\n"
    f"int4_b32 from {got}\ntorch {torch.__version__}\ntriton {triton.__version__}\n")
print("tripwire OK: int4_b32 from", got)
PYT
cat versions.txt | tee -a summary.txt

left=$(( P61_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 600 ] && { say "STOP: no time left"; finish 30; }
[ "$left" -gt 2400 ] && left=2400
say "self-test on this card (each call reads its own layer's store; a captured cell replays into the eager bits)"
perl -e 'alarm 300; exec @ARGV' python $W/sweep_gemv.py --self-test > logs/self_test.log 2>&1; src=$?
tail -2 logs/self_test.log | tee -a summary.txt
[ "$src" = 0 ] || { say "SELF-TEST FAIL rc=$src -- no number is produced"; finish 21; }
say "sweep (alarm ${left}s): grid on per-layer stores + P60's recorded routing (served / dedup, per-layer and shared)"
perl -e "alarm $left; exec @ARGV" python $W/sweep_gemv.py --eids $W/eids_b16.int16.bin --meta $W/eids_b16.int16.json \
  --out $W/p61_rows.json > logs/sweep.log 2>&1
rc=$?
grep -E '^\[p61\]' logs/sweep.log | tail -8 | tee -a summary.txt
[ "$rc" = 0 ] && [ -s "$W/p61_rows.json" ] || { say "SWEEP rc=$rc (rows $( [ -s $W/p61_rows.json ] && echo present || echo ABSENT))"; finish 32; }
say "----- summary -----"; cat summary.txt
finish 0
