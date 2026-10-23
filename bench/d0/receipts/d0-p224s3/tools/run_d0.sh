#!/bin/bash
# D0 inside the d0-cpu container (PREREG-d0.md + Amendment 1): the plumbing gate, then per fixture (dq first) the unseen
# proof, the layer-major pass (Step 1 + Step 2) under the 8 h Step-2 cap, a Step-1 completion pass for any layer the cap
# cut, and the reduce. Records the pip freeze, free RAM and wall times beside the outputs.
#   bash run_d0.sh FIXTURE_DIR OUT_DIR
set -uo pipefail
F=${1:?fixture dir (fixture-dq, fixture-nodq, fixture-base, receipts_fixture-*)}; OUT=${2:?out dir}
T=/home/node/work/d0/tools; V=/home/node/work/d0/venv/bin/python; CK=/home/node/work/d0/olmoe-instruct-7f1c97f4
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"; cd "$T" || exit 2
stamp(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }
stamp "start; free: $(free -g | awk '/Mem:/{print $7" GB available"}'); nproc $(nproc)"
$V -m pip freeze > "$OUT/pip-freeze.txt"
for fx in dq nodq; do
  O="$OUT/$fx"; mkdir -p "$O"
  R=$(ls "$F"/receipts_fixture-$fx/*.json 2>/dev/null | head -1)
  [ -n "$R" ] && [ -d "$F/fixture-$fx" ] || { stamp "$fx: fixture absent, not run"; continue; }
  $V d0_data.py --receipt "$R" --tokenizer "$F/fixture-$fx" --out "$O/heldout.json" > "$O/data.log" 2>&1; dok=$?
  stamp "$fx: data rc $dok ($(tail -1 "$O/data.log"))"
  [ "$fx" = dq ] && [ ! -s "$OUT/gate-rows.json" ] && [ $dok = 0 ] && cp "$O/heldout.json" "$OUT/gate-rows.json"
  if [ ! -s "$OUT/gate.ok" ] && [ -s "$OUT/gate-rows.json" ]; then
    $V d0_gate.py --ckpt $CK --heldout "$OUT/gate-rows.json" > "$OUT/gate.log" 2>&1 && echo ok > "$OUT/gate.ok"
    stamp "gate: $(grep -a D0_GATE "$OUT/gate.log" | tail -1)"
  fi
  [ -s "$OUT/gate.ok" ] || { stamp "gate not passed: Step 2 VOID (the plumbing gate)"; dok=1; }
  DQ=""; [ $fx = dq ] && DQ=--dq
  HO=""; [ $dok = 0 ] && HO="--heldout $O/heldout.json"
  s=$(date +%s)
  timeout 8h $V d0_pass.py --ckpt $CK --fixture-base "$F/fixture-base" $DQ --adapter "$F/fixture-$fx" --out "$O/pass" $HO > "$O/pass.log" 2>&1
  prc=$?; stamp "$fx: pass rc $prc in $(( $(date +%s) - s )) s ($([ $prc = 124 ] && echo 'the 8 h cap: Step 2 INCOMPLETE' || tail -1 "$O/pass.log"))"
  $V d0_retention.py --ckpt $CK --fixture-base "$F/fixture-base" $DQ --adapter "$F/fixture-$fx" --out "$O/pass" > "$O/step1-complete.log" 2>&1
  stamp "$fx: step-1 completion rc $? ($(ls "$O"/pass/cells_L*.csv 2>/dev/null | wc -l) layers)"
  HOK=true; { [ $dok = 0 ] && [ $prc = 0 ]; } || HOK=false
  $V d0_reduce.py --pass-dir "$O/pass" --out "$O/summary.json" --heldout-ok $HOK > "$O/reduce.log" 2>&1
  stamp "$fx: reduce rc $? (heldout-ok $HOK)"
done
stamp "end"
