#!/bin/bash
# Interleaved paired rounds. Every arm is timed ONCE per round in a fixed order,
# and round 0 is dropped, so a machine that warms up or drifts moves all arms
# together instead of rewarding whichever happened to run first.
#
# host_self is the control: the SAME host configuration, timed a second time in
# the same round. Its spread IS the harness's resolution. Any ratio that lands
# inside it is unresolvable, not real -- 0.17.1 lost a published "1.03x" exactly
# this way, and the control is the only thing that catches it.
#
#   usage: rounds.sh <rounds> <out.jsonl> <arm-spec> [arm-spec ...]
#          arm-spec is  label:args...   e.g.  arena128:--arm arena --hot 128
set -u
R=${1:?rounds}; OUT=${2:?out.jsonl}; shift 2
V=/work/venv/bin/python
for r in $(seq 0 "$R"); do
  for spec in "$@"; do
    label="${spec%%:*}"; args="${spec#*:}"
    line=$($V /work/bin/timing.py $args --tag "round=$r,label=$label" 2>/dev/null | grep '^RESULT ' | cut -c8-)
    if [ -z "$line" ]; then
      echo "{\"ok\":false,\"round\":$r,\"label\":\"$label\"}" | tee -a "$OUT"
      continue
    fi
    python3 -c "
import json,sys
d=json.loads(sys.argv[1]); d['round']=int(sys.argv[2]); d['label']=sys.argv[3]
d['scored'] = d['round'] > 0          # round 0 is warmup and is NOT scored
print(json.dumps(d))" "$line" "$r" "$label" | tee -a "$OUT"
  done
done
echo "ROUNDS-DONE"
