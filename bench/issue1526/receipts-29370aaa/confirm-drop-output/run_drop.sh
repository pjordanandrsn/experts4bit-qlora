#!/bin/sh
# #1526 confirmation: T=1024 native, the output held (control, as recorded) and dropped (loss only). Same model and switches.
cd /work
. /work/venv/bin/activate
for mode in held drop; do
  flag=""; [ "$mode" = drop ] && flag="--drop-output"
  echo "== T=1024 native $mode $(date -u +%H:%M:%SZ)"
  python /work/probe_drop.py --model /work/q3red-L4 --tokens 1024 --adapters native $flag \
    --out "/work/out/L4-T1024-native-$mode.json" > "/work/out/L4-T1024-native-$mode.log" 2>&1
  echo "   rc=$? $(grep -o '"peak_allocated_bytes_per_step": \[[^]]*\]' /work/out/L4-T1024-native-$mode.log | head -1)"
done
