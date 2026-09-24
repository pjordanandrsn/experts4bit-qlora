#!/bin/bash
# P66 rehearsal, round 1: plumbing on real CUDA (NOT a reading). /w = /share/Container/gnf4-interp/p66
set -u
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$FREE" -ge 8500 ] || { echo "ABORT: only ${FREE} MiB free on the shared A2000"; exit 75; }
OUT=/w/work/r1; rm -rf $OUT; mkdir -p $OUT/L
cd /w/work
echo "===== calibrate (quick)"
timeout 900 python /w/src/gnf4/bench/calibrate.py --out $OUT/calib.json --nvme-dir /w/work --nvme-gib 1 --quick > $OUT/calibrate.log 2>&1; echo "calibrate rc=$?"; tail -4 $OUT/calibrate.log
C="python /w/src/e4b/bench/p66/p66_census.py --calib $OUT/calib.json --out $OUT/L --warm 2 --tokens 4"
NF="--family olmoe-1b-7b --arena /w/work/olmoe/nf4.arena --snapshot /models/OLMoE-1B-7B-0924"
MX="--family gpt-oss-20b-L0-7 --arena /w/work/gptoss8.arena --snapshot /models/gpt-oss-20b"
run(){ local n=$1; shift; echo "===== $n"; timeout 1800 "$@" > $OUT/$n.log 2>&1; echo "$n rc=$?"; grep -E "^\[p66\]|Error|error|Traceback" $OUT/$n.log | tail -12; }
run L_ref   $C $NF --path ref
run L_pipe  $C $NF --path pipe --hot-fracs 1.0,0.5,0.0 --controlled 0,4,8
run L_hyb   $C $NF --path hyb --hot-fracs 0.5 --controlled 0,4,8 --capture-probe
run L_mref  $C $MX --path mref
run L_mpin  $C $MX --path mpin --hot-fracs 0.5,0.0 --controlled 0,2,4
run L_mnvme $C $MX --path mnvme --hot-fracs 1.0,0.5,0.0 --controlled 0,2,4 --capture-probe
echo "===== reduce"
python /w/src/e4b/bench/p66/p66_reduce.py $OUT/L --json $OUT/L/read.json 2>&1 | tail -60
python /w/src/e4b/bench/hybrid-g9/f1/step_budget.py $OUT/L/table_ref_uniform_eager.txt 2>&1 | head -12
ls -la $OUT/L | head -50
