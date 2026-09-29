# Sourced by run_v5.sh after the A/B, still inside the sdxl-off window, cwd v5-det.
# (a) the PR's GPU tests + a must-fail control (venv34: torch 2.8.0+cu128, triton 3.4.0, pytest),
# (b) the kernel-level receipt with the PR's own helper,
# (c) three K3 prefill processes with the PR's mxfp4_pipelined.py shadowing the installed one,
#     deterministic algorithms OFF: identical results here = the fix alone removes the drift.
P2=/workspace/k3-rel-20260928/v5-det
PYT=/workspace/venv34/bin/python
echo "[$(date -u +%FT%TZ)] phase2 start: gnf4 PR tests" >> status.log
( cd $P2/gnf4-pr/kernel && nice -n 10 $PYT -m pytest test_mxfp4_prefill_combine.py test_mxfp4_pipelined.py -v -p no:cacheprovider ) > $P2/gnf4_pr_tests.log 2>&1
echo "[$(date -u +%FT%TZ)] gnf4 PR tests rc=$?" >> status.log
( cd $P2/gnf4-pr/kernel && nice -n 10 $PYT -u $P2/mustfail.py ) > $P2/gnf4_pr_mustfail.log 2>&1
echo "[$(date -u +%FT%TZ)] mustfail rc=$?" >> status.log
( cd $P2 && nice -n 10 $PYT -u combine_pr.py $P2/gnf4-pr/kernel 6 90 512 ) > $P2/combine_pr.log 2>&1
echo "[$(date -u +%FT%TZ)] combine_pr rc=$?" >> status.log
for i in 1 2 3; do
  run_one ord$i K3_DET=0 K3_SHADOW=$P2/shadow
done
