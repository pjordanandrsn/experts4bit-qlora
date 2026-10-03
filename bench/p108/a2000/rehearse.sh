#!/bin/bash
# P108 rehearsal on the NAS A2000 ($0): (1) p108_run.sh's PROVING path under A2000 knobs; (2) p108_box.measure() on the
# GPU on a tiny Gemma-4 MoE whose window binds (box_rehearsal.py); (3) the reducer on that record. Correctness of the
# path; its numbers are rehearsal numbers (sm_86, stand-in attention, a random tiny model).
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker; C=p108-reh; R=/share/Container/scripts/p108reh; ES=$1
L=$R/reh.log; : > $L
say(){ echo "[$(date -u +%FT%TZ)] $*" >> $L; }
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 21600 >/dev/null
U=$($D exec $C nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d " ")
APPS=$($D exec $C nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
say "preflight utilization ${U:-?} % compute apps ${APPS:-?}"
[ -n "$U" ] && [ "$U" -le 20 ] || { say "ABORT: GPU busy"; echo REH_DONE >> $L; $D rm -f $C >/dev/null 2>&1; exit 1; }
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; mkdir -p /root/p108" >> $L 2>&1
$D cp $R/stage/. $C:/root/p108/
say "(1) the proving path"
DL=$(( $(date +%s) + 20000 ))
$D exec -w /root/p108 $C bash -c "find . -name '._*' -delete; P108_RUN_NONCE=reh P108_RUN_ID=p108-reh P108_DEADLINE_EPOCH=$DL P108_INSTANCE_ID=0 E4B_SHA=$ES P108_PROVE=1 P108_GPU_CLASS=A2000 P108_MIN_DISK_GB=10 P108_MIN_RAM_GB=8 P108_PREMISE_ALLOW_SKIP=1 bash p108_run.sh > outer.log 2>&1; echo RUN_RC=\$?; cat summary.txt | cut -c1-400" >> $L 2>&1
grep -q "^RUN_RC=0" $L || { say "proving path failed; box not run"; mkdir -p $R/out; $D cp $C:/root/p108/. $R/out/ 2>/dev/null; $D rm -f $C >/dev/null 2>&1; echo REH_DONE >> $L; exit 1; }
say "(2) the box: p108_box.measure() on the GPU on a tiny Gemma-4 MoE whose window binds (box_rehearsal.py)"
$D exec -w /root/p108 $C bash -c "mkdir -p reh && cp p108_box.py p108_reduce.py p97_box.py reh/ && cp box_rehearsal.py reh/a2000_box_rehearsal.py && python reh/a2000_box_rehearsal.py /root/p108/reh > logs/box.log 2>&1; echo BOX_RC=\$?; grep -aE 'P108_|Error|error|Traceback' logs/box.log | tail -14 | cut -c1-600" >> $L 2>&1
say "(3) the reducer on the box's record"
$D exec -w /root/p108 $C bash -c "cp summary.txt reh/ && python p108_reduce.py --dir /root/p108/reh --out /root/p108/reh/verdict.json 2>&1 | cut -c1-400" >> $L 2>&1
mkdir -p $R/out && $D cp $C:/root/p108/. $R/out/ 2>/dev/null
$D rm -f $C >/dev/null 2>&1
say done; echo REH_DONE >> $L
