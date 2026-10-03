#!/bin/bash
# P106 rehearsal on the NAS A2000 ($0): (1) p106_run.sh's PROVING path under A2000 knobs; (2) p106_box.py on the real
# Qwen3.6-35B-A3B at reduced sizes, against P98's rehearsal arena (stand-in attention, solver placement); (3) the reducer
# on the box's record. Correctness of the path; its numbers are rehearsal numbers (sm_86, stand-in attention, solver).
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker; C=p106-reh; R=/share/Container/scripts/p106reh; ES=$1
L=$R/reh.log; : > $L
say(){ echo "[$(date -u +%FT%TZ)] $*" >> $L; }
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v /share/models:/models:ro -v /share/ZFS23_DATA/p98r-work:/work pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 21600 >/dev/null
U=$($D exec $C nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d " ")
APPS=$($D exec $C nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
say "preflight utilization ${U:-?} % compute apps ${APPS:-?}"
[ -n "$U" ] && [ "$U" -le 20 ] || { say "ABORT: GPU busy"; echo REH_DONE >> $L; $D rm -f $C >/dev/null 2>&1; exit 1; }
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; mkdir -p /root/p106" >> $L 2>&1
$D cp $R/stage/. $C:/root/p106/
say "(1) the proving path"
DL=$(( $(date +%s) + 20000 ))
$D exec -w /root/p106 $C bash -c "find . -name '._*' -delete; P106_RUN_NONCE=reh P106_RUN_ID=p106-reh P106_DEADLINE_EPOCH=$DL P106_INSTANCE_ID=0 E4B_SHA=$ES P106_PROVE=1 P106_GPU_CLASS=A2000 P106_MIN_DISK_GB=10 P106_PREMISE_ALLOW_SKIP=1 bash p106_run.sh > outer.log 2>&1; echo RUN_RC=\$?; cat summary.txt | cut -c1-400" >> $L 2>&1
grep -q "^RUN_RC=0" $L || { say "proving path failed; box not run"; mkdir -p $R/out; $D cp $C:/root/p106/. $R/out/ 2>/dev/null; $D rm -f $C >/dev/null 2>&1; echo REH_DONE >> $L; exit 1; }
say "(2) the box (reduced sizes)"
$D exec -w /root/p106 $C bash -c "E4B_PAGED_HOT_ROWS=256 python p106_box.py --model /models/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --arena /work/nf4.arena --calib calib.json --out /root/p106/box.json --stand-in-attention --placement solver --vram-gb 1.2 --windows 2 --null-windows 1 --prompt 1024 --cont 16 --mutant-cont 4 --ttft-lengths 512,1024 --ttft-rounds 2 > logs/box.log 2>&1; echo BOX_RC=\$?; grep -aE 'P106_|Error|error' logs/box.log | tail -12 | cut -c1-1200; tail -5 logs/box.log | cut -c1-300" >> $L 2>&1
say "(3) the reducer on the box's record"
$D exec -w /root/p106 $C bash -c "python p106_reduce.py --dir /root/p106 --out /root/p106/verdict.json 2>&1 | cut -c1-1500" >> $L 2>&1
mkdir -p $R/out && $D cp $C:/root/p106/. $R/out/ 2>/dev/null
$D rm -f $C >/dev/null 2>&1
say done; echo REH_DONE >> $L
