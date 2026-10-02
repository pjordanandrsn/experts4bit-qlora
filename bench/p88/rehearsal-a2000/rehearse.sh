#!/bin/bash
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
C=p88-rehearse; R=/share/Container/scripts/k19t1
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 7200 >/dev/null
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; mkdir -p /root/p88"
$D cp $R/stage88/. $C:/root/p88/
E=fe5cbaa9357075ef58fe53be42371ba931af3d88
echo "== run A: class knob only (expect rc 18 on this Intel host)"
$D exec -w /root/p88 $C bash -c "find . -name '._*' -delete; P88_RUN_ID=r P88_RUN_NONCE=a P88_DEADLINE_EPOCH=\$((\$(date +%s)+7200)) P88_INSTANCE_ID=0 E4B_SHA=$E P88_PROVE=1 P88_GPU_CLASS=A2000 P88_MIN_DISK_GB=20 bash p88_run.sh > outerA.log 2>&1; echo RUN_A_RC=\$?; grep -a 'REFUSED\|cpu_vendor' outerA.log forensics.txt | head -3; rm -f summary.txt REFUSAL"
echo "== run B: vendor knob (full proving path)"
$D exec -w /root/p88 $C bash -c "P88_RUN_ID=r P88_RUN_NONCE=b P88_DEADLINE_EPOCH=\$((\$(date +%s)+7200)) P88_INSTANCE_ID=0 E4B_SHA=$E P88_PROVE=1 P88_GPU_CLASS=A2000 P88_CPU_VENDOR=GenuineIntel P88_MIN_DISK_GB=20 bash p88_run.sh > outerB.log 2>&1; echo RUN_B_RC=\$?; cat summary.txt; tail -3 outerB.log"
mkdir -p $R/rehearsal88 && $D cp $C:/root/p88/. $R/rehearsal88/ 2>/dev/null
$D rm -f $C >/dev/null 2>&1
echo P88R_DONE
