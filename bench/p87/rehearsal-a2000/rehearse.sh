#!/bin/bash
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
C=p87-rehearse; R=/share/Container/scripts/k19t1
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 7200 >/dev/null
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; mkdir -p /root/p87"
$D cp $R/stage87/. $C:/root/p87/
$D exec -w /root/p87 $C bash -c "find . -name '._*' -delete; P87_RUN_ID=p87-rehearse-a2000 P87_RUN_NONCE=rehearsal P87_DEADLINE_EPOCH=\$((\$(date +%s)+7200)) P87_INSTANCE_ID=0 E4B_SHA=46a78404921f730a661fcd39b0aad7474a898c47 P87_PROVE=1 P87_GPU_CLASS=A2000 P87_MIN_DISK_GB=20 bash p87_run.sh > outer.log 2>&1; echo RUN_RC=\$?; cat summary.txt; echo ---; tail -5 outer.log; echo ---; tail -3 logs/rowexact.log; tail -3 logs/k19_contract.log"
mkdir -p $R/rehearsal87 && $D cp $C:/root/p87/. $R/rehearsal87/ 2>/dev/null
$D rm -f $C >/dev/null 2>&1
echo P87R_DONE
