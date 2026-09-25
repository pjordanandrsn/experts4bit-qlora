#!/bin/sh
W=/share/Container/gnf4-interp/p70; L=/share/Container/gnf4-interp/a2000.lock; D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
LOG=$W/logs/diag.log
echo "[$(date -u +%FT%TZ)] waiting for $L" > $LOG
until mkdir $L 2>/dev/null; do sleep 60; done
echo "p70-diag $(date -u +%FT%TZ)" > $L/owner
$D run --rm --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -v $W:/w pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel bash /w/diag_job.sh >> $LOG 2>&1
RC=$?; rm -f $L/owner; rmdir $L
echo "P70_DIAG_DONE rc=$RC" >> $LOG
