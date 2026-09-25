#!/bin/sh
# locked_run.sh -- on the NAS host: wait for the shared A2000 lock (mkdir, retry every 60 s, never breaks someone
# else's lock), run the P70 rehearsal job in the devel image with the GPU, release the lock when the container exits.
W=/share/Container/gnf4-interp/p70
L=/share/Container/gnf4-interp/a2000.lock
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
LOG=$W/logs/locked_run.log
echo "[$(date -u +%FT%TZ)] waiting for $L" > $LOG
until mkdir $L 2>/dev/null; do sleep 60; done
echo "p70-rehearsal $(date -u +%FT%TZ)" > $L/owner 2>/dev/null
echo "[$(date -u +%FT%TZ)] lock taken" >> $LOG
$D run --rm --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all --shm-size 8g \
  -v $W:/w -v $W/runner:/root/p70 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel bash /w/job.sh >> $LOG 2>&1
RC=$?
rm -f $L/owner; rmdir $L
echo "[$(date -u +%FT%TZ)] docker rc=$RC; lock released" >> $LOG
echo "P70_REHEARSAL_DONE rc=$RC" >> $LOG
