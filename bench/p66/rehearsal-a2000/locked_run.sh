#!/bin/sh
# locked_run.sh <job> -- on the NAS host. Waits for the shared A2000 lock (mkdir, retry every 60 s, never
# breaks someone else's lock), runs /w/jobs/<job>.sh in the devel image with the GPU, releases the lock the
# moment the container exits (whatever its exit code). Log: /share/Container/gnf4-interp/p66/logs/<job>.log
J=$1
W=/share/Container/gnf4-interp/p66
L=/share/Container/gnf4-interp/a2000.lock
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
mkdir -p $W/logs
LOG=$W/logs/$J.log
echo "[$(date -u +%FT%TZ)] waiting for $L" > $LOG
until mkdir $L 2>/dev/null; do sleep 60; done
echo "[$(date -u +%FT%TZ)] lock taken by p66:$J" >> $LOG
echo "p66:$J $(date -u +%FT%TZ)" > $L/owner 2>/dev/null
$D run --rm --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all --shm-size 8g \
  -v $W:/w -v /share/models/OLMoE-1B-7B-0924:/models/OLMoE-1B-7B-0924:ro \
  -v /share/models/gpt-oss-20b:/models/gpt-oss-20b:ro -v $W/runner:/root/p66 \
  -e PYTHONPATH=/w/pydeps -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel bash /w/jobs/$J.sh >> $LOG 2>&1
RC=$?
rm -f $L/owner; rmdir $L
echo "[$(date -u +%FT%TZ)] job rc=$RC; lock released" >> $LOG
echo "P66_JOB_DONE rc=$RC" >> $LOG
