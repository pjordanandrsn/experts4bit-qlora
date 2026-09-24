#!/bin/sh
# gpu_job.sh <tag> <inner command...>  -- one job on the shared A2000 under a2000.lock
LOCK=/share/Container/gnf4-interp/a2000.lock
W=/share/Container/gnf4-interp/p68
DOCKER=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
TAG=$1; shift
mkdir -p $W/logs
n=0
until mkdir "$LOCK" 2>/dev/null; do
  n=$((n + 1)); [ $n -ge 60 ] && { echo "gave up" >> $W/logs/$TAG.lockwait; exit 75; }
  sleep 30
done
echo "p68 $TAG $(date -u +%FT%TZ)" > "$LOCK/owner"
trap 'rm -f "$LOCK/owner"; rmdir "$LOCK"' EXIT INT TERM
echo "[$(date -u +%FT%TZ)] lock taken" > $W/logs/$TAG.log
$DOCKER run --rm --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all --cpus 6 \
  -e PYTHONUNBUFFERED=1 -e PIP_CACHE_DIR=/w/pipcache -e HF_HUB_OFFLINE=1 \
  -v $W:/w -v /share/Container/gnf4-interp:/g:ro -v /share/models:/models:ro --shm-size 8g \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel \
  bash -lc "$*" >> $W/logs/$TAG.log 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] container rc=$rc" >> $W/logs/$TAG.log
exit $rc
