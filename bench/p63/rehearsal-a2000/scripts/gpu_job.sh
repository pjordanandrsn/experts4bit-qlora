#!/bin/sh
# Run ONE GPU job on the shared A2000 under the a2000.lock convention.
#   gpu_job.sh <tag> <inner command...>
# Waits for the lock (retry every 60 s, never breaks someone else's lock), holds it for exactly
# the docker run, and removes it the moment the container exits. Output: /share/Container/gnf4-interp/p63/logs/<tag>.log
LOCK=/share/Container/gnf4-interp/a2000.lock
W=/share/Container/gnf4-interp/p63
DOCKER=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
TAG=$1; shift
mkdir -p $W/logs
n=0
until mkdir "$LOCK" 2>/dev/null; do
  n=$((n + 1))
  echo "[$(date -u +%FT%TZ)] lock held by someone else ($(cat $LOCK/owner 2>/dev/null)); retry $n in 60 s" >> $W/logs/$TAG.lockwait
  [ $n -ge 90 ] && { echo "gave up after 90 min" >> $W/logs/$TAG.lockwait; exit 75; }
  sleep 60
done
echo "p63 $TAG $(date -u +%FT%TZ)" > "$LOCK/owner"
trap 'rm -f "$LOCK/owner"; rmdir "$LOCK"' EXIT INT TERM
echo "[$(date -u +%FT%TZ)] lock taken" > $W/logs/$TAG.log
$DOCKER run --rm --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all --cpus "${CPUS:-6}" \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e PIP_CACHE_DIR=/w/pipcache \
  -v $W:/w -v /share/models:/models:ro \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel \
  bash -lc "$*" >> $W/logs/$TAG.log 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] container rc=$rc" >> $W/logs/$TAG.log
exit $rc
