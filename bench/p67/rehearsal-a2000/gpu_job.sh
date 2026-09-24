#!/bin/sh
# Run ONE P67 GPU job on the shared A2000 under the a2000.lock convention: wait (retry every 60 s, never break
# another holder's lock), hold it for exactly the docker run, remove it the moment the container exits.
#   gpu_job.sh <tag> <inner command...>        output: /share/Container/gnf4-interp/p67/logs/<tag>.log
LOCK=/share/Container/gnf4-interp/a2000.lock
W=/share/Container/gnf4-interp/p67
DOCKER=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
TAG=$1; shift
mkdir -p $W/logs
n=0
until mkdir "$LOCK" 2>/dev/null; do
  n=$((n + 1))
  echo "[$(date -u +%FT%TZ)] lock held ($(cat $LOCK/owner 2>/dev/null)); retry $n in 60 s" >> $W/logs/$TAG.lockwait
  [ $n -ge 120 ] && { echo "gave up after 120 min" >> $W/logs/$TAG.lockwait; exit 75; }
  sleep 60
done
echo "p67 $TAG $(date -u +%FT%TZ)" > "$LOCK/owner"
trap 'rm -f "$LOCK/owner"; rmdir "$LOCK"' EXIT INT TERM
echo "[$(date -u +%FT%TZ)] lock taken" > $W/logs/$TAG.log
$DOCKER run --rm --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all --cpus 6 \
  -e REH_TAG="$TAG" ${REH_ENV:-} \
  -v $W:/w -v /share/models:/models:ro \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel \
  bash -lc "$*" >> $W/logs/$TAG.log 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] container rc=$rc" >> $W/logs/$TAG.log
exit $rc
