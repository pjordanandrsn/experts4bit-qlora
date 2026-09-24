#!/bin/sh
# NAS side: take the shared A2000 lock (mkdir; wait 60 s while someone else holds it; never break it), run one
# rehearsal GPU job in the devel image, rmdir the lock the moment the job ends (any exit).
FAM=$1; MODELDIR=$2; NSEQ=$3; BUDGET=$4; LAYERS=$5
L=/share/Container/gnf4-interp/a2000.lock
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
until mkdir "$L" 2>/dev/null; do echo "[$(date -u +%FT%TZ)] lock held by another job; retry in 60 s"; sleep 60; done
echo "[$(date -u +%FT%TZ)] lock taken for p65 $FAM"
trap 'rmdir "$L" && echo "[$(date -u +%FT%TZ)] lock released"' EXIT INT TERM
$D run --rm --name "p65-$FAM" --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e BUDGET="$BUDGET" -e FIRST="${FIRST:-}" \
  -v /share/Container/gnf4-interp/p65:/w -v "$MODELDIR":/models/$FAM:ro \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel bash /w/gpu.sh "$FAM" "/models/$FAM" "$NSEQ" $LAYERS
rc=$?
echo "[$(date -u +%FT%TZ)] job rc=$rc"
