#!/bin/sh
# Host side: free the A2000 (stop sdxl-sidecar), run the v5 A/B inside gpu-dev, then give the
# GPU back (start sdxl-sidecar) whatever the runner's exit status.
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
L=/share/gpu-dev/k3-rel-20260928/v5-det/host.log
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stop sdxl-sidecar" >> $L
$D stop sdxl-sidecar >> $L 2>&1
$D exec gpu-dev bash /workspace/k3-rel-20260928/v5-det/run_v5.sh >> $L 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] runner rc=$?; start sdxl-sidecar" >> $L
$D start sdxl-sidecar >> $L 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done" >> $L
