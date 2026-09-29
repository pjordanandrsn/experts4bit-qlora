#!/bin/sh
# Host side, Jordan 2026-09-29 ~01:30Z: stop sdxl-sidecar now rather than wait for it to go idle.
# Only courtesy left: do not cut a burst in flight -- wait until no POST /generate in the last
# 90 s (at most 10 min), then stop it, run the three K3 processes, and start it again whatever
# the runner's status.
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
L=/share/gpu-dev/k3-rel-20260928/v6-rel0336/host.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> $L; }
log "immediate run requested; waiting out any burst in flight (max 10 min)"
i=0
while [ $i -lt 20 ] && [ "$($D logs --since 90s sdxl-sidecar 2>&1 | grep -c 'POST /generate')" -gt 0 ]; do sleep 30; i=$((i+1)); done
log "stop sdxl-sidecar"
$D stop sdxl-sidecar >> $L 2>&1
$D exec gpu-dev bash /workspace/k3-rel-20260928/v6-rel0336/run_v6.sh >> $L 2>&1
rc=$?
log "runner rc=$rc; start sdxl-sidecar"
$D start sdxl-sidecar >> $L 2>&1
log "done"
