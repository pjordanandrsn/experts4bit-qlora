#!/bin/sh
# Host side. sdxl-sidecar was serving bursts of image generations every ~12 min on the evening of
# 2026-09-28, so do not take the A2000 from it while it is in use: wait until it has served no
# POST /generate for 30 minutes (checked each minute, and again just before the stop), then stop
# it, run the three K3 processes, and start it again whatever the runner's status. Give up,
# touching nothing, if it is not idle by 10:00Z.
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
L=/share/gpu-dev/k3-rel-20260928/v6-rel0336/host.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> $L; }
busy() { n=$($D logs --since 30m sdxl-sidecar 2>&1 | grep -c "POST /generate"); [ "$n" -gt 0 ]; }
log "waiting for sdxl-sidecar to be idle 30 min"
while busy; do
  if [ "$(date -u +%H%M)" -ge 1000 ] && [ "$(date -u +%H%M)" -lt 1200 ]; then log "not idle by 10:00Z; giving up, nothing touched"; exit 0; fi
  sleep 60
done
busy && { log "busy again at the last check; giving up"; exit 0; }
log "idle 30 min; stop sdxl-sidecar"
$D stop sdxl-sidecar >> $L 2>&1
$D exec gpu-dev bash /workspace/k3-rel-20260928/v6-rel0336/run_v6.sh >> $L 2>&1
rc=$?
log "runner rc=$rc; start sdxl-sidecar"
$D start sdxl-sidecar >> $L 2>&1
log "done"
