#!/bin/bash
# Run one arm under one host-RAM ceiling.
#
# The verdict comes from docker, not from the container's stdout: a process
# killed by the cgroup dies on SIGKILL and prints nothing, so "no RESULT line"
# has to be readable as a result. State.OOMKilled is the authoritative field.
#
# Free VRAM is sampled before and after because the original box is a shared
# production NAS. A CUDA OOM is a contention artifact, not a host-RAM verdict,
# and the two must not be conflated.
#
#   usage: ./runarm.sh <host|arena> <cap|none> [extra args to cg.py]
#   e.g.   ./runarm.sh host 5120m
#
# Overridable; defaults are the box this was measured on. WORK is a path on the
# DOCKER HOST, which is not necessarily this machine.
set -uo pipefail
: "${DOCKER_CTX:=qnap}"                                   # "" for the local daemon
: "${WORK:=/share/ZFS530_DATA/e4b-hostram}"               # holds venv/, hf/, arena/, bin/
: "${IMAGE:=e4b-bench:2.8.0-cu128}"                       # built from ./Dockerfile
: "${GPU_RUNTIME:=nvidia-runtime}"                        # `docker info | grep Runtimes`
: "${LEDGER:=/tmp/e4b-ladder.jsonl}"
: "${CPUS:=6}"

ARM=${1:?arm: host|arena}; CAP=${2:?cap e.g. 5120m, or none}; shift 2
D="docker"; [ -n "$DOCKER_CTX" ] && D="docker --context $DOCKER_CTX"
NAME="e4b-${ARM}-${CAP}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/e4b-${ARM}-${CAP}-${STAMP}.log"

$D rm -f "$NAME" >/dev/null 2>&1
MEM=()
# --memory alone bounds RESIDENT memory only; the process then pages out to swap
# and survives, which is a different outcome from fitting. Setting --memory-swap
# to the same value is what makes this a ceiling.
[ "$CAP" != "none" ] && MEM=(--memory="$CAP" --memory-swap="$CAP")

vram () { $D run --rm --runtime="$GPU_RUNTIME" -e NVIDIA_VISIBLE_DEVICES=all \
  nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '\r' | tail -1; }

VRAM_BEFORE=$(vram)
T0=$(date +%s)
$D run --name "$NAME" --runtime="$GPU_RUNTIME" -e NVIDIA_VISIBLE_DEVICES=all \
  --cpus="$CPUS" "${MEM[@]}" \
  -v "$WORK":/work \
  -e HF_HOME=/work/hf -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e E4B_ARENA=/work/arena/olmoe-nf4.arena \
  "$IMAGE" \
  /work/venv/bin/python /work/bin/cg.py --arm "$ARM" --tag "cap=${CAP}" "$@" \
  >"$LOG" 2>&1
RC=$?
T1=$(date +%s)
OOM=$($D inspect -f '{{.State.OOMKilled}}' "$NAME" 2>/dev/null)
EXITC=$($D inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null)
VRAM_AFTER=$(vram)
$D rm -f "$NAME" >/dev/null 2>&1

# "completed" means the arm finished all steps, not merely that it exited.
VERDICT=FAILED
[ "$RC" = "0" ] && grep -q '"ok": true' "$LOG" && VERDICT=COMPLETED
[ "$OOM" = "true" ] && VERDICT=OOM_KILLED

python3 - "$ARM" "$CAP" "$RC" "$EXITC" "$OOM" "$VERDICT" "$((T1-T0))" \
        "$VRAM_BEFORE" "$VRAM_AFTER" "$STAMP" "$LOG" <<'PY' | tee -a "$LEDGER"
import json,sys
a=sys.argv[1:]
log=a[10]
res=lastmark=None
for line in open(log,errors="replace"):
    if line.startswith("RESULT "):
        try: res=json.loads(line[7:])
        except Exception: pass
    elif line.startswith("MARK "):
        try: lastmark=json.loads(line[5:])
        except Exception: pass
print(json.dumps(dict(arm=a[0],cap_gb=a[1],rc=int(a[2]),container_exit=a[3],oom_killed=a[4],
    verdict=a[5],wall_s=int(a[6]),vram_free_before_mib=a[7],vram_free_after_mib=a[8],
    stamp=a[9],log=log,last_mark=lastmark,result=res)))
PY
echo "### $ARM cap=${CAP} -> $VERDICT (rc=$RC oom=$OOM) log=$LOG"
