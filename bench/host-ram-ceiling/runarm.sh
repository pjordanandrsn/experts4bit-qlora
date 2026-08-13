#!/bin/bash
# Run one arm under one host-RAM ceiling. Runs on the laptop, drives the QNAP
# through the `qnap` docker context; every path below is a QNAP path.
#
# The verdict comes from docker, not from the container's stdout: a process
# killed by the cgroup dies on SIGKILL and prints nothing, so "no RESULT line"
# has to be readable as a result. State.OOMKilled is the authoritative field.
#
# Free VRAM is sampled before and after because this is a shared production box
# -- SDXL and vLLM hold most of the A2000. A CUDA OOM is a contention artifact,
# not a host-RAM verdict, and the two must not be conflated.
set -uo pipefail
ARM=${1:?arm}; CAP=${2:?capGB or none}; shift 2
D="docker --context qnap"
WORK=/share/ZFS530_DATA/e4b-hostram
OUT=$WORK/out
NAME="e4b-${ARM}-${CAP}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="/tmp/e4b-${ARM}-${CAP}-${STAMP}.log"

$D rm -f "$NAME" >/dev/null 2>&1
MEM=()
[ "$CAP" != "none" ] && MEM=(--memory="$CAP" --memory-swap="$CAP")

vram () { $D run --rm --runtime=nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all \
  nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '\r' | tail -1; }

VRAM_BEFORE=$(vram)
T0=$(date +%s)
$D run --name "$NAME" --runtime=nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all \
  --cpus=6 "${MEM[@]}" \
  -v "$WORK":/work \
  -e HF_HOME=/work/hf -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e E4B_ARENA=/work/arena/olmoe-nf4.arena \
  e4b-bench:2.8.0-cu128 \
  /work/venv/bin/python /work/bin/cg.py --arm "$ARM" --tag "cap=${CAP}" "$@" \
  >"$LOG" 2>&1
RC=$?
T1=$(date +%s)
OOM=$($D inspect -f '{{.State.OOMKilled}}' "$NAME" 2>/dev/null)
EXITC=$($D inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null)
VRAM_AFTER=$(vram)
$D rm -f "$NAME" >/dev/null 2>&1

RESULT=$(grep -m1 '^RESULT ' "$LOG" | cut -c8-)
LASTMARK=$(grep '^MARK ' "$LOG" | tail -1 | cut -c6-)
[ -z "$RESULT" ] && RESULT=null
[ -z "$LASTMARK" ] && LASTMARK=null

# "completed" means the arm finished all steps, not merely that it exited.
VERDICT=FAILED
[ "$RC" = "0" ] && grep -q '"ok": true' "$LOG" && VERDICT=COMPLETED
[ "$OOM" = "true" ] && VERDICT=OOM_KILLED

python3 - "$ARM" "$CAP" "$RC" "$EXITC" "$OOM" "$VERDICT" "$((T1-T0))" \
        "$VRAM_BEFORE" "$VRAM_AFTER" "$STAMP" "$LOG" <<'PY' | tee -a /tmp/e4b-ladder.jsonl
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
rec=dict(arm=a[0],cap_gb=a[1],rc=int(a[2]),container_exit=a[3],oom_killed=a[4],
         verdict=a[5],wall_s=int(a[6]),vram_free_before_mib=a[7],vram_free_after_mib=a[8],
         stamp=a[9],log=log,last_mark=lastmark,result=res)
print(json.dumps(rec))
PY
echo "### $ARM cap=${CAP} -> $VERDICT (rc=$RC oom=$OOM) ${T1}s log=$LOG"
