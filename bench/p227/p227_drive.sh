#!/bin/bash
# P227's CONTROLLER side (bench/p227/PREREG-p227.md): stages the hash-pinned files to the rented box, starts
# p227_run.sh, polls its TP_DONE marker with a heartbeat line, and fetches the run directory back. It does not create,
# destroy, or approve compute: the launcher that calls it (pod-launch for a reading, a direct-API
# launcher for a shakedown) holds the provider key and the run's bus lines carry its instance id.
#   E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH
#   bash bench/p227/p227_drive.sh        (from a clean experts4bit-qlora checkout: the launch commit is its HEAD)
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p227_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set"; exit 78; }
done
case "$E4B_RENT_RUN_ID" in p227-5090-[1-3]|p227-shake-[1-3]) :;; *) say "refusing unregistered run ID: $E4B_RENT_RUN_ID"; exit 78;; esac
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# The launch commit is the e4b this driver ships from: derived from the clean checkout HEAD, never an input (pod-launch
# proves that HEAD is the manifest's heads.e4b; P226's form). An override is refused; so is a dirty tree.
if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi
E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { say "refusing: the checkout is dirty"; exit 78; }
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
for f in bench/p227/p227_bench.py bench/p227/p227_reduce.py bench/p227/p227_run.sh bench/p227/staged-p227.sha256 \
         bench/p153/router-replay/rr.json.pt tests/test_nf4_smallm_bw_gpu.py tests/test_gemv_bw_served_gpu.py; do
  cp "$REPO/$f" "$STAGE/" || { say "stage missing: $f"; exit 78; }
done
(cd "$STAGE" && shasum -a 256 -c staged-p227.sha256 > /dev/null) || { say "refusing: staged files differ from staged-p227.sha256"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
W=/root/p227
NONCE=$(python3 -c "import secrets; print(secrets.token_hex(8))")
if [ "${P227_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage $(ls "$STAGE" | wc -l | tr -d ' ') files -> root@$HOST:$W ; run p227_run.sh ; poll TP_DONE until $DEADLINE ; fetch -> $RUN_DIR/p227"; exit 0; fi
say "run $E4B_RENT_RUN_ID -> $HOST:$PORT; deadline $DEADLINE; e4b $E4B_SHA"
$SSH "test ! -e $W && mkdir -p $W" || { say "stage failed: remote dir exists or ssh refused"; exit 20; }
scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT "$STAGE"/* "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W && P227_RUN_ID=$E4B_RENT_RUN_ID P227_DEADLINE_EPOCH=$DEADLINE E4B_SHA=$E4B_SHA P227_RUN_NONCE=$NONCE \
  nohup bash p227_run.sh > outer.log 2>&1 < /dev/null & echo started" || { say "start failed"; exit 21; }
while :; do
  now=$(date +%s)
  $SSH "test -e $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - 240)) ] && { say "deadline near without TP_DONE: fetching what exists"; break; }
  hb=$($SSH "tail -n 1 $W/outer.log 2>/dev/null | cut -c1-160; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' '" 2>/dev/null | tr '\n' '|')
  say "hb: ${hb:-<no answer>} left $((DEADLINE - now))s"
  sleep 60
done
OUT="$RUN_DIR/p227"
mkdir -p "$OUT" && [ -z "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ] || { say "fetch failed: $OUT is not an empty directory"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" --exclude 'rr.json.pt' "root@$HOST:$W/" "$OUT/" || { say "fetch failed"; exit 22; }
say "fetched $(find "$OUT" -type f | wc -l | tr -d ' ') files into $OUT"
RC=$(cat "$OUT/P227_EXIT_CODE.$NONCE" 2>/dev/null || echo none)
say "box rc=$RC"
[ "$RC" = 0 ] && exit 0
exit 23
