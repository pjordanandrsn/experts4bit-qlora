#!/bin/bash
# P229's CONTROLLER side (bench/p229/PREREG-p229.md): stages the hash-pinned files to the rented box, starts
# p229_run.sh, polls its TP_DONE marker with a heartbeat line, and fetches the run directory back. It does not create,
# destroy, or approve compute: the launcher that calls it (pod-launch for a reading, a direct-API
# launcher for a shakedown) holds the provider key and the run's bus lines carry its instance id.
#   E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH
#   bash bench/p229/p229_drive.sh        (from a clean experts4bit-qlora checkout: the launch commit is its HEAD)
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p229_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set"; exit 78; }
done
case "$E4B_RENT_RUN_ID" in p229-5090-[1-3]|p229-shake-[1-3]) :;; *) say "refusing unregistered run ID: $E4B_RENT_RUN_ID"; exit 78;; esac
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# The launch commit is the e4b this driver ships from: derived from the clean checkout HEAD, never an input (pod-launch
# proves that HEAD is the manifest's heads.e4b; P226's form). An override is refused; so is a dirty tree.
if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi
E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { say "refusing: the checkout is dirty"; exit 78; }
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
for f in bench/p229/p229_bench.py bench/p229/p229_reduce.py bench/p229/p229_run.sh bench/p229/staged-p229.sha256; do
  cp "$REPO/$f" "$STAGE/" || { say "stage missing: $f"; exit 78; }
done
# the premise's files keep the repo's layout under t/ (the CPU suite reads bench/ files beside tests/)
for f in tests/test_int8_head_bw.py tests/test_int8_head_rows_interp.py tests/conftest.py bench/i8dense/i8_fakequant.py \
         bench/p148/p148_kernels.py; do
  mkdir -p "$STAGE/t/$(dirname "$f")" && cp "$REPO/$f" "$STAGE/t/$f" || { say "stage missing: $f"; exit 78; }
done
(cd "$STAGE" && shasum -a 256 -c staged-p229.sha256 > /dev/null) || { say "refusing: staged files differ from staged-p229.sha256"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
W=/root/p229
NONCE=$(python3 -c "import secrets; print(secrets.token_hex(8))")
if [ "${P229_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage $(find "$STAGE" -type f | wc -l | tr -d ' ') files -> root@$HOST:$W ; run p229_run.sh ; poll TP_DONE until $DEADLINE ; fetch -> $RUN_DIR/p229"; exit 0; fi
say "run $E4B_RENT_RUN_ID -> $HOST:$PORT; deadline $DEADLINE; e4b $E4B_SHA"
$SSH "test ! -e $W && mkdir -p $W" || { say "stage failed: remote dir exists or ssh refused"; exit 20; }
scp -q -r -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT "$STAGE"/* "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W && P229_RUN_ID=$E4B_RENT_RUN_ID P229_DEADLINE_EPOCH=$DEADLINE E4B_SHA=$E4B_SHA P229_RUN_NONCE=$NONCE \
  nohup bash p229_run.sh > outer.log 2>&1 < /dev/null & echo started" || { say "start failed"; exit 21; }
while :; do
  now=$(date +%s)
  $SSH "test -e $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - 240)) ] && { say "deadline near without TP_DONE: fetching what exists"; break; }
  hb=$($SSH "tail -n 1 $W/outer.log 2>/dev/null | cut -c1-160; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' '" 2>/dev/null | tr '\n' '|')
  say "hb: ${hb:-<no answer>} left $((DEADLINE - now))s"
  sleep 60
done
OUT="$RUN_DIR/p229"
mkdir -p "$OUT" && [ -z "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ] || { say "fetch failed: $OUT is not an empty directory"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" "root@$HOST:$W/" "$OUT/" || { say "fetch failed"; exit 22; }
say "fetched $(find "$OUT" -type f | wc -l | tr -d ' ') files into $OUT"
RC=$(cat "$OUT/P229_EXIT_CODE.$NONCE" 2>/dev/null || echo none)
say "box rc=$RC"
[ "$RC" = 0 ] && exit 0
exit 23
