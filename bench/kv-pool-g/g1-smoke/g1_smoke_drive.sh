#!/bin/bash
# bench/kv-pool-g/g1-smoke/g1_smoke_drive.sh -- lever G1's GPU smoke, CONTROLLER side: the launcher's --command.
# Stages g1_smoke_run.sh, g1_smoke.py, bench/p39/k8_bake.py and bench/p39/calib.json, each checked against staged.sha256;
# the e4b commit under test is this checkout's HEAD, which must be clean and on the remote (the box installs it from
# GitHub). Starts g1_smoke_run.sh detached under a fresh nonce; polls TP_DONE.<nonce>; fetches receipts (never the venv,
# the checkpoint or the arena). Nothing here creates, destroys or approves compute. Pattern: bench/sc2/sc1g-r/sc1g_r_drive.sh.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [g1s_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); P39="$REPO/bench/p39"
STAGE="$HERE/g1_smoke_run.sh $HERE/g1_smoke.py $P39/k8_bake.py $P39/calib.json $HERE/staged.sha256"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
bash "$HERE/make_pin.sh" --check || { say "refusing: staged.sha256 does not match the sources"; exit 78; }
[ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: the checkout is dirty (the box installs HEAD from the remote)"; exit 78; }
E4B_SHA=$(git -C "$REPO" rev-parse HEAD) || { say "refusing: no HEAD"; exit 78; }
git -C "$REPO" branch -r --contains "$E4B_SHA" 2>/dev/null | grep -q . || { say "refusing: $E4B_SHA is not on the remote"; exit 78; }
GNF4_SHA=${GNF4_SHA:-41ef086e6a3e2be5b158ff1bb38e0de67862a949}   # grouped-nf4-gemm v0.50.0 (boxes N, O and P)
GR_REV=${GR_REV:-a02780686e08a03fe0d2679a293b5c74a90efa89}       # granite-3.1-3b-a800m-instruct, P94's pin (SC1's proof model)
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${G1S_POLL_S:-60}; W=/root/g1s
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="G1S_RUN_ID=$RUN_ID G1S_RUN_NONCE=$NONCE G1S_DEADLINE_EPOCH=$DEADLINE G1S_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA GR_REV=$GR_REV"
if [ "${G1S_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash g1_smoke_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/g1s"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA; gnf4 $GNF4_SHA; granite @$GR_REV; receipts -> $RUN_DIR/g1s; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W || exit 20; nohup env $PASS bash g1_smoke_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat G1S_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; sleep 1; done; echo 'no nonce handshake in 30 s'; exit 21" || { say "lane did not start"; exit 21; }
say "started; polling TP_DONE every ${POLL}s"; LAST=""
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  line=$($SSH "tail -n 1 $W/summary.txt 2>/dev/null" 2>/dev/null | cut -c1-200); [ -n "$line" ] && [ "$line" != "$LAST" ] && { say "box: $line"; LAST=$line; }
  sleep "$POLL"
done
rm -rf "$RUN_DIR/g1s" && mkdir -p "$RUN_DIR/g1s" || { say "fetch failed: local dir"; exit 22; }
rsync -az --no-links -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" --exclude venv --exclude work_granite --exclude .cache --exclude __pycache__ "root@$HOST:$W/" "$RUN_DIR/g1s/" || {
  say "rsync fetch failed -- falling back to tar over the same ssh options"
  $SSH "tar -C $W --exclude=./venv --exclude=./work_granite --exclude=./.cache -cf - ." | tar -C "$RUN_DIR/g1s" -xf - || { say "fetch failed: rsync and tar"; exit 22; }; }
say "fetched $(ls "$RUN_DIR/g1s" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/g1s/G1S_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/g1s/TP_DONE.$NONCE" ] || { say "the smoke did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/g1s/G1S_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
grep -a 'G1SMOKE' "$RUN_DIR/g1s/summary.txt" 2>/dev/null | tail -8
[ -f "$RUN_DIR/g1s/G1S_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "smoke exit rc=$RC without success marker"; exit "$RC"; }
say "smoke complete rc=0 (PASS)"; exit 0
