#!/bin/bash
# bench/kv-pool-g/g3-gpu-check/g3check_drive.sh -- the G3 GPU check (CHECK.md), CONTROLLER side: the launcher's --command.
# Stages g3check_run.sh, g3check.py, bench/p39/k8_bake.py, bench/p39/calib.json and bench/common/card_clock.py, each
# checked against staged.sha256. The e4b commit under test is this checkout's HEAD, which must be clean, merged to
# origin/main (the approval names a merged main sha) and carry e4b#2087 (fc1f630a). Starts g3check_run.sh detached under
# a fresh nonce; polls TP_DONE.<nonce>; fetches receipts by rsync --no-links (bench/common rule 8; never the venv, the
# checkpoint or the arena). Nothing here creates, destroys or approves compute. Pattern: bench/kv-pool-g/g3-smoke.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [g3c_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); P39="$REPO/bench/p39"
STAGE="$HERE/g3check_run.sh $HERE/g3check.py $P39/k8_bake.py $P39/calib.json $REPO/bench/common/card_clock.py $HERE/staged.sha256"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
bash "$HERE/make_pin.sh" --check || { say "refusing: staged.sha256 does not match the sources"; exit 78; }
[ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: the checkout is dirty (the box installs HEAD from the remote)"; exit 78; }
E4B_SHA=$(git -C "$REPO" rev-parse HEAD) || { say "refusing: no HEAD"; exit 78; }
G3_FIX=fc1f630a9672a7e034d0aa2ec13bae4e24ed8648   # e4b#2087's merge: the shared pool's headroom hold-back
git -C "$REPO" cat-file -e "$G3_FIX^{commit}" 2>/dev/null || { say "refusing: e4b#2087's merge ($G3_FIX) is not in this checkout"; exit 78; }
git -C "$REPO" merge-base --is-ancestor "$G3_FIX" "$E4B_SHA" || { say "refusing: $E4B_SHA does not carry e4b#2087 ($G3_FIX)"; exit 78; }
git -C "$REPO" merge-base --is-ancestor "$E4B_SHA" origin/main 2>/dev/null || { say "refusing: $E4B_SHA is not on origin/main (the check runs at a merged main sha)"; exit 78; }
GNF4_SHA=${GNF4_SHA:-30adb1bfa76f95f9ce4a4c29734b8f36de907eca}   # grouped-nf4-gemm: main's CI pin (.github/workflows/ci.yml)
QWEN_REV=${QWEN_REV:-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}   # Qwen3-30B-A3B: SC1's pin (bench/sc1/sc1_run.sh REV)
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${G3C_POLL_S:-60}; W=/root/g3c
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="G3C_RUN_ID=$RUN_ID G3C_RUN_NONCE=$NONCE G3C_DEADLINE_EPOCH=$DEADLINE G3C_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA QWEN_REV=$QWEN_REV"
if [ "${G3C_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash g3check_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/g3c"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA; gnf4 $GNF4_SHA; qwen3 @$QWEN_REV; receipts -> $RUN_DIR/g3c; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W || exit 20; nohup env $PASS bash g3check_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat G3C_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; sleep 1; done; echo 'no nonce handshake in 30 s'; exit 21" || { say "check did not start"; exit 21; }
say "started; polling TP_DONE every ${POLL}s"; LAST=""
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  line=$($SSH "tail -n 1 $W/summary.txt 2>/dev/null" 2>/dev/null | cut -c1-200); [ -n "$line" ] && [ "$line" != "$LAST" ] && { say "box: $line"; LAST=$line; }
  sleep "$POLL"
done
rm -rf "$RUN_DIR/g3c" && mkdir -p "$RUN_DIR/g3c" || { say "fetch failed: local dir"; exit 22; }
rsync -az --no-links -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" --exclude venv --exclude work_qwen3 --exclude .cache --exclude __pycache__ "root@$HOST:$W/" "$RUN_DIR/g3c/" \
  || { say "fetch failed: rsync (no fallback: a tar stream would import links, bench/common rule 8)"; exit 22; }
[ -z "$(find "$RUN_DIR/g3c" -type l | head -1)" ] || { say "refusing: a symlink in the fetched artifacts (bench/common rule 8)"; exit 22; }
say "fetched $(ls "$RUN_DIR/g3c" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/g3c/G3C_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/g3c/TP_DONE.$NONCE" ] || { say "the check did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/g3c/G3C_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
grep -a 'G3CHECK\|CARD\|SERVER' "$RUN_DIR/g3c/summary.txt" 2>/dev/null | tail -8
# PASS needs BOTH rc 0 and this run's success marker; rc 0 without the marker is a malformed record (24), never a PASS
if [ ! -f "$RUN_DIR/g3c/G3C_SUCCESS.$NONCE" ] || [ "$RC" != 0 ]; then
  say "check exit rc=$RC, success marker $([ -f "$RUN_DIR/g3c/G3C_SUCCESS.$NONCE" ] && echo present || echo absent)"
  [ "$RC" = 0 ] && exit 24; exit "$RC"
fi
say "check complete rc=0 (PASS)"; exit 0
