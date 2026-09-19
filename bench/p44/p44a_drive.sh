#!/bin/bash
# bench/p44/p44a_drive.sh -- lane P44-a, CONTROLLER side: the launcher's --command. Stages P39's harness pieces
# (step_decomp.py, k8_bake.py, calib.json), P42's hook, and this lane's serve_stack.py / expert_residuals.py /
# p44a_run.sh; starts the lane detached under a fresh nonce; polls TP_DONE.<nonce>; fetches receipts. Nothing here
# creates, destroys or approves compute. Pattern: bench/p42/p42_drive.sh.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p44a_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); P39="$REPO/bench/p39"
STAGE="$HERE/p44a_run.sh $HERE/serve_stack.py $HERE/expert_residuals.py $P39/step_decomp.py $P39/k8_bake.py $P39/calib.json $HERE/staged-a.sha256"
HOOK="$REPO/bench/p42/hook/usercustomize.py"   # P42's hook, referenced not copied (one file to keep in step)
for f in $STAGE $HOOK; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in p44a_run.sh|serve_stack.py|expert_residuals.py|hook/*) src="$REPO/bench/p42/$name";; *) src="$P39/$name";; esac
  got=$(sha_of "$src"); [ "$got" = "$want" ] || { say "refusing: $src is $got, staged-a.sha256 says $want"; exit 78; }
done < "$HERE/staged-a.sha256"
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex ($E4B_SHA)"; exit 78;; esac; [ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not 40 chars"; exit 78; }
GNF4_SHA=${GNF4_SHA:-24f8c9fb22673a77219b8645b3bb8be50aa158c4}   # grouped-nf4-gemm v0.31.0, the consumer CI pin
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${P44_POLL_S:-60}; W=/root/p44a
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P44A_RUN_ID=$RUN_ID P44A_RUN_NONCE=$NONCE P44A_DEADLINE_EPOCH=$DEADLINE P44A_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA P44_MIN_MBPS=${P44_MIN_MBPS:-20} "
if [ "${P44_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p44a_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p44a"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA gnf4 $GNF4_SHA (from $REPO); receipts -> $RUN_DIR/p44a; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/hook" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" && $SCP "$HOOK" "root@$HOST:$W/hook/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W || exit 20; nohup env $PASS bash p44a_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P44A_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; sleep 1; done; echo 'no nonce handshake in 30 s'; exit 21" || { say "lane did not start"; exit 21; }
say "lane started; polling TP_DONE every ${POLL}s"; LAST=""
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  line=$($SSH "tail -n 1 $W/summary.txt 2>/dev/null" 2>/dev/null | cut -c1-200); [ -n "$line" ] && [ "$line" != "$LAST" ] && { say "box: $line"; LAST=$line; }
  sleep "$POLL"
done
rm -rf "$RUN_DIR/p44a" && mkdir -p "$RUN_DIR/p44a" || { say "fetch failed: local dir"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude 'work_*' --exclude '.cache' --exclude '__pycache__' "root@$HOST:$W/" "$RUN_DIR/p44a/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p44a" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p44a/P44A_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p44a/TP_DONE.$NONCE" ] || { say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p44a/P44A_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p44a/P44A_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
