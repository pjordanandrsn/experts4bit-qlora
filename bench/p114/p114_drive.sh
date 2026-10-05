#!/bin/bash
# bench/p114/p114_drive.sh -- lane P114, CONTROLLER side: the launcher's --command (bench/p114/PREREG-p114.md). K28's driver
# (bench/k28/k28_drive.sh: heartbeat, stall report, dead-lane detection) by named substitution, staging P114's runner, gate,
# reducer and the two energy harnesses -- referenced from where they live, never copied -- and pinning every staged byte in
# staged.sha256. Starts the lane detached under a fresh nonce, polls TP_DONE.<nonce>, fetches receipts. No model, no hf
# token. Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p114_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
STAGE="$HERE/p114_run.sh $HERE/p114_gate.py $HERE/p114_reduce.py $REPO/bench/_upstream/bench_energy.py $REPO/bench/bench_energy_excluded.py $HERE/staged.sha256"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them (flat in $W); resolve each name to its source and compare hashes
# (the same check tests/test_p114_lane.py runs in CI, where it costs nothing).
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    p114_run.sh|p114_gate.py|p114_reduce.py) src="$HERE/$name";;
    bench_energy.py) src="$REPO/bench/_upstream/bench_energy.py";;
    bench_energy_excluded.py) src="$REPO/bench/bench_energy_excluded.py";;
    *) say "refusing: staged.sha256 names an unknown file: $name"; exit 78;;
  esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The e4b the BOX installs is the e4b this driver ships from (the launcher proved it is the manifest's heads.e4b): derived,
# never a literal, refused on a dirty tree (p39-box1b-3's lesson).
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${P114_POLL_S:-60}; W=/root/p114
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P114_RUN_ID=$RUN_ID P114_RUN_NONCE=$NONCE P114_DEADLINE_EPOCH=$DEADLINE P114_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA"
# No knob travels: a rental is the registered run. P114_REHEARSAL is set only on the rehearsal's own box.
if [ "${P114_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p114_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p114"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA (from $REPO); receipts -> $RUN_DIR/p114; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W || exit 20; nohup env $PASS bash p114_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P114_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
# ---- heartbeat + liveness (K28's): a stall is REPORTED, never acted on; two consecutive definite zeros for the lane's own
# process with no TP_DONE end the wait with their own code (25).
STALL_S=${P114_STALL_S:-600}
lane_dead() { [ "${1:-}" = "0" ] && [ "${2:-}" = "0" ] && echo dead; }
say "lane started; polling TP_DONE every ${POLL}s with a heartbeat (stall reported after ${STALL_S}s; never acted on)"
LAST=""; LAST_CHANGE=$(date +%s); LAST_LIVE=""; LANE_DEAD=0
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | live \$(pgrep -f 'bash p114_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p'); live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then say "LANE DEAD: no 'bash p114_run.sh' on the box for two consecutive polls and no TP_DONE"; LANE_DEAD=1; break; fi
  LAST_LIVE=$live
  stall=""; [ $((now - LAST_CHANGE)) -ge "$STALL_S" ] && [ "${util:-0}" -eq 0 ] 2>/dev/null && stall=" STALL? (no summary change for $((now - LAST_CHANGE))s, GPU idle -- LOOK, do not kill)"
  say "hb: ${hb:-<no answer>} | left $((DEADLINE - now))s$stall"
  sleep "$POLL"
done
rm -rf "$RUN_DIR/p114" && mkdir -p "$RUN_DIR/p114" || { say "fetch failed: local dir"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" --exclude '__pycache__' --exclude '.cache' "root@$HOST:$W/" "$RUN_DIR/p114/" \
  || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p114" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p114/P114_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p114/TP_DONE.$NONCE" ] || {
  [ "$LANE_DEAD" = 1 ] && { say "lane DIED on the box: artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p114/P114_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p114/P114_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
