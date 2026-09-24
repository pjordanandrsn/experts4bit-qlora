#!/bin/bash
# bench/p66/p66_drive.sh -- lane P66, CONTROLLER side: the launcher's --command (P66-PREREG.md). Stages the runner, the
# census and its reducer, the Level M shim, and the three repo tools they reuse unchanged (k8_bake.py from bench/p39,
# step_decomp.py from bench/hybrid-g9, step_budget.py from bench/hybrid-g9/f1), plus the HF token (never on a command
# line). Starts p66_run.sh detached under a fresh nonce, polls TP_DONE.<nonce> with a heartbeat (box summary tail,
# GPU util/mem, workdir size, free disk; a stall is REPORTED, never acted on), fetches receipts (never the
# checkpoints, arenas or venvs). Nothing here creates, destroys or approves compute.
# Pattern: bench/p61/p61_drive.sh + bench/tp4/tp4_drive.sh (token staging, lane-dead detection).
#   P66_MODE=prove   the PROVING rental (0.23 h guard): box checks, egress, install, tripwire, pin, one timed shard; it
#                    must end rc 0 with P66_PROVED.<nonce> before the reading rental (P66_MODE=full, the default) runs.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p66_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
STAGE="$HERE/p66_run.sh $HERE/p66_census.py $HERE/p66_reduce.py $HERE/p66_step.py $HERE/staged.sha256 $REPO/bench/p39/k8_bake.py $REPO/bench/hybrid-g9/step_decomp.py $REPO/bench/hybrid-g9/f1/step_budget.py"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them (flat in $W); resolve each name to its source and compare,
# the same mapping tests/test_p66_staged_pin.py applies in CI.
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    k8_bake.py) src="$REPO/bench/p39/$name";;
    step_decomp.py) src="$REPO/bench/hybrid-g9/$name";;
    step_budget.py) src="$REPO/bench/hybrid-g9/f1/$name";;
    *) src="$HERE/$name";;
  esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The e4b the BOX installs is the e4b this driver ships from (the launcher has proven the checkout is the manifest's
# heads.e4b). Derived, never a literal; refused if the tree is dirty (p39-box1b-3's lesson).
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
GNF4_SHA=${GNF4_SHA:?the lane measures a gnf4 cut; pass GNF4_SHA=<40-hex> from the manifest}
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; exit 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char hex sha ($GNF4_SHA)"; exit 78; }
MODE=${P66_MODE:-full}; case "$MODE" in full|prove) ;; *) say "refusing: P66_MODE must be full or prove ($MODE)"; exit 78;; esac
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${P66_POLL_S:-60}; STALL_S=${P66_STALL_S:-900}; W=/root/p66
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P66_RUN_ID=$RUN_ID P66_RUN_NONCE=$NONCE P66_DEADLINE_EPOCH=$DEADLINE P66_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA P66_MODE=$MODE"
if [ "${P66_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p66_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p66"; exit 0; fi
say "run $RUN_ID mode=$MODE nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA (from $REPO); gnf4 $GNF4_SHA; receipts -> $RUN_DIR/p66; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
if [ -s "$HF_TOKEN_FILE" ]; then   # authenticated pulls (unauthenticated shards throttle); the token never appears in a command line
  $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
  say "hf token staged"
else
  say "no hf token file at $HF_TOKEN_FILE -- pulls run unauthenticated (both checkpoints are ungated)"
fi
$SSH "cd $W || exit 20; nohup env $PASS bash p66_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P66_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
say "lane started; polling TP_DONE every ${POLL}s with a heartbeat (a stall is reported after ${STALL_S}s, never acted on)"
LAST=""; LAST_CHANGE=$(date +%s); LAST_LIVE=""; LANE_DEAD=0
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f 'bash p66_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null)
  line=${hb%% | gpu*}
  live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then LAST=$line; LAST_CHANGE=$now; fi
  # A lane that DIED is not a lane that is slow (e4b#634): two consecutive polls without the runner is a death.
  if [ "${live:-}" = 0 ] && [ "${LAST_LIVE:-}" = 0 ]; then
    say "LANE DEAD: no 'bash p66_run.sh' on the box for two consecutive polls and no TP_DONE"; LANE_DEAD=1; break
  fi
  LAST_LIVE=$live
  stall=""; [ $((now - LAST_CHANGE)) -ge "$STALL_S" ] && stall=" STALL? (no summary change for $((now - LAST_CHANGE))s -- LOOK, do not kill)"
  say "hb: ${hb:-<no answer>} | left $((DEADLINE - now))s$stall"
  sleep "$POLL"
done
rm -rf "$RUN_DIR/p66" && mkdir -p "$RUN_DIR/p66" || { say "fetch failed: local dir"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude src --exclude 'venv*' --exclude '.cache' \
  --exclude 'work_qwen3' --exclude '*.arena' --exclude '*.dat' "root@$HOST:$W/" "$RUN_DIR/p66/" || { say "fetch failed: rsync"; exit 22; }
# the arena INDEX is a receipt (geometry, row bytes); the arena itself is not
$SSH "cat $W/work_qwen3/nf4.arena.index.json" > "$RUN_DIR/p66/qwen3.nf4.arena.index.json" 2>/dev/null || true
$SSH "cat $W/work_qwen3/bake.json" > "$RUN_DIR/p66/qwen3.bake.json" 2>/dev/null || true
say "fetched $(ls "$RUN_DIR/p66" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p66/P66_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p66/TP_DONE.$NONCE" ] || {
  [ "$LANE_DEAD" = 1 ] && { say "lane DIED on the box: artifacts fetched, nothing complete"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p66/P66_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p66/P66_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
if [ "$MODE" = prove ]; then
  [ -f "$RUN_DIR/p66/P66_PROVED.$NONCE" ] || { say "prove run ended rc 0 WITHOUT P66_PROVED -- not a proving receipt"; exit 26; }
  say "PROVED: $(grep -a '^PROVE fetch' "$RUN_DIR/p66/summary.txt" | tail -1)"; exit 0
fi
say "lane complete rc=0"; exit 0
