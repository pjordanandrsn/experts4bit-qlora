#!/bin/bash
# bench/d0s/d0s_drive.sh -- lane D0s, CONTROLLER side: the launcher's --command (bench/d0s/PREREG-d0s.md). Stages the
# box runner and staged.sha256 (the kit itself runs from the box's clean checkout of the launch commit, checked against
# staged.sha256), and for the reading D0's two derived adapters; starts the lane detached under a fresh nonce; polls
# TP_DONE.<nonce> with a heartbeat; fetches the receipts with links refused (rule 8 of bench/common/README.md) -- never
# the checkpoint, the arenas or the adapters' weights. D0S_SHAKE=1 is the shakedown (no checkpoint, no adapters).
# Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [d0s_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
SHAKE=${D0S_SHAKE:-0}
STAGE="$HERE/d0s_run.sh $HERE/staged.sha256"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
# staged.sha256 names repository paths; every one must equal this tree's file (the box checks its checkout the same way)
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  [ -s "$REPO/$name" ] || { say "refusing: staged.sha256 names $name, which this tree lacks"; exit 78; }
  [ "$(sha_of "$REPO/$name")" = "$want" ] || { say "refusing: $name differs from staged.sha256"; exit 78; }
done < "$HERE/staged.sha256"
if [ "$SHAKE" != 1 ]; then
  ADAPTERS=${D0S_ADAPTERS:-}
  [ -n "$ADAPTERS" ] || { say "refusing: D0S_ADAPTERS (the directory holding fixture-nodq-v2 and fixture-dq-v2) is not set"; exit 78; }
  for fx in nodq dq; do
    d="$ADAPTERS/fixture-$fx-v2"
    [ -s "$d/adapter_manifest.json" ] && [ -s "$d/adapter.safetensors" ] || { say "refusing: $d is not a derived adapter"; exit 78; }
    [ -z "$(find "$d" -type l)" ] || { say "refusing: $d holds a link"; exit 78; }
    say "adapter fixture-$fx-v2: manifest $(sha_of "$d/adapter_manifest.json" | cut -c1-16) weights $(sha_of "$d/adapter.safetensors" | cut -c1-16)"
  done
fi
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${D0S_POLL_S:-60}; W=/root/d0s
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="D0S_RUN_ID=$RUN_ID D0S_RUN_NONCE=$NONCE D0S_DEADLINE_EPOCH=$DEADLINE D0S_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA D0S_SHAKE=$SHAKE"
if [ "${D0S_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash d0s_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/d0s"; exit 0; fi
say "run $RUN_ID nonce=$NONCE shake=$SHAKE -> $HOST:$PORT; launch e4b $E4B_SHA (from $REPO); receipts -> $RUN_DIR/d0s; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/adapters /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
if [ "$SHAKE" != 1 ]; then
  for fx in nodq dq; do $SCP -r "$ADAPTERS/fixture-$fx-v2" "root@$HOST:$W/adapters/" || { say "stage failed: adapter $fx"; exit 20; }; done
fi
if [ -s "$HF_TOKEN_FILE" ]; then   # the read-only token; it never appears on a command line
  $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
fi
$SSH "cd $W || exit 20; nohup env $PASS bash d0s_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat D0S_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
say "lane started; polling TP_DONE every ${POLL}s (a lane whose process is gone for two polls ends the wait)"
LAST=""; LAST_LIVE=""; LANE_DEAD=0
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | live \$(pgrep -f 'bash d0s_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null)
  line=${hb%% | gpu*}; live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; fi
  if [ "${live:-}" = "0" ] && [ "${LAST_LIVE:-}" = "0" ]; then
    say "LANE DEAD: no 'bash d0s_run.sh' on the box for two consecutive polls and no TP_DONE"; LANE_DEAD=1; break
  fi
  LAST_LIVE=$live
  say "hb: ${hb:-<no answer>} | left $((DEADLINE - now))s"
  sleep "$POLL"
done
rm -rf "$RUN_DIR/d0s" && mkdir -p "$RUN_DIR/d0s" || { say "fetch failed: local dir"; exit 22; }
# links are never imported (rule 8); the checkpoint (hf/), the checkout, the arenas' bytes and every adapter's weights
# stay on the box: the arenas' index, manifest and retention receipt travel, and so does every record and log
rsync -az --no-links -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" \
  --exclude 'hf/' --exclude 'e4bsrc/' --exclude 'adapters/' --exclude '*.arena' --exclude '*.safetensors' \
  --exclude 'model/' --exclude 'hf-cache/' --exclude '__pycache__' --exclude '.cache' \
  "root@$HOST:$W/" "$RUN_DIR/d0s/" || { say "fetch failed: rsync"; exit 22; }
[ -z "$(find "$RUN_DIR/d0s" -type l)" ] || { say "fetch refused: a link reached the run directory"; exit 22; }
say "fetched $(find "$RUN_DIR/d0s" -type f | wc -l | tr -d ' ') files"
[ "$(cat "$RUN_DIR/d0s/D0S_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/d0s/TP_DONE.$NONCE" ] || {
  [ "$LANE_DEAD" = 1 ] && { say "lane DIED on the box: artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/d0s/D0S_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/d0s/D0S_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
