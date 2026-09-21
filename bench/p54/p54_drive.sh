#!/bin/bash
# bench/p54/p54_drive.sh -- lane P54, CONTROLLER side: the launcher's --command. Stages P39's harness pieces,
# P42's hook and this lane's p54_run.sh, starts the lane detached under a fresh nonce, polls TP_DONE.<nonce>,
# fetches receipts. One box, no arguments: the arms are fixed by bench/p54/P54-PREREG.md.
# Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p54_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# step_decomp.py / k8_bake.py / calib.json are P39's, referenced rather than copied: a
# 3,300-line duplicate is not evidence, it is a second thing to keep in step. staged.sha256
# pins them by path, so a drift in either lane refuses here.
P39="$REPO/bench/p39"
STAGE="$HERE/p54_run.sh $P39/step_decomp.py $P39/k8_bake.py $P39/calib.json $HERE/staged.sha256"
HOOK="$REPO/bench/p42/hook/usercustomize.py"
for f in $STAGE $HOOK; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them, so the check here resolves each
# name to its source and compares hashes rather than running `sha256sum -c` against paths
# that only exist after staging.
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in p54_run.sh) src="$HERE/$name";; hook/*) src="$REPO/bench/p42/$name";; *) src="$P39/$name";; esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The e4b the BOX installs must be the e4b this driver ships from -- the launcher has already
# proven that checkout is the manifest's heads.e4b (pod-launch.sh check_tree). A literal default
# here is exactly the unread constant the manifest design exists to remove, and it bit: p39-box1b-3
# installed ef6d532 (pre-#537) while the checkout was 3d22719, so the box rebuilt the same 8-layer
# assignment and VOIDed on its own 48-layer check. Derived, and refused if the tree is dirty.
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
GNF4_SHA=${GNF4_SHA:?set GNF4_SHA to the grouped-nf4-gemm cut the arms run on (>= 0.32.1 / the release commit the e4b CI pins)}
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${P54_POLL_S:-60}; W=/root/p54
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P54_RUN_ID=$RUN_ID P54_RUN_NONCE=$NONCE P54_DEADLINE_EPOCH=$DEADLINE P54_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA "
if [ "${P54_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p54_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p54"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA (from $REPO); receipts -> $RUN_DIR/p54; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/hook" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" && $SCP "$HOOK" "root@$HOST:$W/hook/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W || exit 20; nohup env $PASS bash p54_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P54_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
# ---- heartbeat + liveness (e4b#641 -- ported from tp4_drive.sh, #625/#634). A poller that waits for a
# marker cannot tell a dead lane from a slow one: tp4-c-parity-1's remote process was killed without writing
# its markers and the controller polled the corpse for 53 minutes. Each poll now asks the box (a) the last
# summary line, (b) GPU util, (c) workdir size and free disk, (d) whether `bash p54_run.sh` is still there.
# A stall is REPORTED, never acted on; two consecutive definite zeros on (d) with no TP_DONE end the wait
# with their own exit code (25), so a killed lane is not flattened into "ran out of clock".
STALL_S=${P54_STALL_S:-900}; P54_MIN_PROGRESS_MB=${P54_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$P54_MIN_PROGRESS_MB
  consumed=0; grew=0
  if [ -n "$dfk_now" ] && [ -n "$dfk_prev" ] && [ "$dfk_now" -lt "$dfk_prev" ] 2>/dev/null; then consumed=$(( (dfk_prev - dfk_now) / 1024 )); fi
  if [ -n "$du_now" ] && [ -n "$du_prev" ] && [ "$du_now" -gt "$du_prev" ] 2>/dev/null; then grew=$((du_now - du_prev)); fi
  if [ "$consumed" -ge "$min_mb" ] || [ "$grew" -ge "$min_mb" ]; then
    if [ "$consumed" -ge "$grew" ]; then echo "fetching:-${consumed}M on disk"; else echo "fetching:+${grew}M in tree"; fi
    return 0
  fi
  if [ "$idle_s" -ge "$stall_s" ] && [ "${util:-0}" -eq 0 ] 2>/dev/null; then echo "stall:$idle_s"; fi
}
lane_dead() { [ "${1:-}" = "0" ] && [ "${2:-}" = "0" ] && echo dead; }   # live_now live_prev: two DEFINITE zeros
say "lane started; polling TP_DONE every ${POLL}s with a heartbeat (stall reported after ${STALL_S}s of no change, idle GPU AND no disk movement; never acted on; a lane whose process is gone for two polls ends the wait)"
LAST=""; LAST_CHANGE=$(date +%s); LAST_DFK=""; LAST_DU=""; LAST_LIVE=""; LANE_DEAD=0
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f 'bash p54_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: no 'bash p54_run.sh' on the box for two consecutive polls and no TP_DONE -- the remote process exited without writing its markers; not waiting out the deadline"
    LANE_DEAD=1; break
  fi
  LAST_LIVE=$live
  verdict=$(progress_verdict "$((now - LAST_CHANGE))" "$STALL_S" "${util:-0}" "$dfk" "$LAST_DFK" "$duM" "$LAST_DU")
  LAST_DFK=$dfk; LAST_DU=$duM
  case "$verdict" in
    fetching:*) stall=" fetching (${verdict#fetching:} since last hb)" ;;
    stall:*)    stall=" STALL? (no summary change for ${verdict#stall:}s, GPU idle and no disk movement -- LOOK, do not kill)" ;;
    *)          stall="" ;;
  esac
  say "hb: ${hb:-<no answer>} | left $((DEADLINE - now))s$stall"
  sleep "$POLL"
done
rm -rf "$RUN_DIR/p54" && mkdir -p "$RUN_DIR/p54" || { say "fetch failed: local dir"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude 'artifact*/payloads/layer_*' --exclude work_qwen3 --exclude 'venv*' --exclude '.cache' "root@$HOST:$W/" "$RUN_DIR/p54/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p54" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p54/P54_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p54/TP_DONE.$NONCE" ] || {
  # A lane the box killed is a different fact from a lane that ran out of clock (e4b#634/#641).
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P54_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p54/P54_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p54/P54_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
