#!/bin/bash
# bench/p130/p130_drive.sh -- lane P130, CONTROLLER side: the launcher's --command (bench/p130/PREREG-p130.md; e4b#1313).
# Derived from bench/fam/fam_drive.sh by named substitutions. It stages P130's runner, box, draft and reducer, P109's
# box (its prompts), P39's NF4 bake and host calibration, the pinned chat prompts and the W1 tripwire -- all referenced,
# never copied. It starts the lane detached under a fresh nonce, polls TP_DONE.<nonce> with the shared PID-identity
# liveness check (bench/common/lane_liveness.sh), and fetches receipts -- never the checkpoint, the head or the arena.
# Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p130_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
LANE_HELPER="$REPO/bench/common/lane_liveness.sh"
source "$LANE_HELPER" || { say "refusing: missing lane liveness helper"; exit 78; }
# P130's staging. staged.sha256 pins every piece by the name the box sees.
P39="$REPO/bench/p39"; P109="$REPO/bench/p109"
STAGE="$HERE/p130_run.sh $HERE/p130_box.py $HERE/p130_eagle3.py $HERE/p130_reduce.py $HERE/chat_prompts.json $HERE/expect_w1.json $P109/p109_box.py $P39/k8_bake.py $P39/calib.json $HERE/staged.sha256"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them; resolve each name to its source and compare hashes
# (the same check as tests/test_p130.py, which runs it in CI where it costs nothing).
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    p130_run.sh|p130_box.py|p130_eagle3.py|p130_reduce.py|chat_prompts.json|expect_w1.json) src="$HERE/$name";;
    p109_box.py) src="$P109/$name";;
    k8_bake.py|calib.json) src="$P39/$name";;
    *) say "refusing: staged.sha256 names $name, which this driver does not stage"; exit 78;;
  esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The launch commit is the harness this driver ships from: derived, never a literal, and refused on a dirty tree.
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
# The target stack (e4b 7f044dd9 + gnf4 d769d502), the model and the head are constants in the runner.
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${P130_POLL_S:-60}; W=/root/p130
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P130_RUN_ID=$RUN_ID P130_RUN_NONCE=$NONCE P130_DEADLINE_EPOCH=$DEADLINE P130_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA"
# Nothing that changes WHAT is measured is forwarded: a rental is the registered run.
if [ "${P130_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p130_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p130"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; launch e4b $E4B_SHA (from $REPO); the target is the runner's constant; receipts -> $RUN_DIR/p130; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
if [ -s "$HF_TOKEN_FILE" ]; then
  if python3 "$REPO/bench/common/token_scope.py" --token-file "$HF_TOKEN_FILE"; then
    $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
    say "hf token staged"
  else
    token_rc=$?
    [ "$token_rc" != 78 ] || { say "refusing: token read-only: no"; exit 78; }
    PASS="$PASS HF_HUB_DISABLE_IMPLICIT_TOKEN=1"
    say "token unverified, staging none"
  fi
else
  PASS="$PASS HF_HUB_DISABLE_IMPLICIT_TOKEN=1"
  say "no token file, staging none"
fi
LANE_STARTED_AT=$(date +%s)
LAUNCH_REPLY=$({ cat "$LANE_HELPER"; printf '%s\n' "cd $W || exit 20; nohup env $PASS bash p130_run.sh > outer.log 2>&1 < /dev/null & child=\$!; identity=\$(lane_proc_snapshot \$child); end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P130_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; echo identity:\$identity; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124"; } | $SSH bash -s) || { say "start failed: child did not bind the nonce"; exit 21; }
printf '%s\n' "$LAUNCH_REPLY"
LANE_PID=$(printf '%s\n' "$LAUNCH_REPLY" | sed -n 's/^started:\([0-9][0-9]*\)$/\1/p')
[[ "$LANE_PID" =~ ^[1-9][0-9]*$ ]] || { say "start failed: malformed child PID"; exit 21; }
LANE_INITIAL=$(printf '%s\n' "$LAUNCH_REPLY" | sed -n 's/^identity://p')
say "lane identity: pid=$LANE_PID snapshot=${LANE_INITIAL:-unknown}"
# ---- heartbeat + liveness: a stall is REPORTED, never acted on; two consecutive definite misses of the lane's own
# process (by PID identity) with no TP_DONE end the wait (25); a host reboot ends it at once and writes host-fault.json.
STALL_S=${P130_STALL_S:-900}; P130_MIN_PROGRESS_MB=${P130_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$P130_MIN_PROGRESS_MB
  consumed=0; grew=0
  if [ -n "$dfk_now" ] && [ -n "$dfk_prev" ] && [ "$dfk_now" -lt "$dfk_prev" ] 2>/dev/null; then consumed=$(( (dfk_prev - dfk_now) / 1024 )); fi
  if [ -n "$du_now" ] && [ -n "$du_prev" ] && [ "$du_now" -gt "$du_prev" ] 2>/dev/null; then grew=$((du_now - du_prev)); fi
  if [ "$consumed" -ge "$min_mb" ] || [ "$grew" -ge "$min_mb" ]; then
    if [ "$consumed" -ge "$grew" ]; then echo "fetching:-${consumed}M on disk"; else echo "fetching:+${grew}M in tree"; fi
    return 0
  fi
  if [ "$idle_s" -ge "$stall_s" ] && [ "${util:-0}" -eq 0 ] 2>/dev/null; then echo "stall:$idle_s"; fi
}
lane_dead() { lane_two_missing "$@"; }
say "lane started; polling TP_DONE every ${POLL}s with a heartbeat (stall reported after ${STALL_S}s; never acted on; a lane whose process is gone for two polls ends the wait)"
LAST=""; LAST_CHANGE=$(date +%s); LAST_DFK=""; LAST_DU=""; LAST_LIVE=""; UNKNOWN_PROBES=0; LANE_DEAD=0
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}')\"" 2>/dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  snapshot=$($SSH "bash -s -- --probe $LANE_PID" < "$LANE_HELPER" 2>/dev/null) || snapshot=""
  live=$(lane_snapshot_verdict "$snapshot" "$LANE_INITIAL" "$((now - LANE_STARTED_AT))")
  UNKNOWN_PROBES=$(lane_unknown_streak "$live" "$UNKNOWN_PROBES")
  hb="$hb | live ${live:-unknown} | pid $LANE_PID | unknown_probes $UNKNOWN_PROBES"
  if [ "$live" = reboot ]; then
    say "LANE DEAD: host rebooted (boot identity changed or uptime below lane age/baseline); fetching what exists"
    if lane_write_host_fault "${E4B_RENT_RUN_DIR:-}" reboot "$(lane_reboot_evidence "$snapshot" "$LANE_INITIAL" "$((now - LANE_STARTED_AT))")"; then
      say "host-fault.json written (kind reboot) for the receipt"
    else
      say "host-fault.json NOT written (no run directory)"
    fi
    LANE_DEAD=1; break
  fi
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: launch PID $LANE_PID gone/reused on the box for two consecutive polls and no TP_DONE -- not waiting out the deadline"
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
rm -rf "$RUN_DIR/p130" && mkdir -p "$RUN_DIR/p130" || { say "fetch failed: local dir"; exit 22; }
# The checkpoint and the head stay on the box (the hub cache, outside $W), and so does the arena (work/: only its
# bake.json travels); captures, records, logs and the verdict travel.
rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" \
  --include 'work/' --include 'work/bake.json' --exclude 'work/*' \
  --exclude '__pycache__' --exclude 'venv*' --exclude '.cache' --exclude 'hf' --exclude 'src' \
  "root@$HOST:$W/" "$RUN_DIR/p130/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p130" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p130/P130_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p130/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P130_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p130/P130_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p130/P130_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
