#!/bin/bash
# bench/p47/p47_drive.sh -- lane P47 (Gemma-4 diagnostic, #597), CONTROLLER side: the launcher's --command. Stages the KL instrument
# (bench/kl_fidelity.py, kl_paths.py, kl_prompts.py), P39's k8_bake.py + calib.json, P42's hook, and this lane's
# serve_stack.py / kl_serve.py (P44-b's, shared) / p47_run.sh; starts the lane detached under a fresh nonce; polls TP_DONE.<nonce>; fetches receipts. Nothing here
# creates, destroys or approves compute. Pattern: bench/p42/p42_drive.sh.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p47_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); P39="$REPO/bench/p39"; P44="$REPO/bench/p44"; P49="$REPO/bench/p49"
KL="$REPO/bench"
STAGE="$HERE/p47_run.sh $P44/serve_stack.py $P44/kl_serve.py $P49/act_probe.py $KL/kl_fidelity.py $KL/kl_paths.py $KL/kl_prompts.py $KL/kl_prompts_heldout.py $P39/k8_bake.py $P39/calib.json $HERE/staged.sha256"
HOOK="$REPO/bench/p42/hook/usercustomize.py"   # P42's hook, referenced not copied (one file to keep in step)
for f in $STAGE $HOOK; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in hook/*) src="$REPO/bench/p42/$name";; p47_run.sh) src="$HERE/$name";; serve_stack.py|kl_serve.py) src="$P44/$name";; act_probe.py) src="$P49/$name";; kl_*.py) src="$KL/$name";; *) src="$P39/$name";; esac
  got=$(sha_of "$src"); [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex ($E4B_SHA)"; exit 78;; esac; [ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not 40 chars"; exit 78; }
GNF4_SHA=${GNF4_SHA:-8b1acc9e103ed1798a82307503c1c63d667b59ee}   # grouped-nf4-gemm v0.32.0, the consumer CI pin
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${P47_POLL_S:-60}; W=/root/p47
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P47_RUN_ID=$RUN_ID P47_RUN_NONCE=$NONCE P47_DEADLINE_EPOCH=$DEADLINE P47_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA P47_MIN_MBPS=${P47_MIN_MBPS:-20} P47_MIN_DISK_GB=${P47_MIN_DISK_GB:-120} P47_FAMILY=${P47_FAMILY:-gemma4diag} P47_NEED_S=${P47_NEED_S:-6000} P47_ACT_PROBE=${P47_ACT_PROBE:-0} P47_ACT_PROBE_N=${P47_ACT_PROBE_N:-8} P47_KL_SKIP=${P47_KL_SKIP:-0} P47_PROMPT_SET=${P47_PROMPT_SET:-committed} ${P47_ARMS:+P47_ARMS=\"$P47_ARMS\" }${P47_FAMILY2:+P47_FAMILY2=$P47_FAMILY2 }${P47_ARMS2:+P47_ARMS2=\"$P47_ARMS2\" }${P47_NEED_S2:+P47_NEED_S2=$P47_NEED_S2 }"
if [ "${P47_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p47_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p47"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA gnf4 $GNF4_SHA (from $REPO); receipts -> $RUN_DIR/p47; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/hook" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" && $SCP "$HOOK" "root@$HOST:$W/hook/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W || exit 20; nohup env $PASS bash p47_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P47_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; sleep 1; done; echo 'no nonce handshake in 30 s'; exit 21" || { say "lane did not start"; exit 21; }
# ---- heartbeat + liveness (e4b#641 -- ported from tp4_drive.sh, #625/#634). A poller that waits for a
# marker cannot tell a dead lane from a slow one: tp4-c-parity-1's remote process was killed without writing
# its markers and the controller polled the corpse for 53 minutes. Each poll now asks the box (a) the last
# summary line, (b) GPU util, (c) workdir size and free disk, (d) whether `bash p47_run.sh` is still there.
# A stall is REPORTED, never acted on; two consecutive definite zeros on (d) with no TP_DONE end the wait
# with their own exit code (25), so a killed lane is not flattened into "ran out of clock".
STALL_S=${P47_STALL_S:-900}; P47_MIN_PROGRESS_MB=${P47_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$P47_MIN_PROGRESS_MB
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
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f 'bash p47_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: no 'bash p47_run.sh' on the box for two consecutive polls and no TP_DONE -- the remote process exited without writing its markers; not waiting out the deadline"
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
rm -rf "$RUN_DIR/p47" && mkdir -p "$RUN_DIR/p47" || { say "fetch failed: local dir"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude 'work_*' --exclude 'refcache_*' --exclude '.cache' --exclude '__pycache__' "root@$HOST:$W/" "$RUN_DIR/p47/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p47" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p47/P47_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p47/TP_DONE.$NONCE" ] || {
  # A lane the box killed is a different fact from a lane that ran out of clock (e4b#634/#641).
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P47_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p47/P47_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p47/P47_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
