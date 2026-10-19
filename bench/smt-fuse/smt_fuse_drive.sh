#!/bin/bash
# bench/smt-fuse/smt_fuse_drive.sh -- the SMT fuse measurement, CONTROLLER side: the launcher's --command
# (bench/smt-fuse/PREREG-smt-fuse.md).
# Derived from bench/p146/p146_drive.sh (with #1787's liveness by launch identity) by named substitutions:
#   * it stages the probe and the box script at staged.sha256; there is no data slice and no checkpoint;
#   * the run id must be one the registration declares: smt-fuse-1..3 (bench/smt-fuse/PREREG-smt-fuse.md; -2 and -3 are
#     spares for pre-create refusals, listings gone before the create and NOT_RUN at the pre-flight). Any other id is refused;
#   * SMT_FUSE_DRY=1 is the $0 dry run (ids smt-fuse-dry-<n>) through SMT_FUSE_BOX_EXEC, as P146's.
#   * Amendment 1: smt-sib-1..8 (and smt-sib-dry-<n>; -4..8 are Amendment 1a's pre-create spares) run the pinned sibling probe; smt_fuse_run.sh picks it by run id.
# It starts the lane detached under a fresh nonce, polls TP_DONE.<nonce> with the tp4-style heartbeat/liveness check and
# fetches receipts -- never the checkpoint or the e4b tree. Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [smt_fuse_drive] $*"; }
E4B_PIN=d65793e4a106c7db6ecfc219668a2d6881d207e5      # smt_fuse_run.sh refuses any other
DRY=${SMT_FUSE_DRY:-0}
if [ "$DRY" = 1 ]; then
  for v in E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID SMT_FUSE_BOX_EXEC; do
    [ -n "${!v:-}" ] || { say "refusing: $v is not set -- SMT_FUSE_DRY=1 needs it"; exit 78; }
  done
else
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
fi
# ---- the declared run ids: the reading, its one rerun; the dry run's own
RUN_ID=$E4B_RENT_RUN_ID
case "$RUN_ID" in
  smt-fuse-[1-3]|smt-sib-[1-8]) [ "$DRY" != 1 ] || { say "refusing: $RUN_ID is a reading's id"; exit 78; };;
  smt-fuse-dry-[0-9]|smt-fuse-dry-[0-9][0-9]|smt-sib-dry-[0-9]|smt-sib-dry-[0-9][0-9])
    [ "$DRY" = 1 ] || { say "refusing: $RUN_ID is a dry run's id (SMT_FUSE_DRY=1 only)"; exit 78; };;
  *) say "refusing: run id $RUN_ID is not declared (smt-fuse-1..3, Amendment 1's smt-sib-1..8; *-dry-<n> dry)"; exit 78;;
esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# Lane liveness by PID identity and host uptime (#1512) and the host-fault marker (#1517): the helper is sourced here and
# streamed to the box over stdin for the launch and each probe; it is never staged (staged.sha256 does not list it).
LANE_HELPER="$REPO/bench/common/lane_liveness.sh"
source "$LANE_HELPER" || { say "refusing: missing lane liveness helper"; exit 78; }
STAGE="$HERE/smt_fuse_run.sh $HERE/smt_fuse_box.py $HERE/smt_sib_box.py $HERE/q235_bootstrap.py $HERE/wheels.json $HERE/dq11_require_git.sh $HERE/staged.sha256"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them; resolve each name to its source and compare hashes
# (the same case as tests/test_smt_fuse_staged_pin.py, which runs this check in CI where it costs nothing).
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    smt_fuse_run.sh|smt_fuse_box.py|smt_sib_box.py|q235_bootstrap.py|wheels.json|dq11_require_git.sh) src="$HERE/$name";;
    *) say "refusing: staged.sha256 names $name, which this driver does not stage"; exit 78;;
  esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The e4b the box installs is the registered pin; an E4B_SHA from the caller must be that pin.
[ -z "${E4B_SHA:-}" ] || [ "$E4B_SHA" = "$E4B_PIN" ] || { say "refusing: E4B_SHA=$E4B_SHA is not the registered pin $E4B_PIN"; exit 78; }
E4B_SHA=$E4B_PIN
RUN_DIR=$E4B_RENT_RUN_DIR; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
# Amendment 1's guard is a tested fit (smt_sib_box.time_fit): the launch's own deadline must leave at least that much, so the
# manifest's wallclock and the registration cannot drift apart (refused before anything is staged)
case "$RUN_ID" in smt-sib-[1-8])
  FIT=$(python3 "$HERE/smt_sib_box.py" --fit) || { say "refusing: no time fit from smt_sib_box.py"; exit 78; }
  LEFT_S=$(( DEADLINE - $(date +%s) ))
  [ "$LEFT_S" -ge "$FIT" ] || { say "refusing: the launch leaves ${LEFT_S}s < the registered fit ${FIT}s (smt_sib_box.py GUARD_S)"; exit 78; };;
esac
POLL=${SMT_FUSE_POLL_S:-60}; W=${SMT_FUSE_HOME:-/root/smt_fuse}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
# ---- the transport: ssh/scp/rsync to the rented box, or SMT_FUSE_BOX_EXEC + tar to the dry box
if [ "$DRY" = 1 ]; then
  WHERE="dry box via $SMT_FUSE_BOX_EXEC"
  box(){ "$SMT_FUSE_BOX_EXEC" "$1"; }
  box_stage(){ local args=() f; for f in $STAGE; do args+=(-C "$(dirname "$f")" "$(basename "$f")"); done
    tar -cf - "${args[@]}" | box "tar -C $W -xf -" || return 1
    git -C "$REPO" archive --format=tar --prefix=e4bsrc/ "$E4B_SHA" | box "tar -C $W -xf -"; }
  box_fetch(){ box "cd $W && tar -cf - --exclude=./e4bsrc --exclude='./out_*' --exclude=__pycache__ --exclude=./.cache --exclude=./hf --exclude='./venv*' ."       | tar -C "$1" -xif -; }
else
  HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; WHERE="$HOST:$PORT"
  SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
  SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
  box(){ $SSH "$1"; }
  box_stage(){ $SCP $STAGE "root@$HOST:$W/"; }
  # The checkpoint stays on the box (the hub cache, outside $W), and so do the e4b clone and the fixture; records, logs,
  # the data slice and the verdict travel. The box's venv stays too: smt-fuse-2 fetched its 6.8 GB venv-q235 into the run
  # dir, which made the receipt store read dirty and refused every lane's launch until it was moved.
  box_fetch(){ rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT"       --exclude '__pycache__' --exclude '.cache' --exclude 'hf' --exclude 'e4bsrc' --exclude 'out_*' --exclude '/venv*'       "root@$HOST:$W/" "$1/"; }
fi
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="SMT_FUSE_RUN_ID=$RUN_ID SMT_FUSE_RUN_NONCE=$NONCE SMT_FUSE_DEADLINE_EPOCH=$DEADLINE SMT_FUSE_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA"
# Only the dry-run switch and its lane directory travel: a rental is the registered run, so the knobs that change WHAT is
# measured are not forwarded -- a rehearsal sets them on its own box and the runner marks it REHEARSAL.
for v in SMT_FUSE_DRY SMT_FUSE_HOME; do [ -n "${!v:-}" ] && PASS="$PASS $v=$(printf %q "${!v}")"; done
if [ "${SMT_FUSE_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> $WHERE:$W ; start: env $PASS bash smt_fuse_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/smt_fuse"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $WHERE; e4b $E4B_SHA (the registered pin); stacks are the runner's constants; receipts -> $RUN_DIR/smt_fuse; deadline $DEADLINE"
box "rm -rf -- $W && mkdir -p $W/logs /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
box_stage || { say "stage failed: copy"; exit 20; }
LANE_STARTED_AT=$(date +%s)
# The launch streams the helper ahead of the start, so the child's PID and /proc identity are recorded in the same
# shell that started it (the dry box's command runs it the same way: one shell command, stdin passed through).
LAUNCH_REPLY=$({ cat "$LANE_HELPER"; printf '%s\n' "cd $W || exit 20; nohup env $PASS bash smt_fuse_run.sh > outer.log 2>&1 < /dev/null & child=\$!; identity=\$(lane_proc_snapshot \$child); end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat SMT_FUSE_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; echo identity:\$identity; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124"; } | box "bash -s") || { say "start failed: child did not bind the nonce"; exit 21; }
printf '%s\n' "$LAUNCH_REPLY"
LANE_PID=$(printf '%s\n' "$LAUNCH_REPLY" | sed -n 's/^started:\([0-9][0-9]*\)$/\1/p')
[[ "$LANE_PID" =~ ^[1-9][0-9]*$ ]] || { say "start failed: malformed child PID"; exit 21; }
LANE_INITIAL=$(printf '%s\n' "$LAUNCH_REPLY" | sed -n 's/^identity://p')
say "lane identity: pid=$LANE_PID snapshot=${LANE_INITIAL:-unknown}"
# ---- heartbeat + liveness (e4b#641, tp4_drive.sh's): a stall is REPORTED, never acted on; two consecutive definite
# zeros for the lane's own process with no TP_DONE end the wait with their own code (25).
STALL_S=${SMT_FUSE_STALL_S:-900}; SMT_FUSE_MIN_PROGRESS_MB=${SMT_FUSE_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$SMT_FUSE_MIN_PROGRESS_MB
  consumed=0; grew=0
  if [ -n "$dfk_now" ] && [ -n "$dfk_prev" ] && [ "$dfk_now" -lt "$dfk_prev" ] 2>/dev/null; then consumed=$(( (dfk_prev - dfk_now) / 1024 )); fi
  if [ -n "$du_now" ] && [ -n "$du_prev" ] && [ "$du_now" -gt "$du_prev" ] 2>/dev/null; then grew=$((du_now - du_prev)); fi
  if [ "$consumed" -ge "$min_mb" ] || [ "$grew" -ge "$min_mb" ]; then
    if [ "$consumed" -ge "$grew" ]; then echo "fetching:-${consumed}M on disk"; else echo "fetching:+${grew}M in tree"; fi
    return 0
  fi
  if [ "$idle_s" -ge "$stall_s" ] && [ "${util:-0}" -eq 0 ] 2>/dev/null; then echo "stall:$idle_s"; fi
}
lane_dead() { lane_two_missing "$@"; }   # live_now live_prev: two DEFINITE zeros
say "lane started; polling TP_DONE every ${POLL}s with a heartbeat (stall reported after ${STALL_S}s; never acted on; a lane whose process is gone for two polls ends the wait)"
LAST=""; LAST_CHANGE=$(date +%s); LAST_DFK=""; LAST_DU=""; LAST_LIVE=""; UNKNOWN_PROBES=0; LANE_DEAD=0
while :; do
  now=$(date +%s)
  box "test -f $W/TP_DONE.$NONCE" 2>/dev/null < /dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$(box "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}')\"" 2>/dev/null < /dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  snapshot=$(box "bash -s -- --probe $LANE_PID" < "$LANE_HELPER" 2>/dev/null) || snapshot=""
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
rm -rf "$RUN_DIR/smt_fuse" && mkdir -p "$RUN_DIR/smt_fuse" || { say "fetch failed: local dir"; exit 22; }
box_fetch "$RUN_DIR/smt_fuse" < /dev/null || { say "fetch failed"; exit 22; }
say "fetched $(ls "$RUN_DIR/smt_fuse" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/smt_fuse/SMT_FUSE_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/smt_fuse/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no SMT_FUSE_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/smt_fuse/SMT_FUSE_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/smt_fuse/SMT_FUSE_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
