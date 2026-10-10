#!/bin/bash
# bench/p130/p130_drive.sh -- lane P130, CONTROLLER side: the launcher's --command (bench/p130/PREREG-p130.md;
# e4b#846). Derived from bench/p127/p127_drive.sh by named substitutions: it stages P130's runner, reducer and box,
# P117's box (windows, the teacher-forced pass) with P108's and P97's (whose helpers it imports), P39's NF4 bake and host
# calibration at their registered bytes, the prefill-graph premise test and the fetch watchdog -- all referenced, never
# copied -- starts the lane detached under a fresh nonce; polls TP_DONE.<nonce> with the tp4-style heartbeat/liveness
# check; fetches receipts -- never the checkpoint, the arena or R's log-probs (work/: only bake.json travels). Nothing
# here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p130_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
LANE_HELPER="$REPO/bench/common/lane_liveness.sh"
source "$LANE_HELPER" || { say "refusing: missing lane liveness helper"; exit 78; }
# P130's staging: its runner, reducer and box, P117's, P108's and P97's boxes, P39's k8_bake.py and calib.json, the
# premise test from tests/ and the fetch watchdog from bench/common/. staged.sha256 pins every one by the name the box sees.
TESTS="$REPO/tests"; P39="$REPO/bench/p39"; P117="$REPO/bench/p117"; P108="$REPO/bench/p108"; P97="$REPO/bench/p97"
COMMON="$REPO/bench/common"
STAGE="$HERE/p130_run.sh $HERE/p130_reduce.py $HERE/p130_box.py $P117/p117_box.py $P108/p108_box.py $P97/p97_box.py $P39/k8_bake.py $P39/calib.json $TESTS/test_prefill_graph_gpu.py $COMMON/hf_fetch_watchdog.py $HERE/staged.sha256"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them; resolve each name to its source and compare hashes
# (the same case as tests/test_p130_staged_pin.py, which runs this check in CI where it costs nothing).
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    p130_run.sh|p130_reduce.py|p130_box.py) src="$HERE/$name";;
    p117_box.py) src="$P117/$name";;
    p108_box.py) src="$P108/$name";;
    p97_box.py) src="$P97/$name";;
    k8_bake.py|calib.json) src="$P39/$name";;
    test_prefill_graph_gpu.py) src="$TESTS/$name";;
    hf_fetch_watchdog.py) src="$COMMON/$name";;
    *) say "refusing: staged.sha256 names $name, which this driver does not stage"; exit 78;;
  esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The launch commit is the e4b this driver ships from (the launcher proved it is the manifest's heads.e4b): it stages
# the kit, and the box installs it. Derived, never a literal, and
# refused on a dirty tree (p39-box1b-3's lesson).
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
# The pre-rental fetch gate (PREREG STOP-0): bench/p130/p130_fetch_gate.py ran on the controller, before the rental
# controller quoted, and resolved everything the box will fetch. Nothing is staged without its passing report for THIS
# launch commit, under 24 h old; the report travels with the receipts.
[ -n "${P130_FETCH_GATE:-}" ] || { say "refusing: P130_FETCH_GATE unset -- run bench/p130/p130_fetch_gate.py before quoting"; exit 78; }
python3 "$HERE/p130_fetch_gate.py" --e4b-sha "$E4B_SHA" --check-report "$P130_FETCH_GATE" 1>&2 \
  || { say "refusing: the fetch gate report $P130_FETCH_GATE does not let $E4B_SHA stage"; exit 78; }
# grouped-nf4-gemm's pin is a constant in the runner; the box installs e4b at THIS commit.
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${P130_POLL_S:-60}; W=/root/p130
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P130_RUN_ID=$RUN_ID P130_RUN_NONCE=$NONCE P130_DEADLINE_EPOCH=$DEADLINE P130_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA"
# Only host-limit knobs and the proving-run switch travel: a rental is the registered run, so the knobs that change WHAT
# is measured are not forwarded -- a rehearsal sets them on its own box and the runner marks it REHEARSAL. P130_PROVE=1
# is the proving rental the pre-registration names: the whole box end to end on Qwen3-30B-A3B itself at the proof's sizes
# (no other family takes the int4 K19 prefill route P2 changes), because no local card runs the fp8 paged KV.
for v in P130_PROVE; do [ -n "${!v:-}" ] && PASS="$PASS $v=$(printf %q "${!v}")"; done
# Amendment 1: the corpus commit the gate resolved travels to the box, which reports it beside the one it read
GATE_CORPUS=$(python3 -c 'import json, sys; print((json.load(open(sys.argv[1])).get("corpus") or {}).get("main") or "")' "$P130_FETCH_GATE" 2>/dev/null)
case "$GATE_CORPUS" in ""|*[!0-9a-f]*) ;; *) [ ${#GATE_CORPUS} -eq 40 ] && PASS="$PASS P130_GATE_CORPUS=$GATE_CORPUS";; esac
cp "$P130_FETCH_GATE" "$E4B_RENT_RUN_DIR/p130_fetch_gate.json" 2>/dev/null || say "note: could not copy the fetch gate report to the run directory" 1>&2
if [ "${P130_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p130_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p130"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; launch e4b $E4B_SHA (from $REPO); stacks are the runner's constants; receipts -> $RUN_DIR/p130; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
if [ -s "$HF_TOKEN_FILE" ]; then
  if python3 "$REPO/bench/common/token_scope.py" --token-file "$HF_TOKEN_FILE"; then
    $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
    say "hf token staged"
  else
    token_rc=$?
    [ "$token_rc" != 78 ] || { say "refusing: token read-only: no"; exit 78; }
    # Ignore cached box credentials too when verification is unavailable.
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
# ---- heartbeat + liveness (e4b#641, tp4_drive.sh's): a stall is REPORTED, never acted on; two consecutive definite
# zeros for the lane's own process with no TP_DONE end the wait with their own code (25).
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
# The checkpoint stays on the box (the hub cache, outside $W), and so do the NF4 snapshot, the arena and R's log-probs
# (work/: only its bake.json travels); the process records, logs and the verdict travel.
rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" \
  --include 'work/' --include 'work/bake.json' --exclude 'work/*' \
  --exclude '__pycache__' --exclude 'venv*' --exclude '.cache' --exclude 'hf' \
  "root@$HOST:$W/" "$RUN_DIR/p130/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p130" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p130/P130_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p130/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P130_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p130/P130_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p130/P130_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
