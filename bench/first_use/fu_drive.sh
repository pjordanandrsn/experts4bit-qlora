#!/bin/bash
# T1 step 3's controller, Loggetta's first-use matrix (its PREREG is loggetta bench/first_use/PREREG-first_use.md): HO2's
# driver (bench/ho2/ho2_drive.sh) by counted edits. The kit lives in Loggetta and is pinned here by its MERGE commit and
# its kit hash; every staged file is checked against staged-first_use.sha256 AT that commit before anything is staged.
# The release candidate's wheel and its scripts/first_run.sh are staged with the sha256s this driver computes. Three boxes
# (one per driver band), each with two spares. The fetch keeps first_run.json, the logs and the verdict, never a venv or a
# checkpoint. Refuses unknown run IDs and dirty source trees, polls the nonce-bound completion marker. It does not create,
# destroy, or approve compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [fu_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# Lane liveness by launch identity (PID + /proc identity, host uptime; #1512, #1517): the helper is sourced here and
# streamed to the box over stdin for the launch and each probe; it is never staged (staged-first_use.sha256 does not list it).
# It never decides LANE DEAD from a self-matching pgrep (A3's form, the maintainer's liveness ruling).
LANE_HELPER="$REPO/bench/common/lane_liveness.sh"
source "$LANE_HELPER" || { say "refusing: missing lane liveness helper"; exit 78; }
# The staging: the Loggetta kit at its MERGE commit (loggetta#99). The manifest must hash to the reviewed kit hash, and
# every file it names must hash to its line. As the launcher's --command this runs on a RENTED box, so a mismatch refuses
# before anything is staged there, not before the spend; the USD 0 check is the dry run (FU_DRIVE_DRYRUN=1), run at the ACK.
LOGGETTA_KIT_COMMIT=bf0e5755055fa2c06c93332b9e38b9d446172129
KIT_SHA256=f84c1aabb90bda5aaeac9f3f7e94093a7a1c925e9e71312238d2330a40ca1ad2
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
KIT=$(mktemp -d) || { say "refusing: no temp dir"; exit 78; }
if [ -n "${FU_KIT_SOURCE:-}" ]; then          # a local copy of the kit, for the dry run's tamper check only
  [ "${FU_DRIVE_DRYRUN:-0}" = "1" ] || { say "refusing: FU_KIT_SOURCE is for the dry run only"; exit 78; }
  for f in fu_run.sh fu_reduce.py staged-first_use.sha256; do cp "$FU_KIT_SOURCE/$f" "$KIT/$f" 2>/dev/null; done
else
  for f in fu_run.sh fu_reduce.py staged-first_use.sha256; do
    curl -fsSL --retry 3 -o "$KIT/$f" "https://raw.githubusercontent.com/pjordanandrsn/loggetta/$LOGGETTA_KIT_COMMIT/bench/first_use/$f" \
      || { say "refusing: cannot fetch $f at loggetta $LOGGETTA_KIT_COMMIT"; exit 78; }
  done
fi
for f in fu_run.sh fu_reduce.py staged-first_use.sha256; do [ -s "$KIT/$f" ] || { say "refusing: kit piece missing: $f"; exit 78; }; done
got=$(sha_of "$KIT/staged-first_use.sha256")
[ "$got" = "$KIT_SHA256" ] || { say "refusing: staged-first_use.sha256 is $got, the reviewed kit hash is $KIT_SHA256"; exit 78; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    fu_run.sh|fu_reduce.py) src="$KIT/$name";;
    *) say "refusing: staged-first_use.sha256 names $name, which this driver does not stage"; exit 78;;
  esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $name is $got, staged-first_use.sha256 says $want"; exit 78; }
done < "$KIT/staged-first_use.sha256"
# The release candidate: its commit, its wheel (built by the release flow at that commit) and its first_run.sh.
FU_RC_COMMIT=${FU_RC_COMMIT:-}; FU_RC_WHEEL=${FU_RC_WHEEL:-}
case "$FU_RC_COMMIT" in *[!0-9a-f]*|"") say "refusing: FU_RC_COMMIT is not a 40-char hex sha ($FU_RC_COMMIT)"; exit 78;; esac
[ ${#FU_RC_COMMIT} -eq 40 ] || { say "refusing: FU_RC_COMMIT is not a 40-char hex sha"; exit 78; }
case "$FU_RC_WHEEL" in *.whl) [ -s "$FU_RC_WHEEL" ] || { say "refusing: no wheel at $FU_RC_WHEEL"; exit 78; };;
  *) say "refusing: FU_RC_WHEEL must be the RC's .whl"; exit 78;; esac
if [ -n "${FU_RC_FIRST_RUN:-}" ]; then        # a local copy, for the dry run only
  [ "${FU_DRIVE_DRYRUN:-0}" = "1" ] || { say "refusing: FU_RC_FIRST_RUN is for the dry run only"; exit 78; }
  cp "$FU_RC_FIRST_RUN" "$KIT/first_run.sh" || { say "refusing: cannot read $FU_RC_FIRST_RUN"; exit 78; }
else
  curl -fsSL --retry 3 -o "$KIT/first_run.sh" "https://raw.githubusercontent.com/pjordanandrsn/loggetta/$FU_RC_COMMIT/scripts/first_run.sh" \
    || { say "refusing: cannot fetch scripts/first_run.sh at loggetta $FU_RC_COMMIT"; exit 78; }
fi
# the wheel's path is word-split into the scp and its name reaches the box's shell: plain wheel characters only
case "$FU_RC_WHEEL" in *[!A-Za-z0-9._+/-]*) say "refusing: FU_RC_WHEEL holds characters a wheel path never needs"; exit 78;; esac
WHEEL_SHA=$(sha_of "$FU_RC_WHEEL"); FIRST_RUN_SHA=$(sha_of "$KIT/first_run.sh"); WHEEL_NAME=$(basename "$FU_RC_WHEEL")
STAGE="$KIT/fu_run.sh $KIT/fu_reduce.py $KIT/staged-first_use.sha256 $KIT/first_run.sh $FU_RC_WHEEL"
# The launch commit is the e4b this driver ships from (the launcher proved it is the manifest's heads.e4b). The box does
# not install it (it installs the RC wheel); it is recorded. Derived, never a literal, and refused on a dirty tree
# (p39-box1b-3's lesson).
if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${FU_POLL_S:-60}; W=/root/first_use
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
# Explicit IDs only: each band's reading (-1) and two spares (-2, -3); the band names the box's cells (the PREREG's table).
case "$RUN_ID" in
  fu-below-1|fu-below-2|fu-below-3) CELLS="below-default below-matched cpu-torch";;
  fu-at-1|fu-at-2|fu-at-3) CELLS="at-default at-matched";;
  fu-current-1|fu-current-2|fu-current-3) CELLS="current-default current-matched";;
  *) say "refusing unregistered run ID: $RUN_ID (is not fu-below-N, fu-at-N or fu-current-N)"; exit 78;;
esac
LISTED=${FU_LISTED_DRIVER:-}
case "$LISTED" in *[!0-9.]*) say "refusing: FU_LISTED_DRIVER is not a driver version ($LISTED)"; exit 78;; esac
PASS="FU_RUN_ID=$RUN_ID FU_RUN_NONCE=$NONCE FU_DEADLINE_EPOCH=$DEADLINE FU_CELLS='$CELLS' FU_WHEEL=$WHEEL_NAME FU_WHEEL_SHA256=$WHEEL_SHA FU_FIRST_RUN_SHA256=$FIRST_RUN_SHA FU_LISTED_DRIVER=$LISTED"
if [ "${FU_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN kit $KIT_SHA256 at loggetta $LOGGETTA_KIT_COMMIT verified; RC $FU_RC_COMMIT wheel $WHEEL_SHA first_run $FIRST_RUN_SHA; stage -> root@$HOST:$W ; start: env $PASS bash fu_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/first_use"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; kit $KIT_SHA256 (loggetta $LOGGETTA_KIT_COMMIT); RC $FU_RC_COMMIT wheel $WHEEL_SHA; e4b $E4B_SHA; cells: $CELLS; deadline $DEADLINE"
$SSH "test ! -e $W && mkdir -p $W/logs /root/.cache/huggingface" || { say "stage failed: remote work directory exists"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
if [ -s "$HF_TOKEN_FILE" ]; then   # authenticated pulls (unauthenticated shards throttle); the token never appears on a command line
  $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
  say "hf token staged"
else
  say "no hf token file at $HF_TOKEN_FILE -- pulls run unauthenticated (the checkpoint is ungated)"
fi
LANE_STARTED_AT=$(date +%s)
# The launch streams the helper ahead of the start, so the child's PID and /proc identity are recorded in the same shell
# that started it (P127's form).
LAUNCH_REPLY=$({ cat "$LANE_HELPER"; printf '%s\n' "cd $W || exit 20; nohup env $PASS bash fu_run.sh > outer.log 2>&1 < /dev/null & child=\$!; identity=\$(lane_proc_snapshot \$child); end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat FU_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; echo identity:\$identity; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124"; } | $SSH bash -s) || { say "start failed: child did not bind the nonce"; exit 21; }
printf '%s\n' "$LAUNCH_REPLY"
LANE_PID=$(printf '%s\n' "$LAUNCH_REPLY" | sed -n 's/^started:\([0-9][0-9]*\)$/\1/p')
[[ "$LANE_PID" =~ ^[1-9][0-9]*$ ]] || { say "start failed: malformed child PID"; exit 21; }
LANE_INITIAL=$(printf '%s\n' "$LAUNCH_REPLY" | sed -n 's/^identity://p')
say "lane identity: pid=$LANE_PID snapshot=${LANE_INITIAL:-unknown}"
# ---- heartbeat + liveness (e4b#641, tp4_drive.sh's): a stall is REPORTED, never acted on; two consecutive definite
# zeros for the lane's own process with no TP_DONE end the wait with their own code (25).
STALL_S=${FU_STALL_S:-900}; FU_MIN_PROGRESS_MB=${FU_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$FU_MIN_PROGRESS_MB
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
mkdir -p "$RUN_DIR/first_use" && [ -z "$(find "$RUN_DIR/first_use" -mindepth 1 -maxdepth 1 -print -quit)" ] || { say "fetch failed: local dir"; exit 22; }
# The checkpoints stay on the box (the hub cache, outside $W), and so do the trained adapters (tensors); the receipts,
# logs, phase stamps and the verdict travel.
rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" \
  --exclude 'run/adapter_*' --exclude '*.safetensors' --exclude '*.pt' \
  --exclude '__pycache__' --exclude 'venv*' --exclude '.cache' --exclude 'hf' \
  "root@$HOST:$W/" "$RUN_DIR/first_use/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/first_use" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/first_use/FU_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/first_use/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no FU_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/first_use/FU_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/first_use/FU_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
