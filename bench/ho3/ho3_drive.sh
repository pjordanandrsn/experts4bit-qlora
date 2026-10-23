#!/bin/bash
# Lane HO3's controller (bench/ho3/PREREG-ho3.md): HO2's driver (bench/ho2/ho2_drive.sh) by counted edits -- this
# lane's run ids (Part A: one RTX 5090 and one RTX 4090 box; Part B: two RTX A4000 boxes; retry ids r1..r8 per card),
# the refusal of HO2's own machines (a second box must be another machine), runner and reducer; the staged pieces are
# bench/common/phases.sh and staged-ho3.sha256; the fetch keeps the receipts, the logs and the verdict, never a
# checkpoint or a trained adapter (tensors stay on the box). Stages hash-pinned files, refuses unknown run IDs and dirty
# source trees, polls the nonce-bound completion marker. It does not create, destroy, or approve compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [ho3_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# Lane liveness by launch identity (PID + /proc identity, host uptime; #1512, #1517): the helper is sourced here and
# streamed to the box over stdin for the launch and each probe; it is never staged (staged-ho3.sha256 does not list it).
# It never decides LANE DEAD from a self-matching pgrep (A3's form, the maintainer's liveness ruling).
LANE_HELPER="$REPO/bench/common/lane_liveness.sh"
source "$LANE_HELPER" || { say "refusing: missing lane liveness helper"; exit 78; }
# The staging: this lane's runner and reducer, and the phase stamps. staged-ho3.sha256 pins every one by the name the
# box sees.
COMMON="$REPO/bench/common"
STAGE="$HERE/ho3_run.sh $HERE/ho3_reduce.py $HERE/ho3_drvprobe.py $HERE/loggetta_measure.py $COMMON/phases.sh $HERE/staged-ho3.sha256"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged-ho3.sha256 names the files as the BOX will see them; resolve each name to its source and compare hashes
# (the same map as tests/test_ho3.py, checked in CI).
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    ho3_run.sh|ho3_reduce.py|ho3_drvprobe.py|loggetta_measure.py) src="$HERE/$name";;
    phases.sh) src="$COMMON/$name";;
    *) say "refusing: staged-ho3.sha256 names $name, which this driver does not stage"; exit 78;;
  esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged-ho3.sha256"
# The launch commit is the e4b this driver ships from (the launcher proved it is the manifest's heads.e4b): it stages
# the kit, and the box installs it. Derived, never a literal, and
# refused on a dirty tree (p39-box1b-3's lesson).
if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
# grouped-nf4-gemm's pin is a constant in the runner; the box installs e4b at THIS commit (which must carry the knob).
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${HO3_POLL_S:-60}; W=/root/ho3
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="HO3_RUN_ID=$RUN_ID HO3_RUN_NONCE=$NONCE HO3_DEADLINE_EPOCH=$DEADLINE HO3_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA"
# Explicit IDs only (PREREG "Run ids"): the reading ids and the retry ids r1..r8. No code reads a number out of an id.
case "$RUN_ID" in
  ho3-5090-1|ho3-5090-2|ho3-5090-r[1-8]|ho3-4090-1|ho3-4090-2|ho3-4090-r[1-8]|ho3-a4000-1|ho3-a4000-2|ho3-a4000-3|ho3-a4000-r[1-8]) :;;
  *) say "refusing unregistered run ID: $RUN_ID (is not a registered ho3 reading or retry id)"; exit 78;;
esac
# A second box must be another machine (PREREG "Machines"): HO2's two read machines are refused before anything is
# staged, and so is a box whose machine the launcher did not record in its instance.json.
HO2_MACHINES="124996 154091"
MACHINE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('machine_id') or 'unknown')" "$RUN_DIR/instance.json" 2>/dev/null || echo unknown)
case "$MACHINE" in ""|unknown|*[!0-9]*) say "refusing: the launcher recorded no machine id for this box ($MACHINE)"; exit 78;; esac
for m in $HO2_MACHINES; do [ "$MACHINE" = "$m" ] && { say "refusing: machine $MACHINE ran HO2's rows; a second box must be another machine"; exit 78; }; done
say "machine $MACHINE (not one of HO2's: $HO2_MACHINES)"
# Part B's cohorts need two machines (PREREG "Part B"): an RTX A4000 box is refused on the machine of any earlier A4000
# box that ran this runner (its run directory holds a fetched ho3/), and refused when such a box's machine is unreadable.
case "$RUN_ID" in
  ho3-a4000-*)
    RUNS_ROOT=$(cd "$RUN_DIR/../.." 2>/dev/null && pwd) || { say "refusing: cannot read the runs root above $RUN_DIR"; exit 78; }
    for d in "$RUNS_ROOT"/*/ho3-a4000-*; do
      [ -d "$d/ho3" ] || continue
      [ "$(cd "$d" && pwd)" = "$(cd "$RUN_DIR" && pwd)" ] && continue
      m=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['machine_id'])" "$d/instance.json" 2>/dev/null) \
        || { say "refusing: earlier A4000 box $(basename "$d") ran but its machine is unreadable"; exit 78; }
      [ "$m" = "$MACHINE" ] && { say "refusing: machine $MACHINE already ran $(basename "$d"); Part B needs a second machine"; exit 78; }
    done
    ;;
esac
# Amendment 2: three consecutive driver-peak refusals (the runner's rc 19) on one card stop that card: stop and report,
# never a rental loop. Earlier boxes of this card are ordered by the launcher's instance.json "at"; an earlier box that
# ran this runner but whose exit or time is unreadable refuses (fail closed).
CARD=${RUN_ID#ho3-}; CARD=${CARD%%-*}
RUNS_ROOT=$(cd "$RUN_DIR/../.." 2>/dev/null && pwd) || { say "refusing: cannot read the runs root above $RUN_DIR"; exit 78; }
TAIL19=$(python3 - "$RUNS_ROOT" "$CARD" "$RUN_DIR" <<'PYC'
import glob, json, os, sys
root, card, here = sys.argv[1], sys.argv[2], os.path.realpath(sys.argv[3])
boxes = []
for d in glob.glob(f"{root}/*/ho3-{card}-*"):
    if os.path.realpath(d) == here or not os.path.isdir(f"{d}/ho3"):
        continue
    codes = glob.glob(f"{d}/ho3/HO3_EXIT_CODE.*")
    try:
        at = json.load(open(f"{d}/instance.json"))["at"]
        rc = open(codes[0]).read().strip() if len(codes) == 1 else "none"
    except (OSError, KeyError, ValueError):
        print(f"unreadable {os.path.basename(d)}"); sys.exit()
    boxes.append((at, rc))
n = 0
for _, rc in sorted(boxes, reverse=True):
    if rc != "19":
        break
    n += 1
print(n)
PYC
)
case "$TAIL19" in
  unreadable*) say "refusing: earlier $CARD box ${TAIL19#unreadable } ran this runner but its exit or time is unreadable"; exit 78;;
  ''|*[!0-9]*) say "refusing: cannot count earlier driver-peak refusals ($TAIL19)"; exit 78;;
esac
[ "$TAIL19" -lt 3 ] || { say "refusing: $TAIL19 consecutive driver-peak refusals (rc 19) on the $CARD card -- stop and report (Amendment 2)"; exit 78; }
if [ "${HO3_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash ho3_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/ho3"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; launch e4b $E4B_SHA (from $REPO); stacks are the runner's constants; receipts -> $RUN_DIR/ho3; deadline $DEADLINE"
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
LAUNCH_REPLY=$({ cat "$LANE_HELPER"; printf '%s\n' "cd $W || exit 20; nohup env $PASS bash ho3_run.sh > outer.log 2>&1 < /dev/null & child=\$!; identity=\$(lane_proc_snapshot \$child); end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat HO3_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; echo identity:\$identity; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124"; } | $SSH bash -s) || { say "start failed: child did not bind the nonce"; exit 21; }
printf '%s\n' "$LAUNCH_REPLY"
LANE_PID=$(printf '%s\n' "$LAUNCH_REPLY" | sed -n 's/^started:\([0-9][0-9]*\)$/\1/p')
[[ "$LANE_PID" =~ ^[1-9][0-9]*$ ]] || { say "start failed: malformed child PID"; exit 21; }
LANE_INITIAL=$(printf '%s\n' "$LAUNCH_REPLY" | sed -n 's/^identity://p')
say "lane identity: pid=$LANE_PID snapshot=${LANE_INITIAL:-unknown}"
# ---- heartbeat + liveness (e4b#641, tp4_drive.sh's): a stall is REPORTED, never acted on; two consecutive definite
# zeros for the lane's own process with no TP_DONE end the wait with their own code (25).
STALL_S=${HO3_STALL_S:-900}; HO3_MIN_PROGRESS_MB=${HO3_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$HO3_MIN_PROGRESS_MB
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
mkdir -p "$RUN_DIR/ho3" && [ -z "$(find "$RUN_DIR/ho3" -mindepth 1 -maxdepth 1 -print -quit)" ] || { say "fetch failed: local dir"; exit 22; }
# The checkpoints stay on the box (the hub cache, outside $W), and so do the trained adapters (tensors); the receipts,
# logs, phase stamps and the verdict travel. A fetch never imports links (Amendment 3): a link left on a rented box
# could point a store commit at a file on this controller, so --no-links skips any, and a link that still arrives refuses.
rsync -az --no-links -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" \
  --exclude 'run/adapter_*' --exclude '*.safetensors' --exclude '*.pt' \
  --exclude '__pycache__' --exclude 'venv*' --exclude '.cache' --exclude 'hf' \
  "root@$HOST:$W/" "$RUN_DIR/ho3/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/ho3" | wc -l | tr -d ' ') entries"
# A link that still arrives is recorded (path and target, never followed) in refused_links.txt and removed, so the run
# dir holds no link and its receipt can still be committed; then the fetch refuses (exit 22).
if [ -n "$(find "$RUN_DIR/ho3" -type l -print -quit)" ]; then
  find "$RUN_DIR/ho3" -type l -print0 | while IFS= read -r -d '' l; do
    printf '%q -> %q\n' "${l#"$RUN_DIR/ho3/"}" "$(readlink "$l")" >> "$RUN_DIR/ho3/refused_links.txt"; rm -f -- "$l"
  done
  say "fetch failed: $(wc -l < "$RUN_DIR/ho3/refused_links.txt" | tr -d ' ') symlink(s) arrived; recorded in refused_links.txt and removed"; exit 22
fi
[ "$(cat "$RUN_DIR/ho3/HO3_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/ho3/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no HO3_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/ho3/HO3_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/ho3/HO3_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
