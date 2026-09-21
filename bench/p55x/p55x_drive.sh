#!/bin/bash
# bench/p55x/p55x_drive.sh -- lane P55x, CONTROLLER side: the launcher's --command. Stages P39's harness pieces,
# P42's hook and this lane's p55x_run.sh, PROBES THE UPLOAD PATH, starts the lane detached under a fresh nonce,
# pulls the 15.2 GiB pack artifact in the background as soon as the box says it exists, polls TP_DONE, fetches
# receipts. One box, no arguments: the arms are fixed by bench/p55x/P55X-PREREG.md.
# Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p55x_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# step_decomp.py / k8_bake.py / calib.json are P39's, referenced rather than copied: a 3,300-line duplicate is
# not evidence, it is a second thing to keep in step. staged.sha256 pins them by path.
P39="$REPO/bench/p39"
STAGE="$HERE/p55x_run.sh $P39/step_decomp.py $P39/k8_bake.py $P39/calib.json $HERE/staged.sha256"
HOOK="$REPO/bench/p42/hook/usercustomize.py"
for f in $STAGE $HOOK; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in p55x_run.sh) src="$HERE/$name";; hook/*) src="$REPO/bench/p42/$name";; *) src="$P39/$name";; esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The e4b the BOX installs must be the e4b this driver ships from. A literal default here is exactly the unread
# constant the manifest design exists to remove: p39-box1b-3 installed a pre-fix commit while the checkout was
# three commits ahead, rebuilt the same assignment and VOIDed on its own check.
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
GNF4_SHA=${GNF4_SHA:?set GNF4_SHA to the grouped-nf4-gemm cut the arms run on (the commit e4b CI pins)}
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
RSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${P55X_POLL_S:-60}; W=/root/p55x
MIN_UP_MBS=${P55X_MIN_UP_MBS:-8}; PROBE_MB=${P55X_PROBE_MB:-256}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P55X_RUN_ID=$RUN_ID P55X_RUN_NONCE=$NONCE P55X_DEADLINE_EPOCH=$DEADLINE P55X_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA "
if [ "${P55X_DRIVE_DRYRUN:-0}" = "1" ]; then
  echo "DRYRUN stage -> root@$HOST:$W ; upload probe ${PROBE_MB}MB floor ${MIN_UP_MBS}MB/s ; start: env $PASS bash p55x_run.sh ;"
  echo "DRYRUN poll TP_DONE.$NONCE until $DEADLINE ; on ARTIFACT_READY.$NONCE pull $W/artifact1 -> $RUN_DIR/p55x-artifact ; fetch -> $RUN_DIR/p55x"
  exit 0
fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA (from $REPO); receipts -> $RUN_DIR/p55x; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/hook" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" && $SCP "$HOOK" "root@$HOST:$W/hook/" || { say "stage failed: scp"; exit 20; }

# ---- STOP-0, the half only the controller can measure: this lane's PRODUCT is 15.2 GiB that has to come OFF
# the box, and the rental pre-flight measures the box's DOWNLOAD. Consumer-hosted Vast boxes have been measured
# at 0.2-0.5 MB/s in the other direction, where the artifact would need 11 hours. Probe the real path -- a pull
# of a real file over the real transport -- and refuse before a single byte of checkpoint is fetched.
say "upload probe: ${PROBE_MB} MB off the box (floor ${MIN_UP_MBS} MB/s)"
$SSH "dd if=/dev/urandom of=$W/.uprobe.bin bs=1M count=$PROBE_MB status=none" || { say "upload probe: could not create the probe file"; exit 20; }
PROBE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/p55x-uprobe.XXXXXX"); T0=$(date +%s)
rsync -a --no-compress -e "$RSH" "root@$HOST:$W/.uprobe.bin" "$PROBE_DIR/" >/dev/null 2>&1; PRC=$?
T1=$(date +%s); $SSH "rm -f $W/.uprobe.bin" >/dev/null 2>&1
GOT=$(wc -c < "$PROBE_DIR/.uprobe.bin" 2>/dev/null || echo 0); rm -rf "$PROBE_DIR"
UP=$(python3 -c "
import sys
got, t0, t1 = $GOT, $T0, $T1
secs = max(1.0, t1 - t0)
print(f'{got / 1048576.0 / secs:.2f}')")
say "upload probe: rc=$PRC ${GOT} bytes in $((T1 - T0))s = ${UP} MB/s (floor ${MIN_UP_MBS})"
printf '{"probe_mb": %s, "bytes": %s, "seconds": %s, "mb_s": %s, "floor_mb_s": %s, "rsync_rc": %s}\n' \
  "$PROBE_MB" "$GOT" "$((T1 - T0))" "$UP" "$MIN_UP_MBS" "$PRC" > "${TMPDIR:-/tmp}/p55x-uprobe.json"
if [ "$PRC" != 0 ] || [ "$GOT" -lt $((PROBE_MB * 1048576)) ] \
   || [ "$(python3 -c "print(1 if float('$UP') < float('$MIN_UP_MBS') else 0)")" = 1 ]; then
  say "STOP-0: the box cannot push its own product (${UP} MB/s, rc=$PRC) -- refusing before any fetch; at this"
  say "        rate the 15.2 GiB artifact would need $(python3 -c "print(round(15.2*1024/max(float('$UP'),0.01)/60))") min and the lane's whole purpose is retaining those bytes"
  exit 13
fi

$SSH "cd $W || exit 20; nohup env $PASS bash p55x_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P55X_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }

# ---- heartbeat + liveness (e4b#641, ported from tp4_drive/p54_drive). A poller that waits for a marker cannot
# tell a dead lane from a slow one. Each poll asks the box for the last summary line, GPU util, workdir size and
# free disk, and whether the runner is still there. A stall is REPORTED, never acted on; two consecutive definite
# zeros on liveness with no TP_DONE end the wait with their own code, so a killed lane is not flattened into
# "ran out of clock".
STALL_S=${P55X_STALL_S:-1200}; MIN_PROGRESS_MB=${P55X_MIN_PROGRESS_MB:-16}
ART_LOCAL="$RUN_DIR/p55x-artifact"; ART_PID=""; ART_LOG="$RUN_DIR/p55x-artifact-fetch.log"
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$MIN_PROGRESS_MB
  consumed=0; grew=0
  if [ -n "$dfk_now" ] && [ -n "$dfk_prev" ] && [ "$dfk_now" -lt "$dfk_prev" ] 2>/dev/null; then consumed=$(( (dfk_prev - dfk_now) / 1024 )); fi
  if [ -n "$du_now" ] && [ -n "$du_prev" ] && [ "$du_now" -gt "$du_prev" ] 2>/dev/null; then grew=$((du_now - du_prev)); fi
  if [ "$consumed" -ge "$min_mb" ] || [ "$grew" -ge "$min_mb" ]; then
    if [ "$consumed" -ge "$grew" ]; then echo "working:-${consumed}M on disk"; else echo "working:+${grew}M in tree"; fi
    return 0
  fi
  if [ "$idle_s" -ge "$stall_s" ] && [ "${util:-0}" -eq 0 ] 2>/dev/null; then echo "stall:$idle_s"; fi
}
lane_dead() { [ "${1:-}" = "0" ] && [ "${2:-}" = "0" ] && echo dead; }
start_artifact_fetch() {   # the 15.2 GiB moves WHILE the gate arms and the second build run, not after them
  [ -n "$ART_PID" ] && return 0
  mkdir -p "$ART_LOCAL" || return 1
  say "ARTIFACT_READY seen -- pulling $W/artifact1 -> $ART_LOCAL in the background (log: $ART_LOG)"
  nohup rsync -a --partial --inplace --no-compress --timeout=600 -e "$RSH" \
    "root@$HOST:$W/artifact1/" "$ART_LOCAL/" > "$ART_LOG" 2>&1 &
  ART_PID=$!
}
say "lane started; polling TP_DONE every ${POLL}s (stall reported after ${STALL_S}s idle with no disk movement, never acted on)"
LAST=""; LAST_CHANGE=$(date +%s); LAST_DFK=""; LAST_DU=""; LAST_LIVE=""; LANE_DEAD=0
while :; do
  now=$(date +%s)
  $SSH "test -f $W/ARTIFACT_READY.$NONCE" 2>/dev/null && start_artifact_fetch
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f 'bash p55x_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: no 'bash p55x_run.sh' on the box for two consecutive polls and no TP_DONE -- the remote process exited without writing its markers; not waiting out the deadline"
    LANE_DEAD=1; break
  fi
  LAST_LIVE=$live
  verdict=$(progress_verdict "$((now - LAST_CHANGE))" "$STALL_S" "${util:-0}" "$dfk" "$LAST_DFK" "$duM" "$LAST_DU")
  LAST_DFK=$dfk; LAST_DU=$duM
  case "$verdict" in
    working:*) stall=" working (${verdict#working:} since last hb)" ;;
    stall:*)   stall=" STALL? (no summary change for ${verdict#stall:}s, GPU idle and no disk movement -- LOOK, do not kill)" ;;
    *)         stall="" ;;
  esac
  art=""; [ -n "$ART_PID" ] && { kill -0 "$ART_PID" 2>/dev/null && art=" | artifact-fetch $(du -sm "$ART_LOCAL" 2>/dev/null | cut -f1)M" || art=" | artifact-fetch done"; }
  say "hb: ${hb:-<no answer>} | left $((DEADLINE - now))s$stall$art"
  sleep "$POLL"
done
# one last look for the marker: a lane that dumped and died still has bytes worth saving
$SSH "test -f $W/ARTIFACT_READY.$NONCE" 2>/dev/null && start_artifact_fetch
if [ -n "$ART_PID" ]; then
  say "waiting on the artifact fetch (pid $ART_PID) -- these bytes are the lane's product"
  wait "$ART_PID"; ART_RC=$?
  say "artifact fetch rc=$ART_RC, $(du -sm "$ART_LOCAL" 2>/dev/null | cut -f1)M local"
else
  ART_RC=99; say "no ARTIFACT_READY marker was ever seen -- no pack bytes were retained (STOP-4)"
fi
rm -rf "$RUN_DIR/p55x" && mkdir -p "$RUN_DIR/p55x" || { say "fetch failed: local dir"; exit 22; }
# artifact1's layer payloads came down on their own path above; everything else is small.
rsync -az -e "$RSH" --exclude 'artifact1/payloads/layer_*' --exclude 'artifact2/payloads/layer_*' \
  --exclude work_qwen3 --exclude 'venv*' --exclude '.cache' --exclude '.uprobe.bin' \
  "root@$HOST:$W/" "$RUN_DIR/p55x/" || { say "fetch failed: rsync"; exit 22; }
cp "${TMPDIR:-/tmp}/p55x-uprobe.json" "$RUN_DIR/p55x/upload_probe.json" 2>/dev/null
say "fetched $(ls "$RUN_DIR/p55x" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p55x/P55X_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p55x/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P55X_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p55x/P55X_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
# The lane's product is the BYTES plus the verdict. A run whose gate read cleanly but whose artifact did not
# come off the box is not a success: it is bo6c again (#405), and the marker has to say so.
[ "$ART_RC" = 0 ] || { say "lane rc=$RC but the artifact fetch did not complete (rc=$ART_RC) -- the pack bytes were NOT retained; the gate verdict is an observation, not a licence basis (STOP-4)"; exit 26; }
[ -f "$RUN_DIR/p55x/P55X_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0; artifact at $ART_LOCAL -- verify and place it with bench/p55x/p55x_publish.sh"; exit 0
