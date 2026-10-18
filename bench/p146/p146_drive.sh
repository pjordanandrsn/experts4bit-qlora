#!/bin/bash
# bench/p146/p146_drive.sh -- lane P146, CONTROLLER side: the launcher's --command (bench/p146/PREREG-p146.md).
# Derived from bench/p144/p144_drive.sh by named substitutions:
#   * it stages P146's runner, reducer, box and dry fixture at staged.sha256 (no borrowed pieces);
#   * the e4b the box installs is the registered pin (E4B_PIN), not the launch tree's HEAD; the DRY box gets that pin's
#     tree (git archive) as e4bsrc/, since the dry box installs nothing;
#   * the run id must be one the registration declares: p146-4090-1..8 (bench/p146/PREREG-p146.md; -5..-8 are spares). No paid proof: the
#     $0 dry run is the shakedown. Any other id is refused;
#   * P146_DRY=1 is the $0 dry run: run ids p146-dry-<n>, the box reached through P146_BOX_EXEC (a local command that runs
#     one shell command on the dry box, stdin passed through) with tar for staging and fetching, and P146_HOME naming the
#     dry box's lane directory. Everything else -- staging, the nonce handshake, the heartbeat, the fetch and the exit
#     codes -- is the paid path's.
# It starts the lane detached under a fresh nonce, polls TP_DONE.<nonce> with the tp4-style heartbeat/liveness check and
# fetches receipts -- never the checkpoint or the e4b tree. Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p146_drive] $*"; }
E4B_PIN=4917506a22d960a6586bd5cefe9a0deca662fc46      # p146_run.sh refuses any other
DRY=${P146_DRY:-0}
if [ "$DRY" = 1 ]; then
  for v in E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID P146_BOX_EXEC; do
    [ -n "${!v:-}" ] || { say "refusing: $v is not set -- P146_DRY=1 needs it"; exit 78; }
  done
else
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
fi
# ---- the declared run ids: the reading, its one rerun; the dry run's own
RUN_ID=$E4B_RENT_RUN_ID
case "$RUN_ID" in
  p146-4090-[1-8]) [ "$DRY" != 1 ] || { say "refusing: $RUN_ID is a reading's id"; exit 78; };;
  p146-dry-[0-9]|p146-dry-[0-9][0-9]) [ "$DRY" = 1 ] || { say "refusing: $RUN_ID is a dry run's id (P146_DRY=1 only)"; exit 78; };;
  *) say "refusing: run id $RUN_ID is not declared (p146-4090-1..8; p146-dry-<n> dry)"; exit 78;;
esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# P137 Amendment 1: the training rows travel from the controller (tatsu-lab/alpaca's train parquet @dce01c9b as
# alpaca.parquet, outside the repo; train.py takes its train[:2000] and train[2000:2064]), checked
# here and again on the box against the registered sha256 -- the box downloads no dataset.
DATA_SHA=06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073
P146_DATA=${P146_DATA:-$HOME/p146-data/alpaca.parquet}
[ "$(basename "$P146_DATA")" = alpaca.parquet ] || { say "refusing: P146_DATA must be a file named alpaca.parquet"; exit 78; }
STAGE="$HERE/p146_run.sh $HERE/p146_reduce.py $HERE/p146_box.py $HERE/p146_tiny.py $HERE/staged.sha256 $P146_DATA"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them; resolve each name to its source and compare hashes
# (the same case as tests/test_p146_staged_pin.py, which runs this check in CI where it costs nothing).
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    p146_run.sh|p146_reduce.py|p146_box.py|p146_tiny.py) src="$HERE/$name";;
    *) say "refusing: staged.sha256 names $name, which this driver does not stage"; exit 78;;
  esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The e4b the box installs is the registered pin; an E4B_SHA from the caller must be that pin.
[ -z "${E4B_SHA:-}" ] || [ "$E4B_SHA" = "$E4B_PIN" ] || { say "refusing: E4B_SHA=$E4B_SHA is not the registered pin $E4B_PIN"; exit 78; }
E4B_SHA=$E4B_PIN
RUN_DIR=$E4B_RENT_RUN_DIR; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
POLL=${P146_POLL_S:-60}; W=${P146_HOME:-/root/p146}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
# ---- the transport: ssh/scp/rsync to the rented box, or P146_BOX_EXEC + tar to the dry box
if [ "$DRY" = 1 ]; then
  WHERE="dry box via $P146_BOX_EXEC"
  box(){ "$P146_BOX_EXEC" "$1"; }
  box_stage(){ local args=() f; for f in $STAGE; do args+=(-C "$(dirname "$f")" "$(basename "$f")"); done
    tar -cf - "${args[@]}" | box "tar -C $W -xf -" || return 1
    git -C "$REPO" archive --format=tar --prefix=e4bsrc/ "$E4B_SHA" | box "tar -C $W -xf -"; }
  box_fetch(){ box "cd $W && tar -cf - --exclude=./e4bsrc --exclude=./p146-tiny --exclude='./out_*' --exclude=__pycache__ --exclude=./.cache --exclude=./hf ."       | tar -C "$1" -xif -; }
else
  HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; WHERE="$HOST:$PORT"
  SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
  SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
  box(){ $SSH "$1"; }
  box_stage(){ $SCP $STAGE "root@$HOST:$W/"; }
  # The checkpoint stays on the box (the hub cache, outside $W), and so do the e4b clone and the fixture; records, logs,
  # the data slice and the verdict travel.
  box_fetch(){ rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT"       --exclude '__pycache__' --exclude '.cache' --exclude 'hf' --exclude 'e4bsrc' --exclude 'p146-tiny' --exclude 'out_*'       "root@$HOST:$W/" "$1/"; }
fi
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P146_RUN_ID=$RUN_ID P146_RUN_NONCE=$NONCE P146_DEADLINE_EPOCH=$DEADLINE P146_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA"
# Only the dry-run switch and its lane directory travel: a rental is the registered run, so the knobs that change WHAT is
# measured are not forwarded -- a rehearsal sets them on its own box and the runner marks it REHEARSAL.
for v in P146_DRY P146_HOME; do [ -n "${!v:-}" ] && PASS="$PASS $v=$(printf %q "${!v}")"; done
if [ "${P146_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> $WHERE:$W ; start: env $PASS bash p146_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p146"; exit 0; fi
# P137 Amendment 1: the data slice's sha256, on the controller, before anything is staged (the box checks it again)
got=$(sha_of "$P146_DATA")
[ "$got" = "$DATA_SHA" ] || { say "refusing: $P146_DATA is $got, the registered alpaca parquet is $DATA_SHA"; exit 78; }
say "run $RUN_ID nonce=$NONCE -> $WHERE; e4b $E4B_SHA (the registered pin); stacks are the runner's constants; receipts -> $RUN_DIR/p146; deadline $DEADLINE"
box "rm -rf -- $W && mkdir -p $W/logs /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
box_stage || { say "stage failed: copy"; exit 20; }
if [ "$DRY" = 1 ]; then
  say "dry box: no hf token staged (the fixture's tokenizer and the texts are ungated)"
elif [ -s "$HF_TOKEN_FILE" ]; then   # authenticated pulls (unauthenticated shards throttle); the token never appears on a command line
  $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
  say "hf token staged"
else
  say "no hf token file at $HF_TOKEN_FILE -- pulls run unauthenticated (both checkpoints are ungated)"
fi
box "cd $W || exit 20; nohup env $PASS bash p146_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P146_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
# ---- heartbeat + liveness (e4b#641, tp4_drive.sh's): a stall is REPORTED, never acted on; two consecutive definite
# zeros for the lane's own process with no TP_DONE end the wait with their own code (25).
STALL_S=${P146_STALL_S:-900}; P146_MIN_PROGRESS_MB=${P146_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$P146_MIN_PROGRESS_MB
  consumed=0; grew=0
  if [ -n "$dfk_now" ] && [ -n "$dfk_prev" ] && [ "$dfk_now" -lt "$dfk_prev" ] 2>/dev/null; then consumed=$(( (dfk_prev - dfk_now) / 1024 )); fi
  if [ -n "$du_now" ] && [ -n "$du_prev" ] && [ "$du_now" -gt "$du_prev" ] 2>/dev/null; then grew=$((du_now - du_prev)); fi
  if [ "$consumed" -ge "$min_mb" ] || [ "$grew" -ge "$min_mb" ]; then
    if [ "$consumed" -ge "$grew" ]; then echo "fetching:-${consumed}M on disk"; else echo "fetching:+${grew}M in tree"; fi
    return 0
  fi
  if [ "$idle_s" -ge "$stall_s" ] && [ "${util:-0}" -eq 0 ] 2>/dev/null; then echo "stall:$idle_s"; fi
}
lane_dead() { [ "${1:-}" = "0" ] && [ "${2:-}" = "0" ] && echo dead; }
say "lane started; polling TP_DONE every ${POLL}s with a heartbeat (stall reported after ${STALL_S}s; never acted on; a lane whose process is gone for two polls ends the wait)"
LAST=""; LAST_CHANGE=$(date +%s); LAST_DFK=""; LAST_DU=""; LAST_LIVE=""; LANE_DEAD=0
while :; do
  now=$(date +%s)
  box "test -f $W/TP_DONE.$NONCE" 2>/dev/null < /dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$(box "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f 'bash p146_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null < /dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: no 'bash p146_run.sh' on the box for two consecutive polls and no TP_DONE -- not waiting out the deadline"
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
rm -rf "$RUN_DIR/p146" && mkdir -p "$RUN_DIR/p146" || { say "fetch failed: local dir"; exit 22; }
box_fetch "$RUN_DIR/p146" < /dev/null || { say "fetch failed"; exit 22; }
say "fetched $(ls "$RUN_DIR/p146" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p146/P146_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p146/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P146_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p146/P146_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p146/P146_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
