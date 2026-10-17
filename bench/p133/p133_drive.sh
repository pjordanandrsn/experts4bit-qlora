#!/bin/bash
# bench/p133/p133_drive.sh -- lane P133, CONTROLLER side: the launcher's --command (bench/p133/PREREG-p133.md).
# Derived from bench/p116/p116_drive.sh by named substitutions:
#   * it stages P133's runner, reducer, box and dry fixture with P116's borrowed pieces (P115's reducer and quality box,
#     P109's, P110's, P108's and P97's boxes, P39's bake and calibration, the premise tests), all at staged.sha256;
#   * the e4b the box installs is the registered pin (E4B_PIN), not the launch tree's HEAD;
#   * the run id must be one the registration declares: p133-4090-prove-1 (P133_PROVE=1), then p133-4090-1 and its one
#     rerun p133-4090-2. Any other id is refused;
#   * P133_DRY=1 is the $0 dry run: run ids p133-dry-<n>, and the box is reached through P133_BOX_EXEC (a local command
#     that runs one shell command on the dry box, stdin passed through) with tar for staging and fetching, in place of
#     ssh, scp and rsync. Everything else -- staging, the nonce handshake, the heartbeat, the fetch and the exit codes --
#     is the paid path's.
# It starts the lane detached under a fresh nonce, polls TP_DONE.<nonce> with the tp4-style heartbeat/liveness check and
# fetches receipts -- never the checkpoint, the arena or the reference log-probs. Nothing here creates, destroys or
# approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p133_drive] $*"; }
E4B_PIN=c8d7e629e4c8431fc86970cf7195d8e67f72e22a      # p133_run.sh refuses any other
DRY=${P133_DRY:-0}; PROVE=${P133_PROVE:-0}
if [ "$DRY" = 1 ]; then
  for v in E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID P133_BOX_EXEC; do
    [ -n "${!v:-}" ] || { say "refusing: $v is not set -- P133_DRY=1 needs it"; exit 78; }
  done
else
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
fi
# ---- the declared run ids: the proof, the reading, its one rerun; the dry run's own
RUN_ID=$E4B_RENT_RUN_ID
case "$RUN_ID" in
  p133-4090-prove-1) [ "$PROVE" = 1 ] && [ "$DRY" != 1 ] || { say "refusing: $RUN_ID is the proof's id (P133_PROVE=1 only)"; exit 78; };;
  p133-4090-1|p133-4090-2) [ "$PROVE" != 1 ] && [ "$DRY" != 1 ] || { say "refusing: $RUN_ID is a reading's id"; exit 78; };;
  p133-dry-[0-9]|p133-dry-[0-9][0-9]) [ "$DRY" = 1 ] && [ "$PROVE" != 1 ] || { say "refusing: $RUN_ID is a dry run's id (P133_DRY=1 only)"; exit 78; };;
  *) say "refusing: run id $RUN_ID is not declared (p133-4090-prove-1, p133-4090-1, p133-4090-2; p133-dry-<n> dry)"; exit 78;;
esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
TESTS="$REPO/tests"; P39="$REPO/bench/p39"; P115="$REPO/bench/p115"; P109="$REPO/bench/p109"; P110="$REPO/bench/p110"; P108="$REPO/bench/p108"; P97="$REPO/bench/p97"
STAGE="$HERE/p133_run.sh $HERE/p133_reduce.py $HERE/p133_box.py $HERE/p133_tiny.py $P115/p115_reduce.py $P115/p115_quality.py $P109/p109_box.py $P110/p110_box.py $P108/p108_box.py $P97/p97_box.py $P39/k8_bake.py $P39/calib.json $TESTS/test_decode_graph_buckets.py $TESTS/test_kv_step_select.py $TESTS/test_gemv_bw_served_gpu.py $HERE/staged.sha256"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them; resolve each name to its source and compare hashes
# (the same case as tests/test_p133_staged_pin.py, which runs this check in CI where it costs nothing).
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    p133_run.sh|p133_reduce.py|p133_box.py|p133_tiny.py) src="$HERE/$name";;
    p115_reduce.py|p115_quality.py) src="$P115/$name";;
    p109_box.py) src="$P109/$name";;
    p110_box.py) src="$P110/$name";;
    p108_box.py) src="$P108/$name";;
    p97_box.py) src="$P97/$name";;
    k8_bake.py|calib.json) src="$P39/$name";;
    test_decode_graph_buckets.py|test_kv_step_select.py|test_gemv_bw_served_gpu.py) src="$TESTS/$name";;
    *) say "refusing: staged.sha256 names $name, which this driver does not stage"; exit 78;;
  esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The e4b the box installs is the registered pin; an E4B_SHA from the caller must be that pin.
[ -z "${E4B_SHA:-}" ] || [ "$E4B_SHA" = "$E4B_PIN" ] || { say "refusing: E4B_SHA=$E4B_SHA is not the registered pin $E4B_PIN"; exit 78; }
E4B_SHA=$E4B_PIN
RUN_DIR=$E4B_RENT_RUN_DIR; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
POLL=${P133_POLL_S:-60}; W=/root/p133
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
# ---- the transport: ssh/scp/rsync to the rented box, or P133_BOX_EXEC + tar to the dry box
if [ "$DRY" = 1 ]; then
  WHERE="dry box via $P133_BOX_EXEC"
  box(){ "$P133_BOX_EXEC" "$1"; }
  box_stage(){ local args=() f; for f in $STAGE; do args+=(-C "$(dirname "$f")" "$(basename "$f")"); done
    tar -cf - "${args[@]}" | box "tar -C $W -xf -"; }
  # the rsync rule below, as two tar streams: everything but work/ and the excluded trees, then work/bake.json
  box_fetch(){ box "cd $W && tar -cf - --exclude=./work --exclude=./gnf4src --exclude=./p133-tiny --exclude=__pycache__ --exclude='./venv*' --exclude=./.cache --exclude=./hf . && { [ ! -f work/bake.json ] || tar -cf - work/bake.json; }" \
      | tar -C "$1" -xif -; }
else
  HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; WHERE="$HOST:$PORT"
  SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
  SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
  box(){ $SSH "$1"; }
  box_stage(){ $SCP $STAGE "root@$HOST:$W/"; }
  # The checkpoint stays on the box (the hub cache, outside $W), and so do the NF4 snapshot, the arena and the reference
  # log-probs (work/: only its bake.json travels); records, logs, the prompts and the verdict travel. The gnf4 source
  # clone (gnf4src/, for the kernel's own contract test) stays too.
  box_fetch(){ rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" \
      --include 'work/' --include 'work/bake.json' --exclude 'work/*' \
      --exclude '__pycache__' --exclude 'venv*' --exclude '.cache' --exclude 'hf' --exclude 'gnf4src' --exclude 'p133-tiny' \
      "root@$HOST:$W/" "$1/"; }
fi
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P133_RUN_ID=$RUN_ID P133_RUN_NONCE=$NONCE P133_DEADLINE_EPOCH=$DEADLINE P133_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA"
# Only the proving-run and dry-run switches travel: a rental is the registered run, so the knobs that change WHAT is
# measured are not forwarded -- a rehearsal sets them on its own box and the runner marks it REHEARSAL.
for v in P133_PROVE P133_DRY; do [ -n "${!v:-}" ] && PASS="$PASS $v=$(printf %q "${!v}")"; done
if [ "${P133_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> $WHERE:$W ; start: env $PASS bash p133_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p133"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $WHERE; e4b $E4B_SHA (the registered pin); stacks are the runner's constants; receipts -> $RUN_DIR/p133; deadline $DEADLINE"
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
box "cd $W || exit 20; nohup env $PASS bash p133_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P133_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
# ---- heartbeat + liveness (e4b#641, tp4_drive.sh's): a stall is REPORTED, never acted on; two consecutive definite
# zeros for the lane's own process with no TP_DONE end the wait with their own code (25).
STALL_S=${P133_STALL_S:-900}; P133_MIN_PROGRESS_MB=${P133_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$P133_MIN_PROGRESS_MB
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
  hb=$(box "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f 'bash p133_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null < /dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: no 'bash p133_run.sh' on the box for two consecutive polls and no TP_DONE -- not waiting out the deadline"
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
rm -rf "$RUN_DIR/p133" && mkdir -p "$RUN_DIR/p133" || { say "fetch failed: local dir"; exit 22; }
box_fetch "$RUN_DIR/p133" < /dev/null || { say "fetch failed"; exit 22; }
say "fetched $(ls "$RUN_DIR/p133" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p133/P133_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p133/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P133_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p133/P133_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ "$RC" = 31 ] && { say "FIT VOID (rc=31): W1 does not fit the default server on this card -- no arm ran; no rerun is registered"; exit 31; }
[ -f "$RUN_DIR/p133/P133_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
