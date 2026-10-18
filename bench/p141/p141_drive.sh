#!/bin/bash
# bench/p141/p141_drive.sh -- lane P141, CONTROLLER side: the launcher's --command (bench/p141/PREREG-p141.md).
# Derived from bench/p134/p134_drive.sh by named substitutions: P141's runner, reducer, box and phase-1 selection, with
# P133's dry fixture (bench/p133/p133_tiny.py) and P116's borrowed pieces, all at staged.sha256; the declared run ids
# p141-5090-1 (the reading) and p141-5090-2 (a spare: a refusal before create or a host fault only), and p141-dry-<n>
# under P141_DRY=1. Any other id is refused. e4b is this checkout's HEAD (the launch commit; a dirty tree is refused),
# P135's rule. The transport, nonce handshake, heartbeat, fetch and exit codes are P134's. Nothing here creates, destroys
# or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p141_drive] $*"; }
DRY=${P141_DRY:-0}
if [ "$DRY" = 1 ]; then
  for v in E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID P141_BOX_EXEC; do
    [ -n "${!v:-}" ] || { say "refusing: $v is not set -- P141_DRY=1 needs it"; exit 78; }
  done
else
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
fi
# ---- the declared run ids: the reading and its spare; the dry run's own
RUN_ID=$E4B_RENT_RUN_ID
case "$RUN_ID" in
  p141-5090-1|p141-5090-2) [ "$DRY" != 1 ] || { say "refusing: $RUN_ID is a reading's id"; exit 78; };;
  p141-dry-[0-9]|p141-dry-[0-9][0-9]) [ "$DRY" = 1 ] || { say "refusing: $RUN_ID is a dry run's id (P141_DRY=1 only)"; exit 78; };;
  *) say "refusing: run id $RUN_ID is not declared (p141-5090-1, p141-5090-2; p141-dry-<n> dry)"; exit 78;;
esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
TESTS="$REPO/tests"; P133="$REPO/bench/p133"; P39="$REPO/bench/p39"; P115="$REPO/bench/p115"; P109="$REPO/bench/p109"; P110="$REPO/bench/p110"; P108="$REPO/bench/p108"; P97="$REPO/bench/p97"
STAGE="$HERE/p141_run.sh $HERE/p141_reduce.py $HERE/p141_box.py $HERE/p141_select.py $P133/p133_tiny.py $P115/p115_reduce.py $P115/p115_quality.py $P109/p109_box.py $P110/p110_box.py $P108/p108_box.py $P97/p97_box.py $P39/k8_bake.py $P39/calib.json $TESTS/test_decode_graph_buckets.py $TESTS/test_kv_step_select.py $TESTS/test_gemv_bw_served_gpu.py $HERE/staged.sha256"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them; resolve each name to its source and compare hashes
# (the same case as tests/test_p141_staged_pin.py, which runs this check in CI where it costs nothing).
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    p141_run.sh|p141_reduce.py|p141_box.py|p141_select.py) src="$HERE/$name";;
    p133_tiny.py) src="$P133/$name";;
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
# The e4b the box installs is the launch commit: this checkout's HEAD, clean (P135's rule).
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
RUN_DIR=$E4B_RENT_RUN_DIR; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
POLL=${P141_POLL_S:-60}; W=/root/p141
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
# ---- the transport: ssh/scp/rsync to the rented box, or P141_BOX_EXEC + tar to the dry box
if [ "$DRY" = 1 ]; then
  WHERE="dry box via $P141_BOX_EXEC"
  box(){ "$P141_BOX_EXEC" "$1"; }
  box_stage(){ local args=() f; for f in $STAGE; do args+=(-C "$(dirname "$f")" "$(basename "$f")"); done
    tar -cf - "${args[@]}" | box "tar -C $W -xf -"; }
  # the rsync rule below, as two tar streams: everything but work/ and the excluded trees, then each work/<family>/bake.json
  box_fetch(){ box "cd $W && tar -cf - --exclude=./work --exclude=./gnf4src --exclude=./fam --exclude=__pycache__ --exclude='./venv*' --exclude=./.cache --exclude=./hf . && { set -- work/*/bake.json; [ ! -f \"\$1\" ] || tar -cf - \"\$@\"; }" \
      | tar -C "$1" -xif -; }
else
  HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; WHERE="$HOST:$PORT"
  SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
  SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
  box(){ $SSH "$1"; }
  box_stage(){ $SCP $STAGE "root@$HOST:$W/"; }
  # The checkpoints stay on the box (the hub cache, outside $W), and so do the NF4 snapshots, the arenas and the reference
  # log-probs (work/<family>/: only its bake.json travels); records, logs, the prompts and the verdicts travel. The gnf4
  # source clone (gnf4src/, for the kernel's own contract test) and the dry fixtures (fam/) stay too.
  box_fetch(){ rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" \
      --include 'work/' --include 'work/*/' --include 'work/*/bake.json' --exclude 'work/*/*' --exclude 'work/*' \
      --exclude '__pycache__' --exclude 'venv*' --exclude '.cache' --exclude 'hf' --exclude 'gnf4src' --exclude 'fam' \
      "root@$HOST:$W/" "$1/"; }
fi
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P141_RUN_ID=$RUN_ID P141_RUN_NONCE=$NONCE P141_DEADLINE_EPOCH=$DEADLINE P141_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA"
# Only the dry-run switch travels: a rental is the registered run, so the knobs that change WHAT is measured are not
# forwarded -- a rehearsal sets them on its own box and the runner marks it REHEARSAL.
for v in P141_DRY; do [ -n "${!v:-}" ] && PASS="$PASS $v=$(printf %q "${!v}")"; done
if [ "${P141_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> $WHERE:$W ; start: env $PASS bash p141_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p141"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $WHERE; launch e4b $E4B_SHA (from $REPO); stacks are the runner's constants; receipts -> $RUN_DIR/p141; deadline $DEADLINE"
box "rm -rf -- $W && mkdir -p $W/logs /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
box_stage || { say "stage failed: copy"; exit 20; }
if [ "$DRY" = 1 ]; then
  say "dry box: no hf token staged (the fixture's tokenizer and the texts are ungated)"
elif [ -s "$HF_TOKEN_FILE" ]; then   # authenticated pulls (unauthenticated shards throttle); the token never appears on a command line
  $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
  say "hf token staged"
else
  say "no hf token file at $HF_TOKEN_FILE -- pulls run unauthenticated (the checkpoint is ungated)"
fi
box "cd $W || exit 20; nohup env $PASS bash p141_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P141_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
# ---- heartbeat + liveness (e4b#641, tp4_drive.sh's): a stall is REPORTED, never acted on; two consecutive definite
# zeros for the lane's own process with no TP_DONE end the wait with their own code (25).
STALL_S=${P141_STALL_S:-900}; P141_MIN_PROGRESS_MB=${P141_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$P141_MIN_PROGRESS_MB
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
  hb=$(box "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f 'bash p141_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null < /dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: no 'bash p141_run.sh' on the box for two consecutive polls and no TP_DONE -- not waiting out the deadline"
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
rm -rf "$RUN_DIR/p141" && mkdir -p "$RUN_DIR/p141" || { say "fetch failed: local dir"; exit 22; }
box_fetch "$RUN_DIR/p141" < /dev/null || { say "fetch failed"; exit 22; }
say "fetched $(ls "$RUN_DIR/p141" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p141/P141_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p141/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P141_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p141/P141_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ "$RC" = 31 ] && { say "FIT VOID (rc=31): Mixtral's W1 did not fit the registered server -- VOID with no rerun"; exit 31; }
[ -f "$RUN_DIR/p141/P141_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
