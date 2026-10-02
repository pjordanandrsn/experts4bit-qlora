#!/bin/bash
# bench/p90/p90_drive.sh -- lane P90, CONTROLLER side: the launcher's --command (bench/p90/PREREG-p90.md; e4b#564).
# Derived from bench/p89/p89_drive.sh by named substitutions: it stages P90's runner and reducer, P58's harness pieces
# (bench/p39 step_decomp.py / k8_bake.py / calib.json, bench/p42's hook), P42's census parser, the premise test
# (tests/test_k21_row_exact_gpu.py) and P44's KL instrument (bench/p44 serve_stack.py / kl_serve.py, bench/kl_fidelity.py
# / kl_paths.py / kl_prompts.py) -- all referenced, never copied -- starts the lane detached under a fresh nonce; polls
# TP_DONE.<nonce> with the tp4-style heartbeat/liveness check; fetches receipts -- never the checkpoint, the arena or the
# KL reference cache. Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p90_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# P58's staging: P39's step_decomp.py / k8_bake.py / calib.json, P42's hook (under hook/) and census parser; the premise
# test from tests/. staged.sha256 pins every one by the name the box sees.
P39="$REPO/bench/p39"; P42="$REPO/bench/p42"; P44="$REPO/bench/p44"; KL="$REPO/bench"; TESTS="$REPO/tests"
STAGE="$HERE/p90_run.sh $HERE/p90_reduce.py $P42/p42_reduce.py $TESTS/test_k21_row_exact_gpu.py $P39/step_decomp.py $P39/k8_bake.py $P39/calib.json $P44/serve_stack.py $P44/kl_serve.py $KL/kl_fidelity.py $KL/kl_paths.py $KL/kl_prompts.py $HERE/staged.sha256"
HOOK="$P42/hook/usercustomize.py"
for f in $STAGE $HOOK; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them; resolve each name to its source and compare hashes
# (the same case as tests/test_p90_staged_pin.py, which runs this check in CI where it costs nothing).
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    p90_run.sh|p90_reduce.py) src="$HERE/$name";;
    p42_reduce.py) src="$P42/$name";;
    test_k21_row_exact_gpu.py) src="$TESTS/$name";;
    serve_stack.py|kl_serve.py) src="$P44/$name";;
    kl_fidelity.py|kl_paths.py|kl_prompts.py) src="$KL/$name";;
    hook/usercustomize.py) src="$P42/hook/usercustomize.py";;
    *) src="$P39/$name";;
  esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The launch commit is the e4b this driver ships from (the launcher proved it is the manifest's heads.e4b): it stages
# the kit, and the box installs it (it carries the K21 route, #816). Derived, never a literal, and
# refused on a dirty tree (p39-box1b-3's lesson).
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
# grouped-nf4-gemm's pin (main with K21 + its masked tail) is a constant in the runner; the box installs e4b at THIS commit.
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${P90_POLL_S:-60}; W=/root/p90
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P90_RUN_ID=$RUN_ID P90_RUN_NONCE=$NONCE P90_DEADLINE_EPOCH=$DEADLINE P90_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA"
# Only host-limit knobs and the proving-run switch travel: a rental is the registered run, so the knobs that change WHAT
# is measured (model, rows) are not forwarded -- a rehearsal sets them on its own box and the runner marks it
# REHEARSAL. P90_PROVE=1 is the proving rental the pre-registration names (no model). P90_ARMS (speed | quality,
# amendment 1) picks the run's half of the reading; the box refuses any other value.
for v in P90_PROVE P90_ARMS; do [ -n "${!v:-}" ] && PASS="$PASS $v=$(printf %q "${!v}")"; done
if [ "${P90_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p90_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p90"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; launch e4b $E4B_SHA (from $REPO); stacks are the runner's constants; receipts -> $RUN_DIR/p90; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/hook /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" && $SCP "$HOOK" "root@$HOST:$W/hook/" || { say "stage failed: scp"; exit 20; }
if [ -s "$HF_TOKEN_FILE" ]; then   # authenticated pulls (unauthenticated shards throttle); the token never appears on a command line
  $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
  say "hf token staged"
else
  say "no hf token file at $HF_TOKEN_FILE -- pulls run unauthenticated (openai/gpt-oss-20b is ungated)"
fi
$SSH "cd $W || exit 20; nohup env $PASS bash p90_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P90_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
# ---- heartbeat + liveness (e4b#641, tp4_drive.sh's): a stall is REPORTED, never acted on; two consecutive definite
# zeros for the lane's own process with no TP_DONE end the wait with their own code (25).
STALL_S=${P90_STALL_S:-900}; P90_MIN_PROGRESS_MB=${P90_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$P90_MIN_PROGRESS_MB
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
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f 'bash p90_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: no 'bash p90_run.sh' on the box for two consecutive polls and no TP_DONE -- not waiting out the deadline"
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
rm -rf "$RUN_DIR/p90" && mkdir -p "$RUN_DIR/p90" || { say "fetch failed: local dir"; exit 22; }
# The checkpoint, the NF4 arena, the KL reference cache and the gnf4 clone stay on the box; bake.json travels
# (p57-5090-1 lost its only failure text to a blanket exclusion).
rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" \
  --include 'work_gptoss/' --include 'work_gptoss/bake.json' --exclude 'work_gptoss/*' --exclude 'refcache_gptoss*' --exclude 'artifact/' --exclude 'gnf4/' \
  --exclude '__pycache__' --exclude 'venv*' --exclude '.cache' --exclude 'hf' \
  "root@$HOST:$W/" "$RUN_DIR/p90/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p90" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p90/P90_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p90/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P90_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p90/P90_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p90/P90_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
