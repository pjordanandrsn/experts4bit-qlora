#!/bin/bash
# bench/p84/p84_drive.sh -- lane P84, CONTROLLER side: the launcher's --command (bench/p84/PREREG-p84.md; e4b#674).
# Derived from bench/p83/p83_drive.sh by named substitutions: it stages P84's runner and reducer, P39's k8_bake.py /
# calib.json, and the two harnesses -- o/ = P70's (bench/p39/step_decomp.py, bench/p42/hook) and n/ = P82's
# (bench/p81/step_decomp.py, bench/p81/hook), all referenced, never copied -- starts the lane detached under a fresh nonce;
# polls TP_DONE.<nonce> with the tp4-style heartbeat/liveness check; fetches receipts -- never the checkpoint, the arenas
# or any pack's payloads (the runner copies every pack manifest to manifests/). Nothing here creates, destroys or
# approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p84_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# Harness o/ is P70's (bench/p39/step_decomp.py, bench/p42/hook), harness n/ is P82's (bench/p81/step_decomp.py,
# bench/p81/hook), each referenced unchanged; k8_bake.py / calib.json are P39's. staged.sha256 pins every one by the name
# the box sees (o/..., n/...).
P39="$REPO/bench/p39"; P42="$REPO/bench/p42"; P81="$REPO/bench/p81"
STAGE="$HERE/p84_run.sh $HERE/p84_reduce.py $P39/k8_bake.py $P39/calib.json $HERE/staged.sha256"
O_FILES="$P39/step_decomp.py $P42/hook/usercustomize.py"
N_FILES="$P81/step_decomp.py $P81/hook/usercustomize.py"
for f in $STAGE $O_FILES $N_FILES; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them; resolve each name to its source and compare hashes
# (the same case as tests/test_p84_staged_pin.py, which runs this check in CI where it costs nothing).
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    p84_run.sh|p84_reduce.py) src="$HERE/$name";;
    o/step_decomp.py) src="$P39/step_decomp.py";;
    o/hook/usercustomize.py) src="$P42/hook/usercustomize.py";;
    n/step_decomp.py) src="$P81/step_decomp.py";;
    n/hook/usercustomize.py) src="$P81/hook/usercustomize.py";;
    *) src="$P39/$name";;
  esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
# The e4b the BOX installs is the e4b this driver ships from (the launcher proved it is the manifest's heads.e4b);
# derived, never a literal, and refused on a dirty tree (p39-box1b-3's lesson).
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
# The kernel cut of BOTH stacks is a release-tag commit: v0.33.7 (9407d49). Passed explicitly, never defaulted. Stack A's
# e4b pin (c77aab6, 0.37.4) is a constant in the runner; stack B's e4b is the launch commit.
GNF4_SHA=${GNF4_SHA:?set GNF4_SHA to the registered grouped-nf4-gemm cut of both stacks (PREREG-p84: 9407d499a4d1e0fe8c22b050878a9f869b385b45, v0.33.7)}
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; exit 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${P84_POLL_S:-60}; W=/root/p84
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P84_RUN_ID=$RUN_ID P84_RUN_NONCE=$NONCE P84_DEADLINE_EPOCH=$DEADLINE P84_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA"
# Only host-limit knobs and the proving-run switch travel: a rental is the registered run, so the knobs that change WHAT
# is measured (model, pack, rows, calibration) are not forwarded -- a rehearsal sets them on its own box and the runner
# marks it REHEARSAL. P84_PROVE=1 is the proving rental the pre-registration names (no model, no pack).
for v in P84_PROVE; do [ -n "${!v:-}" ] && PASS="$PASS $v=$(printf %q "${!v}")"; done
if [ "${P84_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p84_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p84"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA (from $REPO); gnf4 $GNF4_SHA; receipts -> $RUN_DIR/p84; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/o/hook $W/n/hook /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SCP "$P39/step_decomp.py" "root@$HOST:$W/o/" && $SCP "$P42/hook/usercustomize.py" "root@$HOST:$W/o/hook/" || { say "stage failed: scp (harness o)"; exit 20; }
$SCP "$P81/step_decomp.py" "root@$HOST:$W/n/" && $SCP "$P81/hook/usercustomize.py" "root@$HOST:$W/n/hook/" || { say "stage failed: scp (harness n)"; exit 20; }
if [ -s "$HF_TOKEN_FILE" ]; then   # authenticated pulls (unauthenticated shards throttle); the token never appears on a command line
  $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
  say "hf token staged"
else
  say "no hf token file at $HF_TOKEN_FILE -- pulls run unauthenticated (Qwen/Qwen3-30B-A3B is ungated)"
fi
$SSH "cd $W || exit 20; nohup env $PASS bash p84_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P84_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
# ---- heartbeat + liveness (e4b#641, tp4_drive.sh's): a stall is REPORTED, never acted on; two consecutive definite
# zeros for the lane's own process with no TP_DONE end the wait with their own code (25).
STALL_S=${P84_STALL_S:-900}; P84_MIN_PROGRESS_MB=${P84_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$P84_MIN_PROGRESS_MB
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
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f 'bash p84_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: no 'bash p84_run.sh' on the box for two consecutive polls and no TP_DONE -- not waiting out the deadline"
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
rm -rf "$RUN_DIR/p84" && mkdir -p "$RUN_DIR/p84" || { say "fetch failed: local dir"; exit 22; }
# The checkpoint, the NF4 arenas and every pack's payloads stay on the box; bake.json and the manifests travel
# (p57-5090-1 lost its only failure text to a blanket exclusion).
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" \
  --include 'work_a/' --include 'work_a/bake.json' --exclude 'work_a/*' \
  --include 'work_b/' --include 'work_b/bake.json' --exclude 'work_b/*' \
  --exclude 'artifact_*/*' --exclude 'attn_*/*' \
  --exclude '__pycache__' --exclude 'venv*' --exclude '.cache' --exclude 'hf' \
  "root@$HOST:$W/" "$RUN_DIR/p84/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p84" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p84/P84_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p84/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P84_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p84/P84_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p84/P84_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
