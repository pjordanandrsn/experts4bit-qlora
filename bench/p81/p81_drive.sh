#!/bin/bash
# bench/p81/p81_drive.sh -- lane P81, CONTROLLER side: the launcher's --command (bench/p81/PREREG-p81.md; e4b#511, e4b#674).
# Derived from bench/p80/p80_drive.sh by named substitutions: it stages P81's runner and reducer, its own copy of
# step_decomp.py (P80's stage plus the pack provenance and --dynb-grouping), its hook (v7, attention through
# int4_attn_calib.enable_from_env) and P39's k8_bake.py / calib.json; starts the lane detached under a fresh nonce;
# polls TP_DONE.<nonce> with the tp4-style heartbeat/liveness check; fetches receipts -- never the checkpoint, the
# arena or either pack's payloads (their manifests travel; the fingerprints are the claim). Nothing here creates,
# destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p81_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# step_decomp.py is P81's own copy of the live harness (bench/hybrid-g9 at 69488f1 plus the P80 stage and P81's two
# additions; the live file is pinned by closed lanes and stays untouched); k8_bake.py / calib.json are P39's,
# referenced, never copied. staged.sha256 pins every one by the name the box sees.
P39="$REPO/bench/p39"
STAGE="$HERE/p81_run.sh $HERE/p81_reduce.py $HERE/step_decomp.py $P39/k8_bake.py $P39/calib.json $HERE/staged.sha256"
HOOK="$HERE/hook/usercustomize.py"
for f in $STAGE $HOOK; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged.sha256 names the files as the BOX will see them; resolve each name to its source and compare hashes
# (the same case as tests/test_p81_staged_pin.py, which runs this check in CI where it costs nothing).
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    p81_run.sh|p81_reduce.py|step_decomp.py|hook/usercustomize.py) src="$HERE/$name";;
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
# The registered kernel cut is a release-tag commit: v0.33.5 (fb15cf5). Passed explicitly, never defaulted.
GNF4_SHA=${GNF4_SHA:?set GNF4_SHA to the registered grouped-nf4-gemm cut (PREREG-p81: fb15cf5f5b9f2fb107fa1a39218fd9d910b34d94, v0.33.5)}
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; exit 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${P81_POLL_S:-60}; W=/root/p81
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P81_RUN_ID=$RUN_ID P81_RUN_NONCE=$NONCE P81_DEADLINE_EPOCH=$DEADLINE P81_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA"
# Only host-limit knobs and the proving-run switch travel: a rental is the registered run, so the knobs that change WHAT
# is measured (model, pack, rows, calibration) are not forwarded -- a rehearsal sets them on its own box and the runner
# marks it REHEARSAL. P81_PROVE=1 is the proving rental the pre-registration names (no model, no pack).
for v in P81_PROVE; do [ -n "${!v:-}" ] && PASS="$PASS $v=$(printf %q "${!v}")"; done
if [ "${P81_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p81_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p81"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA (from $REPO); gnf4 $GNF4_SHA; receipts -> $RUN_DIR/p81; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/hook /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SCP "$HOOK" "root@$HOST:$W/hook/" || { say "stage failed: scp (hook)"; exit 20; }
if [ -s "$HF_TOKEN_FILE" ]; then   # authenticated pulls (unauthenticated shards throttle); the token never appears on a command line
  $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
  say "hf token staged"
else
  say "no hf token file at $HF_TOKEN_FILE -- pulls run unauthenticated (Qwen/Qwen3-30B-A3B is ungated)"
fi
$SSH "cd $W || exit 20; nohup env $PASS bash p81_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P81_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
# ---- heartbeat + liveness (e4b#641, tp4_drive.sh's): a stall is REPORTED, never acted on; two consecutive definite
# zeros for the lane's own process with no TP_DONE end the wait with their own code (25).
STALL_S=${P81_STALL_S:-900}; P81_MIN_PROGRESS_MB=${P81_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$P81_MIN_PROGRESS_MB
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
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f 'bash p81_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: no 'bash p81_run.sh' on the box for two consecutive polls and no TP_DONE -- not waiting out the deadline"
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
rm -rf "$RUN_DIR/p81" && mkdir -p "$RUN_DIR/p81" || { say "fetch failed: local dir"; exit 22; }
# The checkpoint, the NF4 arena and both packs' payloads stay on the box; bake.json and the two manifests travel
# (p57-5090-1 lost its only failure text to a blanket exclusion).
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" \
  --include 'work/' --include 'work/bake.json' --exclude 'work/*' \
  --include 'artifact1/' --include 'artifact1/manifest.json' --exclude 'artifact1/*' \
  --include 'attn1/' --include 'attn1/manifest.json' --exclude 'attn1/*' \
  --exclude '__pycache__' --exclude 'venv*' --exclude '.cache' --exclude 'hf' \
  "root@$HOST:$W/" "$RUN_DIR/p81/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p81" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p81/P81_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p81/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P81_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p81/P81_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p81/P81_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
