#!/bin/bash
# Lane P235's controller (bench/p235/PREREG-p235.md): P233's controller (bench/p233/) by named edits -- this lane's run
# ids (the reading's -1..-3 and the -r1..-r6 after a box that did not start; no shakedown), runner, reducer and
# provenance check; P58's harness staged under the names the runner expects (P37's vLLM driver, P39's step_decomp.py,
# k8_bake.py and calib.json flat, P42's hook under hook/), copied into a stage directory and checked there against
# staged.sha256. The box installs every distribution from PyPI at the runner's constants; this checkout's commit is
# the harness's. pod-launch's contract (P229 Amendments 3 and 6): the launch commit is the clean checkout HEAD, an
# E4B_SHA override is refused, and the box tree is fetched into $RUN_DIR/p235/ with --no-links, a non-empty target
# refused. Polls the nonce-bound completion marker, and fetches receipts without the hub cache, the vLLM venvs or the
# NF4 arena. It does not create, destroy, or approve compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p235_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as pod-launch's --command after a live pre-flight"; exit 78; }
done
# Explicit IDs only: the readings (-1..-3) and the six after a box that did not start (-r1..-r6). No shakedown is
# registered (PREREG-p235.md says why).
RUN_ID=$E4B_RENT_RUN_ID
case "$RUN_ID" in
  p235-5090-1|p235-5090-2|p235-5090-3|p235-5090-r1|p235-5090-r2|p235-5090-r3|p235-5090-r4|p235-5090-r5|p235-5090-r6) :;;
  *) say "refusing unregistered run ID: $RUN_ID (is not p235-5090-N or p235-5090-rN)"; exit 78;;
esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# Lane liveness by launch identity (PID + /proc identity, host uptime; #1512, #1517): the helper is sourced here and
# streamed to the box over stdin for the launch and each probe; it is never staged (staged.sha256 does not list it).
# Launch identity, never a self-matching pgrep (P150's and P154's form).
LANE_HELPER="$REPO/bench/common/lane_liveness.sh"
source "$LANE_HELPER" || { say "refusing: missing lane liveness helper"; exit 78; }
# The launch commit is the e4b this driver ships from (pod-launch proves it is the manifest's heads.e4b): it stages the
# kit, and the box installs it. Derived, never an input: an override is refused, and so is a dirty tree (p39-box1b-3's
# lesson; P226's form).
if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi
E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
[ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
# The staging, under the names p235_run.sh expects: this lane's runner, reducer, provenance check and staged.sha256,
# loggetta's release proof (bench/release_proof.py at f09e5676, #126's merge; byte-identical),
# P37's vLLM driver (byte-identical) and P39's step_decomp.py, k8_bake.py and calib.json flat; P42's hook as hook/usercustomize.py. staged.sha256 pins
# every one by the path the box sees, at P58's digests for every piece but this lane's three and the release proof.
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
for f in bench/p235/p235_run.sh bench/p235/p235_reduce.py bench/p235/provenance.py bench/p235/staged.sha256 \
         bench/p235/release_proof.py bench/h2h-20260905/p37/p37_vllm.py \
         bench/p39/step_decomp.py bench/p39/k8_bake.py bench/p39/calib.json; do
  [ -s "$REPO/$f" ] && cp "$REPO/$f" "$STAGE/" || { say "refusing: staged piece missing: $f"; exit 78; }
done
for f in bench/p42/hook/usercustomize.py; do
  mkdir -p "$STAGE/hook" && [ -s "$REPO/$f" ] && cp "$REPO/$f" "$STAGE/hook/" || { say "refusing: staged piece missing: $f"; exit 78; }
done
# every staged file is pinned, and every pin is staged (the same map as tests/test_p235.py, checked in CI)
(cd "$STAGE" && { sha256sum -c staged.sha256 2>/dev/null || shasum -a 256 -c staged.sha256; } > /dev/null) || { say "refusing: staged files differ from staged.sha256"; exit 78; }
[ "$(cd "$STAGE" && find . -type f ! -name staged.sha256 | wc -l | tr -d ' ')" = "$(grep -c '^[0-9a-f]\{64\}  ' "$STAGE/staged.sha256")" ] \
  || { say "refusing: the stage and staged.sha256 list different files"; exit 78; }
# The box installs experts4bit-qlora 0.58.0 and grouped-nf4-gemm 0.51.0 from PyPI and records their provenance; THIS
# commit is the harness's launch commit, recorded and never installed.
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${P235_POLL_S:-60}; W=/root/p235; OUT="$RUN_DIR/p235"
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P235_RUN_ID=$RUN_ID P235_RUN_NONCE=$NONCE P235_DEADLINE_EPOCH=$DEADLINE P235_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA"
if [ "${P235_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage $(find "$STAGE" -type f | wc -l | tr -d ' ') files -> root@$HOST:$W ; start: env $PASS bash p235_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $OUT"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; harness commit $E4B_SHA (from $REPO); the installs are the runner's PyPI pins, provenance-checked on the box; receipts -> $OUT; deadline $DEADLINE"
$SSH "test ! -e $W && mkdir -p $W/logs /root/.cache/huggingface" || { say "stage failed: remote work directory exists"; exit 20; }
$SCP -r "$STAGE"/* "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
if [ -s "$HF_TOKEN_FILE" ]; then   # authenticated pulls (unauthenticated shards throttle); the token never appears on a command line
  $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
  say "hf token staged"
else
  say "no hf token file -- pulls will be unauthenticated"
fi
LANE_STARTED_AT=$(date +%s)
# The launch streams the helper ahead of the start, so the child's PID and /proc identity are recorded in the same shell
# that started it (P127's form).
LAUNCH_REPLY=$({ cat "$LANE_HELPER"; printf '%s\n' "cd $W || exit 20; nohup env $PASS bash p235_run.sh > outer.log 2>&1 < /dev/null & child=\$!; identity=\$(lane_proc_snapshot \$child); end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P235_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; echo identity:\$identity; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124"; } | $SSH bash -s) || { say "start failed: child did not bind the nonce"; exit 21; }
printf '%s\n' "$LAUNCH_REPLY"
LANE_PID=$(printf '%s\n' "$LAUNCH_REPLY" | sed -n 's/^started:\([0-9][0-9]*\)$/\1/p')
[[ "$LANE_PID" =~ ^[1-9][0-9]*$ ]] || { say "start failed: malformed child PID"; exit 21; }
LANE_INITIAL=$(printf '%s\n' "$LAUNCH_REPLY" | sed -n 's/^identity://p')
say "lane identity: pid=$LANE_PID snapshot=${LANE_INITIAL:-unknown}"
# ---- heartbeat + liveness (e4b#641, tp4_drive.sh's): a stall is REPORTED, never acted on; two consecutive definite
# zeros for the lane's own process with no TP_DONE end the wait with their own code (25).
STALL_S=${P235_STALL_S:-900}; P235_MIN_PROGRESS_MB=${P235_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$P235_MIN_PROGRESS_MB
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
mkdir -p "$OUT" && [ -z "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ] || { say "fetch failed: $OUT is not an empty directory"; exit 22; }
# The checkpoints stay on the box (the hub cache, outside $W), and so do the vLLM venvs and the NF4 arena (work_qwen3/:
# only its bake.json travels); receipts, prompts, logs and the reducer's table travel. --no-links: the fetch
# never imports a symbolic link from the box (a rented host is untrusted, and the receipt writers follow links: bench/common
# rule 8; P229 Amendment 6's form).
rsync -az --no-links -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" \
  --include 'work_qwen3/' --include 'work_qwen3/bake.json' --exclude 'work_qwen3/*' \
  --exclude '__pycache__' --exclude 'pycache-*' --exclude 'venv*' --exclude '.cache' --exclude 'hf' \
  "root@$HOST:$W/" "$OUT/" || { say "fetch failed"; exit 22; }
say "fetched $(find "$OUT" -type f | wc -l | tr -d ' ') files into $OUT"
[ "$(cat "$OUT/P235_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$OUT/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no P235_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$OUT/P235_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$OUT/P235_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
