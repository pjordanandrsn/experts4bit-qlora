#!/bin/bash
# bench/sc1/sc1_drive.sh -- lane SC1, CONTROLLER side: the launcher's --command (SC1-PREREG.md; e4b#846).
# From bench/p58/p58_drive.sh (staging map, nonce start, heartbeat / liveness / stall reporting, fetch) with
# bench/tc1/tc1_drive.sh's forwarded-knob list and %q quoting. Stages sc1_run.sh, sc1_e4b_sched.py, sc1_prompts.py,
# sc1_sampler.sh, sc1_reduce.py (when it exists), the comparator driver directories bench/sc1/{vllm,sglang,llamacpp,
# exl3,lmdeploy}/ AS DIRECTORIES (tar over ssh, COPYFILE_DISABLE so macOS adds no ._* files), P39's step_decomp.py /
# k8_bake.py / calib.json, P42's hook (under hook/), P88's premise test tests/test_k19_row_exact_gpu.py and staged.sha256;
# checks every pin against its source before anything is sent; starts the box script detached under a fresh nonce;
# polls TP_DONE.<nonce>; fetches receipts, logs, samples and bake.json -- never the venvs, caches, arenas, checkpoints or
# the pack's payloads (its manifest.json rides along). Nothing here creates, destroys or approves compute.
# SC1_BOX=A|B|C|D|E|F|G is required (D = SC1b's census box, E = SC2's serving box, F = SC2b's prefill-graph box, G = SC2g's gpt-oss box). SC1_PROVE=1 runs the proving rental. SC1_DRIVE_DRYRUN=1 prints the plan and exits 0.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [sc1_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID SC1_BOX; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
case "$SC1_BOX" in A|B|C|D|E|F|G) ;; *) say "refusing: SC1_BOX must be A, B, C, D, E, F or G (SC1b, SC2, SC2b, SC2g)"; exit 78;; esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
P39="$REPO/bench/p39"; P42="$REPO/bench/p42"; TESTS="$REPO/tests"
# flat pieces (box sees them in $W); the reducer joins when it exists (staged.sha256 pins it then: "pinned at integration")
STAGE="$HERE/sc1_run.sh $HERE/sc1_e4b_sched.py $HERE/sc1_prompts.py $HERE/sc1_sampler.sh $P39/step_decomp.py $P39/k8_bake.py $P39/calib.json $TESTS/test_k19_row_exact_gpu.py $HERE/staged.sha256"
[ -s "$HERE/sc1_reduce.py" ] && STAGE="$STAGE $HERE/sc1_reduce.py"
# SC1b (bench/sc1b): the census box D's pieces, staged flat on EVERY box so staged.sha256 stays one list for A-D
SC1B="$REPO/bench/sc1b"; for f in sc1b_census.py sc1b_e4b_census.py sc1b_vllm_census.py sc1b_serve_census.py sc1b_toy.py kernel_classes.json sc1b_box_d.sh; do STAGE="$STAGE $SC1B/$f"; done
SC2="$REPO/bench/sc2"; for f in sc2_driver.py sc2_prompts.py sc2_reduce.py sc2_box_e.sh sc2_identity.py sc2b_box_f.sh sc2b_reduce.py sc2_trace.py sc2g_box_g.sh sc2g_reduce.py; do STAGE="$STAGE $SC2/$f"; done
HOOK="$P42/hook/usercustomize.py"
COMP_DIRS=""; for d in vllm sglang llamacpp exl3 lmdeploy; do [ -d "$HERE/$d" ] && COMP_DIRS="$COMP_DIRS $d"; done
for f in $STAGE $HOOK; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
case " $COMP_DIRS " in *" vllm "*) ;; *) say "refusing: bench/sc1/vllm/ is missing (every box needs the vLLM anchor)"; exit 78;; esac
# staged.sha256 names the files as the BOX sees them; resolve each name to its source (the same case as make_pin.sh and
# tests/test_sc1_staged_pin.py) and compare, rather than running `sha256sum -c` against paths that only exist after staging.
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in
    sc1_run.sh|sc1_e4b_sched.py|sc1_prompts.py|sc1_sampler.sh|sc1_reduce.py) src="$HERE/$name";;
    vllm/*|sglang/*|llamacpp/*|exl3/*|lmdeploy/*) src="$HERE/$name";;
    hook/usercustomize.py) src="$P42/hook/usercustomize.py";;
    test_k19_row_exact_gpu.py) src="$TESTS/$name";;
    sc1b_*|kernel_classes.json) src="$SC1B/$name";;
    sc2_*|sc2b_*|sc2g_*) src="$SC2/$name";;
    *) src="$P39/$name";;
  esac
  [ -s "$src" ] || { say "refusing: pinned file $name resolves to $src, which is missing"; exit 78; }
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want (run bench/sc1/make_pin.sh after editing a staged file)"; exit 78; }
done < "$HERE/staged.sha256"
# every file under a staged comparator dir must be pinned (a new driver file without a re-pin would reach the box unchecked)
for d in $COMP_DIRS; do
  while read -r rel; do grep -q "  $rel\$" "$HERE/staged.sha256" || { say "refusing: $rel is staged but not pinned in staged.sha256 (run make_pin.sh)"; exit 78; }
  done < <(cd "$HERE" && find "$d" -type f ! -name '*.pyc' ! -path '*/__pycache__/*' ! -name '.DS_Store' | LC_ALL=C sort)
done
# The e4b the BOX installs is the e4b this driver ships from (the launcher proved this checkout is the manifest's heads.e4b):
# derived, never a literal, refused on a dirty tree (p39-box1b-3's lesson).
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
# grouped-nf4-gemm's pin (v0.34.1's COMMIT 34da93d6) is a constant in the runner; the box installs e4b at THIS commit.
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${SC1_POLL_S:-60}; STALL_S=${SC1_STALL_S:-900}; W=/root/sc1
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="SC1_BOX=$SC1_BOX SC1_RUN_ID=$RUN_ID SC1_RUN_NONCE=$NONCE SC1_DEADLINE_EPOCH=$DEADLINE SC1_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA"
# Every knob the box script reads from the environment must be forwardable (tp4's lesson): tests/test_sc1_run_shape.py asserts
# this list against every `${SC1_*` sc1_run.sh reads. Only host-limit knobs, the proving-run switches and install flavours
# travel: a rental is the registered run, and a value off its default marks the box run REHEARSAL.
for v in SC1_PROVE SC1_PROVE_SGLANG_MODEL SC1_GPU_CLASS SC1_MIN_DISK_GB SC1_MIN_DRIVER SC1_CPU_VENDOR SC1_CALIB_NSEQ SC1_REHEARSAL \
         SC1_QUIESCE_S SC1_VLLM_WHEEL; do
  # %q: a value with spaces would otherwise become the remote COMMAND (tp4-b-p46cut-3: rc=127 before the nonce was bound)
  [ -n "${!v:-}" ] && PASS="$PASS $v=$(printf %q "${!v}")"
done
if [ "${SC1_DRIVE_DRYRUN:-0}" = "1" ]; then
  echo "DRYRUN stage [$(for f in $STAGE; do printf '%s ' "${f##*/}"; done)hook/usercustomize.py dirs:$COMP_DIRS] -> root@$HOST:$W ; start: env $PASS bash sc1_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/sc1"; exit 0; fi
say "run $RUN_ID box $SC1_BOX nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA (from $REPO); dirs:$COMP_DIRS; receipts -> $RUN_DIR/sc1; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/hook $W/samples /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" && $SCP "$HOOK" "root@$HOST:$W/hook/" || { say "stage failed: scp"; exit 20; }
COPYFILE_DISABLE=1 tar -C "$HERE" --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' -czf - $COMP_DIRS | $SSH "tar -C $W -xzf -" || { say "stage failed: comparator dirs"; exit 20; }
$SSH "cd $W && sha256sum -c --quiet staged.sha256" || { say "stage failed: the box's bytes do not match staged.sha256"; exit 20; }
if [ -s "$HF_TOKEN_FILE" ]; then   # authenticated pulls (unauthenticated shards throttle); the token never appears in a command line
  $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
  say "hf token staged"
else
  say "no hf token file at $HF_TOKEN_FILE -- pulls run unauthenticated (every registered checkpoint is ungated)"
fi
$SSH "cd $W || exit 20; nohup env $PASS bash sc1_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat SC1_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
# ---- heartbeat + liveness (e4b#641, from tp4_drive.sh / p58_drive.sh): a stall is REPORTED, never acted on; two consecutive
# definite zeros on the lane's own process with no TP_DONE end the wait with their own exit code (25).
SC1_MIN_PROGRESS_MB=${SC1_MIN_PROGRESS_MB:-16}
progress_verdict() {  # idle_s stall_s util dfk_now dfk_prev du_now du_prev -> "" | "fetching:<delta>" | "stall:<idle_s>"
  idle_s=$1; stall_s=$2; util=$3; dfk_now=$4; dfk_prev=$5; du_now=$6; du_prev=$7; min_mb=$SC1_MIN_PROGRESS_MB
  consumed=0; grew=0
  if [ -n "$dfk_now" ] && [ -n "$dfk_prev" ] && [ "$dfk_now" -lt "$dfk_prev" ] 2>/dev/null; then consumed=$(( (dfk_prev - dfk_now) / 1024 )); fi
  if [ -n "$du_now" ] && [ -n "$du_prev" ] && [ "$du_now" -gt "$du_prev" ] 2>/dev/null; then grew=$((du_now - du_prev)); fi
  if [ "$consumed" -ge "$min_mb" ] || [ "$grew" -ge "$min_mb" ]; then
    if [ "$consumed" -ge "$grew" ]; then echo "fetching:-${consumed}M on disk"; else echo "fetching:+${grew}M in tree"; fi
    return 0
  fi
  if [ "$idle_s" -ge "$stall_s" ] && [ "${util:-0}" -eq 0 ] 2>/dev/null; then echo "stall:$idle_s"; fi
}
lane_dead() { [ "${1:-}" = "0" ] && [ "${2:-}" = "0" ] && echo dead; }   # live_now live_prev: two DEFINITE zeros
# SC2g A1: the heartbeat counts '[b]ash sc1_run.sh', not 'bash sc1_run.sh'. The count runs inside an ssh shell whose OWN command
# line carries the pattern, so the plain form always counted itself, `live` was never 0, and LANE DEAD never fired: a lane that
# died under set -u (sc2g-prove-1) waited out its whole deadline. The bracket still matches the process, never this text.
# A10: one pull for both uses (the same keep/leave rules). Keep: every receipt / log / sample csv / summary / quiesce / energy
# json, the pack's manifest.json (payloads stay), work_*/bake.json (k8_bake.py's failure record travels; p57-5090-1 lost the only
# text that said WHY). Leave: venvs, caches, arenas, snapshots, the llama.cpp tree and GGUFs, the pack payloads. Bounded (ssh
# ConnectTimeout + rsync --timeout) so a dead box cannot hang the driver; the remote rsync runs at nice 19 / idle I/O so a pull
# during a timed arm does not perturb it.
pull_box() { local -a low=(); [ "${2:-}" = low ] && low=(--rsync-path="nice -n 19 ionice -c3 rsync" --exclude 'census/')   # mid-run pulls only; SC1b's traces come in the final fetch
  # ${low[@]+...}: the controller runs /bin/bash 3.2 under set -u (the mini), where "${low[@]}" of an empty array is unbound
  rsync -az --timeout=120 -e "ssh -o BatchMode=yes -o ConnectTimeout=20 $E4B_RENT_SSH_OPTS -p $PORT" ${low[@]+"${low[@]}"} \
    --exclude 'artifact*/payloads/' --include 'work_*/' --include 'work_*/bake.json' --exclude 'work_*/*' \
    --exclude 'venv*' --exclude '.cache' --exclude 'llama.cpp/' --exclude 'gguf/' --exclude 'sglang-cache/' \
    "root@$HOST:$W/" "$1/"; }
# A10: an incremental pull every PULL_EVERY_S during the run, into sc1.partial/ -- sc1c-5090-1's box stopped answering at
# 20:55Z and the single end-of-run fetch got nothing (TC2's box B was lost the same way). The partial copy becomes the
# receipt's sc1/ only when the final fetch fails, and is labelled so.
PULL_EVERY_S=${SC1_PULL_EVERY_S:-1200}; LAST_PULL=$(date +%s); PARTIAL_AT=""
say "lane started; polling TP_DONE every ${POLL}s with a heartbeat (stall reported after ${STALL_S}s of no change, idle GPU AND no disk movement; never acted on; a lane whose process is gone for two polls ends the wait)"
LAST=""; LAST_CHANGE=$(date +%s); LAST_DFK=""; LAST_DU=""; LAST_LIVE=""; LANE_DEAD=0
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$($SSH "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}') | dfk \$(df -k /root | tail -1 | awk '{print \$4}') | live \$(pgrep -f '[b]ash sc1_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  dfk=$(echo "$hb" | sed -n 's/.*| dfk \([0-9]*\).*/\1/p'); duM=$(echo "$hb" | sed -n 's/.*| du \([0-9]*\)M.*/\1/p')
  live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  if [ "$line" != "$LAST" ]; then [ -n "$line" ] && say "box: $line"; LAST=$line; LAST_CHANGE=$now; fi
  if [ -n "$(lane_dead "$live" "$LAST_LIVE")" ]; then
    say "LANE DEAD: no 'bash sc1_run.sh' on the box for two consecutive polls and no TP_DONE -- the remote process exited without writing its markers; not waiting out the deadline"
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
  if [ $((now - LAST_PULL)) -ge "$PULL_EVERY_S" ] && [ -n "$hb" ]; then
    LAST_PULL=$now; mkdir -p "$RUN_DIR/sc1.partial"
    if pull_box "$RUN_DIR/sc1.partial" low > /dev/null 2>&1; then PARTIAL_AT=$(date -u +%FT%TZ); say "partial pull ok ($PARTIAL_AT)"; else say "partial pull failed (kept the previous one: ${PARTIAL_AT:-none})"; fi
  fi
  sleep "$POLL"
done
rm -rf "$RUN_DIR/sc1" && mkdir -p "$RUN_DIR/sc1" || { say "fetch failed: local dir"; exit 22; }
if ! pull_box "$RUN_DIR/sc1"; then
  if [ -n "$PARTIAL_AT" ] && [ -d "$RUN_DIR/sc1.partial" ]; then
    rm -rf "$RUN_DIR/sc1" && mv "$RUN_DIR/sc1.partial" "$RUN_DIR/sc1" && echo "$PARTIAL_AT" > "$RUN_DIR/sc1/PARTIAL_PULL_AT"
    say "fetch failed: rsync -- the receipt keeps the last incremental pull ($PARTIAL_AT, labelled PARTIAL_PULL_AT)"
  else say "fetch failed: rsync (no incremental pull to keep)"; fi
  exit 22
fi
rm -rf "$RUN_DIR/sc1.partial"
say "fetched $(ls "$RUN_DIR/sc1" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/sc1/SC1_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/sc1/TP_DONE.$NONCE" ] || {
  [ "${LANE_DEAD:-0}" = 1 ] && { say "lane DIED on the box: its process was gone with no SC1_EXIT_CODE and no TP_DONE -- artifacts fetched, nothing measured"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/sc1/SC1_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/sc1/SC1_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "box $SC1_BOX complete rc=0"; exit 0
