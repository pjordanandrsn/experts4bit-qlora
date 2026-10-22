#!/bin/bash
# bench/gpu-contract/gc_drive.sh -- the GPU contract, CONTROLLER side: the launcher's --command (bench/gpu-contract/README.md).
# Derived from bench/p133/p133_drive.sh's transport (staging, the nonce handshake, the heartbeat, the fetch, the exit codes).
# It runs bench/gpu-contract/gc_run.sh on the box with three inputs, each refused unless well formed:
#   GC_GNF4_SHA  the grouped-nf4-gemm commit under test (40 hex)    GC_E4B_SHA  the experts4bit-qlora commit (40 hex)
#   GC_CARD      the declared card class, MAJOR.MINOR/SMS or MAJOR.MINOR/>=SMS (gc_box.py)
# The launch tree must be clean: the staged tool is the committed one, and its commit goes in the receipt. Run ids are
# gc-<slug>-<n>; GC_DRY=1 is the $0 dry run (gc-dry-<slug>-<n>) through GC_BOX_EXEC and tar, as P133's. Nothing here creates,
# destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [gc_drive] $*"; }
DRY=${GC_DRY:-0}
if [ "$DRY" = 1 ]; then
  for v in E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID GC_BOX_EXEC; do
    [ -n "${!v:-}" ] || { say "refusing: $v is not set -- GC_DRY=1 needs it"; exit 78; }
  done
else
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
fi
RUN_ID=$E4B_RENT_RUN_ID
case "$RUN_ID" in
  gc-dry-*) [ "$DRY" = 1 ] || { say "refusing: $RUN_ID is a dry run's id (GC_DRY=1 only)"; exit 78; };;
  gc-*) [ "$DRY" != 1 ] || { say "refusing: a dry run's id is gc-dry-<slug>-<n>"; exit 78; };;
  *) say "refusing: run id $RUN_ID is not gc-<slug>-<n>"; exit 78;;
esac
[[ "$RUN_ID" =~ ^gc-[a-z0-9]+(-[a-z0-9]+)*-[0-9]+$ ]] || { say "refusing: run id $RUN_ID is not gc-<slug>-<n> (lowercase, digits, hyphens)"; exit 78; }
for v in GC_GNF4_SHA GC_E4B_SHA; do
  val=${!v:-}; [[ "$val" =~ ^[0-9a-f]{40}$ ]] || { say "refusing: $v='$val' is not a full 40-character commit sha (an unpinned ref is refused)"; exit 78; }
done
# GC_REQUIRE (optional): the test ids that must PASS, in junit form <classname>::<name>, space-separated
# (read into an array, never word-split unquoted: an id like t[graph] must not glob against the working directory)
REQ_RE='^[A-Za-z0-9_.]+::[]A-Za-z0-9_.[-]+$'; read -r -a REQ <<< "${GC_REQUIRE:-}"
for id in ${REQ[@]+"${REQ[@]}"}; do
  [[ "$id" =~ $REQ_RE ]] || { say "refusing: GC_REQUIRE id '$id' is not <classname>::<name>"; exit 78; }
done
# GC_E4B_TESTS (optional): the e4b test files to run instead of the served-route premise; GC_NEED_E4B_S: their time need
# (read into an array and never expanded unquoted: from a repo root, tests/test_*.py would glob into real paths here and
# then reach the box as the literal, which refuses it after the rental is paid)
read -r -a E4BT <<< "${GC_E4B_TESTS:-}"
for t in ${E4BT[@]+"${E4BT[@]}"}; do
  [[ "$t" =~ ^tests/test_[a-z0-9_]+\.py$ ]] || { say "refusing: GC_E4B_TESTS entry '$t' is not tests/test_<name>.py"; exit 78; }
done
# no leading zero (the box's $(( )) would read 0900 as octal and exit without TP_DONE), and at most the 1,800 s per-suite alarm
NEED=${GC_NEED_E4B_S:-600}
{ [[ "$NEED" =~ ^[1-9][0-9]{0,5}$ ]] && [ "$NEED" -le 1800 ]; } || { say "refusing: GC_NEED_E4B_S='$NEED' is not 1-1800 whole seconds without a leading zero"; exit 78; }
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# the class must parse: gc_box answers 0 (inside) or 15 (outside) for a well-formed class against this dummy probe; 78 (malformed)
# or anything else (the check itself failed) refuses
python3 "$HERE/gc_box.py" --card "cc=0.0 sms=0" --declared "${GC_CARD:-}" > /dev/null; crc=$?
case "$crc" in 0|15) ;; *) say "refusing: GC_CARD='${GC_CARD:-}' did not parse as MAJOR.MINOR/SMS or MAJOR.MINOR/>=SMS (class check rc $crc)"; exit 78;; esac
# the staged tool is the committed one: a dirty launch tree is refused, and its commit travels into the receipt
[ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: the launch tree $REPO is dirty"; exit 78; }
TOOL_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: $REPO is not a git checkout"; exit 78; }
STAGE="$HERE/gc_run.sh $HERE/gc_box.py"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
RUN_DIR=$E4B_RENT_RUN_DIR; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
# receipts are evidence: an earlier run's receipts in this run dir are refused, never deleted
[ ! -e "$RUN_DIR/gc" ] || [ -z "$(ls -A "$RUN_DIR/gc" 2>/dev/null)" ] || { say "refusing: $RUN_DIR/gc already holds receipts"; exit 78; }
POLL=${GC_POLL_S:-60}; W=/root/gc
if [ "$DRY" = 1 ]; then
  WHERE="dry box via $GC_BOX_EXEC"
  box(){ "$GC_BOX_EXEC" "$1"; }
  box_stage(){ local args=() f; for f in $STAGE; do args+=(-C "$(dirname "$f")" "$(basename "$f")"); done
    COPYFILE_DISABLE=1 tar -cf - "${args[@]}" | box "tar -C $W -xf -"; }
  box_fetch(){ box "cd $W && tar -cf - --exclude=./gnf4src --exclude=./e4bsrc --exclude=__pycache__ --exclude=./.cache ." | tar -C "$1" -xif -; }
else
  HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; WHERE="$HOST:$PORT"
  SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
  SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
  box(){ $SSH "$1"; }
  box_stage(){ $SCP $STAGE "root@$HOST:$W/"; }
  box_fetch(){ rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" \
      --exclude '__pycache__' --exclude '.cache' --exclude 'gnf4src' --exclude 'e4bsrc' "root@$HOST:$W/" "$1/"; }
fi
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="GC_RUN_ID=$RUN_ID GC_RUN_NONCE=$NONCE GC_DEADLINE_EPOCH=$DEADLINE GC_GNF4_SHA=$GC_GNF4_SHA GC_E4B_SHA=$GC_E4B_SHA GC_CARD=$(printf %q "$GC_CARD") GC_REQUIRE=$(printf %q "${GC_REQUIRE:-}") GC_E4B_TESTS=$(printf %q "${E4BT[*]+${E4BT[*]}}") GC_NEED_E4B_S=$NEED"
if [ "${GC_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN tool $TOOL_SHA; stage -> $WHERE:$W ; start: env $PASS bash gc_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/gc"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $WHERE; tool $TOOL_SHA; gnf4 $GC_GNF4_SHA; e4b $GC_E4B_SHA; card $GC_CARD; receipts -> $RUN_DIR/gc; deadline $DEADLINE"
box "rm -rf -- $W && mkdir -p $W/logs" || { say "stage failed: remote cleanup"; exit 20; }
box_stage || { say "stage failed: copy"; exit 20; }
box "cd $W || exit 20; printf '%s\n' '$TOOL_SHA' > GC_TOOL_SHA; nohup env $PASS bash gc_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat GC_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
say "lane started; polling TP_DONE every ${POLL}s (a lane whose process is gone for two polls ends the wait)"
LAST=""; LAST_LIVE=""; LANE_DEAD=0
while :; do
  now=$(date +%s)
  box "test -f $W/TP_DONE.$NONCE" 2>/dev/null < /dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$(box "echo \"\$(grep -v '^[[:space:]]*$' $W/summary.txt 2>/dev/null | tail -n 1 | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | live \$(pgrep -f 'bash gc_run.sh' | wc -l | tr -d ' ')\"" 2>/dev/null < /dev/null)
  line=${hb%% | gpu*}; live=$(echo "$hb" | sed -n 's/.*| live \([0-9]*\).*/\1/p')
  [ "$line" != "$LAST" ] && [ -n "$line" ] && say "box: $line"; LAST=$line
  if [ "${live:-}" = 0 ] && [ "${LAST_LIVE:-}" = 0 ]; then say "LANE DEAD: no 'bash gc_run.sh' on the box for two polls and no TP_DONE"; LANE_DEAD=1; break; fi
  LAST_LIVE=$live
  say "hb: ${hb:-<no answer>} | left $((DEADLINE - now))s"
  sleep "$POLL"
done
mkdir -p "$RUN_DIR/gc" || { say "fetch failed: local dir"; exit 22; }
box_fetch "$RUN_DIR/gc" < /dev/null || { say "fetch failed"; exit 22; }
say "fetched $(ls "$RUN_DIR/gc" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/gc/GC_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/gc/TP_DONE.$NONCE" ] || {
  [ "$LANE_DEAD" = 1 ] && { say "lane DIED on the box: no GC_EXIT_CODE and no TP_DONE -- artifacts fetched"; exit 25; }
  say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/gc/GC_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
grep -E '^(gnf4|e4b) .*tests rc=|^GC_SUITE|^GC CONTRACT|^SKIPPED' "$RUN_DIR/gc/summary.txt" | cut -c1-300 | while read -r l; do say "result: $l"; done
[ -f "$RUN_DIR/gc/GC_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "contract passed rc=0"; exit 0
