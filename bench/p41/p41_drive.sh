#!/bin/bash
# bench/p41/p41_drive.sh -- lane P41, controller-side driver: the launcher's --command (the private control plane's `adertha.compute.rent`, e4b#464; the launcher left this repository under the 2026-09-06 boundary directive).
# Reads the box from the launcher's environment (E4B_RENT_SSH_HOST / _PORT, _RUN_DIR, _RUN_ID, _DEADLINE_EPOCH, _WALLCLOCK_S), stages the
# harness (bench/tp3/tp3_arm.py) and p41_run.sh, synchronously clears the fixed remote workdir, then starts the lane detached. A fresh
# unpredictable nonce binds this controller invocation to the child: p41_run writes P41_RUN_NONCE atomically before doing setup, then
# writes nonce-named P41_EXIT_CODE and TP_DONE files on every normal shell exit and nonce-named P41_SUCCESS only at successful completion.
# The driver polls and fetches only those nonce-named files and returns 0 only for canonical numeric remote rc 0 plus P41_SUCCESS; stale,
# missing, malformed or nonzero remote status is HARNESS_ERROR in the receipt.
# After the fetch it computes the run's VERDICT from the PRE-REGISTERED criteria (admitted arms, VOID rows, fired STOP rules) via
# `p41_admit.py verdict`, writes P41_VERDICT.<nonce>.json beside the receipts, and exits 0 ONLY for `pass` -- 40 fail, 41 inconclusive,
# 24 invalid (e4b#495: a receipt must not read `pass` while the run's own registered failure criteria fired in the same directory).
# Nothing here creates, destroys or approves compute.
# The approval line's rate ceiling and estimate come from the launcher (E4B_RENT_USD_PER_HOUR / E4B_RENT_EST_USD, e4b#465); the run's
# amended PLANNING estimate comes from the controller's environment (P41_PLAN_EST_USD, set per run thread from the pre-registration's
# amended table) and is what STOP-4 works from (P41-PREREG.md amendment, e4b#467) -- the approval line is the governance guard, recorded
# beside it. Two numbers, two jobs; the driver refuses without either. P41_FAMILIES etc. pass through; P41_DRIVE_DRYRUN=1 prints the commands.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p41_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_WALLCLOCK_S E4B_RENT_USD_PER_HOUR E4B_RENT_EST_USD E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run this as rent.py --command after a live pre-flight (e4b#464)"; exit 78; }
done
[ -n "${P41_PLAN_EST_USD:-}" ] || { say "refusing: P41_PLAN_EST_USD is not set -- the run's amended planning estimate (P41-PREREG.md Amendments) is what STOP-4 works from; the approval line (E4B_RENT_EST_USD) is the guard, not the estimate"; exit 78; }
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
HARNESS=$REPO/bench/tp3/tp3_arm.py; RUNSH=$HERE/p41_run.sh; ADMIT=$HERE/p41_admit.py
[ -s "$HARNESS" ] && [ -s "$RUNSH" ] && [ -s "$ADMIT" ] || { say "refusing: harness, lane script or admission rules missing ($HARNESS, $RUNSH, $ADMIT)"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
POLL=${P41_POLL_S:-60}; START_WAIT=${P41_START_WAIT_S:-30}; W=/root/p41
case "$START_WAIT" in ""|*[!0-9]*) say "refusing: P41_START_WAIT_S must be an integer from 1 to 60"; exit 78;; esac
[ "$START_WAIT" -ge 1 ] && [ "$START_WAIT" -le 60 ] || { say "refusing: P41_START_WAIT_S must be from 1 to 60"; exit 78; }
RUN_NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: could not generate run nonce"; exit 78; }
case "$RUN_NONCE" in *[!0-9a-f]*|"") say "refusing: generated run nonce is malformed"; exit 78;; esac
[ ${#RUN_NONCE} -eq 64 ] || { say "refusing: generated run nonce is not 64 hex chars"; exit 78; }
LOCAL_PLAN=""; LOCAL_PLAN_RC=0; LOCAL_PLAN=$(bash "$RUNSH" --plan) || LOCAL_PLAN_RC=$?
[ "$LOCAL_PLAN_RC" -eq 0 ] || { say "refusing: local P41 plan failed rc=$LOCAL_PLAN_RC"; exit 78; }
EXPECTED_ROWS=$(printf '%s\n' "$LOCAL_PLAN" | awk '/^PLAN TOTAL / { for (i=1; i<=NF; i++) if ($i ~ /^arms=/) { sub(/^arms=/, "", $i); print $i } }')
case "$EXPECTED_ROWS" in ""|*[!0-9]*) say "refusing: local P41 plan has no numeric row count"; exit 78;; esac
[ "$EXPECTED_ROWS" -gt 0 ] || { say "refusing: local P41 plan has zero expected rows"; exit 78; }
PASS="P41_RUN_ID=$RUN_ID P41_RUN_NONCE=$RUN_NONCE P41_EXPECTED_ROWS=$EXPECTED_ROWS P41_DEADLINE_EPOCH=$DEADLINE P41_USD_PER_HOUR=$E4B_RENT_USD_PER_HOUR P41_PLAN_EST_USD=$P41_PLAN_EST_USD P41_APPROVAL_EST_USD=$E4B_RENT_EST_USD P41_INSTANCE_ID=$E4B_RENT_INSTANCE_ID P41_PROVIDER=${E4B_RENT_PROVIDER:-UNKNOWN} P41_WALLCLOCK_S=$E4B_RENT_WALLCLOCK_S"
for v in P41_FAMILIES P41_SEQS P41_RANKS P41_PROBE_SEQ P41_STEPS; do [ -n "${!v:-}" ] && PASS="$PASS $v='${!v}'"; done
if [ "${P41_DRIVE_DRYRUN:-0}" = "1" ]; then
  echo "DRYRUN clean: $SSH \"rm -rf -- $W && mkdir -p $W/logs\""
  echo "DRYRUN stage: scp -P $PORT $HARNESS $RUNSH $ADMIT root@$HOST:$W/"
  echo "DRYRUN start: $SSH \"cd $W; nohup env $PASS bash p41_run.sh > outer.log 2>&1 &; wait <=$START_WAIT s for matching P41_RUN_NONCE\""
  echo "DRYRUN poll : $SSH test -f $W/TP_DONE.$RUN_NONCE  (every $POLL s until $DEADLINE)"
  echo "DRYRUN fetch: rsync -az -e 'ssh -p $PORT' root@$HOST:$W/ $RUN_DIR/p41/  (excluding adapters, e4b-src, data)"
  exit 0
fi
say "run $RUN_ID nonce=$RUN_NONCE -> box $HOST:$PORT; receipts -> $RUN_DIR/p41; deadline epoch $DEADLINE"
# The workdir is intentionally fixed for the box-side scripts, so clear it synchronously before staging. The child
# also writes the nonce; the controller never manufactures the child's proof of start.
$SSH "rm -rf -- $W && mkdir -p $W/logs && printf '%s\n' '$E4B_RENT_INSTANCE_ID' > $W/INSTANCE_ID" || { say "stage failed: synchronous remote cleanup"; exit 20; }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P "$PORT" "$HARNESS" "$RUNSH" "$ADMIT" "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W || exit 20; nohup env $PASS bash p41_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s) + $START_WAIT)); while [ \$(date +%s) -lt \$end ]; do observed=\$(cat P41_RUN_NONCE 2>/dev/null || true); [ \"\$observed\" = '$RUN_NONCE' ] && { echo started:\$child; exit 0; }; if ! kill -0 \$child 2>/dev/null; then wait \$child; child_rc=\$?; echo child-exited-before-nonce:rc=\$child_rc >&2; [ \$child_rc -ne 0 ] && exit \$child_rc; exit 125; fi; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the current run nonce"; exit 21; }
say "lane started on the box; polling TP_DONE every $POLL s"
LAST=""
while :; do
  now=$(date +%s)
  if $SSH "test -f $W/TP_DONE.$RUN_NONCE" 2>/dev/null; then say "current-run TP_DONE seen"; break; fi
  if [ "$now" -ge $((DEADLINE - POLL)) ]; then say "deadline reached without TP_DONE -- fetching what exists"; break; fi
  line=$($SSH "tail -n 1 $W/summary.txt 2>/dev/null" 2>/dev/null | cut -c1-200); [ "$line" != "$LAST" ] && [ -n "$line" ] && { say "box: $line"; LAST=$line; }
  sleep "$POLL"
done
rm -rf "$RUN_DIR/p41" || { say "fetch failed: cannot clear local result directory"; exit 22; }
mkdir -p "$RUN_DIR/p41"
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude adapters --exclude e4b-src --exclude data --exclude 'venv*' "root@$HOST:$W/" "$RUN_DIR/p41/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p41" | wc -l | tr -d ' ') entries into $RUN_DIR/p41"
NONCE_FILE="$RUN_DIR/p41/P41_RUN_NONCE"
[ -f "$NONCE_FILE" ] || { say "fetched artifacts have no P41_RUN_NONCE"; exit 24; }
FETCHED_NONCE=$(python3 - "$NONCE_FILE" <<'PYN'
import pathlib, re, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
if re.fullmatch(rb"[0-9a-f]{64}\n", data) is None:
    raise SystemExit(1)
print(data[:-1].decode("ascii"))
PYN
) || { say "malformed P41_RUN_NONCE (expected exactly 64 lowercase hex bytes plus newline)"; exit 24; }
[ "$FETCHED_NONCE" = "$RUN_NONCE" ] || { say "stale or foreign P41_RUN_NONCE '$FETCHED_NONCE' (expected current invocation)"; exit 24; }
[ -f "$RUN_DIR/p41/TP_DONE.$RUN_NONCE" ] || { say "lane did not finish (no current-run TP_DONE)"; exit 23; }
RC_FILE="$RUN_DIR/p41/P41_EXIT_CODE.$RUN_NONCE"
[ -f "$RC_FILE" ] || { say "lane terminal marker has no P41_EXIT_CODE"; exit 24; }
LANE_RC=$(python3 - "$RC_FILE" <<'PYR'
import pathlib, re, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
if re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n", data) is None:
    raise SystemExit(1)
value = int(data[:-1])
if value > 255:
    raise SystemExit(1)
print(value)
PYR
) || { say "malformed P41_EXIT_CODE (expected canonical decimal 0..255 plus newline)"; exit 24; }
[ "$LANE_RC" -eq 0 ] || { say "lane failed with remote rc=$LANE_RC"; exit "$LANE_RC"; }
[ -f "$RUN_DIR/p41/BOX_REFUSED" ] && { say "box refused despite remote rc=0 (BOX_REFUSED)"; exit 12; }
[ -f "$RUN_DIR/p41/P41_SUCCESS.$RUN_NONCE" ] || { say "remote rc=0 without current-run P41_SUCCESS"; exit 24; }
# The run's VERDICT comes from the PRE-REGISTERED criteria the lane recorded -- admitted arms, VOID rows, fired STOP rules --
# and never from "the process exited cleanly" (e4b#495: p41-r1-granite-10 was filed OK/pass while a VOID row and STOP-1 sat in
# the same directory). p41_admit.py verdict re-reads the receipts, the outcome manifest, the STOPn markers, stop_state.json and
# summary.txt, cross-checks them, and fails closed: a criterion it cannot locate, parse or reconcile is `invalid`, not a pass.
# Only `pass` leaves here with the exit status of a clean run; the fetch is already complete, so no evidence is lost either way.
# (#490's guard -- a lane that admits nothing is not a success -- is preserved: zero admitted arms is `inconclusive`, nonzero.)
VERDICT_JSON="$RUN_DIR/p41/P41_VERDICT.$RUN_NONCE.json"
VERDICT_RC=0
VERDICT_OUT=$(python3 "$ADMIT" verdict "$RUN_DIR/p41" --nonce "$RUN_NONCE" --expected "$EXPECTED_ROWS" --out "$VERDICT_JSON" 2>&1) || VERDICT_RC=$?
printf '%s\n' "$VERDICT_OUT" | while IFS= read -r ln; do [ -n "$ln" ] && say "$ln"; done
case "$VERDICT_RC" in
  0) : ;;
  1) say "P41 FAIL: the pre-registration's own failure criteria fired in this run -- the rows stand, the run is not a pass (P41_VERDICT.$RUN_NONCE.json)"; exit 40 ;;
  2) say "P41 INCONCLUSIVE: the lane ran and the registered criteria leave nothing to read -- a real outcome, not a pass (P41_VERDICT.$RUN_NONCE.json)"; exit 41 ;;
  3) say "invalid P41 outcome evidence"; exit 24 ;;
  *) say "invalid P41 outcome evidence (the verdict step itself failed rc=$VERDICT_RC)"; exit 24 ;;
esac
ADMITTED=$(grep -c '^ADMIT OK' "$RUN_DIR/p41/summary.txt" 2>/dev/null || :); ADMITTED=${ADMITTED:-0}
VOIDED=$(grep -c '^ADMIT VOID' "$RUN_DIR/p41/summary.txt" 2>/dev/null || :); VOIDED=${VOIDED:-0}
STUBS=$(grep -c 'STUB ' "$RUN_DIR/p41/summary.txt" 2>/dev/null || :); STUBS=${STUBS:-0}
say "admission: $ADMITTED admitted, $VOIDED VOID, $STUBS stubs"
say "done"; exit 0
