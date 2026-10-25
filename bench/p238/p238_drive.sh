#!/bin/bash
# P238's CONTROLLER side (bench/p238/PREREG-p238.md): P236's driver (bench/p236/p236_drive.sh) by named edits -- this lane's run
# ids (the shakedown, the reading and its rerun, each optionally split by DRAW into an a/b box pair, and the retry ids), its staged
# files (its own kit plus the shared helpers it imports: the Unsloth counter, the card floor, tc1_arm, F15 v2, the release proof),
# and the draws a box runs (P238_DRAWS). The fetch imports no links (rsync --no-links, bench/common rule 8). It stages the
# hash-pinned files to the rented box, starts p238_run.sh, polls its TP_DONE marker with a heartbeat line, and fetches the run
# directory back. It does not create, destroy, or approve compute: pod-launch, which calls it, holds the provider key.
#   E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH
#   bash bench/p238/p238_drive.sh        (from a clean experts4bit-qlora checkout: the launch commit is its HEAD)
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p238_drive] $*"; }
# The driver's OWN exit code is what the launcher reads (it writes "command exited {rc}" and excludes the machine on 13/14/17/18),
# so the box's registered code passes through unchanged. 23: no readable exit file was fetched. 24: the box said 0 but wrote no
# P238_SUCCESS marker (never read as success). 26: the box wrote a code the PREREG does not register.
BOX_CODES=" 9 10 13 15 16 18 27 28 41 78 130 "
finish(){   # finish OUT NONCE -> exits with the driver's code
  local out=$1 nonce=$2 rc
  [ -f "$out/P238_EXIT_CODE.$nonce" ] || { say "no exit file P238_EXIT_CODE.$nonce in the fetch"; exit 23; }
  rc=$(tr -d '[:space:]' < "$out/P238_EXIT_CODE.$nonce")
  case "$rc" in ''|*[!0-9]*) say "unreadable exit file: '$rc'"; exit 23;; esac
  say "box rc=$rc"
  if [ "$rc" = 0 ]; then
    [ -f "$out/P238_SUCCESS.$nonce" ] && exit 0
    say "box rc 0 without P238_SUCCESS.$nonce: not a success"; exit 24
  fi
  case "$BOX_CODES" in *" $rc "*) exit "$rc";; esac
  say "box rc $rc is not a registered code"; exit 26
}
if [ -n "${P238_DRIVE_TAIL_DIR:-}" ]; then finish "$P238_DRIVE_TAIL_DIR" "${P238_DRIVE_TAIL_NONCE:?}"; fi   # the tail, alone (tests)
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set"; exit 78; }
done
# The registered ids: a whole box runs both draws; an "a" box runs draw 1 and a "b" box draw 2 (the split rule, by draw, never by
# arm). The shakedown runs its own reduced schedule (P238_DRAWS=shake).
case "$E4B_RENT_RUN_ID" in
  p238-shake-1|p238-shake-r[12])          DRAWS=shake;;
  p238-5090-[12]|p238-5090-r[1-3])        DRAWS=1,2;;
  p238-5090-[12]a)                        DRAWS=1;;
  p238-5090-[12]b)                        DRAWS=2;;
  *) say "refusing unregistered run ID: $E4B_RENT_RUN_ID"; exit 78;;
esac
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# The launch commit is the e4b this driver ships from: derived from the clean checkout HEAD, never an input. An override is refused;
# so is a dirty tree.
if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi
E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { say "refusing: the checkout is dirty"; exit 78; }
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
for f in bench/p238/p238_run.sh bench/p238/p238_op.py bench/p238/p238_hook.py bench/p238/p238_bytes.py bench/p238/p238_reduce.py \
         bench/p238/p238_schedule.py bench/p238/staged-p238.sha256 bench/common/unsloth_moe_counter.py bench/common/card_clock.py \
         bench/tc1/tc1_arm.py bench/tc1/f15v2.py bench/tc1/release_proof.py; do
  cp "$REPO/$f" "$STAGE/" || { say "stage missing: $f"; exit 78; }
done
(cd "$STAGE" && shasum -a 256 -c staged-p238.sha256 > /dev/null) || { say "refusing: staged files differ from staged-p238.sha256"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
W=/root/p238
NONCE=$(python3 -c "import secrets; print(secrets.token_hex(8))")
if [ "${P238_DRIVE_DRYRUN:-0}" = "1" ]; then
  echo "DRYRUN stage $(ls "$STAGE" | wc -l | tr -d ' ') files -> root@$HOST:$W ; run p238_run.sh draws $DRAWS ; poll TP_DONE until $DEADLINE ; fetch -> $RUN_DIR/p238"
  exit 0
fi
say "run $E4B_RENT_RUN_ID (draws $DRAWS) -> $HOST:$PORT; deadline $DEADLINE; e4b $E4B_SHA"
$SSH "test ! -e $W && mkdir -p $W" || { say "stage failed: remote dir exists or ssh refused"; exit 20; }
scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT "$STAGE"/* "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W && P238_RUN_ID=$E4B_RENT_RUN_ID P238_DRAWS=$DRAWS P238_DEADLINE_EPOCH=$DEADLINE P238_RUN_NONCE=$NONCE E4B_SHA=$E4B_SHA \
  nohup bash p238_run.sh > outer.log 2>&1 < /dev/null & echo started" || { say "start failed"; exit 21; }
while :; do
  now=$(date +%s)
  $SSH "test -e $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - 240)) ] && { say "deadline near without TP_DONE: fetching what exists"; break; }
  hb=$($SSH "tail -n 1 $W/outer.log 2>/dev/null | cut -c1-160; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' '" 2>/dev/null | tr '\n' '|')
  say "hb: ${hb:-<no answer>} left $((DEADLINE - now))s"
  sleep 60
done
OUT="$RUN_DIR/p238"
mkdir -p "$OUT" && [ -z "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ] || { say "fetch failed: $OUT is not an empty directory"; exit 22; }
rsync -az --no-links -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" "root@$HOST:$W/" "$OUT/" || { say "fetch failed"; exit 22; }
say "fetched $(find "$OUT" -type f | wc -l | tr -d ' ') files into $OUT"
finish "$OUT" "$NONCE"
