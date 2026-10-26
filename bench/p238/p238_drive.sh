#!/bin/bash
# P238's CONTROLLER side (bench/p238/PREREG-p238.md): P236's driver (bench/p236/p236_drive.sh) by named edits -- this lane's run
# ids (the shakedown, the reading and its rerun, each optionally split by DRAW into an a/b box pair, and the retry ids), its staged
# files (its own kit plus the shared helpers it imports: the Unsloth counter, the card floor, tc1_arm, F15 v2, the release proof),
# and the box it runs (P238_BOX: all, K-i or shake). The fetch imports no links (rsync --no-links, bench/common rule 8). It stages the
# hash-pinned files to the rented box, starts p238_run.sh, polls its TP_DONE marker with a heartbeat line, and fetches the run
# directory back. It does not create, destroy, or approve compute: pod-launch, which calls it, holds the provider key.
#   E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH
#   bash bench/p238/p238_drive.sh        (from a clean experts4bit-qlora checkout: the launch commit is its HEAD)
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p238_drive] $*"; }
# The driver's OWN exit code is what the launcher reads (it excludes a machine on 13 14 17 18), so its last step is the shared
# bench/common/drive_tail.sh: the host codes always pass through, and so do the codes P238's PREREG registers; 23 no readable exit
# file, 24 the box said 0 without this nonce's P238_SUCCESS marker, 26 any other code.
BOX_CODES="9 10 13 15 16 18 27 41 78 130"
. "$(cd "$(dirname "$0")/../.." && pwd)/bench/common/drive_tail.sh" || { say "refusing: bench/common/drive_tail.sh is missing"; exit 78; }
finish(){ drive_finish "$1" "$2" P238 "$BOX_CODES"; }
if [ -n "${P238_DRIVE_TAIL_DIR:-}" ]; then finish "$P238_DRIVE_TAIL_DIR" "${P238_DRIVE_TAIL_NONCE:?}"; fi   # the tail, alone (tests)
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set"; exit 78; }
done
# The registered ids: a whole reading runs every process (BOX=all); p238-5090-<R>-<K>-<i> runs box i of a K-box split, its whole
# (draw, point) blocks (BOX=K-i; p238_schedule.blocks: by draw and point group, never by arm). The shakedown: BOX=shake.
case "$E4B_RENT_RUN_ID" in
  p238-shake-1|p238-shake-r[1-5])        BOX=shake;;
  p238-5090-[12]|p238-5090-r[1-3])        BOX=all;;
  p238-5090-[12]-2-[12]|p238-5090-[12]-4-[1-4]|p238-5090-[12]-8-[1-8]) BOX=${E4B_RENT_RUN_ID#p238-5090-[12]-};;
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
         bench/tc1/tc1_arm.py bench/tc1/f15v2.py bench/tc1/release_proof.py bench/tp4/tp4_alpaca.py; do
  cp "$REPO/$f" "$STAGE/" || { say "stage missing: $f"; exit 78; }
done
(cd "$STAGE" && shasum -a 256 -c staged-p238.sha256 > /dev/null) || { say "refusing: staged files differ from staged-p238.sha256"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
W=/root/p238
NONCE=$(python3 -c "import secrets; print(secrets.token_hex(8))")
if [ "${P238_DRIVE_DRYRUN:-0}" = "1" ]; then
  echo "DRYRUN stage $(ls "$STAGE" | wc -l | tr -d ' ') files -> root@$HOST:$W ; run p238_run.sh box $BOX ; poll TP_DONE until $DEADLINE ; fetch -> $RUN_DIR/p238"
  exit 0
fi
say "run $E4B_RENT_RUN_ID (box $BOX) -> $HOST:$PORT; deadline $DEADLINE; e4b $E4B_SHA"
$SSH "test ! -e $W && mkdir -p $W" || { say "stage failed: remote dir exists or ssh refused"; exit 20; }
scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT "$STAGE"/* "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W && P238_RUN_ID=$E4B_RENT_RUN_ID P238_BOX=$BOX P238_DEADLINE_EPOCH=$DEADLINE P238_RUN_NONCE=$NONCE E4B_SHA=$E4B_SHA \
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
# the evidence only (TC1's TC1_RSYNC_EXCLUDES, tc1_drive.sh): no venv (GBs), no adapters (tc1_arm --adapter-dir saves
# safetensors; the store's tensor check refuses them), no 40 MB alpaca source, no compile caches; and nothing over 10 MiB (the
# store rule). The reducer reads none of these (the tokens files reach J/token only through the receipt's tokens_per_step).
P238_RSYNC_EXCLUDES=(--exclude 'venv*' --exclude 'adapters' --exclude 'data/alpaca_data_cleaned.json' --exclude 'unsloth_compiled_cache'
  --exclude '__pycache__' --exclude '.cache' --exclude 'hf-cache' --exclude '*.safetensors' --exclude '*.pt' --exclude '*.pth'
  --exclude '*.bin' --max-size=10m)
big=$($SSH "cd $W && find . \( -name 'venv*' -o -name adapters -o -name unsloth_compiled_cache -o -name __pycache__ -o -name .cache \) -prune -o -type f -size +10M -print" 2>/dev/null | head -20 | tr '\n' ' ')
[ -z "$big" ] || say "left on the box (over 10 MiB): $big"
rsync -az --no-links -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" "${P238_RSYNC_EXCLUDES[@]}" "root@$HOST:$W/" "$OUT/" || { say "fetch failed"; exit 22; }
say "fetched $(find "$OUT" -type f | wc -l | tr -d ' ') files into $OUT"
finish "$OUT" "$NONCE"
