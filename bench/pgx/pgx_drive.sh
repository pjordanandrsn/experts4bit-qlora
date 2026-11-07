#!/bin/bash
# PGX's CONTROLLER side (bench/pgx/PREREG-pgx.md, Amendment 1): P238's driver (bench/p238/p238_drive.sh) by named edits -- this
# lane's run ids (the shakedown pgx-shake-1 and its retry ids pgx-shake-r1 to -r3; Amendment 3: the reading's boxes
# pgx-4090-1-2-1 / -2 and their retry ids pgx-4090-r<n>-2-<i>), its staged files (its own kit plus the shared card-clock burn and P39's NF4 bake), and the box it
# runs (PGX_BOX: shake). The fetch imports no links (rsync --no-links, bench/common rule 8). It stages the hash-pinned files to
# the rented box, starts pgx_run.sh, polls its TP_DONE marker with a heartbeat line, and fetches the run directory back. It does
# not create, destroy, or approve compute: pod-launch, which calls it, holds the provider key.
#   E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH
#   bash bench/pgx/pgx_drive.sh        (from a clean experts4bit-qlora checkout: the launch commit is its HEAD)
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [pgx_drive] $*"; }
# The driver's OWN exit code is what the launcher reads (it excludes a machine on 13 14 17 18), so its last step is the shared
# bench/common/drive_tail.sh: the host codes always pass through, and so do the box codes Amendment 1 registers; 23 no readable
# exit file, 24 the box said 0 without this nonce's PGX_SUCCESS marker, 26 any other code.
BOX_CODES="9 10 11 12 15 16 19 25 27 28 33 40 41 78 130"
. "$(cd "$(dirname "$0")/../.." && pwd)/bench/common/drive_tail.sh" || { say "refusing: bench/common/drive_tail.sh is missing"; exit 78; }
finish(){ drive_finish "$1" "$2" PGX "$BOX_CODES"; }
if [ -n "${PGX_DRIVE_TAIL_DIR:-}" ]; then finish "$PGX_DRIVE_TAIL_DIR" "${PGX_DRIVE_TAIL_NONCE:?}"; fi   # the tail, alone (tests)
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set"; exit 78; }
done
# The registered ids (Amendment 1): the shakedown and its three retry ids; (Amendment 3) the reading's two boxes at K = 2 and
# the retry ids that name the box they replace. Every other id is refused here and on the box.
case "$E4B_RENT_RUN_ID" in
  pgx-shake-1|pgx-shake-r[1-3])           BOX=shake;;
  pgx-4090-1-2-[12]|pgx-4090-r[1-3]-2-[12]) BOX=read;;
  pgx2-4090-1-2-[12]|pgx2-4090-r[1-3]-2-[12]) BOX=read;;   # PGX2 (PREREG-pgx2.md): the successor lane's ids
  *) say "refusing unregistered run ID: $E4B_RENT_RUN_ID"; exit 78;;
esac
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# The launch commit is the e4b this driver ships from: derived from the clean checkout HEAD, never an input. An override is refused;
# so is a dirty tree.
if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi
E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { say "refusing: the checkout is dirty"; exit 78; }
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
# staged flat, except the two hooks, which keep their directory (each is loaded by name from its own PYTHONPATH entry)
mkdir -p "$STAGE/hook" "$STAGE/hook_e4b" "$STAGE/prompts"
for f in bench/pgx/pgx_run.sh bench/pgx/pgx_drive.py bench/pgx/pgx_reduce.py bench/pgx/pgx_residency.py bench/pgx/pgx_bytes.py \
         bench/pgx/pgx_place_preview.py bench/pgx/pgx_tripwire.py bench/pgx/pgx_shake.py bench/pgx/pgx_box.py \
         bench/pgx/pgx_schedule.py bench/pgx/pgx_dram_bench.py bench/pgx/pgx_read.sh \
         bench/pgx/fit_profile.jsonl bench/pgx/bytes_per_expert.json bench/pgx/staged-pgx.sha256 \
         bench/train-routing-trace/replay.py bench/common/card_clock.py bench/p39/k8_bake.py; do
  cp "$REPO/$f" "$STAGE/" || { say "stage missing: $f"; exit 78; }
done
cp "$REPO/bench/sc1/sglang/install.sh" "$STAGE/sglang_install.sh" || { say "stage missing: bench/sc1/sglang/install.sh"; exit 78; }
for h in usercustomize sitecustomize; do cp "$REPO/bench/pgx/hook/$h.py" "$STAGE/hook/" || { say "stage missing: bench/pgx/hook/$h.py"; exit 78; }; done
cp "$REPO/bench/pgx/hook_e4b/sitecustomize.py" "$STAGE/hook_e4b/" || { say "stage missing: bench/pgx/hook_e4b/sitecustomize.py"; exit 78; }
for s in warm_c1 warm_c4 shift manifest; do cp "$REPO/bench/pgx/prompts/$s.json" "$STAGE/prompts/" || { say "stage missing: prompts/$s.json"; exit 78; }; done
(cd "$STAGE" && shasum -a 256 -c staged-pgx.sha256 > /dev/null) || { say "refusing: staged files differ from staged-pgx.sha256"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
W=/root/pgx
NONCE=$(python3 -c "import secrets; print(secrets.token_hex(8))")
if [ "${PGX_DRIVE_DRYRUN:-0}" = "1" ]; then
  echo "DRYRUN stage $(find "$STAGE" -type f | wc -l | tr -d ' ') files -> root@$HOST:$W ; run pgx_run.sh box $BOX ; poll TP_DONE until $DEADLINE ; fetch -> $RUN_DIR/pgx"
  exit 0
fi
say "run $E4B_RENT_RUN_ID (box $BOX) -> $HOST:$PORT; deadline $DEADLINE; e4b $E4B_SHA"
$SSH "test ! -e $W && mkdir -p $W" || { say "stage failed: remote dir exists or ssh refused"; exit 20; }
scp -q -r -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT "$STAGE"/* "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
# The run starts detached: `cd $W || exit 20;` first, so what is backgrounded is the nohup command alone, with its own
# redirections. Under `cd $W && ... &` bash backgrounds the whole and-list in a subshell that holds this ssh session's
# stdout until the run ends, and the poll loop below never ran (tests/test_drive_detach.py).
$SSH "cd $W || exit 20; PGX_RUN_ID=$E4B_RENT_RUN_ID PGX_BOX=$BOX PGX_DEADLINE_EPOCH=$DEADLINE PGX_RUN_NONCE=$NONCE E4B_SHA=$E4B_SHA \
  nohup bash pgx_run.sh > outer.log 2>&1 < /dev/null & echo started" || { say "start failed"; exit 21; }
while :; do
  now=$(date +%s)
  $SSH "test -e $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - 240)) ] && { say "deadline near without TP_DONE: fetching what exists"; break; }
  hb=$($SSH "tail -n 1 $W/outer.log 2>/dev/null | cut -c1-160; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' '" 2>/dev/null | tr '\n' '|')
  say "hb: ${hb:-<no answer>} left $((DEADLINE - now))s"
  sleep 60
done
OUT="$RUN_DIR/pgx"
mkdir -p "$OUT" && [ -z "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ] || { say "fetch failed: $OUT is not an empty directory"; exit 22; }
# the evidence only: no venv (GBs), no checkpoint or arena, no compile or JIT caches, no route-dump text over the store's
# 10 MiB rule (the residency summary the box computes from it travels instead); nothing over 10 MiB (the store rule).
# Amendment 2: work/'s JSON records travel (work/bake.json holds the bake's error and traceback); the rest of work/ (the NF4
# snapshot and the arena) does not. rsync's first matching rule wins, so the work/ rules come first.
PGX_RSYNC_EXCLUDES=(--include 'work/' --include 'work/*.json' --exclude 'work/*' --exclude 'venv*' --exclude 'hf-cache' --exclude 'sglang-cache' --exclude 'vllm-cache'
  --exclude '__pycache__' --exclude 'pycache-*' --exclude '.cache' --exclude '*.safetensors' --exclude '*.arena' --exclude '*.pt'
  --exclude '*.bin' --exclude 'gnf4-src' --max-size=10m)
big=$($SSH "cd $W && find . \( -name 'venv*' -o -name work -o -name hf-cache -o -name sglang-cache -o -name __pycache__ -o -name .cache \) -prune -o -type f -size +10M -print" 2>/dev/null | head -20 | tr '\n' ' ')
[ -z "$big" ] || say "left on the box (over 10 MiB): $big"
rsync -az --no-links -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" "${PGX_RSYNC_EXCLUDES[@]}" "root@$HOST:$W/" "$OUT/" || { say "fetch failed"; exit 22; }
say "fetched $(find "$OUT" -type f | wc -l | tr -d ' ') files into $OUT"
finish "$OUT" "$NONCE"
