#!/bin/bash
# P147's driver (bench/p147/PREREG-p147.md): runs on the controller, never on the card's host. It
#   1. refuses any run id but p147-a2000-1 / -2, a dirty checkout, or a HEAD that is not on origin/main;
#   2. claims gpu:a2000 on the bus (exclusive; a held card refuses with 75) for at most 8 h;
#   3. stages the closure listed in staged-p147.sha256 to the card host's gpu-dev share and starts p147_run.sh in the
#      gpu-dev container with E4B_SHA = this checkout's HEAD;
#   4. waits for P147_DONE (polling every 60 s, the 8 h cap), printing the box's P147_ lines as they appear;
#   5. fetches the run directory back and releases the claim, whatever happened.
# P147_DRIVE_DRYRUN=1 prints each remote step instead of running it.
set -uo pipefail
RUN_ID=${P147_RUN_ID:-${1:-}}
case "$RUN_ID" in
  p147-a2000-1|p147-a2000-2) SHAKE=0 ;;
  p147-dry-[0-9]*) SHAKE=1 ;;            # the shakedown: no real window is scored, so it may run before the merge
  *) echo "refusing: run id '$RUN_ID' is not p147-a2000-N (1 or 2) or p147-dry-N"; exit 78;; esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "refusing: the checkout is dirty (the launch commit is HEAD)"; exit 78; }
E4B_SHA=$(git -C "$REPO" rev-parse HEAD)
DRY=${P147_DRIVE_DRYRUN:-0}
if [ "$DRY" != 1 ] && [ "$SHAKE" = 0 ]; then
  git -C "$REPO" fetch -q origin main && git -C "$REPO" merge-base --is-ancestor "$E4B_SHA" origin/main \
    || { echo "refusing: HEAD $E4B_SHA is not on origin/main (launch from the merged registration)"; exit 78; }
fi
QNAP=${P147_CARD_HOST:?set P147_CARD_HOST to the ssh target of the card host (user@host) from the lab configuration}
DOCKER=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
REMOTE=/share/gpu-dev/p147/$RUN_ID; INSIDE=/workspace/p147/$RUN_ID
LOCAL=${P147_LOCAL:-$REPO/bench/p147/receipts/$RUN_ID}
ME=${P147_BUS_ME:?set P147_BUS_ME to the bus label this lane posts as, from the lab configuration}
BUS_CMD=${P147_BUS_CMD:?set P147_BUS_CMD to a command that runs one messenger line read on stdin, from the lab configuration}
bus(){   # bus <args...>: one messenger line through P147_BUS_CMD (stdin-piped)
  if [ "$DRY" = 1 ]; then echo "DRYRUN bus $*"; return 0; fi
  printf 'messenger %s\n' "$*" | bash -c "$BUS_CMD"
}
remote(){ if [ "$DRY" = 1 ]; then echo "DRYRUN remote $*"; else ssh "$QNAP" "$@"; fi; }
release(){ if [ "$DRY" = 1 ]; then bus release --me "$ME" gpu:a2000; else bus release --me "$ME" --text "'p147 $RUN_ID done'" gpu:a2000 >/dev/null 2>&1 || true; fi; }
bus claim --me "$ME" --for 8h --text "'P147 $RUN_ID: Mixtral attention-format quality premise, USD 0, correctness-class (no timing)'" gpu:a2000; rc=$?
[ "$rc" = 0 ] || { echo "refusing: gpu:a2000 is held (claim rc $rc)"; exit 75; }
trap release EXIT
FILES=$(awk '{print $2}' "$HERE/staged-p147.sha256")
src(){ case "$1" in   # every staged name to its one source (explicit: a glob found bench/q235's model_files.json)
  p147_box.py|p147_fakequant.py|p147_reduce.py|p147_run.sh|model_files.json) echo "$HERE/$1";;
  i8_fakequant.py) echo "$REPO/bench/i8dense/$1";;
  p97_box.py|p108_box.py|p109_box.py|p110_box.py|p125_box.py) echo "$REPO/bench/${1%_box.py}/$1";;
  p115_quality.py) echo "$REPO/bench/p115/$1";;
  quality_off_mixtral.json) echo "$REPO/bench/p141/receipts/p141-5090-2/$1";;
  *) echo "";; esac; }
check_sources(){ local f want got
  for f in $FILES; do
    [ -f "$(src "$f")" ] || { echo "refusing: no source for $f"; exit 78; }
    want=$(awk -v n="$f" '$2 == n {print $1}' "$HERE/staged-p147.sha256"); got=$(sha256sum "$(src "$f")" | cut -c1-64)
    [ "$want" = "$got" ] || { echo "refusing: $f's source is not at its pinned bytes"; exit 9; }
  done; }
check_sources
if [ "$DRY" = 1 ]; then
  echo "DRYRUN stage -> $QNAP:$REMOTE ($(echo "$FILES" | wc -w) files + staged-p147.sha256)"
else
  STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"; release' EXIT
  cp "$HERE/staged-p147.sha256" "$STAGE/"
  for f in $FILES; do cp "$(src "$f")" "$STAGE/$f" || { echo "refusing: no source for $f"; exit 78; }; done
  (cd "$STAGE" && sha256sum -c --quiet staged-p147.sha256) || { echo "refusing: a staged file is not at its pinned bytes"; exit 9; }
  remote "rm -rf $REMOTE && mkdir -p $REMOTE"
  tar -C "$STAGE" -cf - . | ssh "$QNAP" "cd $REMOTE && tar -xf -" || { echo "stage failed"; exit 1; }
fi
remote "$DOCKER exec -d gpu-dev bash -c 'cd $INSIDE && E4B_SHA=$E4B_SHA P147_SHAKEDOWN=$SHAKE bash p147_run.sh > outer.log 2>&1'"
echo "started $RUN_ID at $(date -u +%FT%TZ) (e4b $E4B_SHA)"
[ "$DRY" = 1 ] && { echo "DRYRUN wait for $REMOTE/P147_DONE, then fetch to $LOCAL"; exit 0; }
end=$(( $(date +%s) + 8 * 3600 )); seen=0
while [ "$(date +%s)" -lt "$end" ]; do
  lines=$(ssh "$QNAP" "grep -ah '^\[.*p147: \|^P147_' $REMOTE/summary.txt 2>/dev/null")
  n=$(printf '%s\n' "$lines" | grep -c .); [ "$n" -gt "$seen" ] && { printf '%s\n' "$lines" | tail -n $((n - seen)); seen=$n; }
  ssh "$QNAP" "test -f $REMOTE/P147_DONE" && break
  sleep 60
done
mkdir -p "$LOCAL"
ssh "$QNAP" "cd $REMOTE && tar -cf - summary.txt versions.txt closure.log install.log box.log outer.log out P147_DONE NOT_RUN 2>/dev/null" \
  | tar -xf - -C "$LOCAL"
code=$(cat "$LOCAL/P147_DONE" 2>/dev/null || echo "none")
echo "P147 $RUN_ID done: exit $code; records in $LOCAL"
[ "$code" = 0 ] && exit 0 || exit 1
