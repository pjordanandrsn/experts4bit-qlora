#!/usr/bin/env bash
# bench/p43/g4/p43_g4_drive.sh -- CONTROLLER SIDE of P43 T2 (the launcher's --command). Stages the committed probe,
# the box script and the registered tokens file on the rented box, runs the box script synchronously, fetches the
# work dir into the run's receipt dir. Reads the box from the launcher's environment (E4B_RENT_SSH_HOST/_PORT,
# _RUN_DIR). Nothing here creates, destroys or approves compute. Lineage: the mini's untracked probe_drive4.sh.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p43_g4_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR; do [ -n "${!v:-}" ] || { say "refusing: $v unset -- run as rent.py --command after a live pre-flight"; exit 78; }; done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT
SSH="ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
W=/root/probe
# The e4b the box installs is the e4b this driver ships from (pod-launch.sh has proven this checkout is heads.e4b).
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty"; exit 78; }
fi
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha ($E4B_SHA)"; exit 78; }
GNF4_SHA=${GNF4_SHA:-24f8c9fb22673a77219b8645b3bb8be50aa158c4}   # grouped-nf4-gemm v0.31.0 (P43-PREREG "Environment")
MID=${MID:-google/gemma-4-26B-A4B-it}; REV=${REV:-4d7ae4984b7db7de8f8457170b3f1a419ee76d52}
# The registered tokens file: tp4-c-4's tokens_gemma4.json (P43-PREREG pins BOTH hashes; refuse anything else).
TOK=${P43_G4_TOKENS:-$HOME/adertha-receipts/receipts/experts4bit-qlora/2026-09-10/tp4-c-4/tp4/tokens_gemma4.json}
TOK_SHA=${P43_G4_TOKENS_SHA:-4a1e8bf309b2654ca5ae0b56315bc6c37ee104cd524fc0bbf75c845a19684333}
[ -s "$TOK" ] || { say "refusing: tokens file missing: $TOK"; exit 78; }
GOT=$(shasum -a 256 "$TOK" 2>/dev/null | cut -c1-64 || sha256sum "$TOK" | cut -c1-64)
[ "$GOT" = "$TOK_SHA" ] || { say "refusing: tokens file sha256 $GOT != registered $TOK_SHA"; exit 78; }
for f in "$HERE/gemma4_layer1_probe.py" "$HERE/p43_g4_run.sh"; do [ -s "$f" ] || { say "refusing: missing $f"; exit 78; }; done
say "run ${E4B_RENT_RUN_ID:-?} -> $HOST:$PORT; e4b $E4B_SHA gnf4 $GNF4_SHA; $MID @ $REV; tokens sha ok"
$SSH "rm -rf -- $W && mkdir -p $W/logs" || { say "stage failed"; exit 20; }
$SCP "$HERE/gemma4_layer1_probe.py" "$HERE/p43_g4_run.sh" "$TOK" "root@$HOST:$W/" || { say "scp failed"; exit 20; }
say "running (synchronous: pre-flights, venv, fetch, one load + three sweeps + the oracle)"
$SSH "cd $W && E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA MID='$MID' REV=$REV P43_MIN_MBPS=${P43_MIN_MBPS:-20} P43_MIN_VRAM_GB=${P43_MIN_VRAM_GB:-80} P43_ROWS=${P43_ROWS:-128} P43_CHUNK=${P43_CHUNK:-8} bash p43_g4_run.sh" 2>&1 | sed 's/^/    /'
rc=${PIPESTATUS[0]}
say "box returned rc=$rc; fetching"
mkdir -p "$E4B_RENT_RUN_DIR/probe"
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude 'venv*' --exclude '.cache' "root@$HOST:$W/" "$E4B_RENT_RUN_DIR/probe/" || { say "fetch failed"; exit 22; }
say "fetched $(ls "$E4B_RENT_RUN_DIR/probe" | wc -l | tr -d ' ') entries"
[ -f "$E4B_RENT_RUN_DIR/probe/PROBE_SUCCESS" ] || { say "probe did not succeed (rc=$rc; REFUSAL: $(cat "$E4B_RENT_RUN_DIR/probe/REFUSAL" 2>/dev/null || echo none))"; exit "${rc:-1}"; }
say "probe complete"; exit 0
