#!/bin/bash
# bench/tp4/tp4_drive.sh -- lane tp4, CONTROLLER side: the launcher's --command (adertha-agents tools/pod-launch.sh ->
# adertha.compute.rent). Reads the box from the launcher's environment (E4B_RENT_SSH_HOST/_PORT, _RUN_DIR, _RUN_ID,
# _DEADLINE_EPOCH, _INSTANCE_ID), stages the tp4 pieces + the two dataset helpers + the HF token, starts tp4_run.sh
# detached under a fresh nonce, polls TP_DONE.<nonce> with a per-poll HEARTBEAT line (box summary tail + GPU util/mem +
# workdir size; a stall is REPORTED, never acted on), fetches receipts (never the venvs, caches or adapters).
# Pattern: bench/p39/p39_drive.sh.   TP4_BOX=A|B|C (required).   Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [tp4_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID TP4_BOX; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
case "$TP4_BOX" in A|B|C) ;; *) say "refusing: TP4_BOX must be A, B or C"; exit 78;; esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
STAGE="$HERE/tp4_run.sh $HERE/tp4_arm.py $HERE/tp4_reduce.py $HERE/tp4_alpaca.py $REPO/bench/flagship-matrix/drivers/n9_datasets.py $REPO/bench/flagship-matrix/ds_manifest.json"
for f in $STAGE; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# The e4b the BOX installs is the e4b this driver ships from: the launcher has proven this checkout is the manifest's heads.e4b
# (pod-launch.sh check_tree). Derived, never a literal; refused if the tree is dirty (p39-box1b-3's lesson).
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
GNF4_SHA=${GNF4_SHA:-d9fd170d83305fbe3f9b5ae2e661d45307c0a2c5}   # grouped-nf4-gemm main on 2026-09-10 (TP4-PREREG "Environments")
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char hex sha ($GNF4_SHA)"; exit 78; }
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${TP4_POLL_S:-60}; STALL_S=${TP4_STALL_S:-900}; W=/root/tp4
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/hf/token}
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="TP4_BOX=$TP4_BOX TP4_RUN_ID=$RUN_ID TP4_RUN_NONCE=$NONCE TP4_DEADLINE_EPOCH=$DEADLINE TP4_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA"
for v in TP4_FAMILIES TP4_SKIP TP4_UNSLOTH_VERSION TP4_UNSLOTH_ZOO_VERSION TP4_PIN_FALLBACK TP4_STEPS; do [ -n "${!v:-}" ] && PASS="$PASS $v=${!v}"; done
if [ "${TP4_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash tp4_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/tp4"; exit 0; fi
say "run $RUN_ID box $TP4_BOX nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA (from $REPO); gnf4 $GNF4_SHA; receipts -> $RUN_DIR/tp4; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/data /root/.cache/huggingface" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" || { say "stage failed: scp"; exit 20; }
if [ -s "$HF_TOKEN_FILE" ]; then   # HF pulls run authenticated (unauthenticated shards throttle to 400-1100 s); the token never appears in a command line
  $SCP "$HF_TOKEN_FILE" "root@$HOST:/root/.cache/huggingface/token" && $SSH "chmod 600 /root/.cache/huggingface/token" || { say "stage failed: hf token"; exit 20; }
  say "hf token staged"
else
  say "no hf token file at $HF_TOKEN_FILE -- pulls run unauthenticated (every registered checkpoint is ungated)"
fi
$SSH "cd $W || exit 20; nohup env $PASS bash tp4_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat TP4_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
say "lane started; polling TP_DONE every ${POLL}s with a heartbeat (stall reported after ${STALL_S}s of no change and idle GPU; never acted on)"
LAST=""; LAST_CHANGE=$(date +%s)
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  hb=$($SSH "echo \"\$(tail -n 1 $W/summary.txt 2>/dev/null | cut -c1-160) | gpu \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') | du \$(du -sm $W 2>/dev/null | cut -f1)M | disk \$(df -h /root | tail -1 | awk '{print \$4}')\"" 2>/dev/null)
  line=${hb%% | gpu*}; util=$(echo "$hb" | sed -n 's/.*| gpu \([0-9]*\),.*/\1/p')
  if [ "$line" != "$LAST" ]; then LAST=$line; LAST_CHANGE=$now; fi
  stall=""; if [ $((now - LAST_CHANGE)) -ge "$STALL_S" ] && [ "${util:-0}" -eq 0 ] 2>/dev/null; then stall=" STALL? (no summary change for $((now - LAST_CHANGE))s and GPU idle -- LOOK, do not kill)"; fi
  say "hb: ${hb:-<no answer>} | left $((DEADLINE - now))s$stall"
  sleep "$POLL"
done
rm -rf "$RUN_DIR/tp4" && mkdir -p "$RUN_DIR/tp4" || { say "fetch failed: local dir"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude 'venv*' --exclude '.cache' --exclude 'adapters' --exclude 'data/alpaca_data_cleaned.json' "root@$HOST:$W/" "$RUN_DIR/tp4/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/tp4" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/tp4/TP4_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/tp4/TP_DONE.$NONCE" ] || { say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/tp4/TP4_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/tp4/TP4_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "box $TP4_BOX complete rc=0"; exit 0
