#!/bin/bash
# bench/p47/p47_drive.sh -- lane P47 (Gemma-4 diagnostic, #597), CONTROLLER side: the launcher's --command. Stages the KL instrument
# (bench/kl_fidelity.py, kl_paths.py, kl_prompts.py), P39's k8_bake.py + calib.json, P42's hook, and this lane's
# serve_stack.py / kl_serve.py (P44-b's, shared) / p47_run.sh; starts the lane detached under a fresh nonce; polls TP_DONE.<nonce>; fetches receipts. Nothing here
# creates, destroys or approves compute. Pattern: bench/p42/p42_drive.sh.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [p47_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); P39="$REPO/bench/p39"; P44="$REPO/bench/p44"; P49="$REPO/bench/p49"
KL="$REPO/bench"
STAGE="$HERE/p47_run.sh $P44/serve_stack.py $P44/kl_serve.py $P49/act_probe.py $KL/kl_fidelity.py $KL/kl_paths.py $KL/kl_prompts.py $KL/kl_prompts_heldout.py $P39/k8_bake.py $P39/calib.json $HERE/staged.sha256"
HOOK="$REPO/bench/p42/hook/usercustomize.py"   # P42's hook, referenced not copied (one file to keep in step)
for f in $STAGE $HOOK; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in hook/*) src="$REPO/bench/p42/$name";; p47_run.sh) src="$HERE/$name";; serve_stack.py|kl_serve.py) src="$P44/$name";; act_probe.py) src="$P49/$name";; kl_*.py) src="$KL/$name";; *) src="$P39/$name";; esac
  got=$(sha_of "$src"); [ "$got" = "$want" ] || { say "refusing: $src is $got, staged.sha256 says $want"; exit 78; }
done < "$HERE/staged.sha256"
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex ($E4B_SHA)"; exit 78;; esac; [ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not 40 chars"; exit 78; }
GNF4_SHA=${GNF4_SHA:-8b1acc9e103ed1798a82307503c1c63d667b59ee}   # grouped-nf4-gemm v0.32.0, the consumer CI pin
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${P47_POLL_S:-60}; W=/root/p47
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="P47_RUN_ID=$RUN_ID P47_RUN_NONCE=$NONCE P47_DEADLINE_EPOCH=$DEADLINE P47_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA P47_MIN_MBPS=${P47_MIN_MBPS:-20} P47_MIN_DISK_GB=${P47_MIN_DISK_GB:-120} P47_FAMILY=${P47_FAMILY:-gemma4diag} P47_NEED_S=${P47_NEED_S:-6000} P47_ACT_PROBE=${P47_ACT_PROBE:-0} P47_ACT_PROBE_N=${P47_ACT_PROBE_N:-8} P47_KL_SKIP=${P47_KL_SKIP:-0} P47_PROMPT_SET=${P47_PROMPT_SET:-committed} ${P47_ARMS:+P47_ARMS=\"$P47_ARMS\" }${P47_FAMILY2:+P47_FAMILY2=$P47_FAMILY2 }${P47_ARMS2:+P47_ARMS2=\"$P47_ARMS2\" }${P47_NEED_S2:+P47_NEED_S2=$P47_NEED_S2 }"
if [ "${P47_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash p47_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/p47"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA gnf4 $GNF4_SHA (from $REPO); receipts -> $RUN_DIR/p47; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/hook" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" && $SCP "$HOOK" "root@$HOST:$W/hook/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W || exit 20; nohup env $PASS bash p47_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat P47_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; sleep 1; done; echo 'no nonce handshake in 30 s'; exit 21" || { say "lane did not start"; exit 21; }
say "lane started; polling TP_DONE every ${POLL}s"; LAST=""
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  line=$($SSH "tail -n 1 $W/summary.txt 2>/dev/null" 2>/dev/null | cut -c1-200); [ -n "$line" ] && [ "$line" != "$LAST" ] && { say "box: $line"; LAST=$line; }
  sleep "$POLL"
done
rm -rf "$RUN_DIR/p47" && mkdir -p "$RUN_DIR/p47" || { say "fetch failed: local dir"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude 'work_*' --exclude 'refcache_*' --exclude '.cache' --exclude '__pycache__' "root@$HOST:$W/" "$RUN_DIR/p47/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/p47" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/p47/P47_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/p47/TP_DONE.$NONCE" ] || { say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/p47/P47_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/p47/P47_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
