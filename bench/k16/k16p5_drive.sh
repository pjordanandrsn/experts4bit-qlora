#!/bin/bash
# bench/p42/k16p5_drive.sh -- lane K16 P5, CONTROLLER side: the launcher's --command. Stages P39's harness
# pieces, P42's hook and this lane's k16p5_run.sh, starts the lane detached under a fresh nonce, polls TP_DONE.<nonce>,
# fetches receipts. One box, no arguments: the lane is a 2x2 census, not a parameterised experiment.
# Nothing here creates, destroys or approves compute.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [k16p5_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
# step_decomp.py / k8_bake.py / calib.json are P39's, referenced rather than copied: a
# 3,300-line duplicate is not evidence, it is a second thing to keep in step. staged-p5.sha256
# pins them by path, so a drift in either lane refuses here.
P39="$REPO/bench/p39"
STAGE="$HERE/k16p5_run.sh $P39/step_decomp.py $P39/k8_bake.py $P39/calib.json $HERE/staged-p5.sha256"
HOOK="$REPO/bench/p42/hook/usercustomize.py"
for f in $STAGE $HOOK; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
# staged-p5.sha256 names the files as the BOX will see them, so the check here resolves each
# name to its source and compares hashes rather than running `sha256sum -c` against paths
# that only exist after staging.
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in k16p5_run.sh) src="$HERE/$name";; hook/*) src="$REPO/bench/p42/$name";; *) src="$P39/$name";; esac
  got=$(sha_of "$src")
  [ "$got" = "$want" ] || { say "refusing: $src is $got, staged-p5.sha256 says $want"; exit 78; }
done < "$HERE/staged-p5.sha256"
# The e4b the BOX installs must be the e4b this driver ships from -- the launcher has already
# proven that checkout is the manifest's heads.e4b (pod-launch.sh check_tree). A literal default
# here is exactly the unread constant the manifest design exists to remove, and it bit: p39-box1b-3
# installed ef6d532 (pre-#537) while the checkout was 3d22719, so the box rebuilt the same 8-layer
# assignment and VOIDed on its own 48-layer check. Derived, and refused if the tree is dirty.
if [ -z "${E4B_SHA:-}" ]; then
  E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || { say "refusing: cannot read HEAD of $REPO"; exit 78; }
  [ -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] || { say "refusing: $REPO is dirty -- the box would install a commit that does not describe this tree"; exit 78; }
fi
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char hex sha ($E4B_SHA)"; exit 78; }
GNF4_SHA=${GNF4_SHA:?set GNF4_SHA to the grouped-nf4-gemm cut that ships int4_smallm (>= 0.32.0 / the K16 lane merge commit)}
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new -P $PORT"
POLL=${K16P5_POLL_S:-60}; W=/root/k16p5
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="K16P5_RUN_ID=$RUN_ID K16P5_RUN_NONCE=$NONCE K16P5_DEADLINE_EPOCH=$DEADLINE K16P5_INSTANCE_ID=$E4B_RENT_INSTANCE_ID E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA "
if [ "${K16P5_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash k16p5_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/k16p5"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; e4b $E4B_SHA (from $REPO); receipts -> $RUN_DIR/k16p5; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/hook" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" && $SCP "$HOOK" "root@$HOST:$W/hook/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W || exit 20; nohup env $PASS bash k16p5_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat K16P5_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; kill -0 \$child 2>/dev/null || { wait \$child; echo child-exited-early:rc=\$? >&2; exit 125; }; sleep 1; done; echo nonce-handshake-timeout >&2; exit 124" || { say "start failed: child did not bind the nonce"; exit 21; }
say "lane started; polling TP_DONE every ${POLL}s"; LAST=""
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  line=$($SSH "tail -n 1 $W/summary.txt 2>/dev/null" 2>/dev/null | cut -c1-200); [ -n "$line" ] && [ "$line" != "$LAST" ] && { say "box: $line"; LAST=$line; }
  sleep "$POLL"
done
rm -rf "$RUN_DIR/k16p5" && mkdir -p "$RUN_DIR/k16p5" || { say "fetch failed: local dir"; exit 22; }
rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT" --exclude 'artifact*/payloads/layer_*' --exclude work_qwen3 --exclude 'venv*' --exclude '.cache' "root@$HOST:$W/" "$RUN_DIR/k16p5/" || { say "fetch failed: rsync"; exit 22; }
say "fetched $(ls "$RUN_DIR/k16p5" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/k16p5/K16P5_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/k16p5/TP_DONE.$NONCE" ] || { say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/k16p5/K16P5_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/k16p5/K16P5_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0"; exit 0
