#!/bin/bash
# bench/sc2/sc1g-r/sc1g_r_drive.sh -- lane SC1g amendment A4 (#846), box R, CONTROLLER side: the launcher's --command.
# Stages the reference scorer (sc1g_ref.py, sc1g_kl.py, bench/kl_fidelity.py), the five committed windows
# (bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-2/sc1g/k8_window_*.json) and their registered shas, each checked against
# staged-r.sha256; starts sc1g_r_run.sh detached under a fresh nonce; polls TP_DONE.<nonce>; fetches receipts.
# Nothing here creates, destroys or approves compute. Pattern: bench/p44/p44b_drive.sh.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [sc1g_r_drive] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); SC2="$REPO/bench/sc2"
WIN="$REPO/bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-2/sc1g"
STAGE="$HERE/sc1g_r_run.sh $SC2/sc1g_ref.py $SC2/sc1g_kl.py $REPO/bench/kl_fidelity.py $HERE/window_shas.json $HERE/staged-r.sha256"
WINS=""; for s in conv1 conv2 conv3 conv4 wikitext; do WINS="$WINS $WIN/k8_window_$s.json"; done
for f in $STAGE $WINS; do [ -s "$f" ] || { say "refusing: staged piece missing: $f"; exit 78; }; done
sha_of(){ (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d" " -f1; }
while read -r want name; do
  case "$want" in \#*|"") continue;; esac
  case "$name" in windows/*) src="$WIN/${name#windows/}";; sc1g_ref.py|sc1g_kl.py) src="$SC2/$name";; kl_fidelity.py) src="$REPO/bench/$name";; *) src="$HERE/$name";; esac
  got=$(sha_of "$src"); [ "$got" = "$want" ] || { say "refusing: $src is $got, staged-r.sha256 says $want"; exit 78; }
done < "$HERE/staged-r.sha256"
[ -z "$(git -C "$REPO" status --porcelain -- bench/sc2 bench/kl_fidelity.py bench/h2h-2026-10-02/sc1g 2>/dev/null)" ] || { say "refusing: the staged sources are dirty"; exit 78; }
GNF4_SHA=${GNF4_SHA:-dc8f94abfd868f149178623f6eb403dc8b892b02}   # grouped-nf4-gemm v0.41.0 (SC2g / SC1g's pin)
MODEL_REV=${MODEL_REV:-6cee5e81ee83917806bbde320786a8fb61efebee} # openai/gpt-oss-20b, SC2g / SC1g's pin
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR; RUN_ID=$E4B_RENT_RUN_ID; DEADLINE=$E4B_RENT_DEADLINE_EPOCH
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
POLL=${SC1G_R_POLL_S:-60}; W=/root/sc1g_r
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(32))') || { say "refusing: no nonce"; exit 78; }
PASS="SC1G_R_RUN_ID=$RUN_ID SC1G_R_RUN_NONCE=$NONCE SC1G_R_DEADLINE_EPOCH=$DEADLINE SC1G_R_INSTANCE_ID=$E4B_RENT_INSTANCE_ID GNF4_SHA=$GNF4_SHA MODEL_REV=$MODEL_REV"
if [ "${SC1G_R_DRIVE_DRYRUN:-0}" = "1" ]; then echo "DRYRUN stage -> root@$HOST:$W ; start: env $PASS bash sc1g_r_run.sh ; poll TP_DONE.$NONCE until $DEADLINE ; fetch -> $RUN_DIR/sc1g_r"; exit 0; fi
say "run $RUN_ID nonce=$NONCE -> $HOST:$PORT; gnf4 $GNF4_SHA; gpt-oss-20b @$MODEL_REV; receipts -> $RUN_DIR/sc1g_r; deadline $DEADLINE"
$SSH "rm -rf -- $W && mkdir -p $W/logs $W/windows" || { say "stage failed: remote cleanup"; exit 20; }
$SCP $STAGE "root@$HOST:$W/" && $SCP $WINS "root@$HOST:$W/windows/" || { say "stage failed: scp"; exit 20; }
$SSH "cd $W || exit 20; nohup env $PASS bash sc1g_r_run.sh > outer.log 2>&1 < /dev/null & child=\$!; end=\$((\$(date +%s)+30)); while [ \$(date +%s) -lt \$end ]; do [ \"\$(cat SC1G_R_RUN_NONCE 2>/dev/null)\" = '$NONCE' ] && { echo started:\$child; exit 0; }; sleep 1; done; echo 'no nonce handshake in 30 s'; exit 21" || { say "lane did not start"; exit 21; }
say "lane started; polling TP_DONE every ${POLL}s"; LAST=""
while :; do
  now=$(date +%s)
  $SSH "test -f $W/TP_DONE.$NONCE" 2>/dev/null && { say "TP_DONE seen"; break; }
  [ "$now" -ge $((DEADLINE - POLL)) ] && { say "deadline reached without TP_DONE -- fetching what exists"; break; }
  line=$($SSH "tail -n 1 $W/summary.txt 2>/dev/null" 2>/dev/null | cut -c1-200); [ -n "$line" ] && [ "$line" != "$LAST" ] && { say "box: $line"; LAST=$line; }
  sleep "$POLL"
done
rm -rf "$RUN_DIR/sc1g_r" && mkdir -p "$RUN_DIR/sc1g_r" || { say "fetch failed: local dir"; exit 22; }
# A5: the full-vocabulary rows (ref/full/, ~0.82 GB per window) never enter the git receipt store: the receipt fetch excludes
# them and they go to $FULL_STORE/<run id>/ on the controller (outside any repo); their shas are in the receipt's r_calib.json
FULL_STORE=${SC1G_REF_FULL_STORE:-$HOME/sc1g-ref-full}
rsync -az -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" --exclude '.cache' --exclude '__pycache__' --exclude 'ref/full' "root@$HOST:$W/" "$RUN_DIR/sc1g_r/" || {
  say "rsync fetch failed -- falling back to tar over the same ssh options (the box may lack rsync)"
  $SSH "tar -C $W --exclude=./ref/full --exclude=./.cache -cf - ." | tar -C "$RUN_DIR/sc1g_r" -xf - || { say "fetch failed: rsync and tar"; exit 22; }; }
mkdir -p "$FULL_STORE/$RUN_ID" || { say "fetch failed: full-row store dir"; exit 22; }
rsync -a -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" "root@$HOST:$W/ref/full/" "$FULL_STORE/$RUN_ID/" 2>/dev/null ||
  $SSH "tar -C $W/ref/full -cf - ." | tar -C "$FULL_STORE/$RUN_ID" -xf - || say "full rows not fetched (absent if R stopped before writing them)"
(cd "$FULL_STORE/$RUN_ID" 2>/dev/null && (sha256sum ref_full_*.npy 2>/dev/null || shasum -a 256 ref_full_*.npy 2>/dev/null)) | sed "s/^/FULL_FETCHED /"
say "fetched $(ls "$RUN_DIR/sc1g_r" | wc -l | tr -d ' ') entries"
[ "$(cat "$RUN_DIR/sc1g_r/SC1G_R_RUN_NONCE" 2>/dev/null)" = "$NONCE" ] || { say "stale or foreign nonce in fetched artifacts"; exit 24; }
[ -f "$RUN_DIR/sc1g_r/TP_DONE.$NONCE" ] || { say "lane did not finish (no TP_DONE for this run)"; exit 23; }
RC=$(cat "$RUN_DIR/sc1g_r/SC1G_R_EXIT_CODE.$NONCE" 2>/dev/null); case "$RC" in ''|*[!0-9]*) say "malformed exit code"; exit 24;; esac
[ -f "$RUN_DIR/sc1g_r/SC1G_R_SUCCESS.$NONCE" ] && [ "$RC" = 0 ] || { say "lane exit rc=$RC without success marker"; exit "$RC"; }
say "lane complete rc=0 (R_OK)"; exit 0
