#!/bin/bash
# bench/sc2/sc2_qualify_drive.sh -- SC2f Amendment 7's qualify-at-proof gate (SC2h Amendment 5, SC2i Amendment 4),
# CONTROLLER side: the launcher's --command for every SC2f/h/i box, proof and reading. Before any install, on the
# rented card:
#   1. nvidia-smi enforced.power.limit (with power.limit and power.max_limit, recorded) must be 575 W, compared as a
#      float (bench/sc2/sc2_qualify.py power); else rc 17, the launcher's power-cap code;
#   2. bench/common/card_clock.py burn --floor 2700 (hard reason bits refuse on any sample; SwPowerCap clear on >= 95 %
#      of busy samples) in a throwaway venv whose torch 2.8.0+cu128 is a pip DOWNLOAD (the image ships no torch),
#      about 3 GB and a few minutes, counted in each attempt's cost; else rc 18, the launcher's host-floor code. The
#      burn's torch is not the arms' torch: it qualifies the CARD only;
# then it runs bench/sc1/sc1_drive.sh unchanged, with the launcher's environment, and after a CLEAN proof (SC1_PROVE=1,
# exit 0) judges the proof's own host records (sc2_qualify.py judge-proof: forensics.txt's power.limit and the
# launcher's host_facts.json at 575 W); a refusal there is rc 18 too. The proof records no per-block clock samples, so
# the "no capped block" term is the pre-install burn's. Every record lands in <run>/qualify/ (fetched by rsync --no-links, bench/common rule 8) before
# teardown. A refusal is a host fault, never a strike; the create still consumes one of the lane's 3 attempts
# (sc2_qualify.py admit reads that from the receipt store). Nothing here creates, destroys or approves compute, and
# no staged kit entry changes: this script stages only card_clock.py, to /root/qualify, apart from the box's kit.
set -uo pipefail
say(){ echo "[$(date -u +%FT%TZ)] [sc2_qualify] $*"; }
for v in E4B_RENT_SSH_HOST E4B_RENT_SSH_PORT E4B_RENT_SSH_OPTS E4B_RENT_RUN_DIR E4B_RENT_RUN_ID E4B_RENT_DEADLINE_EPOCH E4B_RENT_INSTANCE_ID SC1_BOX; do
  [ -n "${!v:-}" ] || { say "refusing: $v is not set -- run as rent.py --command after a live pre-flight"; exit 78; }
done
case "$SC1_BOX" in N|O|P) ;; *) say "refusing: the qualify gate is SC2f/h/i's (boxes N, O, P), not box $SC1_BOX"; exit 78;; esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
CC="$REPO/bench/common/card_clock.py"; Q="$HERE/sc2_qualify.py"
for f in "$CC" "$Q" "$REPO/bench/sc1/sc1_drive.sh"; do [ -s "$f" ] || { say "refusing: missing $f"; exit 78; }; done
HOST=$E4B_RENT_SSH_HOST; PORT=$E4B_RENT_SSH_PORT; RUN_DIR=$E4B_RENT_RUN_DIR
SSH="ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes $E4B_RENT_SSH_OPTS -P $PORT"
QD="$RUN_DIR/qualify"; QW=/root/qualify
mkdir -p "$QD" || { say "refusing: no local qualify dir"; exit 78; }
if [ "${SC2Q_DRIVE_DRYRUN:-0}" = "1" ]; then
  echo "DRYRUN power -> $QD/power.csv ; burn: stage card_clock.py -> root@$HOST:$QW, venv + torch 2.8.0+cu128, burn --floor 2700 ; then SC1_BOX=$SC1_BOX bash bench/sc1/sc1_drive.sh ; judge-proof if SC1_PROVE=1"; exit 0
fi
fetch_q(){ rsync -az --no-links -e "ssh -o BatchMode=yes $E4B_RENT_SSH_OPTS -p $PORT" "root@$HOST:$QW/" "$QD/box/" > /dev/null 2>&1 || true
  [ -z "$(find "$QD" -type l | head -1)" ] || { say "refusing: a symlink in the fetched qualify records (bench/common rule 8)"; exit 22; }; }

# 1. the enforced power limit, before anything is installed
$SSH "nvidia-smi --query-gpu=enforced.power.limit,power.limit,power.max_limit --format=csv,noheader" > "$QD/power.csv" 2> "$QD/power.err" \
  || { say "nvidia-smi did not answer: $(head -c 200 "$QD/power.err")"; exit 10; }
python3 "$Q" power --csv "$QD/power.csv" --out "$QD/power.json"; rc=$?
[ "$rc" = 0 ] || { say "QUALIFY REFUSED: enforced power limit (rc 17, a host fault, not a strike)"; exit 17; }

# 2. the card's loaded clock (card_clock.py, TC1 amendment 80's floor), on a throwaway venv apart from the box's kit
$SSH "rm -rf -- $QW && mkdir -p $QW" || { say "stage failed: remote dir"; exit 20; }
$SCP "$CC" "root@$HOST:$QW/" || { say "stage failed: scp card_clock.py"; exit 20; }
$SSH "cd $QW || exit 20
  need=0; python3 -c 'import venv, ensurepip' 2> /dev/null || need=1
  if [ \$need = 1 ]; then perl -e 'alarm 900; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq --no-install-recommends python3-venv python3-pip ca-certificates' > apt.log 2>&1 || exit 9; fi
  python3 -m venv venv > venv.log 2>&1 || exit 9
  perl -e 'alarm 1200; exec @ARGV' venv/bin/python -m pip install -q --no-input 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128 > pip_torch.log 2>&1 || exit 9
  perl -e 'alarm 300; exec @ARGV' venv/bin/python card_clock.py burn --floor 2700 --out card_clock.json > card_clock.log 2>&1; echo \$? > burn.rc; exit 0" \
  || { fetch_q; say "the burn's setup failed (apt / venv / torch download): rc 9, not a card verdict"; exit 9; }
fetch_q
BRC=$(cat "$QD/box/burn.rc" 2>/dev/null); case "$BRC" in ''|*[!0-9]*) say "malformed burn rc"; exit 24;; esac
[ "$BRC" = 0 ] || { say "QUALIFY REFUSED: card_clock burn rc $BRC (rc 18, a host fault, not a strike): $(tail -1 "$QD/box/card_clock.log" 2>/dev/null | cut -c1-200)"; exit 18; }
say "qualified: $(python3 -c "import json; d=json.load(open('$QD/power.json')); print('enforced', d['enforced_w'], 'W, limit', d['power_limit_w'], 'W, max', d['power_max_limit_w'], 'W')") ; card_clock PASS"
$SSH "rm -rf -- $QW/venv" > /dev/null 2>&1 || true

# 3. the lane's box, unchanged
bash "$REPO/bench/sc1/sc1_drive.sh"; DRC=$?
[ "${SC1_PROVE:-0}" = 1 ] || exit "$DRC"
# a proof that did not finish clean keeps its own exit code: the lane's rules classify it, and a partial box.json is
# never read as a host fault
[ "$DRC" = 0 ] || exit "$DRC"
# 4. after a clean proof: its own host record
HF=(); [ -s "$RUN_DIR/host_facts.json" ] && HF=(--host-facts "$RUN_DIR/host_facts.json")
python3 "$Q" judge-proof --forensics "$RUN_DIR/sc1/forensics.txt" ${HF[@]+"${HF[@]}"} --out "$QD/proof_judge.json"; jrc=$?
[ "$jrc" = 0 ] || { say "QUALIFY REFUSED after the proof: its host record (rc 18, a host fault, not a strike)"; exit 18; }
exit "$DRC"
