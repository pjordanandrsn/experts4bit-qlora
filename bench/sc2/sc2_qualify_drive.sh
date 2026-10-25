#!/bin/bash
# bench/sc2/sc2_qualify_drive.sh -- SC2f Amendment 7's qualify-at-proof gate (SC2h Amendment 5, SC2i Amendment 4),
# CONTROLLER side: the launcher's --command for every SC2f/h/i box, proof and reading. Exit codes, three ways:
#   17 / 18  a host fault, READ and failed (the launcher's power-cap / host-floor codes: it excludes the machine);
#   25       probe_error: a record missing or unreadable, a HARNESS fault that voids the run and never excludes;
#   9        the burn's setup (apt / venv / torch download) failed: a harness fault, not a card verdict.
#   24       the lane driver exited 0 without THIS run's SC1_RUN_NONCE, SC1_SUCCESS.<nonce> and SC1_EXIT_CODE.<nonce> = 0
#            (SC2f Amendment 8): a malformed record, a harness fault, never a PASS.
# Before any install, on the rented card:
#   1. nvidia-smi enforced.power.limit (with power.limit and power.max_limit, recorded) must be 575 W, compared as a
#      float (bench/sc2/sc2_qualify.py power): READ at another limit 17, unreadable 25;
#   2. bench/common/card_clock.py burn --floor 2700 in a throwaway venv whose torch 2.8.0+cu128 is a pip DOWNLOAD (the
#      image ships no torch), about 3 GB and a few minutes, counted in each attempt's cost; the burn's own refusal
#      field decides: below_floor 18, probe_error 25. The burn's torch is not the arms': it qualifies the CARD only;
# then, on every box, a controller-side sampler (nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,
# power.draw --format=csv -lms 1000) starts on the box, bench/sc1/sc1_drive.sh runs unchanged, and the sampler is
# stopped and fetched. After a CLEAN run (exit 0) the sampler is judged over the whole timed run (sc2_qualify.py
# judge-run, card_clock's rule: no hard bit on any sample, SwPowerCap clear on >= 95 %): READ and failed 18,
# unreadable 25. After a clean PROOF, its own host records too (judge-proof, forensics.txt and host_facts.json, both
# always passed): READ at another limit 18, missing or unreadable 25. A run that did not finish clean keeps its own
# exit code; its records are fetched and judged for the record only. Every record lands in <run>/qualify/ (rsync
# --no-links, bench/common rule 8) before teardown. Each create consumes one of the lane's 3 attempts whatever its
# exit (sc2_qualify.py admit reads that from the receipt store). Nothing here creates, destroys or approves compute,
# and no staged kit entry changes: this script stages only card_clock.py, to /root/qualify, apart from the box's kit.
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
  find "$QD" -type l -print -quit | grep -q . && { say "refusing: a symlink in the fetched qualify records (bench/common rule 8): rc 25, a harness fault"; exit 25; }; return 0; }

# 1. the enforced power limit, before anything is installed
$SSH "nvidia-smi --query-gpu=enforced.power.limit,power.limit,power.max_limit --format=csv,noheader" > "$QD/power.csv" 2> "$QD/power.err" \
  || { say "nvidia-smi did not answer: $(head -c 200 "$QD/power.err")"; exit 10; }
find "$QD" -type l -print -quit | grep -q . && { say "refusing: a symlink in the qualify records (bench/common rule 8): rc 25, a harness fault"; exit 25; }
python3 "$Q" power --csv "$QD/power.csv" --out "$QD/power.json"; rc=$?
[ "$rc" = 25 ] && { say "the power read was unreadable: probe_error (rc 25, a harness fault; the machine is not excluded)"; exit 25; }
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
BRC=$(cat "$QD/box/burn.rc" 2>/dev/null); case "$BRC" in ''|*[!0-9]*) say "no burn rc: probe_error (rc 25)"; exit 25;; esac
if [ "$BRC" != 0 ]; then
  WHY=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('refusal') or '')" "$QD/box/card_clock.json" 2>/dev/null)
  [ "$WHY" = below_floor ] || { say "the burn did not read the card (refusal '${WHY:-none}', rc $BRC): probe_error (rc 25, a harness fault)"; exit 25; }
  say "QUALIFY REFUSED: card_clock below_floor (rc 18, a host fault, not a strike): $(tail -1 "$QD/box/card_clock.log" 2>/dev/null | cut -c1-200)"; exit 18
fi
say "qualified: $(python3 -c "import json; d=json.load(open('$QD/power.json')); print('enforced', d['enforced_w'], 'W, limit', d['power_limit_w'], 'W, max', d['power_max_limit_w'], 'W')") ; card_clock PASS"
$SSH "rm -rf -- $QW/venv" > /dev/null 2>&1 || true

# 3. the run sampler, the lane's box (unchanged), then the sampler stopped and fetched
$SSH "mkdir -p $QW && cd $QW && rm -f sampler.csv sampler.pid && (setsid nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,power.draw --format=csv -lms 1000 > sampler.csv 2> sampler.err < /dev/null & echo \$! > sampler.pid)" \
  || say "the sampler did not start (judged UNREAD after the run)"
bash "$REPO/bench/sc1/sc1_drive.sh"; DRC=$?
# sc1_drive.sh (held, unchanged) exits the box's rc even when it is 0 without SC1_SUCCESS.<nonce> (its last line): a
# clean exit here also needs THIS run's fetched success marker and an exit file reading 0, else the record is
# malformed (24), never a PASS (the class of e4b#2137's driver-tail fix)
if [ "$DRC" = 0 ]; then
  N=$(cat "$RUN_DIR/sc1/SC1_RUN_NONCE" 2>/dev/null)
  { [ -n "$N" ] && [ -f "$RUN_DIR/sc1/SC1_SUCCESS.$N" ] && [ "$(cat "$RUN_DIR/sc1/SC1_EXIT_CODE.$N" 2>/dev/null)" = 0 ]; } \
    || { say "the lane driver exited 0 without this run's success marker and exit code 0: a malformed record (24)"; DRC=24; }
fi
$SSH "cd $QW && kill \$(cat sampler.pid) 2>/dev/null; sleep 1" > /dev/null 2>&1 || true
fetch_q
python3 "$Q" judge-run --sampler "$QD/box/sampler.csv" --out "$QD/run_judge.json"; jr=$?
PJ=0
if [ "${SC1_PROVE:-0}" = 1 ]; then
  python3 "$Q" judge-proof --forensics "$RUN_DIR/sc1/forensics.txt" --host-facts "$RUN_DIR/host_facts.json" --out "$QD/proof_judge.json"; PJ=$?
fi
# a run that did not finish clean keeps its own exit code: the lane's rules classify it, and its records are never
# read as a host fault (they are fetched and judged above for the record only)
[ "$DRC" = 0 ] || exit "$DRC"
for c in "$jr" "$PJ"; do [ "$c" = 18 ] && { say "QUALIFY REFUSED after the run: a host record READ and failed (rc 18, a host fault, not a strike)"; exit 18; }; done
for c in "$jr" "$PJ"; do [ "$c" = 0 ] || { say "a host record after the run was unreadable: probe_error (rc 25, a harness fault; the machine is not excluded)"; exit 25; }; done
exit 0
