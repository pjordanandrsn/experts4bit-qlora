#!/bin/bash
# bench/moegen/rd1/rd1_run.sh -- lane RD1, BOX side (bench/moegen/rd1/RD1-PREREG.md). Started by bench/tc1/tc1_drive.sh as its
# TC1_RUNNER (TC1_EXTRA_STAGE carries this file and its pieces), so it speaks tc1_drive's contract: the nonce handshake
# (TC1_RUN_NONCE within 30 s), summary.txt one line per finished step, and TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> /
# TP_DONE.<nonce> at the end -- a refusal writes them too, so the controller reads a finished lane, never a hang.
#
# install grouped-nf4-gemm at GNF4_SHA -> tripwire -> the train anchor (strict, as tp1: a refused box ends the lane, exit 12)
# -> rd_probe.py over the registered grid -> rd_table.py. No checkpoint is fetched: the probe draws random NF4 stacks at each
# family's registered shapes. Nothing here creates, destroys or approves compute.
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
say(){ echo "[$(date -u +%FT%TZ)] rd1: $*"; }
finish(){ local rc=$1; [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null
  printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}
# RD1_REHEARSAL=1 (the $0 A2000 container rehearsal ONLY; tc1_drive does not forward it, so a rented box can never set it): an
# anchor refusal is recorded and the lane continues, and the grid shrinks to one family at seq 512.
REHEARSAL=${RD1_REHEARSAL:-0}
SEQS=512,2048; FAMS=olmoe,lfm2,ernie,graniteh,qwen3,nemotron,qwen36,mixtral
[ "$REHEARSAL" = 1 ] && { SEQS=512; FAMS=olmoe; echo "REHEARSAL (RD1_REHEARSAL=1): not a registered reading" | tee -a summary.txt; }
echo "RD1 SHAPE: seqs $SEQS; families $FAMS" | tee -a summary.txt

# ---------------------------------------------------------------- install + tripwire
command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: gnf4 @$GNF4_SHA"
perl -e 'alarm 1200; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip.log 2>&1 \
  || { tail -5 logs/pip.log; echo "PIP FAIL" | tee -a summary.txt; finish 9; }
python - <<'PYT' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; echo "TRIPWIRE FAIL" | tee -a summary.txt; finish 9; }
import importlib.metadata as md, json, os, torch, triton
import nf4_grouped, nf4_route
assert hasattr(nf4_route, "dequant_groups") and hasattr(nf4_route, "dense_forward"), "grouped-nf4-gemm without the routes RD1 reads"
assert hasattr(nf4_grouped, "build_group_tiles") and hasattr(nf4_grouped, "dgrad_4bit_grouped")
d = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json"))
got = d.get("vcs_info", {}).get("commit_id")
assert got == os.environ["GNF4_SHA"], f"grouped-nf4-gemm: installed {got} != pinned {os.environ['GNF4_SHA']}"
assert torch.cuda.is_available()
print("rd1 tripwire OK", md.version("grouped-nf4-gemm"), torch.__version__, triton.__version__, torch.cuda.get_device_name())
PYT
cat logs/tripwire.log | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt; lscpu | grep "Model name" | tee -a forensics.txt

# ---------------------------------------------------------------- box class (bench/train-anchor), strict as tp1, load-gated
# Amendment 1 (RD1-PREREG.md): three draws were refused on launch.self_pair alone (1.042, 1.042, 1.060 > 1.03) on two hosts, and
# TC1 amendment 33 measured host load from other tenants driving this kind of instability on these multi-tenant 5090 hosts. The
# anchor stays strict, but it is run only at host load1 <= LOAD_MAX (waiting up to LOAD_WAIT_S), at most ANCHOR_TRIES times;
# the last attempt stands. A sampler records /proc/loadavg every 5 s; the probe's own window is summarised into its receipt,
# and rd_table.py takes no decision from a probe whose median load1 exceeded LOAD_MAX.
LOAD_MAX=5.0; LOAD_WAIT_S=600; ANCHOR_TRIES=3
[ "$REHEARSAL" = 1 ] && LOAD_WAIT_S=30         # the QNAP host runs at load ~17: a rehearsal checks the path, not the wait
( while :; do echo "$(date -u +%s) $(cat /proc/loadavg)"; sleep 5; done ) > logs/loadavg.log 2>&1 &
SAMPLER=$!
load1(){ cut -d' ' -f1 /proc/loadavg; }
wait_load(){ local w=0
  until python -c "import sys; sys.exit(0 if float(open('/proc/loadavg').read().split()[0]) <= $LOAD_MAX else 1)"; do
    [ $w -ge $LOAD_WAIT_S ] && { echo "LOAD WAIT: load1 $(load1) > $LOAD_MAX after ${w}s; the anchor runs anyway" | tee -a summary.txt; return; }
    sleep 15; w=$((w + 15))
  done
  echo "LOAD OK: load1 $(load1) <= $LOAD_MAX after ${w}s" | tee -a summary.txt; }
t=1
while :; do
  wait_load
  say "train anchor (attempt $t of $ANCHOR_TRIES)"
  ANCHOR_OUT=$W/anchor.json perl -e 'alarm 900; exec @ARGV' python train_anchor.py > logs/anchor.log 2>&1
  python train_anchor_gate.py anchor.json | tee logs/anchor_gate.log; arc=${PIPESTATUS[0]}
  echo "ANCHOR attempt $t rc=$arc class=$(grep -E '^\s*class ' logs/anchor_gate.log | awk '{print $2}') load1 $(load1)" | tee -a summary.txt
  [ "$arc" -eq 0 ] || [ $t -ge $ANCHOR_TRIES ] && break
  mv anchor.json anchor.attempt$t.json; mv logs/anchor.log logs/anchor.attempt$t.log; mv logs/anchor_gate.log logs/anchor_gate.attempt$t.log
  t=$((t + 1))
done
echo "ANCHOR rc=$arc class=$(grep -E '^\s*class ' logs/anchor_gate.log | awk '{print $2}')" | tee -a summary.txt
if [ "$arc" -ne 0 ]; then
  [ "$REHEARSAL" = 1 ] || { echo "BOX REFUSED by train anchor" | tee -a summary.txt; finish 12; }
  echo "REHEARSAL: train anchor refused this card (rc=$arc); continuing" | tee -a summary.txt
fi

# ---------------------------------------------------------------- the probe and its table
say "probe"
P0=$(date -u +%s)
perl -e 'alarm 1800; exec @ARGV' python rd_probe.py --out receipts/rd1.json --seqs "$SEQS" --fams "$FAMS" > logs/probe.log 2>&1
prc=$?
P1=$(date -u +%s)
# amendment 1: the host load over the probe's own window, into its receipt (rd_table.py decides nothing above LOAD_MAX)
python - "$P0" "$P1" "$LOAD_MAX" <<'PYL' 2>&1 | tee -a summary.txt
import json, statistics, sys
p0, p1, gate = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
rows = [(int(f[0]), float(f[1])) for f in (l.split() for l in open("logs/loadavg.log")) if len(f) > 1]
vals = [v for t, v in rows if p0 <= t <= p1]
pre = [v for t, v in rows if p0 - 60 <= t < p0]          # the host before the probe's own threads count in load1
try:
    rec = json.load(open("receipts/rd1.json"))
except (OSError, ValueError):
    print("LOAD during probe: no receipt to annotate"); sys.exit(0)
rec["host_load1_probe"] = ({"median": statistics.median(vals), "max": max(vals), "samples": len(vals), "gate": gate,
                            "pre60_median": statistics.median(pre) if pre else None, "pre60_samples": len(pre)}
                           if vals else {"samples": 0, "gate": gate})
json.dump(rec, open("receipts/rd1.json", "w"), indent=1)
print(f"LOAD during probe: {rec['host_load1_probe']}")
PYL
CELLS=$(python -c "import json; print(len(json.load(open('receipts/rd1.json'))['cells']))" 2>/dev/null) || CELLS=0
echo "probe rc=$prc cells=$CELLS" | tee -a summary.txt
python rd_table.py receipts/rd1.json > RESULTS-rd1.txt 2>&1; cat RESULTS-rd1.txt | tee -a summary.txt
[ "$prc" = 0 ] || finish 30
finish 0
