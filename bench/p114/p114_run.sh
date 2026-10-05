#!/bin/bash
# bench/p114/p114_run.sh -- lane P114, BOX side (bench/p114/PREREG-p114.md). Started detached by p114_drive.sh with the
# run's nonce; P114_RUN_NONCE first, then P114_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and P114_SUCCESS.<nonce>
# only when the reducer read the passes. Pattern: bench/k30/k30_run.sh.
#
# The two energy harnesses behind experts4bit-qlora's energy rows, UNCHANGED (sha-pinned in staged.sha256):
#   bench_energy.py           (bench/_upstream/) one OLMoE-dims gate_up projection: native bf16 / dequant -> linear /
#                             bnb.matmul_4bit at decode M=1, prefill M=512, train fwd+bwd M=32; total J/op from
#                             nvidia-smi power.draw sampled at 50 ms over a 6 s op loop
#   bench_energy_excluded.py  (bench/) Part A the memory wall (allocation only), Part B J/token of the fused 4-bit MoE
#                             forward at batch 64 / 256 / 1024 / 4096
# each run three times on one rented RTX 5090, with nvidia-smi pmon logging every process during every pass. No model,
# no checkpoint, no download. Correctness first (p114_gate.py), and no pass runs if it fails.
#
# P114_REHEARSAL=1 (the $0 A2000 rehearsal only): lifts the card check. The reducer then reads VOID by construction
# (the wrong card), which is the rehearsal's expected verdict.
set -uo pipefail
W=/root/p114; mkdir -p $W/logs $W/runs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] p114: $*"; }
NONCE=${P114_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P114_RUN_NONCE.tmp && mv $W/P114_RUN_NONCE.tmp $W/P114_RUN_NONCE
MON=""
finish(){ local rc=$1; [ -n "$MON" ] && kill "$MON" 2>/dev/null; printf '%s\n' "$rc" > P114_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P114_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P114_RUN_ID P114_DEADLINE_EPOCH P114_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
REHEARSAL=${P114_REHEARSAL:-0}
BNB_VER=0.50.2                  # the released build e4b.train.energy-honest.a2000-bnb0502.2026-10-04 read
: > summary.txt; echo "$P114_INSTANCE_ID" > INSTANCE_ID
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p114/staged.sha256"; finish 9; }

python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,power.max_limit,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader \
  | sed "s/^/power.limit,power.max_limit,clocks.max.sm,clocks.max.mem,pcie.gen.max,pcie.width.max /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "card $GPU, rehearsal=$REHEARSAL" | tee -a summary.txt
# An exact match: a substring test on "5090" would also accept other 5090-named parts.
if [ "$REHEARSAL" != 1 ]; then
  [ "$GPU" = "NVIDIA GeForce RTX 5090" ] || { say "REFUSED: card is '$GPU', the lane registers the NVIDIA GeForce RTX 5090"; echo "refused: class $GPU" > REFUSAL; finish 15; }
fi
# The instrument IS nvidia-smi power.draw: a host that does not report it cannot be read (rc 17 names the host).
PW=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1 | tr -d ' ')
case "$PW" in ''|*[!0-9.]*) say "REFUSED: power.draw reads '$PW' on this host -- the energy instrument has no signal"; echo "refused: power.draw '$PW'" > REFUSAL; finish 17;; esac
echo "power.draw readable: $PW W at rest" | tee -a summary.txt

command -v git >/dev/null || { perl -e 'alarm 300; exec @ARGV' sh -c 'apt-get update -q && apt-get install -y -q git' > logs/apt_git.log 2>&1 || { tail -3 logs/apt_git.log; say "NO GIT"; finish 9; }; }
say "install e4b @$E4B_SHA + bitsandbytes $BNB_VER, the image's torch held by a constraint"
echo "torch==$(python -c 'import torch; print(torch.__version__.split("+")[0])')" > logs/constraints.txt
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input -c logs/constraints.txt \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" "bitsandbytes==$BNB_VER" > logs/pip.log 2>&1 \
  || { tail -3 logs/pip.log; say "PIP FAIL"; finish 9; }
WANT_E4B=$E4B_SHA WANT_BNB=$BNB_VER python - <<'PYT' > versions.txt || { cat versions.txt; say "TRIPWIRE FAIL"; finish 9; }
import json, os, subprocess, importlib.metadata as md
import torch, bitsandbytes
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
got = d.get("vcs_info", {}).get("commit_id")
print("experts4bit-qlora", md.version("experts4bit-qlora"), got)
print("bitsandbytes", bitsandbytes.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("driver,power.limit", subprocess.run(["nvidia-smi", "--query-gpu=driver_version,power.limit", "--format=csv,noheader"],
                                           capture_output=True, text=True).stdout.strip())
assert got == os.environ["WANT_E4B"], f"installed e4b is {got}, the pin is {os.environ['WANT_E4B']}"
assert bitsandbytes.__version__ == os.environ["WANT_BNB"], bitsandbytes.__version__
assert torch.__version__.startswith("2.8.0"), torch.__version__
PYT
cat versions.txt | tee -a summary.txt

left(){ echo $(( P114_DEADLINE_EPOCH - $(date +%s) - 600 )); }
say "correctness: the dequant and matmul_4bit arms compute the same projection; every arm and the Part B layer run"
perl -e 'alarm 600; exec @ARGV' python $W/p114_gate.py > logs/gate.log 2>&1; grc=$?
tail -2 logs/gate.log | tee -a summary.txt
[ "$grc" = 0 ] || { say "GATE FAIL rc=$grc -- no energy number is produced"; finish 21; }
python $W/p114_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "RULE SELF-TEST FAILED"; finish 21; }

# One pass: the pre-pass state, a per-process GPU monitor for the pass's whole length, the harness, its stamps.
one_pass(){ local harness=$1 tag=$2 cap=$3
  local a; a=$(left); [ "$a" -gt "$cap" ] && a=$cap
  [ "$a" -ge 120 ] || { say "STOP-2: no time for $tag"; echo "SKIPPED $tag host-limited deadline" >> summary.txt; return 30; }
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader > runs/pre_$tag.txt
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> runs/pre_$tag.txt 2>&1
  ( nvidia-smi pmon -s u -d 1 > runs/pmon_$tag.txt 2>&1 ) & MON=$!
  date -u +%FT%TZ > runs/t0_$tag.txt
  perl -e "alarm $a; exec @ARGV" python $W/$harness > runs/run_$tag.txt 2>&1; local rc=$?
  date -u +%FT%TZ > runs/t1_$tag.txt
  kill $MON 2>/dev/null; wait $MON 2>/dev/null; MON=""
  echo "pass $tag rc=$rc: $(grep -c '|' runs/run_$tag.txt) table lines" | tee -a summary.txt
  return $rc
}
for rep in 1 2 3; do
  sleep 60                                         # cool-down: the card returns toward idle before each pass
  one_pass bench_energy.py proj_$rep 900 || { say "PASS FAIL proj_$rep"; finish 20; }
done
for rep in 1 2 3; do
  sleep 60
  one_pass bench_energy_excluded.py excl_$rep 600 || { say "PASS FAIL excl_$rep"; finish 20; }
done
say "reduce"
python $W/p114_reduce.py $W $W/p114_verdict.json > logs/reduce.log 2>&1; rrc=$?
grep -aE '^P114' logs/reduce.log | tee -a summary.txt
say "----- summary -----"; cat summary.txt
[ "$REHEARSAL" = 1 ] && { grep -q 'P114_VERDICT VOID' logs/reduce.log && { say "rehearsal: VOID as it must be off the target card"; finish 0; } || { say "rehearsal: reducer did not say VOID"; finish 31; }; }
[ "$rrc" = 0 ] || { say "REDUCER rc=$rrc (VOID, NOISY or error)"; finish "$rrc"; }
finish 0
