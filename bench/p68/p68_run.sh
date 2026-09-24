#!/bin/bash
# bench/p68/p68_run.sh -- lane P68, BOX side (bench/p68/P68-PREREG.md; experts4bit-qlora#725). Started detached by
# p68_drive.sh with the run's nonce; P68_RUN_NONCE first, then P68_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit
# and P68_SUCCESS.<nonce> only on clean completion. Pattern: bench/p63/p63_run.sh, whose probe this lane reuses.
#
# Which part of the attention makes a verify or a prefill differ from T = 1 decode, and is the difference over the
# shipped bar once enough positions are read? Two stacks (int4, the served lane; nf4, its control), one process each,
# built the way step_decomp builds the shipped stack. Per stack: the forcing arms (P63's fixture, sites captured) and
# the size reading (P64's committed wikitext rows, logits only). p68_reduce.py applies the registration on the box.
# Nothing here changes a default.
set -uo pipefail
W=/root/p68; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p68: $*"; }
NONCE=${P68_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P68_RUN_NONCE.tmp && mv $W/P68_RUN_NONCE.tmp $W/P68_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P68_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P68_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P68_RUN_ID P68_DEADLINE_EPOCH P68_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$GNF4_SHA" in *[!0-9a-f]*|"") say "refusing: GNF4_SHA is not hex"; finish 78;; esac
[ ${#GNF4_SHA} -eq 40 ] || { say "refusing: GNF4_SHA is not a 40-char sha"; finish 78; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# the P42 lane hook must NOT be on the path: the probe applies the int4 lanes itself (it refuses if the hook is active)
unset PYTHONPATH
MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
FETCH_WORKERS=4; MIN_MBPS=80
: > summary.txt; echo "$P68_INSTANCE_ID" > INSTANCE_ID
# ---- staged pieces, byte-for-byte
for f in p68_probe.py p68_reduce.py p63_probe.py p63_compare.py fixture.txt prompts_wikitext.json serve_stack.py k8_bake.py calib.json staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p68/staged.sha256"; finish 9; }
# ---- the box: a dud or a different class is refused before anything is installed
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; nvcc --version 2>/dev/null | tail -1 | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *5090*) ;; *) say "REFUSED: card is '$GPU', the lane registers the RTX 5090 class"; echo "refused: class $GPU" > REFUSAL; finish 15;; esac
df -h /root | tail -1 | tee -a forensics.txt
# ---- egress, measured the way the fetch runs (P65 Amendment 2's lesson): the fetch below pulls with FETCH_WORKERS
# (4) workers, so the probe is four parallel 50 MB ranges of one HF CDN file, total bytes over the wall time of all
# four, in Python (the stock image ships no curl). The reading needs ~61 GB: ~13 min at the floor.
say "egress pre-flight: HF CDN, $FETCH_WORKERS parallel 50 MB ranges, 30 s cap (floor ${MIN_MBPS} MB/s aggregate)"
MBPS=$(P68_FETCH_WORKERS=$FETCH_WORKERS python3 - <<'PYEG' 2>/dev/null || echo 0
import os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL = "https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors"     # 440 MB: holds 8 x 50 MB
CH, N, CAP = 52428800, int(os.environ["P68_FETCH_WORKERS"]), 30.0
def get(i):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={i * CH}-{(i + 1) * CH - 1}"})
    got, t0 = 0, time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=CAP) as r:
            while time.perf_counter() - t0 < CAP:
                b = r.read(1 << 20)
                if not b:
                    break
                got += len(b)
    except Exception:
        pass
    return got
t0 = time.perf_counter()
with ThreadPoolExecutor(N) as ex:
    total = sum(ex.map(get, range(N)))
print(round(total / 1e6 / max(1e-3, time.perf_counter() - t0), 1))
PYEG
)
say "HF CDN ${MBPS} MB/s ($FETCH_WORKERS parallel ranges)"; echo "hf_cdn_mbps_${FETCH_WORKERS}x=$MBPS" | tee -a forensics.txt >> summary.txt
if python3 -c "import sys; sys.exit(0 if float('${MBPS:-0}') < float('$MIN_MBPS') else 1)"; then
  say "REFUSED: egress ${MBPS} MB/s < ${MIN_MBPS} over $FETCH_WORKERS streams -- host-limited, not a result"; echo "refused: egress ${MBPS}" > REFUSAL; finish 13; fi
# ---- deadline guard + per-stack alarm (P59's): never start a stack that cannot finish 10 min before teardown
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P68_DEADLINE_EPOCH" ] || { say "STOP: $2 needs ${need}s, only $((P68_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local cap=$1 left=$(( P68_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
# ---- install: e4b pinned (image python; P59's toolchain pins) and gnf4 at the lane's cut
export DEBIAN_FRONTEND=noninteractive
say "install e4b @$E4B_SHA, gnf4 @$GNF4_SHA"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL (e4b)"; finish 9; }
import os, importlib.metadata as md
import experts4bit_qlora as e, torch, triton, transformers
import experts4bit_qlora.engines.hybrid as hy
from experts4bit_qlora.engines import hot_residency as hr
from experts4bit_qlora.engines.int4_attn import Int4Linear, _smallm_kernels
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import int4_b32
assert not getattr(hy.enable_hybrid_tier, "_gen_hooked", False), "the P42 hook is active: the lanes would apply twice"
assert "site-packages" in int4_b32.__file__, int4_b32.__file__
assert getattr(Int4Linear, "SMALLM_ROWS_MAX", None) == 16 and Int4Linear.GEMV_ROWS_MAX == 1
_smallm_kernels()                                           # K16 installed: the 2..16-row bucket exists on this box
assert hasattr(hr, "_int4_part_or_none"), "the int4 singleton route needs the buffer fix (0.37.3+)"
# the core forcing replaces the sdpa entry in transformers' attention table; it must be a local-then-global table
assert hasattr(ALL_ATTENTION_FUNCTIONS, "_local_mapping") and callable(ALL_ATTENTION_FUNCTIONS["sdpa"])
open("/root/p68/versions.txt", "a").write(
    f"e4b {e.__version__} @{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\n"
    f"torch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\n"
    f"bitsandbytes {md.version('bitsandbytes')}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"))
PYT
cat versions.txt | tee -a summary.txt
# ---- fetch (pinned) and bake the NF4 arena (P39's k8_bake.py, from the pinned LOCAL snapshot)
say "fetch $MID @ $REV"
perl -e 'alarm 4800; exec @ARGV' python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=$FETCH_WORKERS))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -n 1 logs/fetch.log); [ -d "$SNAP" ] || { say "DL FAIL (no snapshot dir: $SNAP)"; finish 11; }
echo "snapshot $SNAP" | tee -a summary.txt
say "bake NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e 'alarm 5400; exec @ARGV' python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
QA=$W/work/nf4.arena; [ -e "$QA" ] || { say "BAKE FAIL (no arena)"; python -c "import json; r=json.load(open('$W/work/bake.json')); print(r.get('err')); print(r.get('tb'))" 2>/dev/null | tail -n 20; finish 12; }
rm -rf $W/work/nf4snap                                      # the relocation source; the arena is what the stacks load
# ---- stacks: one process each (P68-PREREG "Arms"); the served int4 stack first
rc_any=0; rec(){ local r=$1; [ "$r" = 0 ] || [ "$rc_any" != 0 ] || rc_any=$r; }
stack(){ local S=$1 NEED=$2; can_run "$NEED" "$S" || { rec 30; return; }
  local AL; AL=$(arm_alarm 3000); say "stack $S (alarm ${AL}s)"
  perl -e "alarm $AL; exec @ARGV" python -u $W/p68_probe.py --stack "$S" --model "$SNAP" --arena "$QA" --calib $W/calib.json \
    --fixture $W/fixture.txt --size-rows $W/prompts_wikitext.json --out $W/out/$S > logs/run_$S.log 2>&1
  local r=$?
  grep -aE "^P68 |P68ARM|REFUSED|Traceback|Error" logs/run_$S.log | tail -16 | cut -c1-400 | sed "s/^/    /"
  { echo -n "stack $S rc=$r "; grep -a "P68ARM" logs/run_$S.log | tail -1 | cut -c1-600; echo; } >> summary.txt
  [ -s "$W/out/$S/p68_arm.json" ] || { echo "RECEIPT MISSING $S" >> summary.txt; r=${r:-42}; [ "$r" = 0 ] && r=42; }
  rec $r; }
stack int4 2100
stack nf4 1800
# ---- reduce on the box (the verdict is the reducer's JSON, read against the prereg; never this script's exit code)
say "reduce"; python $W/p68_reduce.py $W/out --md $W/RESULTS-p68-generated.md --json $W/p68_rep.json > RESULTS.txt 2>&1; red=$?
head -c 6000 RESULTS.txt | tee -a summary.txt >/dev/null
[ "$red" = 14 ] && rec 14
say "----- summary -----"; cat summary.txt; echo "----- versions -----"; cat versions.txt
finish "$rc_any"
