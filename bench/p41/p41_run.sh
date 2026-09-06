#!/bin/bash
# bench/p41/p41_run.sh -- lane P41, box-side driver (pre-registration: bench/p41/P41-PREREG.md, verbatim; R1 = the granite grid + p41c probe).
# Form: tp2_run.sh (bench/h2h-20260906/tp2), e4b side only for the e4b-only families (granite, olmoe: Unsloth VOID/HARNESS_ERROR at the tp2
# anchor -> no Unsloth arm, PREREG "Families"). e4b = the shipped cut from PyPI in the image python (versions recorded in every receipt);
# helpers (train anchor, datasets, manifest) from the archive tarball at that cut; the harness is bench/tp3/tp3_arm.py staged by the
# controller. Per family: fetch UNPINNED, prove staged == pin, write refs/main from the pin; tokenise once per seq (512/1024/2048 + the
# 4096 probe); the grid seq-ascending, rank-ascending, the (512, 8) anchor first, fused_attn4 (PRIMARY) then reference_attn4 per cell;
# alpha = 2r; --expect-trainable = (r/8) x the tp2 anchor count; --attn-4bit 1 with the structural census (tp3 T10); one process, one
# JSON, one `perl alarm` per arm from the planning curve; every non-run is a stub row. STOP rules (PREREG): STOP-1 anchor +-10 % of tp2's
# fused s/step; STOP-4 projected spend > 1.5 x estimate; STOP-5 at 80 % of the guard (P41_DEADLINE_EPOCH, from the launcher's
# E4B_RENT_DEADLINE_EPOCH). `--plan` prints the arm plan and exits (no box needed; the controller's tests read it).
set -uo pipefail
PLAN=0; [ "${1:-}" = "--plan" ] && PLAN=1
LANE=p41; W=${P41_WORKDIR:-/root/$LANE}
RUN_ID=${P41_RUN_ID:-p41-r1-granite}
FAMILIES=${P41_FAMILIES:-granite}
SEQS=${P41_SEQS:-"512 1024 2048"}; RANKS=${P41_RANKS:-"8 16 32"}; PROBE_SEQ=${P41_PROBE_SEQ:-4096}
STEPS=${P41_STEPS:-60}; EVAL_EVERY=${P41_EVAL_EVERY:-20}; EVAL_N=${P41_EVAL_N:-48}
LR=${P41_LR:-1e-4}; ACCUM=${P41_ACCUM:-1}; AUTOCAST=${P41_AUTOCAST:-0}; SEED=${P41_SEED:-0}   # P40's fixture as run (PREREG "Fixture")
DATASET=${P41_DATASET:-clinical}; PREREG=${P41_PREREG:-p41/P41-PREREG.md}
E4B_VER=${P41_E4B_VER:-0.35.3}; GNF4_VER=${P41_GNF4_VER:-0.30.2}            # the shipped cut from PyPI at launch (recorded)
TF_VER=${P41_TRANSFORMERS_VER:-5.16.1}; BNB_VER=${P41_BNB_VER:-0.50.1}       # tp1/P38's e4b-side pins, as tp2 ran them
E4B_SRC_REF=${P41_E4B_SRC_REF:-v$E4B_VER}
DEADLINE=${P41_DEADLINE_EPOCH:-0}; RATE=${P41_USD_PER_HOUR:-0}; EST=${P41_EST_USD:-0}
ARM_OVERHEAD=${P41_ARM_OVERHEAD_S:-900}; ALARM_FACTOR=${P41_ALARM_FACTOR:-1.5}; CURVE_EXP=${P41_CURVE_EXP:-1.2}   # planning curve: 60 x s512 x (seq/512)^1.2, x1.5, + load/evals
ANCHOR_STRICT=${P41_ANCHOR_STRICT:-1}; STOP1_TOL=${P41_STOP1_TOL:-0.10}
T_START=$(date +%s)
say(){ echo "[$(date -u +%FT%TZ)] $*"; }

# ---------------------------------------------------------------- the registered families (PREREG "Families"; anchors from RESULTS-tp2.md)
#   row FAM -> MID REV FUSED_S512 REF_S512 EXPECT_R8 N_LAYERS ATTN4_CENSUS OFFLOAD FETCH_ALARM
family_row(){ case "$1" in
  granite) echo "ibm-granite/granite-3.1-3b-a800m-instruct a02780686e08a03fe0d2679a293b5c74a90efa89 0.641 3.587 49807360 32 128 0 3600";;
  olmoe)   echo "allenai/OLMoE-1B-7B-0924-Instruct 7f1c97f440f06ce36705e4f2b843edb5925f4498 0.708 2.644 60817408 16 64 0 3600";;
  mixtral) echo "mistralai/Mixtral-8x7B-Instruct-v0.1 eba92302a2861cdc0098cc54bc9f17cb2c47eb61 2.377 2.929 111673344 32 128 1 10800";;
  qwen3)   echo "Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 4.108 11.067 321257472 48 192 0 7200";;
  *) return 1;; esac; }
alarm_for(){ python3 -c "import math,sys; s,seq,steps,f,ov,e=map(float,sys.argv[1:]); print(int(math.ceil(s*(seq/512.0)**e*steps*f+ov)))" "$1" "$2" "$STEPS" "$ALARM_FACTOR" "$ARM_OVERHEAD" "$CURVE_EXP"; }
expect_for(){ python3 -c "import sys; print(int(sys.argv[1])*int(sys.argv[2])//8)" "$1" "$2"; }

# ---------------------------------------------------------------- the plan (also what --plan prints): one line per arm, in run order
plan(){ local n=0 total=0
  for FAM in $FAMILIES; do
    local row; row=$(family_row $FAM) || { echo "PLAN ERROR unknown family $FAM (the pre-registration names no row: gpt-oss is REFUSED, gemma4 conditional on #426)"; return 1; }
    read -r MID REV FS RS EX NL A4 OFF FAL <<< "$row"
    for SEQ in $SEQS; do for R in $RANKS; do
      for ARM in fused reference; do
        local s=$FS; [ $ARM = reference ] && s=$RS
        local al; al=$(alarm_for $s $SEQ); local ex; ex=$(expect_for $EX $R); n=$((n+1)); total=$((total+al))
        echo "PLAN $n $FAM ${ARM}_attn4_s${SEQ}_r${R} arm=$ARM seq=$SEQ r=$R alpha=$((2*R)) alarm=$al expect_trainable=$ex attn4_census=$A4 offload=$OFF anchor=$([ $SEQ = 512 ] && [ $R = 8 ] && echo yes || echo no)"
      done
    done; done
    local al; al=$(alarm_for $FS $PROBE_SEQ); local ex; ex=$(expect_for $EX 8); n=$((n+1)); total=$((total+al))
    echo "PLAN $n $FAM fused_attn4_s${PROBE_SEQ}_r8_probe arm=fused seq=$PROBE_SEQ r=8 alpha=16 alarm=$al expect_trainable=$ex attn4_census=$A4 offload=$OFF anchor=no probe=p41c"
  done
  echo "PLAN TOTAL arms=$n alarm_sum_s=$total steps=$STEPS prereg=$PREREG e4b=$E4B_VER gnf4=$GNF4_VER"; }
if [ $PLAN = 1 ]; then plan; exit $?; fi

# ---------------------------------------------------------------- box setup (tp2's form)
mkdir -p $W $W/logs $W/adapters; cd $W || exit 9
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
: > summary.txt; : > versions.txt; touch STARTED; echo "$T_START" > STARTED
echo "RUN $RUN_ID prereg=$PREREG families=$FAMILIES seqs=$SEQS ranks=$RANKS probe=$PROBE_SEQ steps=$STEPS eval_every=$EVAL_EVERY eval_n=$EVAL_N lr=$LR accum=$ACCUM autocast=$AUTOCAST seed=$SEED dataset=$DATASET deadline_epoch=$DEADLINE rate_usd_h=$RATE est_usd=$EST" | tee -a summary.txt
plan | tee -a summary.txt
[ -s $W/tp3_arm.py ] || { echo "STAGE MISSING: tp3_arm.py"; touch TP_DONE; exit 9; }
say "install e4b (image python, PyPI): experts4bit-qlora==$E4B_VER grouped-nf4-gemm==$GNF4_VER transformers==$TF_VER bitsandbytes==$BNB_VER"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "experts4bit-qlora==$E4B_VER" "grouped-nf4-gemm==$GNF4_VER" "transformers==$TF_VER" "bitsandbytes==$BNB_VER" \
  datasets accelerate safetensors "huggingface_hub>=0.23" sentencepiece tiktoken rouge-score > logs/pip_e4b.log 2>&1
rc=$?; echo "pip(e4b) rc=$rc"; [ $rc -ne 0 ] && { tail -4 logs/pip_e4b.log; echo "PIP FAIL (e4b)" | tee -a summary.txt; touch TP_DONE; exit 9; }
E4B_VER="$E4B_VER" GNF4_VER="$GNF4_VER" TF_VER="$TF_VER" python - <<'PYT' || { echo "TRIPWIRE FAIL (e4b)" | tee -a summary.txt; touch TP_DONE; exit 9; }
import importlib.metadata as md, inspect, os
import experts4bit_qlora as e, torch, triton, transformers, bitsandbytes
from experts4bit_qlora import enable_fast_train, ExpertsLoRA, load_moe_4bit_streaming, verify_moe_4bit
from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit, detect_attention_projections   # the structural census (e4b#435)
from nf4_qlora import fused_grouped_lora
assert "dgrad_kernel" in inspect.signature(fused_grouped_lora).parameters
assert e.__version__ == os.environ["E4B_VER"], (e.__version__, os.environ["E4B_VER"])
assert md.version("grouped-nf4-gemm") == os.environ["GNF4_VER"], md.version("grouped-nf4-gemm")
assert transformers.__version__ == os.environ["TF_VER"], transformers.__version__
print("p41 tripwire OK (e4b):", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__, "triton", triton.__version__, "transformers", transformers.__version__, "bnb", bitsandbytes.__version__)
open("versions.txt", "a").write(f"e4b {e.__version__} (PyPI)\ngnf4 {md.version('grouped-nf4-gemm')} (PyPI)\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {bitsandbytes.__version__}\n")
PYT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|NUMA node\(s\)" | tee -a forensics.txt; grep MemTotal /proc/meminfo | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')" || { echo "DUD BOX" | tee -a summary.txt; touch TP_DONE; exit 10; }
# helpers at the cut (archive tarball: tp1 amendment 2), the train anchor gate, the registered text (sha-verified)
case "$E4B_SRC_REF" in *[!0-9a-f]*|"") SRC_URL="https://github.com/pjordanandrsn/experts4bit-qlora/archive/refs/tags/$E4B_SRC_REF.tar.gz";; *) SRC_URL="https://github.com/pjordanandrsn/experts4bit-qlora/archive/$E4B_SRC_REF.tar.gz";; esac
say "fetching repo helpers from $SRC_URL"
rm -rf $W/e4b-src && mkdir -p $W/e4b-src && perl -e 'alarm 600; exec @ARGV' bash -c "curl -sL $SRC_URL | tar xz -C $W/e4b-src --strip-components=1"; rc=$?
{ [ $rc -ne 0 ] || [ ! -s $W/e4b-src/bench/flagship-matrix/ds_manifest.json ]; } && { echo "SRC FETCH FAIL ($SRC_URL)" | tee -a summary.txt; touch TP_DONE; exit 9; }
cp $W/e4b-src/bench/flagship-matrix/drivers/n9_datasets.py $W/e4b-src/bench/flagship-matrix/ds_manifest.json $W/e4b-src/bench/train-anchor/train_anchor.py $W/e4b-src/bench/train-anchor/train_anchor_gate.py $W/
echo "HELPERS e4b-src ref=$E4B_SRC_REF ds_manifest sha=$(sha256sum $W/ds_manifest.json | awk '{print $1}')" | tee -a summary.txt
say "train anchor"
ANCHOR_OUT=$W/anchor.json perl -e 'alarm 900; exec @ARGV' python $W/train_anchor.py > logs/anchor.log 2>&1; tail -3 logs/anchor.log
python $W/train_anchor_gate.py $W/anchor.json | tee logs/anchor_gate.log; arc=${PIPESTATUS[0]}
export TP2_ANCHOR_JSON=$W/anchor.json TP2_BOX_CLASS="$(grep -E '^\s*class ' logs/anchor_gate.log | awk '{print $2}')"
echo "ANCHOR rc=$arc class=$TP2_BOX_CLASS" | tee -a summary.txt
if [ "$arc" -ne 0 ] && [ "$ANCHOR_STRICT" = "1" ]; then echo "BOX REFUSED by train anchor (rc=$arc)" | tee -a summary.txt; touch BOX_REFUSED TP_DONE; exit 12; fi
say "dataset: $DATASET (n9_datasets.py, sha-verified against ds_manifest.json)"
mkdir -p $W/data && (cd $W/data && python $W/n9_datasets.py $W/data > $W/logs/datasets.log 2>&1); tail -2 logs/datasets.log
DATA=$W/data/ds_$DATASET.json
DATA_SHA=$(python -c "import json; print(json.load(open('$W/ds_manifest.json'))['$DATASET']['sha256'])")
GOT_SHA=$(sha256sum $DATA | awk '{print $1}'); [ "$GOT_SHA" = "$DATA_SHA" ] || { echo "DATASET MISMATCH: $GOT_SHA != $DATA_SHA" | tee -a summary.txt; touch TP_DONE; exit 13; }
echo "DATASET $DATASET sha=$DATA_SHA" | tee -a summary.txt

# ---------------------------------------------------------------- helpers
vram_start(){ ( while :; do echo "$(date -u +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)"; sleep 1; done ) > $W/vram_$1.txt 2>/dev/null & echo $!; }
vram_stop(){ kill $1 2>/dev/null; wait $1 2>/dev/null; }
# stubw FAM TAG ARM STATUS REASON SEQ R: a row for an attempt that never reached the harness (not_run / refused / load_fault / alarm / harness_error)
stubw(){ python3 - "$W" "$STEPS" "$ACCUM" "$PREREG" "$@" <<'PYS'
import json, os, sys
W, steps, accum, prereg, fam, tag, arm, status, reason, seq, r = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], *sys.argv[5:12]
rec = {"framework": "e4b", "fam": fam, "arm": arm, "tag": tag, "status": status, "reason": reason[:800], "steps": steps, "seq": int(seq), "r": int(r), "alpha": 2 * int(r), "accum": accum,
       "written_by": "p41_run.sh", "prereg": prereg}
json.dump(rec, open(os.path.join(W, f"{fam}_e4b_{tag}.json"), "w"), indent=1)
print(f"STUB {status.upper()} {fam}/e4b/{tag}: {reason[:160]}")
PYS
  echo "$1/e4b/$2 STUB $4: $5" | cut -c1-300 >> summary.txt; }
elapsed(){ echo $(( $(date +%s) - T_START )); }
stop5_due(){ [ "$DEADLINE" -gt 0 ] || return 1; local now; now=$(date +%s); [ $(( now - T_START )) -ge $(( (DEADLINE - T_START) * 80 / 100 )) ]; }
stop4_due(){ [ "$RATE" != "0" ] && [ "$EST" != "0" ] || return 1
  python3 -c "import sys; el,rate,est,rem=map(float,sys.argv[1:]); proj=(el+rem)/3600.0*rate; sys.exit(0 if proj > 1.5*est else 1)" "$(elapsed)" "$RATE" "$EST" "$1"; }
remaining_alarm_sum(){ python3 -c "
import sys; lines=open('$W/summary.txt').read().splitlines(); done=set(l.split()[0] for l in lines if l.startswith('$1/e4b/'))
tot=0
for l in lines:
    if l.startswith('PLAN ') and ' $1 ' in l and 'TOTAL' not in l:
        tag=l.split()[3]
        if '$1/e4b/'+tag not in done: tot+=int([p for p in l.split() if p.startswith('alarm=')][0][6:])
print(tot)"; }
fetch(){ local FAM=$1 MID=$2 REV=$3 AL=$4; say "fetch $FAM ($MID, unpinned; pin $REV)"
  perl -e "alarm $AL; exec @ARGV" python - "$MID" <<'PYF' > logs/fetch_$FAM.log 2>&1
import os, sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "merges.txt", "vocab.json"], max_workers=4)
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}"); print("STAGED", os.path.basename(os.path.realpath(p)))
PYF
  local rc=$?; tail -2 logs/fetch_$FAM.log | head -1
  [ $rc -ne 0 ] && { echo "$FAM: FETCH FAILED rc=$rc" | tee -a summary.txt; FETCH_REASON="fetch failed rc=$rc (alarm $AL s; logs/fetch_$FAM.log)"; return 1; }
  local GOT; GOT=$(grep -a "^STAGED " logs/fetch_$FAM.log | tail -1 | awk '{print $2}')
  local RDIR=/root/.cache/huggingface/hub/models--${MID//\//--}
  if [ "$GOT" = "$REV" ]; then mkdir -p $RDIR/refs && printf '%s' "$REV" > $RDIR/refs/main; echo "PIN OK $FAM staged=$GOT == pin; refs/main written from the pin (P38 amendment 2 / e4b#404)" | tee -a summary.txt
  else echo "PIN MISMATCH $FAM: staged=$GOT != pin=$REV -- main moved past the pin; the family is ABORTED with load_fault stubs (not coerced)" | tee -a summary.txt; FETCH_REASON="staged snapshot $GOT != pinned revision $REV; family aborted, not coerced"; return 2; fi
  df -h /root | tail -1; return 0; }
free_family(){ rm -rf /root/.cache/huggingface/hub/models--$2; say "freed $1 (disk: $(df -h /root | tail -1 | awk '{print $4}') free)"; }
# arm FAM TAG ARM SEQ R ALARM EXPECT MID REV OFFLOAD: one process, one JSON (<FAM>_e4b_<TAG>.json), one alarm; the result line -> summary.txt
arm(){ local FAM=$1 TAG=$2 ARM=$3 SEQ=$4 R=$5 AL=$6 EXP=$7 MID=$8 REV=$9 OFF=${10}
  local TOK=$W/tokens_${FAM}_s${SEQ}.json; local TOK_SHA; TOK_SHA=$(python3 -c "import json; print(json.load(open('$TOK'))['sha256'])" 2>/dev/null) || { stubw $FAM $TAG $ARM harness_error "no tokens file for seq $SEQ" $SEQ $R; return 0; }
  say "arm $FAM/e4b/$TAG (arm=$ARM seq=$SEQ r=$R alpha=$((2*R)) steps=$STEPS lr=$LR accum=$ACCUM autocast=$AUTOCAST offload=$OFF alarm=$AL expect_trainable=$EXP)"
  local sp; sp=$(vram_start ${FAM}_e4b_$TAG)
  HF_HUB_OFFLINE=1 perl -e "alarm $AL; exec @ARGV" python -u $W/tp3_arm.py --framework e4b --arm $ARM --tag $TAG --fam $FAM --model "$MID" --revision $REV \
      --steps $STEPS --seq $SEQ --accum $ACCUM --autocast $AUTOCAST --lr $LR --r $R --alpha $((2*R)) --seed $SEED --offload $OFF --attn-4bit 1 \
      --tokens $TOK --tokens-sha $TOK_SHA --eval-every $EVAL_EVERY --eval-n $EVAL_N --expect-trainable $EXP --prereg $PREREG \
      --out $W --adapter-dir $W/adapters > logs/run_${FAM}_e4b_$TAG.log 2>&1
  local rc=$?; vram_stop $sp
  if [ $rc -eq 142 ] && [ ! -s $W/${FAM}_e4b_$TAG.json ]; then stubw $FAM $TAG $ARM alarm "arm alarm $AL s (SIGALRM; the process could not write its own stub)" $SEQ $R; fi
  grep -aE "^CELL |^LOAD OK|^ENGAGE|^STUB|Error|error:" logs/run_${FAM}_e4b_$TAG.log | tail -3 | cut -c1-300 | sed "s/^/    /"
  { echo -n "$FAM/e4b/$TAG rc=$rc "; grep -aE "^CELL " logs/run_${FAM}_e4b_$TAG.log | tail -1 | cut -c1-400; echo; } >> summary.txt
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
# STOP-1: the (512, 8) fused anchor vs tp2's fused s/step, +-10 % (P40's anchor rule); a disagreement halts the family's remaining cells
stop1_check(){ local FAM=$1 FS=$2; local f=$W/${FAM}_e4b_fused_attn4_s512_r8.json
  python3 - "$f" "$FS" "$STOP1_TOL" <<'PY1'
import json, sys
p, ref, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
try: r = json.load(open(p))
except Exception as e: print(f"STOP-1 UNDECIDED: anchor receipt unreadable ({e})"); sys.exit(2)
if r.get("status") != "ok": print(f"STOP-1 UNDECIDED: anchor status {r.get('status')}"); sys.exit(2)
s = float(r.get("s_per_step_median_11plus", 0) or 0); d = (s - ref) / ref
print(f"ANCHOR fused s512 r8 = {s:.3f} s/step vs tp2 {ref:.3f} ({d*100:+.1f} %)")
sys.exit(0 if abs(d) <= tol else 1)
PY1
}

# ---------------------------------------------------------------- the lane
STOPPED=""
for FAM in $FAMILIES; do
  row=$(family_row $FAM) || { echo "UNKNOWN FAMILY $FAM (no row in the pre-registration's table; nothing run)" | tee -a summary.txt; continue; }
  read -r MID REV FS RS EX NL A4 OFF FAL <<< "$row"
  say "===== family $FAM ($MID @ $REV; offload=$OFF; fused anchor $FS s, reference anchor $RS s)"
  FETCH_REASON=""; fetch $FAM $MID $REV $FAL; frc=$?
  if [ $frc -ne 0 ]; then st=not_run; [ $frc -eq 2 ] && st=load_fault
    for SEQ in $SEQS; do for R in $RANKS; do stubw $FAM fused_attn4_s${SEQ}_r${R} fused $st "$FETCH_REASON" $SEQ $R; stubw $FAM reference_attn4_s${SEQ}_r${R} reference $st "$FETCH_REASON" $SEQ $R; done; done
    stubw $FAM fused_attn4_s${PROBE_SEQ}_r8_probe fused $st "$FETCH_REASON" $PROBE_SEQ 8; free_family $FAM ${MID//\//--}; continue; fi
  for SEQ in $SEQS $PROBE_SEQ; do
    say "tokenise ($FAM, seq $SEQ) -> tokens_${FAM}_s${SEQ}.json"
    HF_HUB_OFFLINE=1 python $W/tp3_arm.py --prepare --fam $FAM --model "$MID" --revision $REV --data $DATA --data-sha $DATA_SHA --seq $SEQ --eval-n $EVAL_N --tokens $W/tokens_${FAM}_s${SEQ}.json > logs/prepare_${FAM}_s${SEQ}.log 2>&1 \
      && echo "TOKENS $FAM seq=$SEQ sha=$(python3 -c "import json; print(json.load(open('$W/tokens_${FAM}_s${SEQ}.json'))['sha256'])")" | tee -a summary.txt \
      || { tail -2 logs/prepare_${FAM}_s${SEQ}.log; echo "$FAM: TOKENS FAIL seq=$SEQ" | tee -a summary.txt; }
  done
  for SEQ in $SEQS; do for R in $RANKS; do
    EXP=$(expect_for $EX $R); FAL_F=$(alarm_for $FS $SEQ); FAL_R=$(alarm_for $RS $SEQ)
    for ARM in fused reference; do
      TAG=${ARM}_attn4_s${SEQ}_r${R}; AL=$FAL_F; [ $ARM = reference ] && AL=$FAL_R
      if [ -n "$STOPPED" ]; then stubw $FAM $TAG $ARM not_run "$STOPPED" $SEQ $R; continue; fi
      if stop5_due; then STOPPED="STOP-5: 80 % of the guard reached ($(elapsed) s of $((DEADLINE - T_START)) s); remaining cells NOT_RUN, a new run needs a new approval"; echo "$STOPPED" | tee -a summary.txt; touch STOP5; stubw $FAM $TAG $ARM not_run "$STOPPED" $SEQ $R; continue; fi
      if stop4_due "$(remaining_alarm_sum $FAM)"; then STOPPED="STOP-4: projected spend > 1.5 x the estimate (elapsed $(elapsed) s at \$$RATE/h, estimate \$$EST); remaining cells NOT_RUN, continuing needs a new approval"; echo "$STOPPED" | tee -a summary.txt; touch STOP4; stubw $FAM $TAG $ARM not_run "$STOPPED" $SEQ $R; continue; fi
      arm $FAM $TAG $ARM $SEQ $R $AL $EXP "$MID" $REV $OFF
      if [ $SEQ = 512 ] && [ $R = 8 ] && [ $ARM = fused ]; then
        stop1_check $FAM $FS | tee -a summary.txt; s1=${PIPESTATUS[0]}
        if [ "$s1" = "1" ] && [ "$ANCHOR_STRICT" = "1" ]; then STOPPED="STOP-1: the (512, 8) fused anchor disagrees with tp2 beyond +-10 % -- the box/stack is not comparable; remaining $FAM cells NOT_RUN"; echo "$STOPPED" | tee -a summary.txt; touch STOP1; fi
      fi
    done
  done; done
  TAG=fused_attn4_s${PROBE_SEQ}_r8_probe
  if [ -n "$STOPPED" ]; then stubw $FAM $TAG fused not_run "$STOPPED" $PROBE_SEQ 8
  elif stop5_due; then echo "STOP-5 before the probe" | tee -a summary.txt; touch STOP5; stubw $FAM $TAG fused not_run "STOP-5: 80 % of the guard reached before the p41c probe" $PROBE_SEQ 8
  else arm $FAM $TAG fused $PROBE_SEQ 8 $(alarm_for $FS $PROBE_SEQ) $(expect_for $EX 8) "$MID" $REV $OFF; fi
  echo "$(echo $FAM | tr a-z A-Z) DONE" | tee -a summary.txt
  free_family $FAM ${MID//\//--}
  [ -n "$STOPPED" ] && case "$STOPPED" in STOP-1*) STOPPED="";; esac   # STOP-1 halts one family; the next family gets its own anchor
done
echo "----- summary.txt -----"; cat summary.txt; echo "----- versions.txt -----"; cat versions.txt
say "TP_DONE elapsed=$(elapsed)s"; touch TP_DONE
