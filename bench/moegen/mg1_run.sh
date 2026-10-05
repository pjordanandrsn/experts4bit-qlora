#!/bin/bash
# bench/moegen/mg1_run.sh -- lane MG1, BOX side (bench/moegen/MG1-PREREG.md). Started by bench/tc1/tc1_drive.sh as its
# TC1_RUNNER (TC1_EXTRA_STAGE carries this file and its pieces), so it speaks tc1_drive's contract: the nonce handshake
# (TC1_RUN_NONCE within 30 s), summary.txt one line per finished step, and TC1_EXIT_CODE.<nonce> / TC1_SUCCESS.<nonce> /
# TP_DONE.<nonce> at the end -- a refusal writes them too, so the controller reads a finished lane, never a hang.
#
# Per family, in order (cheap first; OLMoE-Instruct and Gemma-4-26B-A4B-it are the regression anchors -- licensed families re-read
# on this branch, on the checkpoints tp1 licensed; amendment 1):
#   fetch the PINNED revision -> tp1's arm driver UNCHANGED (tp1_train_smoke.py: reference, fused; the registered clinical
#   fixture, N=60, seq 512, r=8) -> the ladder (bench/moegen/ladder.py: fused, fused_pre, keep, interleaved A..Z Z..A).
# Nothing here creates, destroys or approves compute.
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
say(){ echo "[$(date -u +%FT%TZ)] mg1: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts data
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}
STEPS=${MG1_STEPS:-60}; SEQ=512
FAMS=${MG1_FAMILIES:-"olmoe gemma4 lfm2 graniteh ernie nemotron qwen3_5"}
# MG1_PROVE=1 (the proving run, forwarded by tc1_drive): install + tripwire, then a clean finish -- no anchor, no fetch, no arm.
# MG1_REHEARSAL=1 (the $0 A2000 container rehearsal ONLY; tc1_drive does not forward it, so a rented box can never set it): a
# train-anchor refusal is recorded and the lane continues, so the arms and the ladder run on a card outside the anchor band.
PROVE=${MG1_PROVE:-0}; REHEARSAL=${MG1_REHEARSAL:-0}
# Amendment 2: MG1_LADDER_ONLY=1 skips the anchor and the arms and runs only the ladder, with MG1_LADDER_ARGS appended (argparse:
# the last value wins). Its registered shape reads P2 for Qwen3.6 -- the fused rung's DGRAD_STATS -- at the licensed arms'
# configuration. P2 is an engagement count, not a timing, so the box class does not bear on it.
LADDER_ONLY=${MG1_LADDER_ONLY:-0}; LADDER_ARGS=${MG1_LADDER_ARGS:-}
A2_ARGS="--rungs fused --r 8 --alpha 16 --adapter-dtype fp32 --attn4 0 --attn-lora 1 --profile 0 --warmup 1 --steps 2"
# Amendment 3: MG1_P2_ARM=1 skips the anchor and the ladder and runs only tp1's fused arm, UNCHANGED, under p2_hook.py, which
# writes grouped-nf4-gemm's DGRAD_STATS at the driver's exit (receipts/<fam>_p2_dgrad.json). Amendment 2's ladder OOMed at the
# licensed configuration; the licensed driver itself fits, so P2 is read on the code path the PASS was read on.
P2_ARM=${MG1_P2_ARM:-0}
if [ "$P2_ARM" = 1 ]; then
  [ "$FAMS" = qwen3_5 ] && [ "$STEPS" = 60 ] && [ "$LADDER_ONLY" != 1 ] && echo "AMENDMENT 3 SHAPE (registered): P2 for qwen3_5 on tp1's fused arm" | tee -a summary.txt \
    || echo "NON-REGISTERED SHAPE: MG1_P2_ARM=1 MG1_FAMILIES=$FAMS MG1_STEPS=$STEPS MG1_LADDER_ONLY=$LADDER_ONLY (not a registered reading)" | tee -a summary.txt
elif [ "$LADDER_ONLY" = 1 ]; then
  [ "$FAMS" = qwen3_5 ] && [ "$LADDER_ARGS" = "$A2_ARGS" ] && echo "AMENDMENT 2 SHAPE (registered): P2 for qwen3_5" | tee -a summary.txt \
    || echo "NON-REGISTERED SHAPE: MG1_LADDER_ONLY=1 MG1_FAMILIES=$FAMS MG1_LADDER_ARGS=$LADDER_ARGS (not a registered reading)" | tee -a summary.txt
else
  [ "$STEPS" = 60 ] && [ "$FAMS" = "olmoe gemma4 lfm2 graniteh ernie nemotron qwen3_5" ] || echo "NON-REGISTERED SHAPE: MG1_STEPS=$STEPS MG1_FAMILIES=$FAMS (not a registered reading)" | tee -a summary.txt
fi
[ "$REHEARSAL" = 1 ] && echo "REHEARSAL (MG1_REHEARSAL=1): not a registered reading" | tee -a summary.txt
# family -> model id | pinned revision | offload (0 resident; qwen3_5 retries under offload on an OOM stub)
declare -A MID=( [olmoe]=allenai/OLMoE-1B-7B-0924-Instruct [gemma4]=google/gemma-4-26B-A4B-it [lfm2]=LiquidAI/LFM2-8B-A1B [graniteh]=ibm-granite/granite-4.0-h-tiny
                 [ernie]=baidu/ERNIE-4.5-21B-A3B-PT [nemotron]=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
                 [qwen3_5]=Qwen/Qwen3.6-35B-A3B )
declare -A REV=( [olmoe]=7f1c97f440f06ce36705e4f2b843edb5925f4498 [gemma4]=4d7ae4984b7db7de8f8457170b3f1a419ee76d52
                 [lfm2]=c1c44ff9fc00db3ebf4516970563f5f383d23670
                 [graniteh]=791e0d3d28c86e106c9b6e0b4cecdee0375b6124 [ernie]=87db95487941cb39592ee0abca3b9155a6d19c5c
                 [nemotron]=a9904d24bcc1d289a1950fa9d2b978c47cf903b9 [qwen3_5]=995ad96eacd98c81ed38be0c5b274b04031597b0 )

# ---------------------------------------------------------------- install + tripwire
# pip's git+https needs git; the pytorch/pytorch images carry none (P55's lesson: only Vast's runtime layer happened to add it).
command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: gnf4 @$GNF4_SHA + e4b @$E4B_SHA, transformers 5.18.0"
perl -e 'alarm 2400; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.18.0" "bitsandbytes==0.50.2" datasets accelerate safetensors "huggingface_hub>=0.23" sentencepiece tiktoken > logs/pip.log 2>&1 \
  || { tail -5 logs/pip.log; echo "PIP FAIL" | tee -a summary.txt; finish 9; }
# The Mamba-2 / short-conv kernels the hybrid families' recurrent blocks use (Granite-H, Nemotron-H): without them transformers runs its
# reference PyTorch scan, unfused, so an arm that pays the fallback is not the fast path. Upstream's own release wheels, no-deps
# (the sdists need nvcc even for metadata, and a resolver pass could move torch). A failed install is recorded, never fatal:
# the arms then run the fallback and say so in their logs.
PYTAG=$(python -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
python -m pip install -q --no-input einops > logs/pip_mamba.log 2>&1
# Qwen3.5/3.6's Gated DeltaNet: flash-linear-attention (pure Triton), no-deps for the same reason.
python -m pip install -q --no-input --no-deps flash-linear-attention==0.5.2 fla-core==0.5.2 >> logs/pip_mamba.log 2>&1
python -m pip install -q --no-input --no-deps \
  "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.8cxx11abiTRUE-$PYTAG-$PYTAG-linux_x86_64.whl" \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-$PYTAG-$PYTAG-linux_x86_64.whl" >> logs/pip_mamba.log 2>&1
echo "RECURRENT_KERNELS mamba2/causal_conv1d/fla $(python -c 'import transformers.utils.import_utils as u; print(u.is_mamba_2_ssm_available(), u.is_causal_conv1d_available(), u.is_flash_linear_attention_available())' 2>/dev/null)" | tee -a summary.txt
python - <<'PYT' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; echo "TRIPWIRE FAIL" | tee -a summary.txt; finish 9; }
import importlib.metadata as md, json, os, torch
from experts4bit_qlora.engines.fast import FAST_TRAIN_STATS, enable_fast_train   # this branch's engagement census
from experts4bit_qlora.engines.rope_train import ROPE_TRAIN_STATS_REFUSED        # this branch's RoPE semantics probe
import nf4_qlora
assert hasattr(nf4_qlora, "DGRAD_STATS"), "grouped-nf4-gemm without DGRAD_STATS"
for dist, want in (("experts4bit-qlora", os.environ["E4B_SHA"]), ("grouped-nf4-gemm", os.environ["GNF4_SHA"])):
    d = json.loads(md.distribution(dist).read_text("direct_url.json"))
    got = d.get("vcs_info", {}).get("commit_id")
    assert got == want, f"{dist}: installed {got} != pinned {want}"
assert torch.cuda.is_available()
print("mg1 tripwire OK", md.version("experts4bit-qlora"), md.version("grouped-nf4-gemm"), torch.__version__, torch.cuda.get_device_name())
PYT
cat logs/tripwire.log | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt; lscpu | grep "Model name" | tee -a forensics.txt
[ "$PROVE" = 1 ] && { echo "PROVE: install + tripwire OK; no anchor, no fetch, no arm" | tee -a summary.txt; finish 0; }

# ---------------------------------------------------------------- box class (bench/train-anchor), strict as tp1
if [ "$P2_ARM" = 1 ]; then echo "ANCHOR skipped (MG1_P2_ARM: an engagement count, not a timing)" | tee -a summary.txt
elif [ "$LADDER_ONLY" = 1 ]; then echo "ANCHOR skipped (MG1_LADDER_ONLY: an engagement count, not a timing)" | tee -a summary.txt; else
say "train anchor"
ANCHOR_OUT=$W/anchor.json perl -e 'alarm 900; exec @ARGV' python train_anchor.py > logs/anchor.log 2>&1
python train_anchor_gate.py anchor.json | tee logs/anchor_gate.log; arc=${PIPESTATUS[0]}
export TP1_ANCHOR_JSON=$W/anchor.json TP1_BOX_CLASS="$(grep -E '^\s*class ' logs/anchor_gate.log | awk '{print $2}')"
echo "ANCHOR rc=$arc class=$TP1_BOX_CLASS" | tee -a summary.txt
if [ "$arc" -ne 0 ]; then
  [ "$REHEARSAL" = 1 ] || { echo "BOX REFUSED by train anchor" | tee -a summary.txt; finish 12; }
  echo "REHEARSAL: train anchor refused this card (rc=$arc); continuing" | tee -a summary.txt
fi
fi

# ---------------------------------------------------------------- the fixture: regenerate, refuse unless the registered bytes
(cd data && python ../n9_datasets.py . > ../logs/datasets.log 2>&1)
DATA=$W/data/ds_clinical.json; DATA_SHA=$(python -c "import json; print(json.load(open('ds_manifest.json'))['clinical']['sha256'])")
[ "$(sha256sum $DATA | awk '{print $1}')" = "$DATA_SHA" ] || { echo "DATASET MISMATCH" | tee -a summary.txt; finish 13; }
echo "DATASET clinical sha=$DATA_SHA" | tee -a summary.txt

arm(){ local fam=$1 a=$2 snap=$3 off=$4 al=$5 py=python
  [ "$P2_ARM" = 1 ] && py="python p2_hook.py"          # amendment 3: the same driver file, run as __main__ under the census hook
  say "arm $fam/$a offload=$off"
  MG1_P2_OUT=$W/receipts/${fam}_p2_dgrad.json perl -e "alarm $al; exec @ARGV" $py tp1_train_smoke.py --model "$snap" --fam "$fam" --arm "$a" --steps $STEPS --seq $SEQ \
    --offload "$off" --data "$DATA" --data-sha "$DATA_SHA" --out receipts > "logs/run_${fam}_${a}.log" 2>&1
  local rc=$?; echo "$fam/$a offload=$off rc=$rc" | tee -a summary.txt; return $rc; }

for fam in $FAMS; do
  say "fetch $fam ${MID[$fam]}@${REV[$fam]}"
  snap=$(perl -e 'alarm 3600; exec @ARGV' python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('${MID[$fam]}', revision='${REV[$fam]}', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt','*.jinja','*.py']))" 2> logs/fetch_$fam.log | tail -1)
  [ -d "$snap" ] || { echo "$fam: FETCH FAILED" | tee -a summary.txt; continue; }
  echo "$fam: fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt
  off=0
  if [ "$P2_ARM" = 1 ]; then
    arm $fam fused "$snap" $off 5400
    python - "$W/receipts/${fam}_p2_dgrad.json" <<'PYP' | tee -a summary.txt
import json, sys
try:
    c = json.load(open(sys.argv[1]))
except (OSError, ValueError) as e:
    print(f"P2 census ABSENT ({type(e).__name__}): the driver did not reach interpreter shutdown -- a row, status unchanged")
    sys.exit(0)
print(f"P2 census dgrad={json.dumps(c.get('dgrad'), sort_keys=True)} "
      f"recurrent_fallbacks={(c.get('fast_train') or {}).get('recurrent_fallbacks')} driver_sha256={c['driver_sha256']}")
PYP
    rm -rf "$(dirname "$(dirname "$snap")")"
    continue
  fi
  if [ "$LADDER_ONLY" != 1 ]; then
  arm $fam reference "$snap" $off 5400
  if [ "$fam" = qwen3_5 ] && grep -q '"status": "oom"' receipts/${fam}_train_reference.json 2>/dev/null; then
    mv receipts/${fam}_train_reference.json receipts/${fam}_train_reference.resident_oom.json; off=1
    echo "$fam: resident reference OOM -> both arms under offload (the registered fallback)" | tee -a summary.txt
    arm $fam reference "$snap" $off 9000
  fi
  arm $fam fused "$snap" $off 5400
  fi
  say "ladder $fam"
  perl -e 'alarm 3600; exec @ARGV' python ladder.py --model "$snap" --rungs fused,fused_pre,keep --offload $off --steps 8 --warmup 3 \
    --seq $SEQ $LADDER_ARGS --out receipts/${fam}_ladder.json > logs/ladder_$fam.log 2>&1
  echo "$fam/ladder rc=$? $(grep -c SUMMARY logs/ladder_$fam.log) rungs summarised" | tee -a summary.txt
  rm -rf "$(dirname "$(dirname "$snap")")"          # the family's cache: the box disk holds one checkpoint at a time
done
python mg1_reduce.py receipts > RESULTS-mg1.txt 2>&1; cat RESULTS-mg1.txt | tee -a summary.txt
finish 0
