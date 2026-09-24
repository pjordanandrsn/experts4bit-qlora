#!/bin/bash
# P67 REHEARSAL on the NAS A2000 -- NOT a reading (bench/p67/P67-PREREG.md, "Rehearsal"). Inside
# pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel with the GPU mapped. tp4_arm.py's own arms, one process each (the
# way tp4_run.sh runs them), on granite-3.0-1b-a400m-instruct from the NAS store, at a SHORTENED fixture (seq 512,
# 16 held-out rows) so the whole set fits a shared card. gnf4 at the registered cut (9206352f, v0.32.1).
set -uo pipefail
R=/w/rehearsal/${REH_TAG:?}; mkdir -p $R/logs $R/receipts $R/adapters; cd $R
export PYTHONPATH=/w/site-run:/w/e4b-reh:/w/pyenv PATH=/w/pyenv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
M=${REH_MODEL:-/models/granite-3.0-1b-a400m-instruct}; FAM=${REH_FAM:-granite}; ARM=/w/e4b-reh/bench/tp4/tp4_arm.py
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version,compute_cap --format=csv,noheader | tee gpu.txt
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "${free_mb:-0}" -ge 4000 ] || { echo "REFUSED: only ${free_mb} MiB free on the shared card"; exit 75; }
python -c "import importlib.metadata as m, torch, experts4bit_qlora as e; print('torch', torch.__version__, 'e4b', e.__version__, e.__file__, 'gnf4', m.version('grouped-nf4-gemm'), 'bnb', m.version('bitsandbytes'), 'transformers', m.version('transformers'))" | tee versions.txt
sha256sum /w/e4b-reh/experts4bit_qlora/lora.py /w/e4b-reh/bench/tp4/tp4_arm.py | sed "s|/w/e4b-reh/||" | tee -a versions.txt
python $ARM --prepare --fam $FAM --model $M --revision local --data /w/rehearsal/ds_alpaca.json \
  --data-sha 5324987afa4042556953026289e8dbdbe8a936b32832ed9e603b9192b706a2fb --seq 512 --eval-n 16 --template alpaca \
  --tokens $R/tokens_$FAM.json > logs/prepare.log 2>&1 || { tail -5 logs/prepare.log; exit 3; }
tail -1 logs/prepare.log | tee -a summary.txt
TS=$(python -c "import json; print(json.load(open('$R/tokens_$FAM.json'))['sha256'])")
COMMON="--framework e4b --fam $FAM --model $M --revision local --steps 20 --seq 512 --micro-batch 2 --accum 4 --autocast 0 --lr 2e-4 --r 16 --alpha 16 --seed 3407 --offload 0 --optim adamw_8bit --weight-decay 0.001 --lr-schedule linear --warmup-steps 5 --tokens $R/tokens_$FAM.json --tokens-sha $TS --eval-every 20 --eval-n 16 --attn-4bit 1 --no-sampler 1 --prereg bench/p67/P67-PREREG.md --out $R/receipts --adapter-dir $R/adapters"
run(){ local tag=$1 arm=$2; shift 2
  env "$@" python -u $ARM $COMMON --arm $arm --tag $tag > logs/run_$tag.log 2>&1; local rc=$?
  { echo -n "$tag rc=$rc "; grep -aE "^CELL " logs/run_$tag.log | tail -1 | cut -c1-240; echo; } | tee -a summary.txt
  rm -rf $R/adapters/*; }
for spec in ${REH_ARMS:-"reference_attn4:reference" "reference_attn4_repeat:reference" "reference_attn4_perm1:reference:E4B_REFERENCE_EXPERT_ORDER=perm:1" "reference_attn4_perm2:reference:E4B_REFERENCE_EXPERT_ORDER=perm:2" "reference_attn4_perm3:reference:E4B_REFERENCE_EXPERT_ORDER=perm:3" "batched_attn4:batched:E4B_BATCHED_PAD_WASTE_LIMIT=64" "fused_attn4:fused"}; do
  IFS=: read -r tag arm envv <<< "$spec"
  if [ -n "${envv:-}" ]; then run "$tag" "$arm" "$envv"; else run "$tag" "$arm"; fi
done
echo REHEARSAL DONE | tee -a summary.txt
