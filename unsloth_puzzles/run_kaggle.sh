#!/usr/bin/env bash
# One-shot Task B runner for a Kaggle 2x T4 notebook cell:
#   !curl -sSL https://raw.githubusercontent.com/pjordanandrsn/experts4bit-qlora/triton-nf4/unsloth_puzzles/run_kaggle.sh | bash
# Installs deps, fetches the SFT script, writes a bf16 FSDP2 config, runs the single-GPU
# reference + the 2x T4 FSDP2 job, and prints the two loss curves side by side (equivalence).
set -eo pipefail
# Pin to an immutable commit SHA (not the branch) so raw.githubusercontent.com's ~5 min branch cache
# can't serve a stale fsdp2_qlora_sft.py. Bump this SHA when the script changes.
BASE=https://raw.githubusercontent.com/pjordanandrsn/experts4bit-qlora/4c02db2a9ddddf7646897dd46d907853197a3641/unsloth_puzzles
MAX_STEPS="${MAX_STEPS:-20}"
export MAX_SEQ="${MAX_SEQ:-512}"  # keep the demo tractable on a T4
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # reduce load-time fragmentation

pip install -q -U bitsandbytes accelerate peft trl datasets
wget -qO fsdp2_qlora_sft.py "$BASE/fsdp2_qlora_sft.py"

cat > fsdp2_config.yaml <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16          # bf16 (emulated but functional on a T4); avoids the fp16 GradScaler
num_machines: 1
num_processes: 2
rdzv_backend: static
use_cpu: false
fsdp_config:
  fsdp_version: 2
  fsdp_offload_params: true
  fsdp_activation_checkpointing: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_cpu_ram_efficient_loading: false
  fsdp_use_orig_params: true
YAML

echo "########## 1) SINGLE-GPU REFERENCE (${MAX_STEPS} steps) ##########"
MAX_STEPS="$MAX_STEPS" CUDA_VISIBLE_DEVICES=0 python fsdp2_qlora_sft.py --single

echo "########## 2) 2x T4 FSDP2 + QLoRA (${MAX_STEPS} steps) ##########"
MAX_STEPS="$MAX_STEPS" accelerate launch --config_file fsdp2_config.yaml fsdp2_qlora_sft.py

echo "########## 3) LOSS EQUIVALENCE ##########"
python - <<'PY'
import json
s = dict(json.load(open("losses_single.json")))
f = dict(json.load(open("losses_fsdp2.json")))
ks = sorted(set(s) & set(f))
print("step   single    fsdp2")
for k in ks:
    print(f"{k:>4}   {s[k]:.4f}   {f[k]:.4f}")
print("MAX_ABS_LOSS_DIFF =", round(max(abs(s[k] - f[k]) for k in ks), 4) if ks else "n/a")
PY
