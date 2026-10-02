#!/bin/bash
# TC3 amendment 5: the 12 GB HF mb1 arm re-run with the e4b arms' budget (alarm 7200 s); same staged harness, venv, snapshot, tokens, init.
W=/workspace/tc1-frontier12; O=/workspace/tc1-frontier12-rerun; A=7200
cd $W || exit 9
export HF_HUB_CACHE=$W/hf-cache TC1_W=$O HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
echo "[$(date -u +%FT%TZ)] rerun start: hf/hf_peft_m_mb1 alarm=$A out=$O"
env HF_HUB_OFFLINE=1 UNSLOTH_ENABLE_LOGGING=1 OMP_NUM_THREADS=6 TC1_BOX_CLASS="RTX A2000" TC1_ARM_ALARM_S=$A perl -e "alarm $A; exec @ARGV" $W/venv-e4b/bin/python -u $W/tc1_arm.py --framework hf --arm hf --tag hf_peft_m_mb1 --fam qwen3frontier12 --model Qwen/Qwen3-30B-A3B --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --steps 20 --seq 2048 --micro-batch 1 --accum 8 --autocast 0 --lr 2e-4 --r 16 --alpha 16 --seed 3407 --offload 0 \
  --optim adamw_8bit --weight-decay 0.001 --lr-schedule linear --warmup-steps 5 \
  --tokens tokens_qwen3frontier12.json --tokens-sha bfc742f67e376ecfee191e2c6fd34170ab2ed310260a8c89bc79c46e3e7521ac --eval-every 20 --eval-n 8 --unsloth-loader FastLanguageModel \
  --prereg tc1/TC3-PREREG.md --out $O --adapter-dir $O/adapters --adapter-dtype fp32 --lora-init matched:3407 \
  --note "TC1_LOCAL_BOX hand run, TC3 amendment 5: the HF mb1 arm re-run with alarm 7200 s; snapshot /models/Qwen3-30B-A3B (pin_proof: config.json sha == hub@ad44e777bcd1 and 16 safetensors (61066575648 bytes) match the Hub listing)" \
  > $O/logs/run_qwen3frontier12_hf_hf_peft_m_mb1.log 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] rerun rc=$rc"
grep -aE "^CELL |^LOAD OK|^PROLOGUE |^PHASE ALARM|^STUB|Error|error:" $O/logs/run_qwen3frontier12_hf_hf_peft_m_mb1.log | tail -3 | cut -c1-400
[ $rc -eq 142 ] && [ ! -s $O/qwen3frontier12_hf_hf_peft_m_mb1.json ] && echo "ALARM: SIGALRM at $A s and no stub written"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "RERUN_DONE rc=$rc"
