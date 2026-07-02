"""FSDP2 + QLoRA SFT of Llama-3.1-8B on 2+ GPUs — Unsloth Puzzles "Task B".

A single script, launched with ``accelerate launch`` using an FSDP2 config (see fsdp2_config.yaml).
It stays fully transformers-native (``SFTTrainer`` / ``SFTConfig``), so the loss curve is identical
to single-GPU training given the same seed/data.

Two things make FSDP2 + bitsandbytes-NF4 QLoRA actually work (both handled below):

1. ``bnb_4bit_quant_storage`` is set to the training dtype (bf16/fp16). bitsandbytes then stores the
   packed 4-bit weights *inside* a tensor of that dtype, so FSDP2 can flatten + shard them uniformly
   with the (bf16) LoRA / norm params. Without this, the uint8 4-bit params can't join an FSDP2
   flat-parameter group and sharding fails.
2. The model is loaded with **no** ``device_map`` — FSDP2 places and shards it. (``device_map="auto"``
   pins layers to devices and is incompatible with FSDP.)

FSDP2 features exercised (all via fsdp2_config.yaml): parameter CPU offload, activation checkpointing,
bf16 mixed precision, and transformer-layer auto-wrap.

Run (2x GPU, e.g. Kaggle 2x T4 or any 2x NVIDIA / WSL2 box):

    accelerate launch --config_file unsloth_puzzles/fsdp2_config.yaml \
        unsloth_puzzles/fsdp2_qlora_sft.py

Single-GPU reference loss (to prove equivalence):

    CUDA_VISIBLE_DEVICES=0 python unsloth_puzzles/fsdp2_qlora_sft.py --single
"""

import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

# A 16-bit checkpoint (non-gated) — bitsandbytes quantizes it to NF4 on load *with* a shardable
# quant_storage (below). A pre-quantized `...-bnb-4bit` model can't set quant_storage, so FSDP2
# can't flatten its 4-bit params — hence a full-precision source is required here.
MODEL = os.environ.get("MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct")
MAX_SEQ = int(os.environ.get("MAX_SEQ", "2048"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "60"))
SEED = 3407
# fp16 on T4 (no bf16 on Turing); bf16 on Ampere+.
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
        # (1) pack the 4-bit weights inside a bf16/fp16 tensor so FSDP2 can shard them uniformly.
        bnb_4bit_quant_storage=DTYPE,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_config,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
        # (2) NO device_map — FSDP2 places + shards the model itself.
    )
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    return model, tok


def main():
    single = "--single" in sys.argv
    set_seed(SEED)
    model, tok = build_model_and_tokenizer()

    url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
    dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]")

    args = SFTConfig(
        output_dir="outputs-fsdp2-qlora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        logging_steps=1,
        seed=SEED,
        max_length=MAX_SEQ,  # SFTConfig renamed max_seq_length -> max_length as of trl 1.x
        bf16=DTYPE == torch.bfloat16,
        fp16=DTYPE == torch.float16,
        report_to="none",
        dataset_num_proc=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Under `accelerate launch` with fsdp2_config.yaml, the Trainer picks up the FSDP2 plugin
        # (sharding, CPU offload, activation checkpointing, bf16 mixed precision) automatically.
    )
    trainer = SFTTrainer(model=model, train_dataset=dataset, processing_class=tok, args=args)
    trainer.train()

    # Dump the loss curve (main rank) so single-GPU vs FSDP2 can be compared for equivalence.
    if trainer.accelerator.is_main_process:
        import json

        tag = "single" if single else "fsdp2"
        curve = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
        json.dump(curve, open(f"losses_{tag}.json", "w"))
        print(f"[{tag}] dtype={DTYPE} steps={MAX_STEPS} | loss curve ({len(curve)} pts) -> losses_{tag}.json")


if __name__ == "__main__":
    main()
