"""Create a tiny random-weight Qwen3-MoE checkpoint for mechanism testing.

Real Qwen tokenizer (tiny download) so FastModel's tokenizer load succeeds;
model weights are random — this tests the quantizer/adapter path, not quality.
"""

import sys
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-tiny-qwen3moe"

from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

cfg = Qwen3MoeConfig(
    vocab_size=len(tok),
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_experts=8,
    num_experts_per_tok=2,
    decoder_sparse_step=1,
    max_position_embeddings=512,
    tie_word_embeddings=True,
    initializer_range=0.2,  # default 0.02 makes expert outputs ~1e-5: bf16 noise drowns parity SNR
)

torch.manual_seed(0)
model = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("saved tiny Qwen3-MoE to", OUT, "| params:", sum(p.numel() for p in model.parameters()))
