"""The instrument's model: a random two-layer Qwen3-MoE at Qwen3-30B-A3B's per-layer dimensions, saved in bf16 for e4b's loader."""
import os
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
D = os.environ.get("P129_MODEL_DIR", "qwen3moe-2l")
cfg = Qwen3MoeConfig(vocab_size=4096, hidden_size=2048, intermediate_size=6144, moe_intermediate_size=768, num_hidden_layers=2,
                     num_attention_heads=32, num_key_value_heads=4, head_dim=128, num_experts=128, num_experts_per_tok=8,
                     decoder_sparse_step=1, mlp_only_layers=[], max_position_embeddings=4096, norm_topk_prob=True, tie_word_embeddings=False)
torch.manual_seed(0)
Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).save_pretrained(D)
print("wrote", D)
