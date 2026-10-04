"""A tiny random Qwen3-MoE with Qwen3-30B-A3B's attention geometry (head_dim 128, GQA), its tokenizer, saved as a local
checkpoint for the prefill sync census. Weights are random: the census reads code paths, not numerics."""
import os, sys
import torch
from huggingface_hub import hf_hub_download
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
cfg = Qwen3MoeConfig(vocab_size=151936, hidden_size=512, intermediate_size=1024, num_hidden_layers=4,
                     num_attention_heads=8, num_key_value_heads=2, head_dim=128, moe_intermediate_size=256,
                     num_experts=32, num_experts_per_tok=8, norm_topk_prob=True, decoder_sparse_step=1,
                     mlp_only_layers=[], max_position_embeddings=4096, rope_theta=1000000.0, tie_word_embeddings=False,
                     bos_token_id=151643, eos_token_id=151645, torch_dtype="bfloat16")
torch.manual_seed(0)
m = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
m.save_pretrained(out, safe_serialization=True)
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
    p = hf_hub_download("Qwen/Qwen3-30B-A3B", f)
    open(os.path.join(out, f), "wb").write(open(p, "rb").read())
print("TINY_OK", out, sum(p.numel() for p in m.parameters()))
