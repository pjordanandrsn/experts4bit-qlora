"""A tiny random GraniteMoE (the NF4-store family SC1's $0 proofs serve), with Granite 3.1's tokenizer, saved locally."""
import os, sys
import torch
from huggingface_hub import hf_hub_download
from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
REPO = "ibm-granite/granite-3.1-3b-a800m-instruct"
cfg = GraniteMoeConfig(vocab_size=49155, hidden_size=512, intermediate_size=256, num_hidden_layers=4,
                       num_attention_heads=8, num_key_value_heads=2, num_local_experts=32, num_experts_per_tok=8,
                       max_position_embeddings=4096, tie_word_embeddings=True, bos_token_id=0, eos_token_id=0,
                       torch_dtype="bfloat16")
torch.manual_seed(0)
m = GraniteMoeForCausalLM(cfg).to(torch.bfloat16)
m.save_pretrained(out, safe_serialization=True)
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
    try:
        p = hf_hub_download(REPO, f)
        open(os.path.join(out, f), "wb").write(open(p, "rb").read())
    except Exception as e:  # noqa: BLE001
        print("skip", f, repr(e)[:120])
print("TINY_OK", out, sum(p.numel() for p in m.parameters()))
