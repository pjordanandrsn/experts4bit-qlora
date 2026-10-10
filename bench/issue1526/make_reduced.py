"""A reduced-layer Qwen3-30B-A3B with random weights: real hidden size, heads, vocabulary, experts and top-k."""
import json
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

layers, out = int(sys.argv[1]), sys.argv[2]
REV = "ad44e777bcd1"        # the revision TC1's Qwen3-30B-A3B arms recorded (RESULTS-tc1-padbk.md)
cfg = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B", revision=REV)
cfg.num_hidden_layers = layers
cfg.mlp_only_layers = []
cfg.max_window_layers = layers
torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
model.save_pretrained(out, safe_serialization=True)
json.dump({"source": "Qwen/Qwen3-30B-A3B", "revision": REV, "num_hidden_layers": layers, "weights": "random, seed 0"},
          open(f"{out}/REDUCED.json", "w"))
print("saved", out, sum(p.numel() for p in model.parameters()))
