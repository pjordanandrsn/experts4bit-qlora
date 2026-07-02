"""Module-level parity for any model/loader combo.

Usage: parity_only.py [--model PATH_OR_ID] [--truncate N] [--no4bit] [--loader auto|fastmodel]
"""

import argparse
import json
import os

import unsloth  # noqa: F401  (applies zoo patches)
import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.activations import ACT2FN

from audit_lib import find_expert_modules, parity_check, audit_experts

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/tmp/claude-tiny-qwen3moe")
ap.add_argument("--truncate", type=int, default=None)
ap.add_argument("--no4bit", action="store_true")
ap.add_argument("--loader", default="fastmodel", choices=["fastmodel", "auto"])
args = ap.parse_args()

if args.loader == "fastmodel":
    from unsloth import FastLanguageModel
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=512, load_in_4bit=not args.no4bit, dtype=None,
    )
else:
    cfg = AutoConfig.from_pretrained(args.model)
    if args.truncate:
        getattr(cfg, "text_config", cfg).num_hidden_layers = args.truncate
    kw = {}
    if not args.no4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, dtype=torch.bfloat16, device_map={"": 0}, **kw)

model.eval()
cfg = model.config.get_text_config()
_, summary = audit_experts(model)
_, mod = find_expert_modules(model)[0]
act = getattr(mod, "act_fn", None) or ACT2FN[cfg.hidden_act]
n_exp = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts")
torch.manual_seed(0)
try:
    res = parity_check(mod, cfg.hidden_size, n_exp, cfg.num_experts_per_tok, act)
except Exception as e:
    res = {"error": f"{type(e).__name__}: {e}"}
print(f"MODEL={args.model} 4bit={not args.no4bit} trunc={args.truncate} "
      f"quantized={summary['n_quantized']}/{summary['n_expert_modules']} "
      f"BACKEND={os.environ.get('UNSLOTH_MOE_BACKEND', '(auto)')}\nPARITY={json.dumps(res)}")
