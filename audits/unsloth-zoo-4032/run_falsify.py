"""Falsification driver: does unsloth-zoo's shipped MoE bnb-4bit fix actually work?

Per arch: load through the patched path with load_in_4bit=True, then
  AUDIT  - are fused expert tensors bnb Params4bit (quant_state) or plain bf16?
  PARITY - patched expert forward vs independent fp32 loop over DEQUANTIZED
           stored weights (never re-quantized).
  TRAIN  - two adapter routes; assert WHERE LoRA landed and that expert-adapter
           grads are nonzero (a finite loss alone is a hollow pass).

Usage: run_falsify.py --arch {qwen3_tiny,olmoe,gemma4} [--model PATH]
"""

import argparse
import gc
import json
import sys
import traceback

import unsloth  # must precede transformers: applies zoo temporary patches ("init" phase)
import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.activations import ACT2FN

from audit_lib import audit_experts, find_expert_modules, parity_check, train_probe, lora_placement, dump

ARCHS = {
    "qwen3_tiny": {"model": "/tmp/claude-tiny-qwen3moe", "loader": "fastmodel", "truncate": None},
    "olmoe": {"model": "allenai/OLMoE-1B-7B-0924", "loader": "fastmodel", "truncate": None},
    "qwen3_30b": {"model": "/home/node/venvs/qwen30b-slice", "loader": "auto", "truncate": None,
               "probe": {"seq_len": 16, "batch": 1}},
    "gemma4": {"model": "google/gemma-4-26B-A4B", "loader": "auto", "truncate": 4,
               "probe": {"seq_len": 8, "batch": 1}},  # 262k-vocab CE probe OOMs a 10 GB card at 64x2
}

REPORTER_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(spec, result):
    """Load through the most reporter-faithful path that works; record which."""
    name = spec["model"]
    if spec["loader"] == "fastmodel":
        try:
            from unsloth import FastLanguageModel
            model, tok = FastLanguageModel.from_pretrained(
                model_name=name, max_seq_length=512, load_in_4bit=True, dtype=None,
            )
            result["loader_used"] = "FastLanguageModel"
            return model, tok
        except Exception as e:
            result["fastmodel_error"] = f"{type(e).__name__}: {e}"
            print(f"[FastLanguageModel failed: {e!r} -- falling back to AutoModel]")
    cfg = AutoConfig.from_pretrained(name)
    if spec["truncate"]:
        # set on the raw text_config: zoo's gemma4 patch wraps get_text_config()
        # in a read-only proxy (_Gemma4KVSharedSafeProxy)
        tc = getattr(cfg, "text_config", cfg)
        tc.num_hidden_layers = spec["truncate"]
        result["truncated_layers"] = spec["truncate"]
    model = AutoModelForCausalLM.from_pretrained(
        name, config=cfg, quantization_config=bnb_config(),
        dtype=torch.bfloat16, device_map={"": 0},
    )
    result["loader_used"] = "AutoModelForCausalLM (post-import-unsloth, patched quantizer)"
    return model, None


def run(arch, model_override=None):
    spec = dict(ARCHS[arch])
    if model_override:
        spec["model"] = model_override
    result = {"arch": arch, "model": spec["model"]}

    model, tok = load_model(spec, result)
    model.eval()
    cfg = model.config.get_text_config()

    # --- AUDIT ---
    rows, summary = audit_experts(model)
    result["audit"] = summary
    result["audit_first_module"] = rows[0] if rows else None
    print(f"[AUDIT] expert modules: {summary['n_expert_modules']}, "
          f"quantized: {summary['n_quantized']}, "
          f"expert bytes: {summary['expert_bytes_total']/2**30:.2f} GiB, "
          f"cuda alloc: {summary['cuda_allocated_gb']} GiB")

    # --- PARITY (first experts module) ---
    if rows:
        _, mod = find_expert_modules(model)[0]
        act = getattr(mod, "act_fn", None) or ACT2FN[cfg.hidden_act]
        n_exp = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts")
        top_k = getattr(cfg, "num_experts_per_tok", 2)
        try:
            result["parity"] = parity_check(mod, cfg.hidden_size, n_exp, top_k, act)
            print(f"[PARITY] {result['parity']}")
        except Exception as e:
            result["parity"] = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()

    # --- TRAIN probe (a): unsloth get_peft_model, reporter-style targets ---
    if result["loader_used"] == "FastLanguageModel":
        try:
            from unsloth import FastLanguageModel
            model = FastLanguageModel.get_peft_model(
                model, r=8, lora_alpha=16, target_modules=REPORTER_TARGETS,
                use_gradient_checkpointing=False,
            )
            result["lora_placement_unsloth"] = {
                k: {"n": len(v), "sample": v[:4]} for k, v in lora_placement(model).items()
            }
            model.train()
            result["train_unsloth_path"] = train_probe(model, cfg.vocab_size, **spec.get("probe", {}))
            print(f"[TRAIN unsloth-path] {json.dumps(result['train_unsloth_path'])}")
            print(f"[LORA placement] { {k: v['n'] for k, v in result['lora_placement_unsloth'].items()} }")
        except Exception as e:
            result["train_unsloth_path"] = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        # fresh model for probe (b)
        del model
        gc.collect(); torch.cuda.empty_cache()
        model, tok = load_model(spec, result)

    # --- TRAIN probe (b): raw peft, explicit expert-parameter targeting ---
    try:
        from peft import LoraConfig, get_peft_model
        pcfg = LoraConfig(
            r=8, lora_alpha=16, task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
        )
        model = get_peft_model(model, pcfg)
        result["lora_placement_peft"] = {
            k: {"n": len(v), "sample": v[:4]} for k, v in lora_placement(model).items()
        }
        model.train()
        result["train_peft_params"] = train_probe(model, cfg.vocab_size, **spec.get("probe", {}))
        print(f"[TRAIN peft target_parameters] {json.dumps(result['train_peft_params'])}")
        print(f"[LORA placement] { {k: v['n'] for k, v in result['lora_placement_peft'].items()} }")
    except Exception as e:
        result["train_peft_params"] = {"error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    res = run(args.arch, args.model)
    dump(f"results_{args.arch}.json", res)
    print("\n=== ROW ===")
    a = res.get("audit", {})
    print(json.dumps({
        "arch": res["arch"], "loader": res.get("loader_used"),
        "quantized": f"{a.get('n_quantized')}/{a.get('n_expert_modules')}",
        "expert_GiB": round(a.get("expert_bytes_total", 0) / 2**30, 2),
        "parity_rel_err": res.get("parity", {}).get("max_rel_err"),
        "unsloth_expert_lora": res.get("lora_placement_unsloth", {}).get("expert", {}).get("n"),
        "peft_expert_lora": res.get("lora_placement_peft", {}).get("expert", {}).get("n"),
    }))
