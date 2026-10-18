#!/usr/bin/env python3
"""families_probe.py -- the $0 correctness leg of bench/layer-major-families/DESIGN.md on the real path: a tiny model of
each family written in its on-disk layout, loaded by load_moe_4bit_streaming with NF4 experts offloaded, gradient
checkpointing enabled and then enable_fast_train(dgrad=True), on one CUDA card. No timing is read.

    python families_probe.py --out probe.json      (run from the repository root; tests/ on sys.path)

STRICT per family (Mixtral, OLMoE, Gemma-4; Qwen3 as the control):
  1. the first layer_major_step passes its drift guard: every micro-batch's loss torch.equal to the model's own stock
     forward, the stock forward bit-reproducible (otherwise the step raises before any backward);
  2. stagings: layer-major 2n - 1 per steady step (2n + the guard's forwards on the first), the stock loop
     GRAD_ACCUM x (2n - 1) per steady step, n the family's offloaded MoE layers;
  3. every decoder layer runs e4b's reentrant checkpoint (enable_fast_train's routing) on the fused path;
  4. Gemma-4: the mutant sharing one shared_kv_states dict across micro-batches raises LayerMajorDriftError.
REPORTING: the largest relative gradient gap, layer-major against the stock loop (bf16: a band, not equality).
"""
import argparse
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, "tests")
VOCAB, M = 97, 3


def _models():
    import transformers as tr
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    return {
        "qwen3": lambda: tr.Qwen3MoeForCausalLM(tr.Qwen3MoeConfig(
            hidden_size=128, intermediate_size=256, moe_intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, num_experts=8, num_experts_per_tok=2, vocab_size=VOCAB,
            decoder_sparse_step=1, mlp_only_layers=[], max_position_embeddings=64)),
        "mixtral": lambda: tr.MixtralForCausalLM(tr.MixtralConfig(
            hidden_size=128, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            num_local_experts=8, num_experts_per_tok=2, vocab_size=VOCAB, max_position_embeddings=64)),
        "olmoe": lambda: tr.OlmoeForCausalLM(tr.OlmoeConfig(
            hidden_size=128, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
            num_experts=8, num_experts_per_tok=2, vocab_size=VOCAB, max_position_embeddings=64)),
        "gemma4": lambda: Gemma4ForCausalLM(Gemma4TextConfig(
            vocab_size=VOCAB, hidden_size=128, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, global_head_dim=64, num_global_key_value_heads=1,
            layer_types=["sliding_attention", "full_attention"] * 3, sliding_window=4, num_kv_shared_layers=2,
            enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=64, attention_k_eq_v=True,
            hidden_size_per_layer_input=0, vocab_size_per_layer_input=VOCAB, max_position_embeddings=512)),
    }


def _write(family, model, d):
    from test_loader_architectures import _write_block_sparse_ckpt, _write_ckpt
    if family == "mixtral":
        _write_block_sparse_ckpt(model, d)
    else:
        _write_ckpt(model, d, per_expert=family in ("qwen3", "olmoe"))


def _batches(widths, seed=3):
    g = torch.Generator().manual_seed(seed)
    out = []
    for w in widths:
        ids = torch.randint(1, VOCAB, (2, w), generator=g)
        att = torch.ones(2, w, dtype=torch.long)
        att[1, w - 2:] = 0
        out.append({"input_ids": ids.cuda(), "labels": ids.masked_fill(att == 0, -100).cuda(), "attention_mask": att.cuda()})
    return out


def _load(path):
    from experts4bit_qlora.engines.fast import enable_fast_train
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    torch.manual_seed(0)
    model, _ = load_moe_4bit_streaming(path, "cuda", torch.bfloat16, 4, 8, offload=True, pin=True, quant_type="nf4")
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_fast = enable_fast_train(model, dgrad=True, absmax_dq=False)
    model.config.use_cache = False
    model.train()
    return model, n_fast


def _ckpt_routing(model):
    from transformers.modeling_layers import GradientCheckpointingLayer

    from experts4bit_qlora.engines.ckpt_offload import reentrant_checkpoint
    layers = [m for m in model.modules() if isinstance(m, GradientCheckpointingLayer) and m is not model]
    return [getattr(m, "_gradient_checkpointing_func", None) is reentrant_checkpoint for m in layers]


def run_family(family, build, mutant=False):
    from experts4bit_qlora.engines import layer_major as lm
    from experts4bit_qlora.engines import offload as off
    os.environ["E4B_OFFLOAD_STATS"] = "1"
    d = tempfile.mkdtemp(prefix=f"lmf-{family}-")
    torch.manual_seed(0)
    _write(family, build(), d)
    widths = (12, 12, 12) if family == "gemma4" else (9, 14, 6)
    rec = {}
    model, rec["n_fast"] = _load(d)
    handles = off.offload_handles(model)
    rec["n_offloaded"] = n = len(handles)
    rec["ckpt_reentrant"] = _ckpt_routing(model)
    if mutant:
        shared = lm._shared_kv_states()
        real = lm._shared_kv_states
        lm._shared_kv_states = lambda: shared
        try:
            lm.layer_major_step(model, _batches(widths), [1.0 / M] * M)
            rec["mutant_raised"] = False
        except lm.LayerMajorDriftError:
            rec["mutant_raised"] = True
        finally:
            lm._shared_kv_states = real
        return rec
    # the stock loop, two steps, for its stagings and the reference gradients
    stock_stag = []
    for step in range(2):
        model.zero_grad(set_to_none=True)
        off.reset_offload_stats()
        for b in _batches(widths, seed=3 + step):
            (model(**b).loss / M).backward()
        stock_stag.append(len(off._stats().copies))
        if step == 0:
            ref = {k: p.grad.detach().float().clone() for k, p in model.named_parameters() if p.grad is not None}
    rec["stock_stagings"] = stock_stag
    lm_stag = []
    for step in range(2):
        model.zero_grad(set_to_none=True)
        off.reset_offload_stats()
        lm.layer_major_step(model, _batches(widths, seed=3 + step), [1.0 / M] * M)   # step 0 runs the drift guard
        lm_stag.append(len(off._stats().copies))
        if step == 0:
            got = {k: p.grad.detach().float().clone() for k, p in model.named_parameters() if p.grad is not None}
    rec["layer_major_stagings"] = lm_stag
    rec["verified"] = model in lm._VERIFIED
    rec["grad_keys_equal"] = got.keys() == ref.keys() and len(got) > 0
    rec["max_rel_grad_gap"] = max(float((got[k] - ref[k]).norm() / (ref[k].norm() + 1e-12)) for k in ref) if ref else None
    rec["checks"] = {
        "1_drift_guard_passed": rec["verified"],
        "2_stagings": n > 0 and lm_stag[1] == 2 * n - 1 and lm_stag[0] > 2 * n
        and stock_stag[1] == M * (2 * n - 1) and stock_stag[0] == M * (2 * n - 1) + 1,
        "3_reentrant_routed": rec["n_fast"] > 0 and len(rec["ckpt_reentrant"]) > 0 and all(rec["ckpt_reentrant"]),
        "grads_present": rec["grad_keys_equal"],
    }
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    import transformers
    out = {"transformers": transformers.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "families": {}}
    for family, build in _models().items():
        if a.only and family != a.only:
            continue
        try:
            out["families"][family] = run_family(family, build)
        except Exception as e:
            out["families"][family] = {"error": f"{type(e).__name__}: {str(e)[:400]}", "checks": {"ran": False}}
    try:
        out["gemma4_mutant"] = run_family("gemma4", _models()["gemma4"], mutant=True)
    except Exception as e:
        out["gemma4_mutant"] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
    checks = {f"{f}.{k}": v for f, r in out["families"].items() for k, v in r["checks"].items()}
    checks["gemma4.4_mutant_fails_the_equality_check"] = bool(out["gemma4_mutant"].get("mutant_raised"))
    out["checks"], out["pass"] = checks, all(checks.values())
    json.dump(out, open(a.out, "w"), indent=1)
    print("LMF_PROBE " + json.dumps({"pass": out["pass"], "checks": checks,
                                     "gaps": {f: r.get("max_rel_grad_gap") for f, r in out["families"].items()},
                                     "errors": {f: r["error"] for f, r in out["families"].items() if "error" in r}}),
          flush=True)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
