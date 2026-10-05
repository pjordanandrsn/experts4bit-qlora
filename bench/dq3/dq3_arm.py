"""bench/dq3/dq3_arm.py -- lane DQ3, BOX side, ONE arm per process (bench/dq3/DQ3-PREREG.md, Amendment 1).

Arms: R (resident), S (``enable_dense_offload(train_prefetch=True)``: overlapped), S0 (``enable_dense_offload()``: today's
synchronous grad-mode staging). The subject is Qwen3-32B's architecture built from its config with random NF4 weights
(DQ2's layer construction at full depth; parity, step time and memory do not depend on weight values), wrapped by PEFT
``get_peft_model`` -- so PEFT's ``lora.bnb.Linear4bit`` with fp32 adapters -- under HF non-reentrant gradient
checkpointing.

Two passes, same seed, same initial adapters:
  parity  2 steps, deterministic (use_deterministic_algorithms, CUBLAS_WORKSPACE_CONFIG set by the runner, SDPA math);
          loss + sha256 of every LoRA gradient after each step
  timing  default SDPA, adapters and optimizer reset: 2 warm + 6 timed steps; per-step wall time, loss, allocator peak,
          pinned-host stats, dense-offload counters (S, S0)
Writes one JSON receipt (``--out``); the verdict is dq3_reduce.py's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch

QWEN3_32B = dict(hidden_size=5120, intermediate_size=25600, num_attention_heads=64, num_key_value_heads=8, head_dim=128,
                 rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=40960, attention_bias=False,
                 hidden_act="silu", num_hidden_layers=64, vocab_size=151936, tie_word_embeddings=False)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_model(cfg_overrides: dict, seed: int = 0):
    """Qwen3ForCausalLM from its config, every decoder projection a bnb Linear4bit (nf4, blocksize 64, double-quant,
    bf16 compute) quantized from random bf16 weights, layer by layer so the bf16 model never exists whole."""
    import bitsandbytes as bnb
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    cfg = Qwen3Config(**{**QWEN3_32B, **cfg_overrides})
    cfg._attn_implementation = "sdpa"
    n_layers = cfg.num_hidden_layers
    cfg_shell = Qwen3Config(**{**QWEN3_32B, **cfg_overrides, "num_hidden_layers": 0})
    cfg_shell._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    with torch.device("cuda"):
        model = Qwen3ForCausalLM(cfg_shell).to(torch.bfloat16)
    model.config.num_hidden_layers = n_layers
    layers = []
    for i in range(n_layers):
        with torch.device("cuda"):
            lay = Qwen3DecoderLayer(cfg, layer_idx=i).to(torch.bfloat16)
        for parent in (lay.self_attn, lay.mlp):
            for name in TARGETS:
                lin = getattr(parent, name, None)
                if lin is None:
                    continue
                q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=False, compute_dtype=torch.bfloat16,
                                      compress_statistics=True, quant_type="nf4", device="cpu")
                q.weight = bnb.nn.Params4bit(lin.weight.data.detach().to("cpu"), requires_grad=False,
                                             compress_statistics=True, quant_type="nf4", blocksize=64)
                setattr(parent, name, q.to("cuda"))
                del lin
        layers.append(lay)
        torch.cuda.empty_cache()
    model.model.layers = torch.nn.ModuleList(layers)
    model.config.layer_types = ["full_attention"] * n_layers if hasattr(model.config, "layer_types") else None
    for p in model.parameters():
        p.requires_grad_(False)
    model.is_loaded_in_4bit = True          # what a BitsAndBytesConfig load sets; PEFT's bnb dispatch reads it
    return model, cfg


def lora_wrap(model, seed: int = 0):
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    torch.manual_seed(seed)
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS,
                                          task_type="CAUSAL_LM"))
    return pm


def engagement(pm) -> dict:
    wr = [type(m).__module__ + "." + type(m).__name__ for n, m in pm.named_modules() if n.rsplit(".", 1)[-1] in TARGETS]
    lora = [p for n, p in pm.named_parameters() if "lora_" in n]
    return {"wrapped": len(wr), "wrapper_kinds": sorted(set(wr)), "lora_params": len(lora),
            "lora_dtypes": sorted({str(p.dtype) for p in lora}), "trainable": sum(p.numel() for p in lora),
            "gradient_checkpointing": bool(getattr(pm.base_model.model, "is_gradient_checkpointing", False))}


def lora_state(pm):
    return {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_" in n}


def load_lora_state(pm, st):
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if n in st:
                p.copy_(st[n])


def grad_hashes(pm) -> dict:
    out = {}
    for n, p in pm.named_parameters():
        if "lora_" in n:
            g = p.grad
            out[n] = None if g is None else hashlib.sha256(g.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    return out


def host_stats() -> dict:
    try:
        st = torch.cuda.host_memory_stats()
        return {k: v for k, v in st.items() if "reserved" in k or "allocated" in k}
    except Exception as exc:  # recorded
        return {"error": str(exc)}


def run_steps(pm, ids, n_steps, opt, timed: bool, handles):
    from experts4bit_qlora.engines.dense_offload import dense_offload_report
    out = []
    for _ in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = pm(input_ids=ids, labels=ids).loss
        loss.backward()
        rec = {"loss": float(loss.detach())}
        if not timed:
            rec["grads"] = grad_hashes(pm)
            rec["loss_bits"] = hashlib.sha256(loss.detach().float().cpu().numpy().tobytes()).hexdigest()
        opt.step()
        torch.cuda.synchronize()
        rec["s"] = time.perf_counter() - t0
        if handles:
            rep = dense_offload_report(handles)
            rec["counters"] = rep["train_prefetch"]
        out.append(rec)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=("R", "S", "S0"))
    p.add_argument("--out", required=True)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--warm", type=int, default=2)
    p.add_argument("--timed", type=int, default=6)
    p.add_argument("--parity-steps", type=int, default=2)
    p.add_argument("--layers", type=int, default=64, help="rehearsal only: fewer layers")
    p.add_argument("--hidden", type=int, default=0, help="rehearsal only: a smaller hidden size (0 = Qwen3-32B's)")
    p.add_argument("--rehearsal", action="store_true")
    args = p.parse_args()

    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experts4bit_qlora.engines.dense_offload import dense_offload_report, enable_dense_offload

    over = {"num_hidden_layers": args.layers}
    if args.hidden:
        over.update(hidden_size=args.hidden, intermediate_size=args.hidden * 5, num_attention_heads=max(1, args.hidden // 128),
                    num_key_value_heads=max(1, args.hidden // 640), vocab_size=32000)
    rec = {"schema": "dq3-arm/1", "arm": args.arm, "rehearsal": bool(args.rehearsal), "seq": args.seq,
           "started_at": time.strftime("%FT%TZ", time.gmtime()), "device": torch.cuda.get_device_name(),
           "config_overrides": over, "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "E4B_SHA")}}

    def flush():
        with open(args.out + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(args.out + ".tmp", args.out)

    t0 = time.time()
    model, cfg = build_model(over)
    pm = lora_wrap(model)
    rec["build_s"] = round(time.time() - t0, 1)
    rec["engagement"] = engagement(pm)
    handles = []
    if args.arm in ("S", "S0"):
        handles = enable_dense_offload(pm, pin=True, prefetch=False, train_prefetch=(args.arm == "S"))
        torch.cuda.empty_cache()
        rep = dense_offload_report(handles)
        rec["offload"] = {k: rep[k] for k in ("layers", "tensors", "host_bytes", "per_layer_bytes", "all_pinned",
                                              "late_bound_4bit")}
    rec["host_after_setup"] = host_stats()
    flush()

    g = torch.Generator(device="cpu").manual_seed(1234)
    ids = torch.randint(0, cfg.vocab_size, (1, args.seq), generator=g).to("cuda")
    init = lora_state(pm)

    # ---- parity pass: deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)
    opt = torch.optim.AdamW([q for q in pm.parameters() if q.requires_grad], lr=2e-4)
    with sdpa_kernel(SDPBackend.MATH):
        rec["parity"] = run_steps(pm, ids, args.parity_steps, opt, timed=False, handles=handles)
    torch.use_deterministic_algorithms(False)
    flush()

    # ---- timing pass: realistic
    load_lora_state(pm, init)
    opt = torch.optim.AdamW([q for q in pm.parameters() if q.requires_grad], lr=2e-4)
    torch.cuda.empty_cache()      # the timing peaks, reserved included, start clean of the parity pass's cache
    torch.cuda.reset_peak_memory_stats()
    steps = run_steps(pm, ids, args.warm + args.timed, opt, timed=True, handles=handles)
    rec["timing"] = {"warm": steps[:args.warm], "timed": steps[args.warm:],
                     "peak_alloc": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()}
    rec["host_after_timing"] = host_stats()
    rec["finished_at"] = time.strftime("%FT%TZ", time.gmtime())
    flush()
    print(f"DQ3 arm {args.arm} done: build {rec['build_s']} s, step {sum(s['s'] for s in steps[args.warm:]) / max(1, args.timed):.3f} s, "
          f"peak {rec['timing']['peak_alloc'] / 2**30:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
