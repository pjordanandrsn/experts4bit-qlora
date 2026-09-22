#!/usr/bin/env python3
"""kl_b16.py -- P59: KL between two SERVED arms at B=16, teacher-forced, one token per forward, sixteen rows at once.

Why a new scorer: `kl_fidelity.decode_teacher_forced_logits` (P44's) is batch=1 by design, and at one row the int4
attention takes the GEMV route. P54 found that fusing q/k/v changes the generated TOKENS at B=16 and not at B=1, and
P57 traced it to the K16 small-M GEMM (`Int4Linear.SMALLM_ROWS_MAX = 16`) tiling over a 3x wider N -- a route that
only exists for 2..16 rows. A quality read of that lever must therefore run the batched route: sixteen distinct
rows through the model together, one token per forward with a carried KV cache, exactly the shape the timed B=16
arms decode. Same scorer on both sides of every comparison (kl_fidelity's rule), stated in every receipt.

Fixture: the harness's own B=16 prompt rows (`step_decomp._k8_window`, wikitext-2 test, 16 distinct 512-token rows;
P54/P57/P58's `prompts_b16.json`, digest `f67e7e4d...`). The first `--prefix` tokens (384) are prefilled as one
[16, 384] forward -- the prefill path is shared by every arm (Int4Linear dequantises to bf16 above 16 rows, the
fusion only concatenates) and its last-position logits are kept as the PREFILL CONTROL; the remaining 128 positions
are decoded one token per forward at B=16 with the ground-truth token fed back (teacher forcing, never the model's
own argmax), so every arm's logits are aligned token-for-token. Logits are saved fp32 [16, 128, vocab] per arm; the
reducer computes KL(P_ref || P_test) in fp64 over the full vocabulary with kl_fidelity's primitives.

    python kl_b16.py --arm int4 --arena ... --calib ... --prompts prompts_b16.json --out out/int4      # on the box
    python kl_b16.py --arm int4_fqkv --fuse-qkv ...                                                    # the lever
    python kl_b16.py --self-test        # CPU: tiny Qwen3-MoE -- batched decode == per-row B=1 decode; self-KL 0;
                                        #      a perturbed copy reads KL > 0 and top-1 < 1

The arm is BUILT by `serve_stack.build_served_model` (P44's: the model exactly as the harness builds it, the staged
P42 hook applying the env lanes at load), then `--fuse-qkv` calls `experts4bit_qlora.engines.qkv_fuse.fuse_qkv` and
REFUSES unless it fused every attention module. The census (Int4Linear count, int4 expert layers, fusion counts) is
saved beside the logits: a KL against a stack whose lever silently did not apply is the number this lane must never
publish (kl_serve.py's rule).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "p44")):     # staged flat on the box; bench/ + bench/p44 in the repo
    if cand not in sys.path:
        sys.path.insert(0, cand)


def batched_teacher_forced(model, ids: torch.Tensor, prefix: int, device, prefill_chunk: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """ids [B, T] int64. Prefill ids[:, :prefix] in one forward; then one token per forward for positions
    prefix..T-1 with the carried KV cache, feeding the ground-truth token. Returns (prefill_last_logits [B, V],
    decode_logits [B, T-prefix, V]) as fp32 on CPU. decode_logits[:, j] is the model's prediction AFTER seeing
    ids[:, :prefix+j+1], i.e. the distribution over token prefix+j+1 -- identical alignment for every arm."""
    B, T = ids.shape
    assert 0 < prefix < T, (prefix, T)
    ids = ids.to(device)
    steps = []
    with torch.no_grad():
        # P59 amendment 1: ``prefill_chunk`` > 0 prefills in chunks of that many tokens PER ROW (B x chunk rows per
        # forward) -- the harness steps 128 prefill tokens at a time, and above 16 rows Int4Linear runs cuBLAS on its
        # cached bf16 weight, whose kernel choice depends on (M, N): the one path where fusing q/k/v can change bits.
        step = prefill_chunk if prefill_chunk > 0 else prefix
        past = None
        for s0 in range(0, prefix, step):
            out = model(input_ids=ids[:, s0:min(prefix, s0 + step)], past_key_values=past, use_cache=True)
            past = out.past_key_values if hasattr(out, "past_key_values") else out[1]
            if past is None:
                raise RuntimeError("the model returned no past_key_values after prefill: the batched scorer needs a carried KV cache")
        pre_last = (out.logits if hasattr(out, "logits") else out[0])[:, -1].float().cpu()
        for t in range(prefix, T):
            out = model(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values if hasattr(out, "past_key_values") else out[1]
            if past is None:
                raise RuntimeError(f"no past_key_values after decode step {t}")
            logits = out.logits if hasattr(out, "logits") else out[0]
            steps.append(logits[:, -1].float().cpu())
    return pre_last, torch.stack(steps, dim=1)


def run_arm(a) -> int:
    import serve_stack
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    pf = json.load(open(a.prompts))
    prompts = pf["prompts"]
    assert pf["batch"] == len(prompts) == a.batch, (pf["batch"], len(prompts), a.batch)
    assert len({len(p) for p in prompts}) == 1, "rows must be equal length (no padding, by design)"
    assert len({tuple(p) for p in prompts}) == a.batch, "rows must be distinct"
    psha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    assert psha == pf["prompts_sha256"], "prompt file digest mismatch"
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    model, info = serve_stack.build_served_model(a.model, a.arena, a.calib, device=a.device)
    info["build_s"] = round(time.time() - t0, 1)
    n_int4_before = sum(1 for m in model.modules() if isinstance(m, Int4Linear))
    if a.fuse_qkv:
        from experts4bit_qlora.engines.qkv_fuse import fuse_qkv
        n = fuse_qkv(model)
        info["fuse_qkv_n"] = n
        expected = a.expect_fused_modules
        if n != expected:
            raise SystemExit(f"fuse_qkv fused {n} attention modules, expected {expected}: refusing to score a half-fused arm")
        print(f"P59 fused q/k/v projections on {n} attention modules", flush=True)
    else:
        info["fuse_qkv_n"] = 0
    info["int4_attn_projections_before_fuse"] = n_int4_before
    info["int4_attn_projections"] = sum(1 for m in model.modules() if isinstance(m, Int4Linear))
    # amendment 1: the route each Int4Linear takes at 2..16 rows (K16 small-M GEMM vs cached-bf16 cuBLAS) -- the
    # module count alone cannot say whether the lever's arithmetic ran
    info["int4_attn_smallm_routed"] = sum(1 for m in model.modules() if isinstance(m, Int4Linear) and getattr(m, "_smallm", None) is not None)
    info["prefill_chunk"] = a.prefill_chunk
    info["env"] = {k: v for k, v in os.environ.items() if k.startswith("E4B_") or k.startswith("GNF4_")}
    info["arm"] = a.arm
    info["prompts_sha256"] = psha
    info["prefix"] = a.prefix
    ids = torch.tensor(prompts, dtype=torch.long)
    t1 = time.time()
    pre_last, dec = batched_teacher_forced(model, ids, a.prefix, a.device, prefill_chunk=a.prefill_chunk)
    info["score_s"] = round(time.time() - t1, 1)
    info["decode_logits_shape"] = list(dec.shape)
    info["vocab"] = int(dec.shape[-1])
    torch.save({"prefill_last": pre_last, "decode": dec, "prefix": a.prefix, "prompts_sha256": psha, "arm": a.arm},
               os.path.join(a.out, "logits.pt"))
    json.dump(info, open(os.path.join(a.out, "census.json"), "w"), indent=1)
    print("P59ARM " + json.dumps({k: info[k] for k in ("arm", "fuse_qkv_n", "int4_attn_projections", "int4_expert_layers",
                                                        "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n",
                                                        "decode_logits_shape", "build_s", "score_s") if k in info}), flush=True)
    return 0


def _self_test() -> int:
    """CPU: a tiny random Qwen3-MoE. (1) batched teacher forcing at B=5 equals kl_fidelity's per-row B=1 decode scorer
    on every row (alignment + cache handling); (2) the same model scored twice reads KL exactly 0; (3) a perturbed
    copy (one expert's weights nudged) reads KL > 0 and top-1 < 1 -- the harness detects the change it exists for."""
    import kl_fidelity
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_experts=16,
                         num_experts_per_tok=4, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, vocab_size=512, max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).eval()
    B, T, prefix = 5, 24, 16
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (B, T), generator=g)
    pre, dec = batched_teacher_forced(model, ids, prefix, "cpu")
    assert dec.shape == (B, T - prefix, cfg.vocab_size), dec.shape
    worst = 0.0
    for b in range(B):
        ref = kl_fidelity.decode_teacher_forced_logits(model, ids[b])          # [T, V]: position i predicts token i+1
        # batched decode logit j (after seeing prefix+j+1 tokens) == per-row logit at index prefix+j
        d = (dec[b] - ref[prefix:].float()).abs().max().item()
        d_pre = (pre[b] - ref[prefix - 1].float()).abs().max().item()
        worst = max(worst, d, d_pre)
    assert worst < 2e-3, f"batched decode differs from per-row B=1 decode by {worst:.3e}"
    pre_c, dec_c = batched_teacher_forced(model, ids, prefix, "cpu", prefill_chunk=5)     # 16 = 5+5+5+1: an uneven last chunk
    worst_c = max((dec_c - dec).abs().max().item(), (pre_c - pre).abs().max().item())
    assert worst_c < 2e-3, f"chunked prefill differs from one-forward prefill by {worst_c:.3e}"
    acc = kl_fidelity.KLAccumulator()
    pre2, dec2 = batched_teacher_forced(model, ids, prefix, "cpu")
    for b in range(B):
        acc.add(dec[b], dec2[b])
    s = acc.summary()
    assert s["exactly_zero"] and s["kl_mean"] == 0.0, s
    pert = Qwen3MoeForCausalLM(cfg).eval()
    pert.load_state_dict(model.state_dict())
    with torch.no_grad():
        for m in pert.modules():
            if type(m).__name__ == "Qwen3MoeSparseMoeBlock":
                for p in list(m.parameters())[:1]:
                    p.add_(0.5 * torch.randn_like(p))
    _, dec3 = batched_teacher_forced(pert, ids, prefix, "cpu")
    acc2 = kl_fidelity.KLAccumulator()
    for b in range(B):
        acc2.add(dec[b], dec3[b])
    s2 = acc2.summary()
    assert s2["kl_mean"] > 0 and s2["top1_agreement"] < 1.0, s2
    print(json.dumps({"batched_vs_per_row_max_abs_diff": worst, "chunked_vs_one_forward_max_abs_diff": worst_c, "self_kl": s["kl_mean"], "perturbed_kl": s2["kl_mean"],
                      "perturbed_top1": s2["top1_agreement"], "tokens_scored": s["n_tokens_scored"]}, indent=1))
    print("self-test OK: batched B=5 teacher forcing == per-row decode scorer; chunked prefill == one forward; self-KL exactly 0; perturbation detected")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--arena")
    ap.add_argument("--calib")
    ap.add_argument("--prompts")
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--prefix", type=int, default=384)
    ap.add_argument("--prefill-chunk", type=int, default=0, help="tokens per row per prefill forward; 0 = one forward (P59 run 1)")
    ap.add_argument("--fuse-qkv", action="store_true")
    ap.add_argument("--expect-fused-modules", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    for k in ("arm", "arena", "calib", "prompts", "out"):
        if getattr(a, k) is None:
            ap.error(f"--{k.replace('_', '-')} is required in arm mode")
    return run_arm(a)


if __name__ == "__main__":
    sys.exit(main())
