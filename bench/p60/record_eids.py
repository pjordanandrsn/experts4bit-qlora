#!/usr/bin/env python3
"""record_eids.py -- P60: record the expert ids Qwen3-30B-A3B's routers choose at B=16 decode, per layer per step.

The replay bench (`replay_gemv.py`) must replay the workload's real routing, never a synthetic one: the R=128 GEMV
sweep of P23 picked configs on UNIFORM expert ids and they lost 0.892x in serving, because real routing is skewed
(`finding_gemv_plan_microbench_refuted_by_real_routing`). So the ids come from the served int4 stack itself, on the
same 16 wikitext rows every B=16 lane decoded (P54/P57/P58/P59 `prompts_b16.json`): prefill 384 tokens per row in
8-token-per-row chunks (128 rows per forward, the harness's step), then decode 128 positions one token per forward at
B=16 with the ground-truth token fed back (teacher forcing -- the routing of the model on these tokens, with no
sampling in it). A forward hook on every MoE router records the routed ids of every 16-row call, host-side (this
path captures no CUDA graph, so the host read is legal). `E4B_FUSE_ROUTER_EPI=0` so the router module is called.

    python record_eids.py --arena ... --calib ... --prompts prompts_b16.json --out eids_b16.pt     # on the box
    python record_eids.py --self-test                                                            # CPU, tiny Qwen3-MoE

Output: {"eids": int16 [steps, layers, 16, top_k], "prompts_sha256", "prefix", "prefill_chunk", "distinct": per layer
per step, "mean_distinct"} -- the replay reads `eids`.
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
for cand in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "p44")):
    if cand not in sys.path:
        sys.path.insert(0, cand)


class RouterRecorder:
    """Records routed ids of every call with exactly ``batch`` rows on every ``*SparseMoeBlock.gate``."""

    def __init__(self, model, top_k: int, batch: int, block_suffix: str = "SparseMoeBlock"):
        self.top_k, self.batch = top_k, batch
        blocks = [m for _, m in model.named_modules() if type(m).__name__.endswith(block_suffix)]
        if not blocks:
            raise RuntimeError(f"no module whose class name ends in {block_suffix!r}")
        self.calls = [[] for _ in blocks]
        self.other = [0] * len(blocks)
        self.handles = [blk.gate.register_forward_hook(self._hook(i)) for i, blk in enumerate(blocks)]

    def _hook(self, li):
        def hook(_mod, _inp, out):
            if isinstance(out, tuple) and len(out) == 3 and not torch.is_floating_point(out[2]):
                ids = out[2]
            else:
                logits = out[0] if isinstance(out, tuple) else out
                ids = torch.topk(logits.reshape(-1, logits.shape[-1]), self.top_k, dim=-1).indices
            ids = ids.reshape(-1, ids.shape[-1])
            if ids.shape[0] == self.batch:
                self.calls[li].append(ids.to(torch.int16).cpu())
            else:
                self.other[li] += 1
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def tensor(self) -> torch.Tensor:
        steps = min(len(c) for c in self.calls)
        return torch.stack([torch.stack(c[:steps]) for c in self.calls], dim=1)      # [steps, layers, batch, k]


def run(model, ids: torch.Tensor, prefix: int, chunk: int, device) -> None:
    ids = ids.to(device)
    past = None
    with torch.no_grad():
        for s0 in range(0, prefix, chunk):
            out = model(input_ids=ids[:, s0:min(prefix, s0 + chunk)], past_key_values=past, use_cache=True)
            past = out.past_key_values
        for t in range(prefix, ids.shape[1]):
            out = model(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values


def summarise(eids: torch.Tensor) -> dict:
    S, L = eids.shape[:2]
    d = torch.tensor([[len(set(eids[s, li].flatten().tolist())) for li in range(L)] for s in range(S)], dtype=torch.float64)
    return {"steps": S, "layers": L, "mean_distinct": float(d.mean()), "layer_means": [round(float(x), 2) for x in d.mean(0)],
            "min_layer_mean": float(d.mean(0).min()), "max_layer_mean": float(d.mean(0).max())}


def main_box(a) -> int:
    import serve_stack
    pf = json.load(open(a.prompts))
    prompts = pf["prompts"]
    psha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    assert psha == pf["prompts_sha256"] and len(prompts) == a.batch
    if os.environ.get("E4B_FUSE_ROUTER_EPI", "0") == "1":
        raise SystemExit("E4B_FUSE_ROUTER_EPI=1 bypasses the router module (the fused epilogue never calls it): set it to 0")
    t0 = time.time()
    model, info = serve_stack.build_served_model(a.model, a.arena, a.calib, device=a.device)
    rec = RouterRecorder(model, top_k=info["top_k"], batch=a.batch)
    run(model, torch.tensor(prompts, dtype=torch.long), a.prefix, a.prefill_chunk, a.device)
    rec.remove()
    eids = rec.tensor()
    s = summarise(eids)
    out = {"eids": eids, "prompts_sha256": psha, "prefix": a.prefix, "prefill_chunk": a.prefill_chunk, "census": info,
           "prefill_or_other_calls_per_layer": rec.other[0], **s}
    torch.save(out, a.out)
    json.dump({k: v for k, v in out.items() if k != "eids"} | {"eids_shape": list(eids.shape), "wall_s": round(time.time() - t0, 1)},
              open(a.out + ".json", "w"), indent=1, default=str)
    print("P60EIDS " + json.dumps({"shape": list(eids.shape), "mean_distinct": round(s["mean_distinct"], 2),
                                   "layer_mean_range": [round(s["min_layer_mean"], 2), round(s["max_layer_mean"], 2)]}), flush=True)
    return 0


def self_test() -> int:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_experts=16, num_experts_per_tok=4,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=512,
                         max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).eval()
    B, T, prefix = 5, 20, 12
    rec = RouterRecorder(model, top_k=4, batch=B)
    ids = torch.randint(0, 512, (B, T), generator=torch.Generator().manual_seed(1))
    run(model, ids, prefix, 4, "cpu")
    rec.remove()
    e = rec.tensor()
    assert e.shape == (T - prefix, 3, B, 4), e.shape
    assert rec.other == [3, 3, 3], rec.other                               # 12 prefill tokens in chunks of 4 -> 3 calls each
    assert all(len(set(r.tolist())) == 4 for r in e.reshape(-1, 4)), "a row routed to a repeated expert"
    s = summarise(e)
    assert 1 <= s["mean_distinct"] <= min(16, B * 4), s
    print(json.dumps(s, indent=1)[:400])
    print("self-test OK: 3 layers x", T - prefix, "decode steps recorded; prefill chunks excluded by row count")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--arena")
    ap.add_argument("--calib")
    ap.add_argument("--prompts")
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--prefix", type=int, default=384)
    ap.add_argument("--prefill-chunk", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    for k in ("arena", "calib", "prompts", "out"):
        if getattr(a, k) is None:
            ap.error(f"--{k} is required")
    return main_box(a)


if __name__ == "__main__":
    sys.exit(main())
