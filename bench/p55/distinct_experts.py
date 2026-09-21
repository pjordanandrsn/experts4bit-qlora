#!/usr/bin/env python3
"""The distinct-expert count per layer per decode step -- the one soft number in #564's expert-tier roofline.

P54 registered this as an untimed ``--amort on --series-out`` arm and the harness refused it: ``step_decomp``'s
B>1 stage asserts ``--amort off`` ("amort-armed runs keep the baseline dispatch path (not capturable)"), so
the count could not come from the census harness at B=16 at all. This is the replacement (P54 amendment 1):
a forward hook on every MoE block's router that recomputes the top-k from the router logits and records, per
layer and per decode step, how many DISTINCT experts the batch touched. It is model-generic (any block whose
router is an ``nn.Linear`` named ``gate`` and whose config carries ``num_experts_per_tok``), it never touches
the serving kernels, and its step time is not quoted -- it is an accounting run.

Run under the serving stack exactly as a census arm would (the P42 hook applies the int4 lanes at load), with
``E4B_FUSE_ROUTER_EPI=0``: the fused router epilogue computes its logits by ``F.linear`` on the gate's weight
and never calls the gate module, so the hook would not fire under it. Routing is a property of the weights and
the prompts, not of which kernel computes it, so the count read here is the count the fused stack routes too.

    python distinct_experts.py --model Qwen/Qwen3-30B-A3B --revision <sha> --batch 16 --prompt-len 512 \
        --steps 32 --out distinct_experts.json                # on the box, after the harness's own load path
    python distinct_experts.py --self-test                    # CPU: a tiny random Qwen3MoE, the hook end to end

Output: ``{"batch", "top_k", "num_experts", "layers", "steps", "per_layer": [{"mean_distinct", "min", "max",
"series": [...]}], "mean_distinct_over_layers", "uniform_random_expectation"}``. The expectation for a uniform
router is ``E * (1 - (1 - k/E)**B)`` for B rows drawing k of E experts each -- 80.9 for E=128, k=8, B=16 --
and is printed beside the measurement so the skew is visible.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch
from torch import nn


class DistinctExpertCounter:
    """Hooks every ``gate`` Linear under a module whose class name ends in ``SparseMoeBlock`` (transformers'
    Qwen3-MoE naming; ``--block-suffix`` overrides) and records distinct top-k experts per call."""

    def __init__(self, model: nn.Module, top_k: int, block_suffix: str = "SparseMoeBlock"):
        self.top_k = top_k
        self.series: list[list[int]] = []          # per layer: list of distinct counts, one per forward call
        self.handles = []
        blocks = [(n, m) for n, m in model.named_modules() if type(m).__name__.endswith(block_suffix)]
        if not blocks:
            raise RuntimeError(f"no module whose class name ends in {block_suffix!r}: cannot place the router hook")
        for li, (name, blk) in enumerate(blocks):
            gate = getattr(blk, "gate", None)
            if not isinstance(gate, nn.Module):
                raise RuntimeError(f"{name}: expected the router module at .gate, found {type(gate).__name__}")
            self.series.append([])
            self.handles.append(gate.register_forward_hook(self._make_hook(li)))
        self.layers = len(blocks)

    def _make_hook(self, li: int):
        def hook(_mod, _inp, out):
            # transformers >= 5.x: the router is its own module returning (logits, scores, indices) -- take the
            # indices it actually routed on. Older trees: a bare nn.Linear `gate` returning logits -- the
            # selection is argmax-invariant to the softmax, so top-k on the logits IS that router's top-k.
            if isinstance(out, tuple) and len(out) == 3 and not torch.is_floating_point(out[2]):
                ids = out[2]
            else:
                logits = out[0] if isinstance(out, tuple) else out
                ids = torch.topk(logits.reshape(-1, logits.shape[-1]), self.top_k, dim=-1).indices
            self.series[li].append(int(torch.unique(ids).numel()))
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def report(self, batch: int, num_experts: int, skip_first: int = 0) -> dict:
        per_layer = []
        for s in self.series:
            s = s[skip_first:]
            per_layer.append({"steps": len(s), "mean_distinct": statistics.mean(s) if s else None,
                              "min": min(s) if s else None, "max": max(s) if s else None, "series": s})
        means = [x["mean_distinct"] for x in per_layer if x["mean_distinct"] is not None]
        e, k, b = num_experts, self.top_k, batch
        return {"batch": batch, "top_k": k, "num_experts": e, "layers": self.layers,
                "steps": min((x["steps"] for x in per_layer), default=0),
                "mean_distinct_over_layers": statistics.mean(means) if means else None,
                "min_layer_mean": min(means) if means else None, "max_layer_mean": max(means) if means else None,
                "uniform_random_expectation": e * (1 - (1 - k / e) ** b),
                "per_layer": per_layer}


def _self_test() -> int:
    """CPU: a tiny random Qwen3MoE; the hook must see every layer on every step and count sensibly."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_experts=16,
                         num_experts_per_tok=4, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, vocab_size=512, max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).eval()
    counter = DistinctExpertCounter(model, top_k=cfg.num_experts_per_tok)
    B, T, steps = 5, 7, 4
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        for _ in range(steps):
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
    counter.remove()
    rep = counter.report(batch=B, num_experts=cfg.num_experts, skip_first=1)      # drop the prefill call
    assert rep["layers"] == 3 and rep["steps"] == steps, rep
    for pl in rep["per_layer"]:
        assert all(1 <= d <= min(cfg.num_experts, B * cfg.num_experts_per_tok) for d in pl["series"]), pl
    print(json.dumps({k: v for k, v in rep.items() if k != "per_layer"}, indent=1))
    print("self-test OK: 3 layers x", steps, "decode steps hooked; uniform expectation",
          round(rep["uniform_random_expectation"], 2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--revision")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--out", default="distinct_experts.json")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    # On the box: load through the SAME path the census arms use (the P42 hook applies the int4 lanes when the
    # hybrid tier is enabled), so the model, revision and store are the census's. The load is the harness's
    # business; this script only places the hook and counts. It refuses rather than guessing a loader.
    raise SystemExit("box mode is wired by the lane's runner (p55_run.sh) around the harness's own load path; "
                     "run --self-test here")


if __name__ == "__main__":
    sys.exit(main())
