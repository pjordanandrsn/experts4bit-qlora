# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Why FP8 KV survives and naive INT4 KV does not — measured, not asserted.

The G7 quality run shows E4M3 costing +0.18% perplexity while symmetric
INT4 on the SAME scale axis costs 993x. A number that extreme deserves a
mechanism, so this probe captures real K/V tensors from a live model and
measures the property that separates the two formats:

`amax / rms` per (token, head) row. A row whose largest element dwarfs its
typical element wastes a UNIFORM grid — INT4's step is amax/7, so an
element at 1 rms lands on step round(rms/(amax/7)), and when amax is 10x
rms that is step 1 of 7, i.e. under three effective levels for the values
that carry the signal. FP8 does not care: E4M3 is floating point, so its
precision is RELATIVE and an outlier costs the row nothing.

Reported per tensor kind, because K and V are known to differ — the
outlier phenomenon lives mostly in keys.
"""
import argparse
import json
from pathlib import Path

import torch


def main(model_id, n_layers_probe, out_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).cuda().eval()

    grabbed = {}

    class _Grab:
        """Capture what the attention layer hands the cache — the exact
        tensors a KV format has to store."""

        def __init__(self, idx):
            self.idx = idx

        def update(self, k, v, layer_idx, cache_kwargs=None):
            if layer_idx not in grabbed:
                grabbed[layer_idx] = (k.detach().float().cpu(),
                                      v.detach().float().cpu())
            return k, v

        def get_seq_length(self, layer_idx=0):
            return 0

        def get_mask_sizes(self, q_len, layer_idx):
            return q_len, 0

        def get_query_offset(self, layer_idx=0):
            return 0

        def get_max_length(self):
            return -1

        @property
        def is_sliding(self):
            return [False] * 64

        @property
        def is_compileable(self):
            return False

    text = ("The three tiers of a memory hierarchy, from fastest to "
            "slowest, are registers, caches, and main memory. " * 40)
    ids = tok(text, return_tensors="pt").input_ids[:, :512].cuda()
    with torch.no_grad():
        model(input_ids=ids, past_key_values=_Grab(0), use_cache=True)

    rows = []
    for layer in sorted(grabbed)[:n_layers_probe]:
        for kind, x in zip(("K", "V"), grabbed[layer]):
            amax = x.abs().amax(dim=-1)
            rms = x.pow(2).mean(dim=-1).sqrt().clamp_min(1e-12)
            ratio = (amax / rms).flatten()
            # effective INT4 levels for a typical (1-rms) element: the grid
            # step is amax/7, so this is how many steps the signal spans
            eff = (7.0 / ratio).flatten()
            rows.append({
                "layer": int(layer), "kind": kind,
                "amax_over_rms_p50": float(ratio.median()),
                "amax_over_rms_p99": float(ratio.quantile(0.99)),
                "amax_over_rms_max": float(ratio.max()),
                "int4_effective_levels_at_1rms_p50": float(eff.median()),
                "int4_effective_levels_at_1rms_p01": float(eff.quantile(0.01)),
            })

    k_rows = [r for r in rows if r["kind"] == "K"]
    v_rows = [r for r in rows if r["kind"] == "V"]
    rep = {
        "model": model_id,
        "per_layer": rows,
        "summary": {
            "K_amax_over_rms_p50": sum(r["amax_over_rms_p50"]
                                       for r in k_rows) / len(k_rows),
            "V_amax_over_rms_p50": sum(r["amax_over_rms_p50"]
                                       for r in v_rows) / len(v_rows),
            "K_int4_levels_p50": sum(r["int4_effective_levels_at_1rms_p50"]
                                     for r in k_rows) / len(k_rows),
            "V_int4_levels_p50": sum(r["int4_effective_levels_at_1rms_p50"]
                                     for r in v_rows) / len(v_rows),
        },
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    name = model_id.split("/")[-1]
    (Path(out_dir) / f"g7_outliers_{name}.json").write_text(
        json.dumps(rep, indent=2))
    s = rep["summary"]
    print(f"K amax/rms p50 = {s['K_amax_over_rms_p50']:.2f}  "
          f"-> int4 levels at 1 rms = {s['K_int4_levels_p50']:.2f}")
    print(f"V amax/rms p50 = {s['V_amax_over_rms_p50']:.2f}  "
          f"-> int4 levels at 1 rms = {s['V_int4_levels_p50']:.2f}")
    print("G7_OUTLIERS " + json.dumps(rep["summary"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    main(a.model, a.layers, a.out)
