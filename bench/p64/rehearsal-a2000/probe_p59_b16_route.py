#!/usr/bin/env python3
"""probe_p59_b16_route.py -- which expert route does P59's B=16 scorer run? (P64 pre-registration, "A correction").

Builds P59's `int4` arm (RTN int4 experts + RTN int4 attention + folds; env set by the caller, as P59's runner sets
it) through P44's `serve_stack.build_served_model`, installs kl_a16.py's phase counters, and scores 16 rows at once
through P59's `kl_b16.batched_teacher_forced` exactly as P59 did (prefill in 8-token-per-row chunks, then one token
per forward at B = 16). It counts the expert dispatch's `quant_x_rows` / `dequant_int4_ref` calls and the
Int4Linear quantises in the DECODE phase:

  * as P59 ran it (`hot_residency.DEVICE_GROUPING` off -- nothing in serve_stack / kl_b16 / the hook sets it);
  * with `DEVICE_GROUPING` on, the flag `step_decomp.py`'s timed B = 16 stages set (the route the licensed B = 16
    speed rows run).

A correctness probe on the A2000, not a timing and not a KL: it says which arithmetic ran.
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("P64_STAGE", os.path.join(HERE, "..")))       # the staged run dir (flat) or bench/p64

import kl_a16  # noqa: E402
import serve_stack  # noqa: E402
from kl_b16 import batched_teacher_forced  # noqa: E402


def main():
    model_id, arena, calib, prompts, out = sys.argv[1:6]
    decode_steps = int(os.environ.get("PROBE_DECODE_STEPS", "8"))
    rec = json.load(open(prompts))
    ids = torch.tensor(rec["prompts"], dtype=torch.long)
    B = ids.shape[0]
    prefix = 384
    ids = ids[:, :prefix + decode_steps]
    t0 = time.time()
    model, info = serve_stack.build_served_model(model_id, arena, calib, device="cuda")
    census = kl_a16.Census()
    kl_a16.install_counters(model, census)
    from experts4bit_qlora.engines import hot_residency as hr
    res = {"model": model_id, "B": B, "prefix": prefix, "decode_steps": decode_steps, "prefill_chunk_per_row": 8,
           "build": {k: info.get(k) for k in ("moe_layers", "top_k", "int4_expert_layers", "int4_attn_projections",
                                              "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")},
           "env": {k: v for k, v in os.environ.items() if k.startswith("E4B_")}, "build_s": round(time.time() - t0, 1), "routes": {}}
    for label, dg in (("as_P59_ran_it (DEVICE_GROUPING off)", False), ("timed_B16_route (DEVICE_GROUPING on)", True)):
        hr.DEVICE_GROUPING[0] = dg
        census.reset()
        t1 = time.time()
        batched_teacher_forced(model, ids, prefix, "cuda", prefill_chunk=8)
        c = census.snapshot()
        res["routes"][label] = {"decode_counts": {k: v for k, v in c.items() if k.endswith("_decode")},
                                "prefill_counts": {k: v for k, v in c.items() if k.endswith("_prefill")}, "wall_s": round(time.time() - t1, 1)}
        hr.DEVICE_GROUPING[0] = False
    L, k = info["moe_layers"], info["top_k"]
    off = res["routes"]["as_P59_ran_it (DEVICE_GROUPING off)"]["decode_counts"]
    on = res["routes"]["timed_B16_route (DEVICE_GROUPING on)"]["decode_counts"]
    res["reading"] = {
        "P59_scorer_experts_int8_quantises_at_decode": off.get("expert_quant_decode", 0),
        "P59_scorer_experts_dequants_at_decode": off.get("expert_dequant_decode", 0),
        "P59_scorer_attention_int8_quantises_at_decode": off.get("attn_quant_decode", 0),
        "timed_route_experts_int8_quantises_at_decode": on.get("expert_quant_decode", 0),
        "timed_route_expected_expert_quantises": 2 * L * decode_steps,
        "note": f"L={L} MoE layers, top_k={k}; one decode forward = {B} rows"}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["reading"], indent=1))


if __name__ == "__main__":
    main()
