"""Quantize Qwen3-30B-A3B to nf4 via e4b's loader, emit a per-expert nf4
snapshot in the layout nvme_arena expects, then bake the arena.

bake_expert_tensors is a RELOCATION bake -- it moves pre-quantized tensors.
The HF checkpoint ships bf16 experts, so the nf4 snapshot has to be produced
first. That is what this does.
"""
import json, os, sys, time, gc, traceback
import torch
from safetensors.torch import save_file

import os as _os
WORK = _os.environ["K8_WORK"]
MODEL = _os.environ["K8_MODEL"]
OUT = WORK + "/bake.json"
SNAP = WORK + "/nf4snap"
ARENA = WORK + "/nf4.arena"
R = {"step": "start"}
def dump(status, **kw):
    R["status"] = status; R.update(kw); json.dump(R, open(OUT, "w"), indent=1)
    print("BAKE", status, flush=True)

try:
    os.makedirs(SNAP, exist_ok=True)
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.engines.hot_residency import target_modules
    from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS
    from nvme_arena import bake_expert_tensors

    t0 = time.time()
    model, _ = load_moe_4bit_streaming(MODEL, "cuda",
                                       torch.bfloat16, 8, 16, quant_type="nf4")
    R["load_s"] = round(time.time() - t0, 1)
    mods = target_modules(model)
    L, E = len(mods), mods[0].num_experts
    R["layers"], R["experts"] = L, E
    print(f"layers={L} experts={E}", flush=True)

    weight_map, total = {}, 0
    t1 = time.time()
    for li, mod in enumerate(mods):
        n1, k1 = mod._gate_up_shape
        n2, k2 = mod._down_shape
        payload = {
            "nf4.gate_up_blocks":  mod.gate_up_proj.view(E, n1, k1 // 2),
            "nf4.gate_up_absmax":  mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
            "nf4.down_blocks":     mod.down_proj.view(E, n2, k2 // 2),
            "nf4.down_absmax":     mod.down_absmax.view(E, n2, k2 // 64).float(),
        }
        shard = {}
        for kind, stack in payload.items():
            for e in range(E):
                nm = f"model.layers.{li}.mlp.experts.{e}.{kind}"
                shard[nm] = stack[e].contiguous().cpu().clone()
                weight_map[nm] = f"model-{li:05d}.safetensors"
        fn = os.path.join(SNAP, f"model-{li:05d}.safetensors")
        save_file(shard, fn)
        total += sum(v.numel() * v.element_size() for v in shard.values())
        del shard, payload; gc.collect()
        if li % 8 == 0:
            print(f"  shard {li}/{L}  {total/2**30:.1f} GiB  {time.time()-t1:.0f}s", flush=True)
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(SNAP, "model.safetensors.index.json"), "w"))
    R["snapshot_gib"] = round(total / 2**30, 2)
    R["snapshot_s"] = round(time.time() - t1, 1)
    print(f"snapshot done {R['snapshot_gib']} GiB in {R['snapshot_s']}s", flush=True)

    del model, mods; gc.collect(); torch.cuda.empty_cache()

    t2 = time.time()
    info = bake_expert_tensors(SNAP, ARENA,
                               name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
                               kinds=tuple(NF4_SEGMENTS.values()), align=4096)
    R["bake_s"] = round(time.time() - t2, 1)
    R["arena"] = ARENA
    R["bake_info"] = {k: (v if isinstance(v, (int, float, str)) else str(v)[:200])
                      for k, v in (info or {}).items()}
    dump("OK")
except Exception as e:
    dump("FAILED", err=repr(e)[:1200], tb=traceback.format_exc()[-2000:])
