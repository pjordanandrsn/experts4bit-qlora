#!/usr/bin/env python3
"""Lane P98's arena bake (bench/p98/PREREG-p98.md; e4b#564). ``bench/p39/k8_bake.py``'s steps, with the checkpoint
pinned to a revision: quantize the experts to NF4 through e4b's loader, write a per-expert NF4 snapshot in the layout
``nvme_arena`` expects, and bake the arena that ``serve_paged.build_engine`` serves from.

``--offload`` (REHEARSAL only: the A2000's 12 GB cannot hold Qwen3.6's experts) loads with the experts in host RAM.
An offloaded module holds 0-element placeholders, so the tensors are read from the offload handle's pinned host copies
(the first rehearsal read the placeholders and failed: "shape [256, 1024, 1024] is invalid for input of size 0").
The bytes written are the same NF4 tensors either way.

    python p98_bake.py --model ID --revision SHA --work DIR [--offload]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--offload", action="store_true", help="REHEARSAL: experts in host RAM while baking")
    a = ap.parse_args()
    out, snap, arena = a.work + "/bake.json", a.work + "/nf4snap", a.work + "/nf4.arena"
    rec = {"step": "start", "model": a.model, "revision": a.revision, "offload": a.offload}

    def dump(status, **kw):
        rec["status"] = status
        rec.update(kw)
        json.dump(rec, open(out, "w"), indent=1)
        print("BAKE", status, flush=True)

    try:
        import torch
        from nvme_arena import bake_expert_tensors
        from safetensors.torch import save_file

        from experts4bit_qlora.engines.hot_residency import target_modules
        from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS
        from experts4bit_qlora.loader import load_moe_4bit_streaming

        os.makedirs(snap, exist_ok=True)
        t0 = time.time()
        model, cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, 8, 16, quant_type="nf4",
                                             revision=a.revision, offload=a.offload)
        rec["load_s"] = round(time.time() - t0, 1)
        rec["loaded_commit"] = getattr(cfg, "_commit_hash", None)
        mods = target_modules(model)
        n_layers, n_exp = len(mods), mods[0].num_experts
        # an offloaded layer's module holds 0-element placeholders; its NF4 tensors live in the offload handle's
        # pinned host copies ("homes"), keyed by the same names
        from experts4bit_qlora.engines.offload import offload_handles
        homes = {id(h.base): h.home for h in offload_handles(model)}
        if a.offload and len(homes) != n_layers:
            raise RuntimeError(f"--offload: {len(homes)} offload handles for {n_layers} expert layers")
        rec["read_from"] = "offload homes" if homes else "the modules"

        def tensor(mod, name):
            home = homes.get(id(mod))
            return home[name] if home is not None else getattr(mod, name)
        rec["layers"], rec["experts"] = n_layers, n_exp
        print(f"layers={n_layers} experts={n_exp}", flush=True)
        weight_map, total = {}, 0
        t1 = time.time()
        for li, mod in enumerate(mods):
            n1, k1 = mod._gate_up_shape
            n2, k2 = mod._down_shape
            payload = {
                "nf4.gate_up_blocks": tensor(mod, "gate_up_proj").view(n_exp, n1, k1 // 2),
                "nf4.gate_up_absmax": tensor(mod, "gate_up_absmax").view(n_exp, n1, k1 // 64).float(),
                "nf4.down_blocks": tensor(mod, "down_proj").view(n_exp, n2, k2 // 2),
                "nf4.down_absmax": tensor(mod, "down_absmax").view(n_exp, n2, k2 // 64).float(),
            }
            shard = {}
            for kind, stack in payload.items():
                for e in range(n_exp):
                    nm = f"model.layers.{li}.mlp.experts.{e}.{kind}"
                    shard[nm] = stack[e].contiguous().cpu().clone()
                    weight_map[nm] = f"model-{li:05d}.safetensors"
            save_file(shard, os.path.join(snap, f"model-{li:05d}.safetensors"))
            total += sum(v.numel() * v.element_size() for v in shard.values())
            del shard, payload
            gc.collect()
            if li % 8 == 0:
                print(f"  shard {li}/{n_layers}  {total / 2**30:.1f} GiB  {time.time() - t1:.0f}s", flush=True)
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
                  open(os.path.join(snap, "model.safetensors.index.json"), "w"))
        rec["snapshot_gib"] = round(total / 2**30, 2)
        rec["snapshot_s"] = round(time.time() - t1, 1)
        del model, mods
        gc.collect()
        torch.cuda.empty_cache()
        t2 = time.time()
        info = bake_expert_tensors(snap, arena, name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
                                   kinds=tuple(NF4_SEGMENTS.values()), align=4096)
        rec["bake_s"] = round(time.time() - t2, 1)
        rec["arena"] = arena
        rec["bake_info"] = {k: (v if isinstance(v, (int, float, str)) else str(v)[:200]) for k, v in (info or {}).items()}
        dump("OK")
        return 0
    except Exception as e:  # noqa: BLE001  (the bake's failure is the record)
        dump("FAILED", err=repr(e)[:1200], tb=traceback.format_exc()[-2000:])
        return 12


if __name__ == "__main__":
    raise SystemExit(main())
