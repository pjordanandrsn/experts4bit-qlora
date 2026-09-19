#!/usr/bin/env python3
"""bench/p44/expert_residuals.py -- P44-a's per-expert reconstruction-error census (Granite, Mixtral).

For every (layer, expert, role) the relative activation-weighted reconstruction error of the packed int4 rows
against the bf16 expert on the calibration activations,

    rel_act = || W_q X - W X ||_F / || W X ||_F  =  sqrt( tr(D H D^T) / tr(W H W^T) ),   D = W_q - W,  H = 2 X X^T,

computed from the SAME Hessians the calibrated recipe packs with (`calibrate_expert_hessians`, the fused forward's
tap, `H` the running-mean `2 X X^T` over the rows each expert actually saw -- the factor cancels in the ratio), for
BOTH packers the recipe can choose between: round-to-nearest (`int4_pack_ref.pack_int4_b32`, what the RTN arms serve)
and GPTQ (`gptq_pack.gptq_pack_int4_b32`, what the calibrated arms serve where an expert saw >= min_rows rows; the
recipe's own choice is recorded as `recipe_method`). A weight-only `rel_frob = ||D||_F / ||W||_F` sits beside it so an
expert the calibration text never routed to (no Hessian) still has a row.

This is DATA for a per-expert NF4 fallback (a third method in the recorded gptq/rtn assignment, #530); no fallback is
built or gated here and no verdict is read here. `p44_reduce.py` reads P3 (tail concentration) from these rows.

Memory: Hessians are calibrated in passes sized by E4B_INT4_HESSIAN_BUDGET_GB (default 24) -- Mixtral-8x7B's eight
14336^2 down Hessians are ~7 GB a layer, so it runs one layer per pass and a pass is a full run over the batches.
The bf16 expert stacks are read one layer at a time through the loader's own plan reader, so the census and the
enabler can never disagree on which bytes an expert is.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from serve_stack import MODELS, apply_env, build_served_model, calib_batches  # noqa: E402


def _trace_quad(D: torch.Tensor, H: torch.Tensor) -> float:
    """tr(D H D^T) in fp32 on the tensors' device."""
    return float(((D @ H) * D).sum())


def census_row(W: torch.Tensor, H, rows: int, *, min_rows: int, damp: float, dev: str) -> dict:
    from gptq_pack import gptq_pack_int4_b32
    from int4_pack_ref import dequant_int4_ref, pack_int4_b32
    W = W.to(dev, torch.float32)
    N, K = W.shape
    out = {"N": N, "K": K, "rows": int(rows), "recipe_method": "gptq" if rows >= min_rows else "rtn"}
    wn = float(W.norm())
    p, s = pack_int4_b32(W.cpu())                                   # the enabler packs RTN on the CPU copy
    Wr = dequant_int4_ref(p, s, N, K).to(dev, torch.float32)
    Dr = Wr - W
    out["rtn"] = {"rel_frob": float(Dr.norm()) / wn if wn > 0 else None, "rel_act": None, "sq_err_act": None}
    out["gptq"] = None
    if H is not None:
        H = H.to(dev, torch.float32)
        denom = _trace_quad(W, H)
        out["denom_act"] = denom
        num_r = _trace_quad(Dr, H)
        out["rtn"]["sq_err_act"] = num_r
        out["rtn"]["rel_act"] = math.sqrt(max(num_r, 0.0) / denom) if denom > 0 else None
        if rows >= min_rows:
            pg, sg = gptq_pack_int4_b32(W, H, damp=damp)            # E4B_INT4_GPTQ_DEVICE=cuda: the GPU solve
            Wg = dequant_int4_ref(pg.cpu(), sg.cpu(), N, K).to(dev, torch.float32)
            Dg = Wg - W
            num_g = _trace_quad(Dg, H)
            out["gptq"] = {"rel_frob": float(Dg.norm()) / wn if wn > 0 else None,
                           "rel_act": math.sqrt(max(num_g, 0.0) / denom) if denom > 0 else None,
                           "sq_err_act": num_g}
            del pg, sg, Wg, Dg
        del H
    del W, Wr, Dr, p, s
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="P44-a: per-expert reconstruction-error census")
    ap.add_argument("--family", required=True, choices=["granite", "mixtral", "olmoe", "gemma4", "gptoss"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--source", default="c4", choices=["c4", "wikitext"])
    ap.add_argument("--nseq", type=int, default=32, help="calibration windows of 512 tokens (hook default 32 = 16k)")
    ap.add_argument("--min-rows", type=int, default=32, help="the recipe's GPTQ threshold (enable_serve_experts_int4)")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--layers", default="", help="debug: comma list of layer indices")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model_id, revision = MODELS[a.family]
    model_id = a.model or model_id
    revision = a.revision or revision
    dev = "cuda"
    os.environ.setdefault("E4B_INT4_GPTQ_DEVICE", "cuda")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from experts4bit_qlora.arch.moe_load import make_plan_reader, read_fused_expert_layer
    from experts4bit_qlora.engines.int4_experts import (_expert_layers, calibrate_expert_hessians,
                                                        safetensors_reader)

    apply_env({"E4B_MODEL_ID": model_id})                            # NO lanes: the census needs the NF4 tap + the bf16 bytes
    t0 = time.time()
    model, info = build_served_model(model_id, a.arena, a.calib, device=dev)
    if info["int4_expert_layers"] or info["int4_attn_projections"]:
        raise RuntimeError("the census model carries int4 stores -- a lane env leaked in; refusing")
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    src = snapshot_download(model_id, revision=revision, allow_patterns=["*.json", "*.safetensors"])
    plan, layer_ws = _expert_layers(model, src)
    if not plan.experts:
        raise RuntimeError(f"{a.family}: the plan has no per-expert projections (prefused stacks) -- census not defined here")
    keys, read_tensor = safetensors_reader(src)
    read = make_plan_reader(plan, read_tensor, torch.float32)
    batches = calib_batches(tok, a.nseq, a.source)
    order = [layer for layer, _w in layer_ws]
    if a.layers:
        keep = {int(x) for x in a.layers.split(",")}
        order = [x for x in order if x in keep]
    cfg = model.config
    tcfg = getattr(cfg, "text_config", None) or cfg
    hid = getattr(tcfg, "hidden_size")
    inter = getattr(tcfg, "moe_intermediate_size", None) or getattr(tcfg, "intermediate_size")
    E = info["experts"]
    per_layer = int(E) * (int(hid) ** 2 + int(inter) ** 2) * 4
    budget = int(float(os.environ.get("E4B_INT4_HESSIAN_BUDGET_GB", "24")) * (1 << 30))
    lpp = max(1, budget // per_layer)
    rep = {"lane": "P44-a census", "family": a.family, "model": model_id, "revision": revision,
           "calibration": {"source": a.source, "nseq": a.nseq, "seq_len": 512, "n_batches": len(batches),
                           "tokens": a.nseq * 512},
           "min_rows": a.min_rows, "damp": a.damp, "gptq_device": os.environ["E4B_INT4_GPTQ_DEVICE"],
           "layers_total": len(layer_ws), "layers_censused": order, "experts": E, "hidden": hid, "inter": inter,
           "hessian_bytes_per_layer": per_layer, "layers_per_pass": lpp, "engagement": info,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": []}

    def flush():
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)

    flush()
    for i in range(0, len(order), lpp):
        chunk = order[i:i + lpp]
        t1 = time.time()
        hs = calibrate_expert_hessians(model, src, batches, only_layers=chunk, layers_per_pass=len(chunk),
                                       hessian_device="cpu")
        print(f"  hessians for layers {chunk[0]}..{chunk[-1]} in {time.time() - t1:.0f}s", flush=True)
        for layer in chunk:
            first, down = read_fused_expert_layer(plan, layer, read, device="cpu", dtype=torch.float32)
            hl = hs.get(layer, {})
            for e in range(first.shape[0]):
                H_gu, H_dn, rows = hl.get(e, (None, None, 0))
                for role, W, H in (("gu", first[e], H_gu), ("dn", down[e], H_dn)):
                    row = census_row(W, H, rows, min_rows=a.min_rows, damp=a.damp, dev=dev)
                    row.update({"layer": int(layer), "expert": int(e), "role": role})
                    rep["rows"].append(row)
            del first, down
            n_routed = sum(1 for e in range(E) if e in hl)
            print(f"  layer {layer}: {n_routed}/{E} experts routed; {len(rep['rows'])} rows so far", flush=True)
            flush()
        del hs
        gc.collect()
        torch.cuda.empty_cache()
    rep["wall_s"] = round(time.time() - t0, 1)
    flush()
    print(f"census -> {a.out}: {len(rep['rows'])} rows over {len(order)} layers in {rep['wall_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
