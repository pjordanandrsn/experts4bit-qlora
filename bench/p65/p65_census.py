#!/usr/bin/env python3
"""bench/p65/p65_census.py -- P65's census: P44-a's per-expert rows, plus activation entropy, on two calibration texts
in two disjoint halves each (bench/p65/P65-PREREG.md).

What is REUSED, byte-for-byte, and not re-implemented:
  * the served model (NF4 through the arena, hybrid tier, all-VRAM): ``bench/p44/serve_stack.build_served_model``;
  * the wikitext calibration windows: ``serve_stack.calib_batches(tok, nseq, "wikitext")`` (wikitext-2 TRAIN);
  * the Hessian tap and its passes: ``experts4bit_qlora.engines.int4_experts.calibrate_expert_hessians``, now with
    ``activation_means=`` so the same pass also returns each expert's mean input rows;
  * every census row's error fields: ``bench/p44/expert_residuals.census_row``, called unchanged. It is called with a
    ``min_rows`` above any possible row count, so it takes its RTN branch only: ``rel_act`` is the int4-b32
    round-to-nearest error, whose bytes do not depend on the calibration text (only ``H`` does), and no GPTQ solve runs.

What is NEW here:
  * the ``entropy`` field of each row (``expert_entropy.entropy_fields``);
  * the c4val1 text: C4 validation shard 00001, first 2000 documents -- the exact text K8 scores as ``c4val1``
    (``bench/hybrid-g9/step_decomp.py:_k8_window``), windowed by ``calib_batches``' own rule (``windows``; a test holds
    the two equal on the same text);
  * each text is split into two disjoint, interleaved halves (batches 0, 2, 4, ... and 1, 3, 5, ...), each censused
    on its own; the ``full`` row of a text is the exact combination of its halves (moments and trace terms are means
    over rows, so the union's are the row-weighted means). The halves are what let the reducer tell a ranking that a
    new domain moves from one that re-sampling the SAME text already moves.

Rows: one per (text, half in {0, 1, "full"}, layer, expert, role in {gu, dn}). Routing frequency is each row's ``rows``
(token-slots the tap routed to the expert), the quantity ``hot_sets_from_profile`` ranks as ``tokens_routed``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P44 = os.path.join(os.path.dirname(HERE), "p44")
for _p in (HERE, P44):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from expert_entropy import combine_moments, direct_entropy_ratio, entropy_fields, entropy_ratio  # noqa: E402
from expert_residuals import census_row  # noqa: E402
from serve_stack import MODELS, apply_env, build_served_model, calib_batches  # noqa: E402

NO_GPTQ = 1 << 62                      # census_row's min_rows: no expert reaches it, so every row is RTN
FAMILIES = ("granite", "olmoe", "mixtral")
TEXTS = ("wikitext", "c4val1")
C4VAL1 = {"repo": "allenai/c4", "data_files": "en/c4-validation.00001-of-00008.json.gz", "docs": 2000}


# ------------------------------------------------------------------------------------------- texts
def windows(ids: torch.Tensor, n_seq: int, seq_len: int = 512, bsz: int = 4):
    """``calib_batches``' windowing rule, for a token-id vector: ``n_seq`` windows of ``seq_len`` at an even stride,
    stacked ``bsz`` at a time. Kept literally equal to it (tests/test_p65_census.py)."""
    step = max(1, (ids.numel() - seq_len) // n_seq)
    rows = [ids[i * step:i * step + seq_len] for i in range(n_seq)]
    return [torch.stack(rows[i:i + bsz]) for i in range(0, n_seq, bsz)]


def c4val1_text() -> str:
    """K8's c4val1 corpus, as ``step_decomp._k8_window`` builds it."""
    from datasets import load_dataset
    ds = load_dataset(C4VAL1["repo"], data_files={"v": C4VAL1["data_files"]}, split="v")
    return "\n\n".join(ds["text"][:C4VAL1["docs"]])


def c4val1_batches(tok, n_seq: int, seq_len: int = 512, bsz: int = 4):
    text = c4val1_text()[:6_000_000]                         # calib_batches' cap (the c4val1 text is under it)
    ids = tok(text, return_tensors="pt").input_ids[0]
    return windows(ids, n_seq, seq_len, bsz)


def text_batches(tok, text: str, n_seq: int):
    if text == "wikitext":
        return calib_batches(tok, n_seq, "wikitext")
    if text == "c4val1":
        return c4val1_batches(tok, n_seq)
    raise ValueError(f"unknown text {text!r}")


def split_halves(batches):
    """Two disjoint halves that both span the text: even-indexed and odd-indexed batches."""
    return list(batches[0::2]), list(batches[1::2])


def batches_digest(batches) -> dict:
    h = hashlib.sha256()
    n_tok = 0
    for b in batches:
        t = b.to(torch.int64).contiguous()
        h.update(t.numpy().tobytes())
        n_tok += t.numel()
    return {"batches": len(batches), "windows": sum(int(b.shape[0]) for b in batches), "tokens": n_tok,
            "ids_sha256": h.hexdigest()}


# ------------------------------------------------------------------------------------------- rows
def combine_rows(r0: dict | None, r1: dict | None) -> dict:
    """The ``full`` row of a text from its two halves. ``rel_frob`` is text-independent (copied); the RTN trace terms
    ``sq_err_act = tr(D H D^T)`` and ``denom_act = tr(W H W^T)`` are linear in ``H``, a mean over rows, so the union's are
    the row-weighted means and ``rel_act = sqrt(num / den)`` of those. The entropy field is set by the caller from the
    combined moment vectors (a ratio does not combine; its moments do)."""
    parts = [r for r in (r0, r1) if r is not None]
    base = parts[0]
    out = {k: base[k] for k in ("N", "K", "layer", "expert", "role", "text") if k in base}
    n = [int(r.get("rows", 0)) for r in parts]
    N = sum(n)
    out.update({"half": "full", "rows": N, "recipe_method": "rtn", "gptq": None,
                "derived": "row-weighted combination of halves 0 and 1 (exact: every term is a mean over rows)"})
    rtn = {"rel_frob": base["rtn"]["rel_frob"], "rel_act": None, "sq_err_act": None}
    have = [(k, r) for k, r in zip(n, parts) if k > 0 and r["rtn"].get("sq_err_act") is not None
            and r.get("denom_act") is not None]
    if have:
        Nh = sum(k for k, _ in have)
        num = sum(r["rtn"]["sq_err_act"] * (k / Nh) for k, r in have)
        den = sum(r["denom_act"] * (k / Nh) for k, r in have)
        rtn["sq_err_act"] = num
        rtn["rel_act"] = math.sqrt(max(num, 0.0) / den) if den > 0 else None
        out["denom_act"] = den
    out["rtn"] = rtn
    out["device"] = ",".join(sorted({str(r.get("device")) for r in parts}))
    return out


class _RowCapture:
    """A calibration tap that KEEPS the down-input rows of a few experts of one layer (selfcheck only)."""

    def __init__(self, gu_p_id: int, experts, cap: int = 4096):
        self.gu_p_id, self.experts, self.cap = gu_p_id, set(experts), cap
        self.h = {e: [] for e in experts}

    def __call__(self, gu_p, sorted_ids, x_sorted, h):
        if id(gu_p) != self.gu_p_id:
            return
        ids = sorted_ids.to("cpu")
        for e in self.experts:
            sel = (ids == e).nonzero().flatten().to(h.device)
            if sel.numel():
                self.h[e].append(h.index_select(0, sel).to("cpu", torch.float64))


def main() -> int:
    ap = argparse.ArgumentParser(description="P65: per-expert census + activation entropy on two texts x two halves")
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--model", default=None, help="override the pinned id (a local path for the rehearsal)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True, help="the placement calibration (bench/p39/calib.json)")
    ap.add_argument("--nseq", type=int, default=64, help="windows of 512 tokens PER TEXT; each half gets nseq/2")
    ap.add_argument("--texts", default=",".join(TEXTS))
    ap.add_argument("--layers", default="", help="comma list of layer indices (default: every expert layer, in order)")
    ap.add_argument("--selfcheck", type=int, default=1, help="1: determinism + moment-vs-direct entropy on real rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.nseq % 8:
        raise SystemExit("--nseq must be a multiple of 8 (two halves of whole 4-window batches)")
    model_id, revision = MODELS[a.family]
    if a.model:
        model_id, revision = a.model, a.revision
    texts = [t for t in a.texts.split(",") if t]
    dev = "cuda"

    from transformers import AutoTokenizer

    from experts4bit_qlora.arch.moe_load import make_plan_reader, read_fused_expert_layer
    from experts4bit_qlora.engines import hot_residency as _hr
    from experts4bit_qlora.engines.int4_experts import (_expert_layers, _prefused_layers, calibrate_expert_hessians,
                                                        safetensors_reader)

    apply_env({"E4B_MODEL_ID": model_id})               # NO lanes: the census needs the NF4 tap and the bf16 bytes
    t0 = time.time()
    model, info = build_served_model(model_id, a.arena, a.calib, device=dev)
    if info["int4_expert_layers"] or info["int4_attn_projections"]:
        raise RuntimeError("the census model carries int4 stores -- a lane env leaked in; refusing")
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if os.path.isdir(model_id):
        src = model_id
    else:
        from huggingface_hub import snapshot_download
        src = snapshot_download(model_id, revision=revision, allow_patterns=["*.json", "*.safetensors"])
    plan, layer_ws = _expert_layers(model, src)
    keys, read_tensor = safetensors_reader(src)
    read = make_plan_reader(plan, read_tensor, torch.float32)
    prefused = _prefused_layers(plan) if not plan.experts else {}
    if not plan.experts and not prefused:
        raise RuntimeError(f"{a.family}: neither per-expert projections nor prefused stacks in the plan")
    layout = "per-expert" if plan.experts else "prefused stacks"

    def read_layer(layer):                               # P44's read_layer, the same bytes the enabler packs
        if plan.experts:
            return read_fused_expert_layer(plan, layer, read, device="cpu", dtype=torch.float32)
        gu_key, dn_key = prefused[layer]
        return read(gu_key).to(torch.float32), read(dn_key).to(torch.float32)

    order = [layer for layer, _w in layer_ws]
    if a.layers:
        keep = {int(x) for x in a.layers.split(",")}
        order = [x for x in order if x in keep]
    cfg = model.config
    tcfg = getattr(cfg, "text_config", None) or cfg
    hid = int(getattr(tcfg, "hidden_size"))
    inter = int(getattr(tcfg, "moe_intermediate_size", None) or getattr(tcfg, "intermediate_size"))
    E = int(info["experts"])
    per_layer_h = E * (hid ** 2 + inter ** 2) * 4                   # P44's Hessian bytes per layer
    per_layer_w = E * 3 * hid * inter * 4                           # the chunk's fp32 expert weights, read once
    budget = int(float(os.environ.get("E4B_INT4_HESSIAN_BUDGET_GB", "24")) * (1 << 30))
    lpp = max(1, budget // (per_layer_h + per_layer_w))

    tb = {}
    tdig = {}
    for t in texts:
        full = text_batches(tok, t, a.nseq)
        halves = split_halves(full)
        tb[t] = halves
        tdig[t] = {"source": ("wikitext-2-raw-v1 TRAIN (serve_stack.calib_batches)" if t == "wikitext" else
                              f"{C4VAL1['repo']} {C4VAL1['data_files']} first {C4VAL1['docs']} docs (K8's c4val1)"),
                   "halves": [batches_digest(h) for h in halves]}
    rep = {"lane": "P65 census", "family": a.family, "model": model_id, "revision": revision,
           "method": "rtn (census_row with min_rows above any row count: no GPTQ solve)",
           "entropy_definition": "Colla-Q rho = sigma2_within / sigma2_total of the expert output y = W_dn h "
                                 "(bench/p65/expert_entropy.py)",
           "texts": tdig, "nseq_per_text": a.nseq, "seq_len": 512, "layers_total": len(layer_ws),
           "layers_requested": order, "layers_censused": [], "expert_layout": layout, "experts": E, "hidden": hid,
           "inter": inter, "hessian_bytes_per_layer": per_layer_h, "weight_bytes_per_layer": per_layer_w,
           "layers_per_pass": lpp, "engagement": info, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "timing": [], "selfcheck": None, "rows": []}

    def flush():
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rep, f, indent=1)
        os.replace(tmp, a.out)

    flush()
    if a.selfcheck:
        rep["selfcheck"] = selfcheck(model, src, layer_ws, order[0], read_layer, tb[texts[0]][0][0],
                                     calibrate_expert_hessians, _hr, dev)
        flush()
        if not rep["selfcheck"]["ok"]:
            print("SELFCHECK FAILED", json.dumps(rep["selfcheck"]), flush=True)
            return 3
        print("selfcheck ok:", json.dumps({k: v for k, v in rep["selfcheck"].items() if k != "experts"}), flush=True)

    for i in range(0, len(order), lpp):
        chunk = order[i:i + lpp]
        t1 = time.time()
        weights = {layer: read_layer(layer) for layer in chunk}
        t_read = time.time() - t1
        halves_rows, halves_mom = {}, {}
        for t in texts:
            for hi, batches in enumerate(tb[t]):
                t2 = time.time()
                means = {}
                hs = calibrate_expert_hessians(model, src, batches, only_layers=chunk, layers_per_pass=len(chunk),
                                               hessian_device="cpu", activation_means=means)
                t_h = time.time() - t2
                gc.collect()
                torch.cuda.empty_cache()
                t3 = time.time()
                for layer in chunk:
                    first, down = weights[layer]
                    hl, ml = hs.get(layer, {}), means.get(layer, {})
                    for e in range(first.shape[0]):
                        H_gu, H_dn, rows = hl.get(e, (None, None, 0))
                        mx, mh = ml.get(e, (None, None))
                        for role, W, H, m in (("gu", first[e], H_gu, mx), ("dn", down[e], H_dn, mh)):
                            row = census_row(W, H, rows, min_rows=NO_GPTQ, damp=0.01, dev=dev)
                            ent = entropy_fields(role, W, H, m, dev=dev)
                            row["entropy"] = ent[0] if ent else None
                            row.update({"layer": int(layer), "expert": int(e), "role": role, "text": t, "half": hi})
                            rep["rows"].append(row)
                            halves_rows[(t, hi, layer, e, role)] = row
                            if ent:
                                halves_mom[(t, hi, layer, e, role)] = (int(rows),) + ent[1]
                del hs, means
                gc.collect()
                torch.cuda.empty_cache()
                rep["timing"].append({"chunk": chunk, "text": t, "half": hi, "hessian_s": round(t_h, 1),
                                      "rows_s": round(time.time() - t3, 1)})
                print(f"  layers {chunk[0]}..{chunk[-1]} {t}/{hi}: hessians {t_h:.0f}s rows {time.time() - t3:.0f}s",
                      flush=True)
                flush()
            for layer in chunk:                                     # the text's full rows, from its two halves
                for e in range(weights[layer][0].shape[0]):
                    for role in ("gu", "dn"):
                        r0, r1 = halves_rows.get((t, 0, layer, e, role)), halves_rows.get((t, 1, layer, e, role))
                        row = combine_rows(r0, r1)
                        n, m2, mu = combine_moments([halves_mom[k] for k in ((t, 0, layer, e, role), (t, 1, layer, e, role))
                                                     if k in halves_mom])
                        if n:
                            ent = entropy_ratio(m2, mu)
                            ent["of"] = "output y = W_dn h" if role == "dn" else "input x"
                            row["entropy"] = ent
                        else:
                            row["entropy"] = None
                        rep["rows"].append(row)
        rep["timing"].append({"chunk": chunk, "read_s": round(t_read, 1)})
        rep["layers_censused"].extend(int(x) for x in chunk)
        del weights, halves_rows, halves_mom
        gc.collect()
        print(f"  layers {chunk[0]}..{chunk[-1]} done; {len(rep['rows'])} rows", flush=True)
        flush()
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["cuda_max_memory_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
    flush()
    print(f"census -> {a.out}: {len(rep['rows'])} rows over {len(rep['layers_censused'])} layers in {rep['wall_s']} s")
    return 0


def selfcheck(model, src, layer_ws, layer, read_layer, batch, calibrate_expert_hessians, _hr, dev) -> dict:
    """Two instrument checks on the box's own silicon, before any row is written:
      (a) the moment-derived ratio equals the ratio of the SAME expert outputs computed from raw rows (one batch, the
          two most-routed experts of the first layer): the tap feeds what the definition says it does;
      (b) recomputing a row gives the same bits (census_row's rel_act and the entropy ratio): the census is
          deterministic on this box."""
    w = dict(layer_ws)[layer]
    first, down = read_layer(layer)
    means = {}
    hs = calibrate_expert_hessians(model, src, [batch], only_layers=[layer], layers_per_pass=1,
                                   hessian_device="cpu", activation_means=means)
    routed = sorted(hs.get(layer, {}).items(), key=lambda kv: (-kv[1][2], kv[0]))[:2]
    cap = _RowCapture(id(w.h_gu_p), [e for e, _ in routed])
    prev = _hr._CALIB_SINK
    try:
        _hr._CALIB_SINK = cap
        with torch.no_grad():
            model(batch.to(dev))
    finally:
        _hr._CALIB_SINK = prev
    out = {"layer": int(layer), "experts": [], "ok": True}
    for e, (H_gu, H_dn, rows) in routed:
        hrows = torch.cat(cap.h[e]) if cap.h[e] else torch.empty(0)
        y = hrows @ down[e].to(torch.float64).t()
        direct = direct_entropy_ratio(y)
        ent, _mom = entropy_fields("dn", down[e], H_dn, means[layer][e][1], dev=dev)
        ent2, _ = entropy_fields("dn", down[e], H_dn, means[layer][e][1], dev=dev)
        r1 = census_row(down[e], H_dn, rows, min_rows=NO_GPTQ, damp=0.01, dev=dev)
        r2 = census_row(down[e], H_dn, rows, min_rows=NO_GPTQ, damp=0.01, dev=dev)
        rel = abs(ent["rho"] - direct["rho"]) / max(abs(direct["rho"]), 1e-30)
        rec = {"expert": int(e), "tap_rows": int(rows), "captured_rows": int(hrows.shape[0]),
               "rho_moments": ent["rho"], "rho_direct": direct["rho"], "rel_diff": rel,
               "entropy_bitwise_repeat": ent == ent2, "rel_act_bitwise_repeat": r1["rtn"]["rel_act"] == r2["rtn"]["rel_act"]}
        rec["ok"] = (rec["tap_rows"] == rec["captured_rows"] and rel <= 1e-4
                     and rec["entropy_bitwise_repeat"] and rec["rel_act_bitwise_repeat"])
        out["experts"].append(rec)
        out["ok"] = out["ok"] and rec["ok"]
    out["max_rel_diff"] = max((r["rel_diff"] for r in out["experts"]), default=None)
    del first, down, hs
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
