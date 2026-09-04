"""Decompose a Phase-9 engine step into its cost buckets.

The fixbox run left ~84 ms/step unattributed between attention and engine
overhead (197 ms/step in the engine vs 113 ms bare forward, experts 66 ms).
This instrument splits a step without touching repo code:

  decode step = scheduler python
              + forward submission   (host time inside model(...))
                  |- attention host  (paged-attention calls, host ns)
                  |- dram experts    (synchronous CPU kernel wall)
                  |- other submission (router/norms/embed/lm_head launches)
              + drain                (argmax+tolist sync absorbing GPU tail)
              + bookkeeping

Device-side truth comes separately from CUDA events (attention kernels,
GPU expert kernels) — reported as device occupancy, never subtracted from
host buckets, because overlap makes subtraction a lie.

Methodology validated on the dev box (OLMoE); constants only bind on a
serving-class box (G6 tiny-model trap applies to MAGNITUDES, not shape).
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

import torch

PROF = {"attn_host_ns": 0, "attn_calls": 0, "attn_events": [],
        "mode": "decode"}
COMPILE_GRAPH_STEP = [False]


def _materialize_from_arena(mods, arena_path):
    """R1 mechanics (PREREG-b1): the streaming loader leaves module
    expert tensors as META stubs (the bytes live in the gnf4 arena), and
    the pipelined engine sources its pinned arena from MODULE tensors.
    Fill the modules from the SAME arena file -- identical packed bytes
    by construction, no requantization. Byte counts are asserted per
    segment, and the R0==R1 bitwise token gate downstream is the
    semantic backstop: wrong bytes cannot pass it."""
    import numpy as np
    import torch.nn as nn

    from nvme_arena import _seg_len, _seg_off, load_index, row_offset

    idx = load_index(arena_path)
    layer_ids = sorted({l for l, _e, _o in idx["rows"]})
    assert len(layer_ids) == len(mods), (len(layer_ids), len(mods))
    seg_map = (("gate_up_proj", "nf4.gate_up_blocks", True),
               ("gate_up_absmax", "nf4.gate_up_absmax", False),
               ("down_proj", "nf4.down_blocks", True),
               ("down_absmax", "nf4.down_absmax", False))
    mm = np.memmap(arena_path, dtype=np.uint8, mode="r")
    for mi, wrapped in enumerate(mods):
        base = getattr(wrapped, "base", wrapped)
        E = base.num_experts
        li = layer_ids[mi]
        for attr, suffix, is_param in seg_map:
            meta = getattr(base, attr)
            t = torch.empty(meta.shape, dtype=meta.dtype)
            flat = t.view(torch.uint8).reshape(E, -1) \
                if t.dtype != torch.uint8 else t.reshape(E, -1)
            off, ln = _seg_off(idx, suffix), _seg_len(idx, suffix)
            assert flat.shape[1] == ln, \
                (attr, tuple(meta.shape), flat.shape[1], ln)
            for e in range(E):
                lo = row_offset(idx, li, e) + off
                flat[e] = torch.from_numpy(
                    np.ascontiguousarray(mm[lo:lo + ln]))
            if is_param:
                setattr(base, attr,
                        nn.Parameter(t, requires_grad=False))
            else:
                setattr(base, attr, t)
    del mm
    print(f"materialized {len(mods)} modules from {arena_path} "
          f"(byte-exact, per-segment lengths asserted)", flush=True)


@torch.no_grad()
def ppl_oracle_score(model, ids, prompt_len: int, steps: int,
                     chunk: int = 256, device=None):
    """Teacher-forced NLL of ``ids[prompt_len + 1 : prompt_len + steps + 1]``
    through the model's OWN attention with its own cache: prefill the
    prompt, then feed the continuation in chunks. Every scored token is
    predicted from exactly the tokens before it (a chunk's logits at row
    j predict ids[start + j + 1]), so this equals a single full forward's
    NLL -- checked by a CPU test -- while fitting eager attention's
    [H, chunk, ctx] scores in memory at 8k context."""
    dev = device or next(model.parameters()).device
    ids = ids.to(dev)
    end = prompt_len + steps + 1
    assert ids.numel() >= end, (ids.numel(), end)
    out = model(input_ids=ids[None, :prompt_len], use_cache=True)
    cache = out.past_key_values
    last_logit = out.logits[0, -1].float()          # predicts ids[prompt_len]
    nll = 0.0
    pos = prompt_len
    # the first scored token is ids[prompt_len + 1], predicted after
    # feeding ids[prompt_len]; the prompt's last logit predicts
    # ids[prompt_len] which the paged instrument does NOT score (it
    # starts at cont[1]) -- keep the two windows identical
    while pos < end - 1:
        n = min(chunk, end - 1 - pos)
        chunk_ids = ids[pos:pos + n]
        out = model(input_ids=chunk_ids[None], use_cache=True,
                    past_key_values=cache,
                    position_ids=torch.arange(pos, pos + n, device=dev)[None])
        cache = out.past_key_values
        lg = torch.log_softmax(out.logits[0].float(), -1)   # [n, V]
        tgt = ids[pos + 1:pos + n + 1]
        nll += -lg[torch.arange(n, device=dev), tgt].sum().item()
        pos += n
    return nll / steps


@torch.no_grad()
def ppl_oracle_score_full(model, ids, prompt_len: int, steps: int, device=None):
    """Teacher-forced NLL of the same window as :func:`ppl_oracle_score`,
    computed in ONE forward with no cache and no chunking.

    Why this exists: the chunked scorer and a single full forward are the
    same mathematics in a different order, and on a mixture-of-experts
    model that reordering flips a few percent of router top-k choices --
    4.52% on gpt-oss, 6.77% on Qwen3 -- which disagrees regardless of
    correctness (METHODOLOGY 13.1). A single forward has no such
    ordering, so it is the reference an incremental decode should be
    compared against, and the chunked-vs-full difference is that model's
    floor.

    Cost is quadratic in the window: eager attention materialises
    [heads, T, T]. Keep `steps` in the hundreds, not thousands -- this
    is a reference measurement, not the 8192-step production window.
    """
    dev = device or next(model.parameters()).device
    end = prompt_len + steps + 1
    assert ids.numel() >= end, (ids.numel(), end)
    out = model(input_ids=ids[None, :end].to(dev))
    # row j predicts ids[j + 1]; the scored window starts at prompt_len + 1
    lg = out.logits[0, prompt_len:end - 1].cpu().float()
    tgt = ids[prompt_len + 1:end].cpu()
    nll = -torch.log_softmax(lg, -1).gather(1, tgt[:, None]).sum()
    return float(nll) / steps


# ---------------------------------------------------------------------------
# Fake-quantised eager attention: the fp8 paged kernel's PRECISION MODEL run
# inside the chunk-free full forward. Every rounding the sm_120 fp8 compute
# path performs is reproduced on the bf16 tensors HF's own attention module
# hands over, and nothing else changes -- same weights, same mask, same
# softmax order as ``eager``. So the arm isolates "what the fp8 cache and
# fp8 dot cost" from "what the paged shim/geometry does", which a paged-vs-
# full delta cannot (e4b#359). What the kernel does, mirrored here:
#   q  -> one scale per (kv head, query row) over its G query heads x D
#         (``q_amax = max|q[G, D]|``, ``q_s = 448 / q_amax``), e4m3
#   k  -> stored e4m3 with ``k_groups`` sub-row absmax scales (fp8_kv)
#   v  -> stored e4m3 with one absmax scale per (token, head)   (fp8_kv)
#   p  -> the softmax weights times the V scale, range-folded by
#         ``c = 448 / max(vs)`` over the row's valid keys, e4m3
# ``from`` rows attend unmodified: the paged prefill computes the prompt's
# own attention on staged bf16 K/V and only the decode steps read fp8.
_FQ = {"spec": "", "kg": 4, "vg": 1, "layers": "all", "frm": 0}
_FQ_IMPL = "eager_fq"
_E4M3_MAX = 448.0


def _fq_quant_dequant(x, group):
    """fp8_kv.quantize_kv_fp8 + dequant_kv_fp8_ref in one step (fp32 in and
    out): absmax per ``group`` consecutive values of the last dim, scale =
    amax/448 with the all-zero group pinned to 1.0, e4m3 round trip.
    Kept local so the CPU oracle needs no kernel wheel; a test pins it
    bit-exact to fp8_kv when that module imports."""
    d = x.shape[-1]
    group = d if group is None else group
    xg = x.reshape(*x.shape[:-1], d // group, group)
    amax = xg.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / _E4M3_MAX, torch.ones_like(amax))
    q = (xg / scale).to(torch.float8_e4m3fn).to(torch.float32)
    return (q * scale).reshape(x.shape), scale.squeeze(-1)


def _fq_eager_attention(module, query, key, value, attention_mask,
                        scaling=None, dropout: float = 0.0,
                        softcap=None, **kwargs):
    """``[B, H_q, T, D]`` in, ``([B, T, H_q, D], None)`` out -- transformers'
    eager attention with the fp8 kernel's roundings applied per _FQ."""
    B, Hq, T, D = query.shape
    Hkv = key.shape[1]
    G = Hq // Hkv
    if attention_mask is None and T > 1:
        raise RuntimeError("eager_fq received no attention mask for a "
                           f"{T}-token forward: the causal mask was not built "
                           "for this attention name, and attending without "
                           "one is silent and non-causal")
    spec = _FQ["spec"]
    is_sliding = bool(getattr(module, "is_sliding", None)
                      or getattr(module, "sliding_window", None))
    want = _FQ["layers"]
    apply = bool(spec) and (want == "all" or (want == "sliding") == is_sliding)
    scale = float(D ** -0.5 if scaling is None else scaling)

    def attend(fq: bool):
        qf, kf, vf = query.float(), key.float(), value.float()
        vs = None
        if fq and "k" in spec:
            kf, _ = _fq_quant_dequant(kf, None if _FQ["kg"] == 1 else D // _FQ["kg"])
        if fq and "v" in spec:
            vf, vs = _fq_quant_dequant(vf, None if _FQ["vg"] == 1 else D // _FQ["vg"])
            if vs.ndim == 4:                      # grouped V: kernel folds one
                vs = vs.amax(-1)                  # scale per row; approximate
        if fq and "q" in spec:
            qt = qf.view(B, Hkv, G, T, D)
            amax = qt.abs().amax(dim=(2, 4), keepdim=True)          # per (b, kv head, row)
            qs = torch.where(amax > 0, _E4M3_MAX / amax, torch.ones_like(amax))
            qf = ((qt * qs).to(torch.float8_e4m3fn).to(torch.float32) / qs).view(B, Hq, T, D)
        k_rep = kf.repeat_interleave(G, dim=1)
        v_rep = vf.repeat_interleave(G, dim=1)
        sc = torch.matmul(qf, k_rep.transpose(-1, -2)) * scale
        if softcap is not None:
            sc = torch.tanh(sc / softcap) * softcap
        if attention_mask is not None:
            sc = sc + attention_mask[:, :, :, : k_rep.shape[-2]].to(sc.dtype)
        p = torch.softmax(sc, dim=-1, dtype=torch.float32)
        if fq and "p" in spec and vs is not None:
            vs_q = vs.repeat_interleave(G, dim=1)[:, :, None, :]     # [B, Hq, 1, Tk]
            valid = p > 0
            cmax = (vs_q * valid).amax(-1, keepdim=True)
            c = torch.where(cmax > 0, _E4M3_MAX / cmax, torch.ones_like(cmax))
            pv = p * vs_q * c
            p = pv.to(torch.float8_e4m3fn).to(torch.float32) / (vs_q * c)
        return torch.matmul(p, v_rep)                                  # [B, Hq, T, D]

    out = attend(apply)
    frm = int(_FQ["frm"])
    if apply and 0 < frm < T:
        out = torch.cat([attend(False)[:, :, :frm], out[:, :, frm:]], dim=2)
    return out.transpose(1, 2).contiguous().to(query.dtype), None


def _fq_register(spec: str, kg: int, vg: int, layers: str, frm: int) -> str:
    """Install the precision model under ``eager_fq`` (attention AND mask
    interfaces -- a custom name with no mask entry gets no causal mask)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    if any(ch not in "qkvp" for ch in spec):
        raise ValueError(f"--ppl-fq letters must be from qkvp, got {spec!r}")
    _FQ.update(spec=spec, kg=int(kg), vg=int(vg), layers=layers, frm=int(frm))
    ALL_ATTENTION_FUNCTIONS[_FQ_IMPL] = _fq_eager_attention
    # transformers' mask preprocessor consults the CLASS-level mapping only
    # (masking_utils._preprocess_mask_arguments: "not in
    # ALL_MASK_ATTENTION_FUNCTIONS._global_mapping" -> no mask is built and
    # the attention function is left to mask for itself). A name registered
    # through __setitem__ lands in the instance's local mapping, is skipped,
    # and the model runs NON-causal with no error -- caught by the tiny-model
    # test. So register in both.
    eager_mask = ALL_MASK_ATTENTION_FUNCTIONS["eager"]
    ALL_MASK_ATTENTION_FUNCTIONS[_FQ_IMPL] = eager_mask
    ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[_FQ_IMPL] = eager_mask
    return _FQ_IMPL


# ---------------------------------------------------------------------------
# Per-layer output diff: the paged decode step against the chunk-free eager
# forward at the SAME position, hooked at every decoder layer's attention,
# dense MLP, expert block and router (e4b#359 next test under H0). The
# paged step is captured from the K8 loop's OWN forward (a second paged
# forward would append that token's K/V twice); the reference is one eager
# forward over ids[:pos+1] with the paged context cleared, i.e. the `full`
# arm's path, and only its last position is compared.
def _decoder_layers(model):
    """The first ModuleList whose members carry `self_attn`."""
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) and hasattr(m[0], "self_attn"):
            return m
    raise RuntimeError("no decoder layer list with self_attn found")


def _layer_diff_install(model, cap):
    """Forward hooks that record each part's LAST-position output (fp32,
    cpu) into `cap[(layer, part)]`; router hooks record (weights, index)."""
    handles = []

    def rec(i, part):
        def hook(_m, _inp, out):
            if part == "router":
                pos = _router_out_positions(out)
                if pos is None:
                    return
                w, idx = out[pos[0]], out[pos[1]]
                cap[(i, "router")] = (w.detach().float().cpu()[-1], idx.detach().cpu()[-1])
                return
            y = out[0] if isinstance(out, tuple) else out
            y = y.detach().float()
            cap[(i, part)] = (y[:, -1] if y.ndim == 3 else y[-1:]).reshape(-1).cpu()
        return hook

    for i, L in enumerate(_decoder_layers(model)):
        handles.append(L.register_forward_hook(rec(i, "layer")))
        handles.append(L.self_attn.register_forward_hook(rec(i, "attn")))
        for part in ("mlp", "experts", "router"):
            sub = getattr(L, part, None)
            if isinstance(sub, torch.nn.Module):
                handles.append(sub.register_forward_hook(rec(i, part)))
    return handles


def _layer_diff_compare(cap_paged, cap_ref, layers_meta):
    """Per-layer relative errors ||paged - ref|| / ||ref|| and router
    agreement; returns rows plus a summary."""
    rows = []
    for i, kind in layers_meta:
        row = {"layer": i, "kind": kind}
        # the decoder layer's own output is reported as "out" so the row's
        # "layer" key stays the layer INDEX (Bugbot, #361)
        for part, key in (("attn", "attn"), ("mlp", "mlp"), ("experts", "experts"), ("layer", "out")):
            a_, b_ = cap_paged.get((i, part)), cap_ref.get((i, part))
            if a_ is not None and b_ is not None:
                row[key] = float((a_ - b_).norm() / (b_.norm() + 1e-12))
        if (i, "router") in cap_paged and (i, "router") in cap_ref:
            (wp, ip), (wr, ir) = cap_paged[(i, "router")], cap_ref[(i, "router")]
            sp, sr = set(ip.tolist()), set(ir.tolist())
            row["router_overlap"] = len(sp & sr) / max(len(sr), 1)
            if sp == sr:
                order = {int(e): k for k, e in enumerate(ir.tolist())}
                wr_al = torch.stack([wr[order[int(e)]] for e in ip.tolist()])
                row["router_w_maxabs"] = float((wp - wr_al).abs().max())
        rows.append(row)

    def mean_of(part, kind=None):
        v = [r[part] for r in rows if part in r and (kind is None or r["kind"] == kind)]
        return sum(v) / len(v) if v else None
    summ = {"attn_rel_sliding": mean_of("attn", "sliding"), "attn_rel_full": mean_of("attn", "full"),
            "mlp_rel": mean_of("mlp"), "experts_rel": mean_of("experts"), "out_rel": mean_of("out"),
            "router_overlap_mean": mean_of("router_overlap"),
            "first_layer_over_5pct": next((r["layer"] for r in rows if r.get("out", 0.0) > 0.05), None)}
    return rows, summ


def _layer_diff_reference(model, ids_upto, cap_ref):
    """One eager forward over `ids_upto` (the `full` arm's path) with the
    paged context cleared; hooks fill cap_ref with the last position."""
    try:
        from experts4bit_qlora.engines.paged_attention import set_context
    except ImportError:          # no engine in this environment: nothing to clear
        set_context = lambda _ctx: None  # noqa: E731
    prev_ctx = set_context(None)
    # the reference forward must not consume routing-replay rows: pause
    # replay/record for its duration and restore afterwards (Bugbot, #365)
    route_mode = _ROUTE["mode"]
    _ROUTE["mode"] = None
    # One save per DISTINCT config object: HF modules share the model's
    # config, so saving per module would record "eager" for every module
    # after the first and the restore would leave the model on eager --
    # every paged step after the first diff would then run one-token
    # eager attention with no cache (Bugbot, #361, High).
    saved, seen = [], set()
    for m in [model] + list(model.modules()):
        cfg = getattr(m, "config", None)
        if cfg is None or not hasattr(cfg, "_attn_implementation") or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        saved.append((cfg, cfg._attn_implementation))
        cfg._attn_implementation = "eager"
    handles = _layer_diff_install(model, cap_ref)
    try:
        with torch.no_grad():
            model(input_ids=ids_upto[None].to(next(model.parameters()).device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
        for cfg, impl in reversed(saved):
            cfg._attn_implementation = impl
        _ROUTE["mode"] = route_mode
        set_context(prev_ctx)


def _layer_kinds(model):
    out = []
    for i, L in enumerate(_decoder_layers(model)):
        at = L.self_attn
        sliding = bool(getattr(at, "is_sliding", None) or getattr(at, "sliding_window", None))
        out.append((i, "sliding" if sliding else "full"))
    return out


def _layer_diff_report(t, pos_abs, cap_paged, cap_ref, model, out_path):
    import json as _json
    rows, summ = _layer_diff_compare(cap_paged, cap_ref, _layer_kinds(model))
    fmt = lambda d: " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())  # noqa: E731
    print(f"LAYER_DIFF t={t} pos={pos_abs} {fmt(summ)}", flush=True)
    for r in rows:
        print("  " + fmt(r), flush=True)
    with open(f"{out_path}.layerdiff_t{t}.json", "w") as f:
        _json.dump({"t": t, "pos": pos_abs, "summary": summ, "rows": rows}, f, indent=1)
    return summ


def _oracle_label(a, impl: str) -> str:
    """The arm's identity as recorded in `attn_path` and the K8 line. The
    upstream (HF-loaded) arms keep their `upstream-` prefix under every
    modifier -- chunk size or fake-quant letters -- so a verdict can always
    tell the HF-loaded control from the e4b-loaded oracle (Bugbot, #361)."""
    base = {"upstream": "upstream-eager", "full": "eager-fullforward",
            "upstream-full": "upstream-eager-fullforward"}.get(a.ppl_oracle, impl)
    ck = int(getattr(a, "ppl_chunk", 0) or 0)
    if a.ppl_oracle in ("eager", "upstream") and ck > 0:
        base = f"{base}-chunk{ck}"
    fq = getattr(a, "ppl_fq", "none")
    if fq and fq != "none" and a.ppl_oracle in ("full", "upstream-full"):
        frm = a.prompt_len if getattr(a, "fq_from", -1) < 0 else a.fq_from
        base = (f"{base}-fq{fq}-kg{a.fq_kgroups}-vg{a.fq_vgroups}"
                f"-{a.fq_layers}-from{frm}")
    return base


# ---------------------------------------------------------------------------
# Matched routing: record every router's top-k choice (index + weight) per
# layer and position from a reference forward, then REPLAY them into another
# path scoring the same tokens. On a mixture-of-experts model two
# arithmetically different paths route a few percent of tokens to different
# experts, and Gemma-4 amplifies that into 0.1-0.3 nats per 512-token window
# (P27, e4b#359) -- so a parity delta at that resolution measures the
# router's chaos, not the path. With routing matched, what remains is the
# attention + kernel difference, which is the thing being asked about.
#
# Positions are tracked by CONSUMPTION: each layer's router serves rows in
# order (a full forward serves all T rows at once; a paged path serves the
# prompt in prefill chunks, then one row per decode step), and the harness
# resets the counters when the KV cache is rewound to the prompt boundary.
# Every router call outside the recorded range passes through untouched
# and is counted, so the K8 line can say how much of the window was
# actually matched.
_ROUTE = {"mode": None, "store": {}, "consumed": {}, "served": 0, "passed": 0,
          "k": None, "handles": []}


def _router_modules(model):
    """(layer index, router module) for every decoder layer that has one --
    the first submodule whose class name carries 'Router' (Gemma-4
    `layer.router`, Qwen3-MoE / OLMoE / Mixtral `layer.mlp.gate`)."""
    out = []
    for i, L in enumerate(_decoder_layers(model)):
        for _n, m in L.named_modules():
            if "Router" in type(m).__name__:
                out.append((i, m))
                break
    return out


def _router_out_positions(out):
    """Where a router's (weights, index) sit in its output tuple. Most
    families return (logits_or_probs, weights, index); GraniteMoe returns
    (index, weights, logits). The index is the only integer tensor, and
    the weights are the float tensor with its shape."""
    if not isinstance(out, tuple):
        return None
    ipos = [i for i, t in enumerate(out) if torch.is_tensor(t) and not t.is_floating_point()]
    if len(ipos) != 1:
        return None
    idx = out[ipos[0]]
    wpos = [i for i, t in enumerate(out) if i != ipos[0] and torch.is_tensor(t)
            and t.is_floating_point() and tuple(t.shape) == tuple(idx.shape)]
    if len(wpos) != 1:
        return None
    return wpos[0], ipos[0]


def _route_hook(layer):
    def hook(_m, _inp, out):
        pos = _router_out_positions(out)
        if pos is None:
            raise RuntimeError(f"layer {layer}: router output {type(out).__name__} of "
                               f"{len(out) if isinstance(out, tuple) else 'n/a'} carries no "
                               "single (weights, index) pair -- refusing to record/replay")
        wpos, ipos = pos
        w, idx = out[wpos], out[ipos]
        n = int(idx.shape[0])
        st = _ROUTE
        if st["mode"] == "record":
            st["store"].setdefault(layer, {"idx": [], "w": []})
            st["store"][layer]["idx"].append(idx.detach().cpu())
            st["store"][layer]["w"].append(w.detach().float().cpu())
            st["consumed"][layer] = st["consumed"].get(layer, 0) + n
            return None
        if st["mode"] == "replay":
            rec = st["store"].get(layer)
            c = st["consumed"].get(layer, 0)
            st["consumed"][layer] = c + n
            if rec is None or c + n > int(rec["idx"].shape[0]):
                st["passed"] += n
                return None
            st["served"] += n
            new_idx = rec["idx"][c:c + n].to(idx.device)
            new_w = rec["w"][c:c + n].to(device=w.device, dtype=w.dtype)
            new = list(out)
            new[wpos], new[ipos] = new_w, new_idx
            return tuple(new)
        return None
    return hook


def _route_install(model, mode: str, store=None):
    """Arm record or replay on every router; returns the number armed."""
    _route_clear()
    _ROUTE["mode"] = mode
    _ROUTE["store"] = {} if store is None else store
    for layer, m in _router_modules(model):
        _ROUTE["handles"].append(m.register_forward_hook(_route_hook(layer)))
    return len(_ROUTE["handles"])


def _route_clear():
    for h in _ROUTE["handles"]:
        h.remove()
    _ROUTE.update(mode=None, store={}, consumed={}, served=0, passed=0, handles=[])


def _route_reset(pos: int = 0):
    """Point every layer's counter at absolute position `pos` (after a KV
    rewind to the prompt boundary) and zero the served/passed tallies."""
    for layer in list(_ROUTE["consumed"]) + [l for l, _ in _ROUTE.get("_layers", [])]:
        _ROUTE["consumed"][layer] = pos
    _ROUTE["served"] = 0
    _ROUTE["passed"] = 0


def _route_save(path: str, meta: dict) -> dict:
    """Concatenate the recorded chunks per layer and write them."""
    rec = {int(l): {"idx": torch.cat(v["idx"]), "w": torch.cat(v["w"])}
           for l, v in _ROUTE["store"].items()}
    torch.save({"layers": rec, "meta": meta}, path)
    return rec


def _route_load(path: str):
    d = torch.load(path, map_location="cpu")
    return d["layers"], d.get("meta", {})


def _ppl_oracle_main(a, model, ppl_ids, ppl_sha):
    """The oracle arm: same window, same sha, transformers' attention."""
    import json as _json
    import time as _time
    impl = "eager" if a.ppl_oracle in ("upstream", "full", "upstream-full") else a.ppl_oracle
    label = _oracle_label(a, impl)
    fq = getattr(a, "ppl_fq", "none")
    if fq and fq != "none":
        if a.ppl_oracle not in ("full", "upstream-full"):
            raise SystemExit("--ppl-fq models the fp8 kernel inside the "
                             "chunk-free forward: use it with --ppl-oracle full")
        frm = a.prompt_len if a.fq_from < 0 else a.fq_from
        impl = _fq_register(fq, a.fq_kgroups, a.fq_vgroups, a.fq_layers, frm)
    model.config._attn_implementation = impl
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = impl
    model.eval()
    t0 = _time.perf_counter()
    route = getattr(a, "ppl_route", "none")
    if a.ppl_oracle in ("full", "upstream-full"):
        if route == "record":
            n_armed = _route_install(model, "record")
            print(f"ROUTE record armed on {n_armed} routers", flush=True)
        elif route == "replay":
            raise SystemExit("--ppl-route replay is for the paged arm; the "
                             "full forward is what gets RECORDED")
        mean_nll = ppl_oracle_score_full(model, ppl_ids, a.prompt_len,
                                         a.ppl_steps)
        if route == "record":
            rec = _route_save(a.ppl_route_file, {"text_sha": ppl_sha,
                                                 "prompt_len": a.prompt_len,
                                                 "steps": a.ppl_steps,
                                                 "model": a.model})
            n_pos = next(iter(rec.values()))["idx"].shape[0] if rec else 0
            print(f"ROUTE recorded {len(rec)} layers x {n_pos} positions -> "
                  f"{a.ppl_route_file}", flush=True)
            _route_clear()
    else:
        ck = int(getattr(a, "ppl_chunk", 0) or 0)
        mean_nll = ppl_oracle_score(model, ppl_ids, a.prompt_len, a.ppl_steps,
                                    **({"chunk": ck} if ck > 0 else {}))
    rec = {"k8": "ppl", "attn_path": f"{label}-oracle", "steps": a.ppl_steps,
           "mean_nll": mean_nll, "ppl": float(torch.exp(torch.tensor(mean_nll))),
           "prompt_len": a.prompt_len, "prompt_offset": a.prompt_offset,
           "ppl_source": a.ppl_source, "tokens_scored": a.ppl_steps,
           "text_sha": ppl_sha, "wall_s": round(_time.perf_counter() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        _json.dump(rec, f, indent=1)
    print(f"K8_PPL steps={a.ppl_steps} nll={mean_nll:.5f} ppl={rec['ppl']:.5f} "
          f"compute={label}-oracle sha={ppl_sha[:12]} out={a.out}", flush=True)


def _chat_prefix_ids(tok, suffix: str = "") -> "torch.Tensor":
    """Token ids of the chat-template prefix that turns the K8 corpus into
    an assistant reply: system/developer defaults, one user turn, the
    generation prompt, then ``suffix`` (special tokens honoured)."""
    msgs = [{"role": "user",
             "content": "Continue the following text verbatim, without commentary."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids)
    if suffix:
        ids += list(tok(suffix, add_special_tokens=False)["input_ids"])
    return torch.tensor(ids, dtype=torch.long)


def _k8_window(a, tok):
    """The K8 corpus, the per-row prompts and the scored window with its
    digest -- one function so every arm (paged, e4b-loaded oracle,
    upstream control) scores byte-identical text."""
    from datasets import load_dataset
    if a.ppl_source == "c4val1":
        ds = load_dataset("allenai/c4",
                          data_files={"v": "en/c4-validation.00001-of-00008.json.gz"},
                          split="v")
        text = "\n\n".join(ds["text"][:2000])
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="test")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
    chat = bool(getattr(a, "ppl_chat", False))
    # inside the chat template the corpus is the assistant's reply: it must
    # carry no BOS of its own (Gemma/Llama tokenizers add one by default;
    # a BOS mid-sequence is exactly the out-of-distribution token the chat
    # window exists to avoid)
    ids = tok(text, return_tensors="pt", add_special_tokens=not chat).input_ids[0]
    if chat:
        pre = _chat_prefix_ids(tok, getattr(a, "ppl_chat_suffix", ""))
        assert pre.numel() < a.prompt_len, \
            f"chat prefix ({pre.numel()} tokens) must fit inside --prompt-len"
        print(f"K8 chat window: {pre.numel()}-token template prefix, corpus as the "
              f"assistant reply (suffix={getattr(a, 'ppl_chat_suffix', '')!r})",
              flush=True)
        ids = torch.cat([pre, ids])
    if a.prompt_offset or a.prompt_span:
        end = (a.prompt_offset + a.prompt_span) if a.prompt_span \
            else ids.numel()
        assert a.prompt_offset + a.batch * a.prompt_len < end <= ids.numel(), \
            "prompt slice leaves too little corpus for the windows"
        ids = ids[a.prompt_offset:end]
    step = max(1, (ids.numel() - a.prompt_len) // max(1, a.batch))
    prompts = [ids[i * step:i * step + a.prompt_len].tolist()
               for i in range(a.batch)]
    # PREREG-k8 G5: the scored window and a digest of it, so the
    # verdict can REFUSE two arms that evaluated different text
    ppl_sha = hashlib.sha256(
        ids[:a.prompt_len + max(a.ppl_steps, 0) + 1].numpy().tobytes()
    ).hexdigest()
    return ids, step, prompts, ids, ppl_sha


def _upstream_oracle_main(a, tok):
    """Positive control for the oracle: the SAME window through the model
    exactly as transformers loads it -- its own expert weights (MXFP4
    dequantised to bf16 where no kernel is installed), its own router,
    its own attention -- with no e4b loader anywhere in the process.
    Where this disagrees with the e4b-loaded oracle, the loader or the
    expert tier owns the gap, not attention."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    devs = sorted({str(d) for d in getattr(model, "hf_device_map", {}).values()}) \
        or [str(next(model.parameters()).device)]
    print(f"UPSTREAM loaded {type(model).__name__} params={n_params:.2f}B "
          f"devices={devs} quant={getattr(model.config, 'quantization_config', None)}",
          flush=True)
    _, _, _, ppl_ids, ppl_sha = _k8_window(a, tok)
    _ppl_oracle_main(a, model, ppl_ids, ppl_sha)


def _attn_compute_ran() -> str:
    """The compute mode the decode attention ACTUALLY used, from the
    kernel's own tally: "fp8", "f32", "fp8+f32 (n/m)" when a window
    mixed them, or "none (no decode attention entered)". Never the
    environment request -- an unset `GNF4_ATTN_COMPUTE` selects
    capability-conditionally, so the request is silent about the event."""
    try:
        import fp8_paged_attn
        counts = fp8_paged_attn.compute_counts() or {}
    except Exception:                                    # noqa: BLE001
        return "unknown (kernel tally unavailable)"
    hot = {k: v for k, v in counts.items() if v}
    if not hot:
        return "none (no decode attention entered)"
    if len(hot) == 1:
        return next(iter(hot))
    total = sum(hot.values())
    parts = ", ".join(f"{k} {v}" for k, v in sorted(hot.items()))
    return f"mixed ({parts}; {total} calls)"


def _kv_geometry(cfg):
    """(kv heads, head_dim) for the paged cache: scalars for a uniform
    model, per-layer lists for a heterogeneous config (transformers 5.16
    exposes Gemma-4's sliding layers at 256/8 and full layers at 512/2
    through ``per_layer_config``). Fp8PagedKV sizes its rows for the
    largest layer and serves each layer in its own geometry."""
    def _one(c):
        heads = getattr(c, "num_key_value_heads")
        hd = getattr(c, "head_dim", None) or (
            c.hidden_size // getattr(c, "num_attention_heads"))
        return int(heads), int(hd)
    try:
        return _one(cfg)
    except Exception as e:                    # AmbiguousGlobalPerLayerAttributeError
        if "per-layer attribute" not in str(e):
            raise
    per = [_one(lc) for lc in cfg.per_layer_config]
    heads, dims = [h for h, _ in per], [d for _, d in per]
    if len(set(heads)) == 1 and len(set(dims)) == 1:
        return heads[0], dims[0]
    print(f"KV geometry varies per layer: {sorted(set(zip(heads, dims)))} "
          f"-- the paged cache sizes rows for the largest and serves each "
          f"layer in its own", flush=True)
    return heads, dims


def _uniform_layer_attr(cfg, key, default=None):
    """A per-layer attribute read once for the whole model. transformers
    5.16 marks heterogeneous configs (Gemma-4: sliding and full attention
    layers) and REFUSES a global read of a per-layer attribute; this
    harness sizes one KV cache for every layer, so it reads the value
    from each layer config and insists they agree (P24-GEN-B: Gemma-4's
    NF4 arms died on ``config.num_key_value_heads``)."""
    try:
        from transformers.integrations.heterogeneity.configuration_utils import (
            AmbiguousGlobalPerLayerAttributeError as _Ambiguous)
    except ImportError:                       # older transformers: no heterogeneity
        _Ambiguous = ()
    try:
        return getattr(cfg, key, default)
    except _Ambiguous:
        pass
    except Exception as e:                    # the base class has moved between 5.x releases
        if "per-layer attribute" not in str(e):
            raise
    vals = {getattr(lc, key, default) for lc in cfg.per_layer_config}
    if len(vals) != 1:
        raise SystemExit(
            f"{key} varies across layers ({sorted(map(str, vals))}): the paged "
            "KV cache and the fp8 decode kernel take ONE geometry for every "
            "layer, so this family cannot be served through the paged path "
            "until per-layer KV geometry exists -- refusing rather than "
            "sizing the cache for the wrong layers")
    return vals.pop()


def _routed_topk(cfg):
    """The routed top-k under whatever name this family's config uses.
    Extend the alias list when onboarding a family, never hardcode a
    key at a call site (docs/hybrid/PORTABILITY.md)."""
    # multimodal configs (gemma4) keep the MoE fields under text_config
    for c in (cfg, getattr(cfg, "text_config", None)):
        for key in ("num_experts_per_tok", "num_experts_per_token",
                    "moe_top_k", "moe_topk", "top_k_experts", "top_k"):
            v = getattr(c, key, None)
            if isinstance(v, int) and v > 0:
                return v
    raise ValueError("cannot find the routed top-k in this config; add "
                     "its key to _routed_topk")
     # set when --compile-layers uses cudagraphs
PER_MODE = {"prefill": {"attn_host_ns": 0, "attn_calls": 0},
            "decode": {"attn_host_ns": 0, "attn_calls": 0}}


HOST_BRACKETS = [False]     # --host-brackets: region walls + record_function
REGION_PROF = {
    "moe": {"prefill": {"ns": 0, "calls": 0}, "decode": {"ns": 0, "calls": 0}},
    "moe_block": {"prefill": {"ns": 0, "calls": 0},
                  "decode": {"ns": 0, "calls": 0}},
    "lmhead": {"prefill": {"ns": 0, "calls": 0}, "decode": {"ns": 0, "calls": 0}},
}


def _wrap_region(fn, key, region_name):
    """Host-wall bracket + a profiler region around one call site
    (T5b Phase A). Region names feed the event-tree op counter; the
    wall feeds the per-step decomposition. Only installed when
    --host-brackets is set, so timing arms never carry the overhead."""
    def timed(*a, **k):
        t0 = time.perf_counter_ns()
        with torch.profiler.record_function(region_name):
            out = fn(*a, **k)
        d = REGION_PROF[key][PROF["mode"]]
        d["ns"] += time.perf_counter_ns() - t0
        d["calls"] += 1
        return out
    return timed


def wrap_attention(impl_name):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS[impl_name]

    def timed(*a, **k):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        if HOST_BRACKETS[0]:
            with torch.profiler.record_function("e4b::attn"):
                t0 = time.perf_counter_ns()
                e0.record()
                out = orig(*a, **k)
                e1.record()
                dt = time.perf_counter_ns() - t0
                PROF["attn_host_ns"] += dt
                PROF["attn_calls"] += 1
                PROF["attn_events"].append((PROF["mode"], e0, e1))
                PER_MODE[PROF["mode"]]["attn_host_ns"] += dt
                PER_MODE[PROF["mode"]]["attn_calls"] += 1
                return out
        t0 = time.perf_counter_ns()
        e0.record()
        out = orig(*a, **k)
        e1.record()
        dt = time.perf_counter_ns() - t0
        PROF["attn_host_ns"] += dt
        PROF["attn_calls"] += 1
        PROF["attn_events"].append((PROF["mode"], e0, e1))
        PER_MODE[PROF["mode"]]["attn_host_ns"] += dt
        PER_MODE[PROF["mode"]]["attn_calls"] += 1
        return out

    timed._orig = orig      # b1d unwraps: timing-event records are
    ALL_ATTENTION_FUNCTIONS[impl_name] = timed   # capture-invalid


def _grouping_parity(a, model, runner, sched, kv):
    """PREREG-bv3b: per-layer device-vs-eager grouping parity on REAL
    decode inputs. Drain the scheduler until B rows decode, record one
    step's per-layer MoE inputs via forward hooks, then replay each
    layer's own forward twice -- DEVICE_GROUPING off, then on -- and
    record max|delta| / max|ref| per layer. The flag flip exercises
    the exact live dispatch both ways on identical tensors."""
    from experts4bit_qlora.engines import hot_residency as _hr
    from experts4bit_qlora.engines.hot_residency import target_modules

    B = a.batch
    # the DEVICE_GROUPING flag is read only inside _forward_collapsed,
    # which runs only on the collapsed all-resident path -- without
    # this bind both replays take the SAME code and every delta is
    # exactly zero: a vacuous probe the defect gate would green-light
    # (review, e4b#269)
    assert a.engine == "hybrid" and a.placement_override == "all-vram", \
        "parity binds to the collapsed all-resident point"
    assert a.amort == "off", \
        "parity binds amort-off: _forward_collapsed requires it too"
    while (sched.active or sched.queue) and (
            len(runner.slot_of) < B
            or any(len(runner.tokens[r]) <= a.prompt_len
                   for r in runner.slot_of)):
        if sched.step().is_empty:
            break
    assert len(runner.slot_of) == B, (len(runner.slot_of), B)
    mods = target_modules(model)
    rec = {}
    hooks = []
    for li, m in enumerate(mods):
        def _mk(li_, m_):
            orig = m_.forward

            def _wrap(hidden, top_k_index, top_k_weights):
                if li_ not in rec:
                    rec[li_] = (hidden.detach().clone(),
                                top_k_index.detach().clone(),
                                top_k_weights.detach().clone())
                return orig(hidden, top_k_index, top_k_weights)
            return orig, _wrap
        o, w = _mk(li, m)
        m.forward = w
        hooks.append((m, o))
    try:
        if sched.step().is_empty:
            raise SystemExit("parity: no live step to record")
    finally:
        for m, o in hooks:
            m.forward = o
    missing = [li for li in range(len(mods)) if li not in rec]
    if missing:
        raise SystemExit(f"parity: layers without recorded inputs: "
                         f"{missing} -- refusing a partial probe")
    for li, m in enumerate(mods):
        st_ = m._hot_residency
        if not (getattr(st_, "collapse_resident", False)
                and st_._all_hot()
                and getattr(st_, "amort", None) is None):
            raise SystemExit(f"parity: layer {li} is not on the "
                             "collapsed all-resident path -- the "
                             "grouping flag would not dispatch and "
                             "the probe would be vacuous")
    out = {}
    prev_dg = _hr.DEVICE_GROUPING[0]
    prev_sg = _hr.FORCE_SINGLETON_GROUPS[0]
    with torch.no_grad():
        for li, m in enumerate(mods):
            hidden, idx, wts = rec[li]
            _hr.DEVICE_GROUPING[0] = False
            _hr.FORCE_SINGLETON_GROUPS[0] = False
            y_e = m.forward(hidden, idx, wts)
            if isinstance(y_e, tuple):
                y_e = y_e[0]
            _hr.DEVICE_GROUPING[0] = True
            y_d = m.forward(hidden, idx, wts)
            if isinstance(y_d, tuple):
                y_d = y_d[0]
            d = (y_e.float() - y_d.float()).abs().max().item()
            r = y_e.float().abs().max().item()
            out[str(li)] = {"max_abs_delta": d, "max_abs_ref": r,
                            "rows": int(hidden.reshape(
                                -1, hidden.shape[-1]).shape[0])}
    _hr.DEVICE_GROUPING[0] = prev_dg
    _hr.FORCE_SINGLETON_GROUPS[0] = prev_sg
    rep = {"parity": out, "batch": B, "layers": len(mods),
           "frame": "max|delta| <= max|ref| * 2^-7 per layer"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    worst = max(out.values(),
                key=lambda c: c["max_abs_delta"] / max(c["max_abs_ref"],
                                                       1e-30))
    print(f"GROUPING_PARITY layers={len(mods)} worst_ratio="
          f"{worst['max_abs_delta']/max(worst['max_abs_ref'],1e-30):.3e}"
          f" out={a.out}", flush=True)


def _bv3_stage(a, model, runner, sched, kv):
    """PREREG-bv3: the B>1 CUDA-graph decode loop. Prefill runs through
    the production scheduler until every row is decoding; the decode
    loop is then driven manually with static [B, 1] buffers, one step
    captured, replayed for the timed window. Per-row token logs land in
    the receipt for the offline identity gate against the eager arm.
    A sibling of the certified b1d lane, deliberately branch-free of
    its S2 probes."""
    from experts4bit_qlora.engines.paged_attention import set_context

    B = a.batch
    assert B > 1, "bv3 is the batched lane; use b1d at --batch 1"
    if a.compile_layers:
        # the scheduler's drain decodes at GROWING batch sizes (1..B)
        # -- B distinct shapes under dynamic=False -- and the default
        # recompile_limit (8) exhausts before the manual loop's [B, 1]
        # shape compiles; dynamo then falls back mid-capture and the
        # graph invalidates (hit live at B=16: capture_end
        # cudaErrorStreamCaptureInvalidated after 'hit
        # config.recompile_limit (8)'). Budget = B shapes + prefill +
        # margin; replays never re-enter python, and the
        # recompiles_in_window guard still refuses live recompiles.
        import torch._dynamo as _dyn
        _dyn.config.recompile_limit = max(2 * B + 16,
                                          _dyn.config.recompile_limit)
        # ... and the GLOBAL cap: 48 compiled layer bodies x the
        # growing shapes exceeds accumulated_recompile_limit's default
        # 256 long before [B,1] compiles (review, e4b#263). The lane
        # tolerates compile churn pre-capture; live recompiles inside
        # the timed window still refuse via recompiles_in_window.
        _dyn.config.accumulated_recompile_limit = max(
            4096, _dyn.config.accumulated_recompile_limit)
    # the registered treatment routes T=B MoE rows through the
    # CAPTURE-SAFE device grouping (PREREG-bv3 item 2; S3 machinery).
    # The default at T>1 is EAGER grouping, whose host-size sync
    # invalidates capture -- hit live on graph_a attempt 2: capture
    # died with NO recompile warning once the dynamo budget was
    # raised, leaving the grouping path as the remaining host sync.
    from experts4bit_qlora.engines import hot_residency as _hr
    _hr.DEVICE_GROUPING[0] = True
    _hr.FORCE_SINGLETON_GROUPS[0] = False
    assert a.engine == "hybrid" and a.placement_override == "all-vram", \
        "bv3 binds to the collapsed all-resident point"
    assert a.amort == "off", \
        "amort-armed runs keep the baseline dispatch path (not capturable)"
    # rids come from slot_of -- decode_rows holds per-step TIMING
    # dicts in TimedRunner, not request ids (Bugbot HIGH, e4b#261;
    # certified b1d reads slot_of the same way). Drain until all B
    # admitted rows have generated at least one decode token.
    while (sched.active or sched.queue) and (
            len(runner.slot_of) < B
            or any(len(runner.tokens[r]) <= a.prompt_len
                   for r in runner.slot_of)):
        if sched.step().is_empty:
            break
    rids = sorted(runner.slot_of)
    assert len(rids) == B, (len(rids), B)
    assert all(len(runner.tokens[r]) > a.prompt_len for r in rids), \
        "a row reached the manual loop without any decoded token"
    slots = [runner.slot_of[r] for r in rids]
    # Same contract as Stage A's replay profiler (e4b#275): post-window
    # profiled replays advance pos/KV, so their slots are reserved HERE
    # and the timed window stays byte-identical to a no-flag run.
    profile_replays = 8 if (a.replay_profile_out and a.b1d_timed) else 0
    cap_tokens = a.prompt_len + a.gen_tokens + 8 + profile_replays
    kv.graph_mode_init_batch(slots, upto_tokens=cap_tokens)
    dev = "cuda"
    in_ids = torch.tensor([[int(runner.tokens[r][-1])] for r in rids],
                          dtype=torch.long, device=dev)
    pos = torch.tensor([[runner.pos_of[r] - 1] for r in rids],
                       dtype=torch.long, device=dev)
    runner.ctx.mode = "decode"
    runner.ctx.slots = slots
    prev = set_context(runner.ctx)

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from experts4bit_qlora.engines.paged_attention import IMPL_NAME
    _cur = ALL_ATTENTION_FUNCTIONS[IMPL_NAME]
    _orig = getattr(_cur, "_orig", None)
    if _orig is not None:
        # same unwrap-then-re-disable dance as b1d: the timing shim's
        # cudaEventRecord is illegal inside capture, but the RAW shim
        # would discard the dynamo.disable compile-layers applied
        # (feedback: disable-wrappers-get-unwrapped)
        # PREREG-k12 arm 3: --compile-attn-tier lifts this exclusion,
        # and it must be lifted HERE too. Skipping only the initial
        # disable leaves this re-wrap to silently restore it on the
        # very loops arm 3 measures, so the arm would run as arm 2
        # while its receipt recorded compile_attn_tier=True
        # (review, e4b#289) -- the disable-wrappers-get-unwrapped trap
        # in mirror image.
        if a.compile_layers and not getattr(a, "compile_attn_tier", False):
            import torch._dynamo as dynamo
            ALL_ATTENTION_FUNCTIONS[IMPL_NAME] = dynamo.disable(_orig)
        else:
            ALL_ATTENTION_FUNCTIONS[IMPL_NAME] = _orig

    def one_step():
        out = model(input_ids=in_ids, position_ids=pos, use_cache=False)
        tok = out.logits[:, -1].argmax(-1)
        in_ids.copy_(tok.reshape(B, 1))
        pos.add_(1)

    n_warm = 3
    base_pos = max(runner.pos_of[r] for r in rids)
    # profile_replays excluded from the budget, mirroring Stage A: when
    # capacity binds, the reserved slots must survive the timed window
    # or the profiled replays walk off the reserved KV (review, round 1).
    n_steps = min(a.gen_tokens,
                  cap_tokens - profile_replays - (base_pos + n_warm) - 2)
    assert n_steps >= 16, f"window too small for bv3 ({n_steps})"
    # per-row decode tokens generated BEFORE the manual loop, so the
    # receipt carries the FULL greedy stream and the verdict can align
    # it against the eager arm from decode step 0 (Bugbot HIGH,
    # e4b#261: window-only logs cannot be identity-compared)
    pre_tokens = {str(i): [int(t) for t in
                           runner.tokens[r][a.prompt_len:]]
                  for i, r in enumerate(rids)}
    warm_log = torch.zeros(n_warm, B, dtype=torch.long, device=dev)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for w in range(n_warm):
            one_step()
            warm_log[w].copy_(in_ids.reshape(B))
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    tok_log = torch.zeros(n_steps, B, dtype=torch.long, device=dev)
    _frames_before = _dynamo_frame_count()
    torch.cuda.empty_cache()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        one_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for i in range(n_steps):
        g.replay()
        tok_log[i].copy_(in_ids.reshape(B))
    torch.cuda.synchronize()
    total_ms = (time.perf_counter_ns() - t0) / 1e6
    if profile_replays:
        # Kernel census of the SHIPPED batched replay -- kineto records
        # kernels inside graph replays. AFTER the timed window, in the
        # slots cap_tokens reserved; CUDA activity only (TR1 capacity
        # lesson); the timed loop above stays clean. Mirrors Stage A.
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as rp:
            for _ in range(profile_replays):
                g.replay()
            torch.cuda.synchronize()
        tbl = rp.key_averages().table(
            sort_by="cuda_time_total", row_limit=120)
        hdr = (f"profiled replay steps: {profile_replays} "
               f"(batched B={B} graph replay)\n")
        Path(a.replay_profile_out).write_text(hdr + tbl)
        print(f"REPLAY_PROFILE_OUT {a.replay_profile_out}", flush=True)
    set_context(prev)
    _frames_after = _dynamo_frame_count()
    step_ms = total_ms / n_steps
    rep = {
        "bv3": True, "batch": B, "n_steps": n_steps,
        "slots": slots,
        "tokens": {str(i): (pre_tokens[str(i)]
                            + warm_log[:, i].cpu().tolist()
                            + tok_log[:, i].cpu().tolist())
                   for i in range(B)},
        "pre_len": {k: len(v) for k, v in pre_tokens.items()},
        "warm_steps": n_warm,
        "step_ms_clean": step_ms,
        "window_ms": total_ms,
        "aggregate_tok_s": 1000.0 / step_ms * B,
        "compile_layers": bool(a.compile_layers),
        "compile_moe_tier": bool(a.compile_moe_tier),
        "compile_attn_tier": bool(a.compile_attn_tier),
        "compile_mode": a.compile_mode if a.compile_layers else None,
        "fuse_qkv": bool(a.fuse_qkv),
        "device_grouping": True,
        "dyn_limits_env": [os.environ.get("E4B_RECOMPILE_LIMIT"),
                           os.environ.get("E4B_ACCUM_RECOMPILE_LIMIT")],
        "recompile_limit": (None if not a.compile_layers else
                            __import__("torch._dynamo", fromlist=["config"])
                            .config.recompile_limit),
        "dynamo_frames_before": _frames_before,
        "dynamo_frames_after": _frames_after,
        "recompiles_in_window": (
            _frames_after["total"] - _frames_before["total"]
            if _frames_before and _frames_after else None),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"BV3_GRAPH batch={B} steps={n_steps} "
          f"step={step_ms:.2f}ms agg={rep['aggregate_tok_s']:.1f}tok/s "
          f"recompiles={rep['recompiles_in_window']} out={a.out}",
          flush=True)



def _capturing() -> bool:
    """True while the current stream is capturing a CUDA graph."""
    try:
        return bool(torch.cuda.is_available()
                    and torch.cuda.is_current_stream_capturing())
    except Exception:                       # no CUDA build, or no stream
        return False


class _TopkProbe:
    """PREREG-k10 Stage A: name the owner of the census `router` row.

    A CAPTURED REPLAY carries no Python frames, so a stack-attributed
    profile cannot say who launched `sbtopk::gatherTopK` /
    `bitonicSortKVInPlace`. The decisive test is a counted ablation
    instead: wrap `torch.topk`, count its calls and record the shapes
    it sees, and optionally force `sorted=False`. If the kernels are
    this call site, forcing sorted=False must drive
    `bitonicSortKVInPlace` to ZERO in a profiled window -- and if it
    does not, the attribution is wrong and Stage A refuses rather than
    handing a treatment an unidentified cost (K9 died twice that way).

    `sorted_override=False` changes the ORDER of the selected experts,
    not the set; downstream `w / w.sum()` then sums the same k floats
    in a different order, so it is numerics-changing and gated as such
    by Stage B1 -- never shipped from this probe.
    """

    def __init__(self, sorted_override=None):
        self.sorted_override = sorted_override
        self.calls = 0
        self.shapes = {}
        self._orig = None
        self.window_steps = None
        # B1-C is a REFUSE gate on the SELECTED SET, so measure the set
        # -- not a proxy. Token streams are neither necessary nor
        # sufficient: reordered weights can flip a token without
        # changing the set, and an unchanged set can still shift a
        # token through fp re-association in w/w.sum(). Digest the
        # SORTED indices of every call so two arms compare exactly.
        self._set_digest = hashlib.sha256()

    def begin_window(self, steps):
        """Zero the counters and declare how many STEPS the window is.

        The probe wraps the whole stage, so scheduled prefill and warm
        decode would otherwise fold into counts divided by the
        window's step count -- breaking the one-topk-per-layer
        attribution and letting B1-C refuse two arms whose scored sets
        agree but whose warm tokens diverged (Bugbot, e4b#280 High).

        The graph arm declares ONE step and opens its window around the
        CAPTURE, not the replay loop: `g.replay()` never enters Python,
        so a window around the replays would count exactly nothing.

        """
        self.calls = 0
        self.shapes = {}
        self.window_steps = steps
        self._set_digest = hashlib.sha256()

    def __enter__(self):
        self._orig = torch.topk

        def _wrapped(input, k, dim=-1, largest=True, sorted=True,
                     *a, **kw):
            self.calls += 1
            key = f"{tuple(input.shape)}|k={k}"
            self.shapes[key] = self.shapes.get(key, 0) + 1
            if self.sorted_override is not None:
                sorted = self.sorted_override
            out = self._orig(input, k, dim=dim, largest=largest,
                             sorted=sorted, *a, **kw)
            idx = getattr(out, "indices", None)
            # A D2H copy is ILLEGAL inside stream capture, and the
            # ablation arms capture. Skip the digest there; B1-C is
            # adjudicated on the EAGER ppl arms, which is where the
            # scored sets actually live (Bugbot, e4b#280 High).
            if idx is not None and not _capturing():
                # sort so the digest is order-INVARIANT: the whole
                # point is that sorted=False permutes, and B1 asks
                # whether the SET survived that permutation
                v = idx.detach().to("cpu", torch.int64).sort(-1).values
                self._set_digest.update(v.numpy().tobytes())
            return out

        torch.topk = _wrapped
        return self

    def __exit__(self, *exc):
        torch.topk = self._orig
        return False

    def report(self, layers):
        steps = self.window_steps
        if not steps:
            raise RuntimeError(
                "report() before begin_window(): the counts would "
                "include prefill and warm decode and be divided by a "
                "step count that never applied to them")
        return {"topk_calls": self.calls,
                "steps": steps, "layers": layers,
                "calls_per_step": self.calls / max(steps, 1),
                "calls_per_step_per_layer": (self.calls
                                             / max(steps * layers, 1)),
                "shapes": self.shapes,
                "selected_set_digest": self._set_digest.hexdigest(),
                "sorted_override": self.sorted_override}


_BREAK_FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
#: dynamo prefixes each break record; the reason is the text after it
_BREAK_HEAD = re.compile(r"(Graph break[^\n]*|Skip calling[^\n]*)")


def _graph_break_census(one_step, steps: int = 3) -> dict:
    """PREREG-k13 Stage A: name the graph breaks in the REAL compile.

    K12 established that compiling the MoE tier costs 0.672 ms and
    adds ~670 launches/step because dynamo cannot trace it as one
    graph. It did NOT establish WHICH host read breaks it -- a frame
    stack is not an attribution.

    **This traces the actual compilation, not a wrapper around it.**
    The first version called `torch._dynamo.explain(one_step)()`, and
    on the box that named exactly one break --
    `qkv_fuse.py:60 "Skip calling torch.compiler.disable()d function"`
    -- IDENTICALLY in both cells, with the MoE tier absent entirely.
    `explain` traced the OUTER step, saw the deliberate disable
    boundaries, and never descended into the layer bodies that
    `--compile-layers` had already wrapped in `torch.compile`, which
    is where the MoE work lives. It was pointed one level too high,
    and 2 counter reasons for a step with 48 MoE layers was the tell.
    (`break_reasons` also came back EMPTY on the box while non-empty
    in CPU probes, so it is not relied on at all now.)

    So: install a capture on dynamo's logger, reset the counters, and
    run the step FOR REAL. The layer bodies compile on that first
    call and every break they hit is recorded with its own reason and
    its own stack -- from the same code path K12 measured.

    Each dynamo record carries the reason AND the frames, so they are
    parsed together per record. The previous version matched counter
    reasons to separately-captured sites by word overlap, which with
    1016 captured sites collapsed every reason onto one location.
    """
    import logging

    import torch._dynamo as _dyn
    from torch._dynamo.utils import counters

    seen: dict = {}

    def _record(msg):
        frames = _BREAK_FRAME.findall(msg)
        head = _BREAK_HEAD.search(msg)
        if not head:
            return
        reason = " ".join(head.group(1).split())[:240]
        if frames:
            fn, ln, func = frames[-1]
            stack = [f"{a}:{b} in {c}" for a, b, c in frames][-6:]
        else:
            fn, ln, func, stack = "<no-frame>", -1, "<no-frame>", []
        key = (fn, int(ln), func, reason)
        if key in seen:
            seen[key]["count"] += 1
        else:
            seen[key] = {"file": fn, "line": int(ln), "func": func,
                         "reason": reason, "count": 1, "stack": stack}

    class _Cap(logging.Handler):
        def emit(self, rec):
            try:
                _record(rec.getMessage())
            except Exception:                    # noqa: BLE001
                pass

    handler = _Cap()
    dynlog = logging.getLogger("torch._dynamo")
    prev_level = dynlog.level
    dynlog.addHandler(handler)
    dynlog.setLevel(logging.DEBUG)

    # STOP THE STACK SUPPRESSION. Run 3 of K13 came back with records
    # reading "Graph break (user stack suppressed due to duplicate
    # graph break)" -- a reason with NO file/line. The only LOCATED
    # break was qkv_fuse.py:60, the deliberate disable boundary,
    # present identically in both cells, so k13_verdict refused with
    # "no break inside the MoE tier's frames". The MoE break is
    # plausibly among the suppressed ones.
    #
    # This is dynamo describing its own behaviour, not a frame stack
    # being read as an attribution -- which is what the three earlier
    # (and all refuted) explanations were.
    _prev_verbose = getattr(_dyn.config, "verbose", None)
    try:
        _dyn.config.verbose = True
    except Exception:                            # noqa: BLE001
        _prev_verbose = None
    try:
        import torch._logging as _tlog
        _tlog.set_logs(graph_breaks=True)
    except Exception:                            # noqa: BLE001
        pass
    _dyn.reset()
    counters["graph_break"].clear()
    # The REAL compile. Its own try: a failure here means there is no
    # census at all.
    try:
        one_step()                               # layer bodies compile here
        first_counts = dict(counters["graph_break"])
    except Exception as ex:                      # noqa: BLE001
        dynlog.removeHandler(handler)
        dynlog.setLevel(prev_level)
        if _prev_verbose is not None:
            try:
                _dyn.config.verbose = _prev_verbose
            except Exception:                    # noqa: BLE001
                pass
        return {"error": f"{type(ex).__name__}: {ex}"[:300], "breaks": []}

    # PHASE, per break, by COUNT rather than by set membership.
    # `set(seen) - trace_keys` labelled every site seen during compile
    # as "trace" forever, even one that re-fires on every step -- the
    # receipt then could not tell compile-only from keeps-firing
    # (review, e4b#292). Snapshot each key's count, re-run, and a key
    # whose count GREW re-fired.
    # NOT COVERED BY ANY TEST. Five fixtures were tried and each
    # passed under a mutation that broke this classification, so no
    # green check here means it is right. What protects a verdict is
    # that PREREG-k13 RECORDS phase and never scores it, and that
    # k13_verdict refuses a blank one -- a misclassification cannot
    # change a verdict, only mislead a reader.
    at_compile = {k: v["count"] for k, v in seen.items()}
    before_n = len(seen)
    again, phase_err = None, None
    # Its own try: losing an already-captured census to a failure in
    # the PHASE probe is the worse outcome. k13_verdict REFUSES an
    # unknown phase, which is right; the receipt says why.
    try:
        counters["graph_break"].clear()
        for _ in range(max(1, steps)):
            one_step()
        again = sum(counters["graph_break"].values())
    except Exception as ex:                      # noqa: BLE001
        phase_err = f"{type(ex).__name__}: {ex}"[:200]
    finally:
        dynlog.removeHandler(handler)
        dynlog.setLevel(prev_level)
        if _prev_verbose is not None:
            try:
                _dyn.config.verbose = _prev_verbose
            except Exception:                    # noqa: BLE001
                pass

    breaks = []
    for key, b in seen.items():
        b = dict(b)
        if phase_err is not None:
            b["phase"] = None                    # unknown, and said so
        else:
            # DEFENSIVE: measurement shows a recompile logs under a
            # NEW key rather than incrementing an old one, so this
            # clause is not exercised in practice. Kept because the
            # review's concern is sound if that ever changes; the
            # test does not claim to cover it.
            grew = b["count"] > at_compile.get(key, 0)
            b["phase"] = "step" if (grew or key not in at_compile) else "trace"
        b["count_at_compile"] = at_compile.get(key, 0)
        breaks.append(b)
    breaks.sort(key=lambda b: -b["count"])
    return {"breaks": breaks,
            "source": "dynamo-log/real-compile",
            "counter_reasons_first": {str(k)[:120]: v
                                      for k, v in first_counts.items()},
            "counter_total_first": sum(first_counts.values()),
            "counter_total_recompiled": again,
            "records_seen": before_n,
            # how many records arrived WITHOUT a location: if this is
            # still nonzero the suppression is not fully lifted and
            # the census is still partly blind
            "no_frame_records": sum(1 for b in seen.values()
                                    if b["file"] == "<no-frame>"),
            "verbose_was_set": _prev_verbose is not None,
            "phase_error": phase_err,
            "phase_basis": "the step is run FOR REAL so the layer "
                           "bodies compile under the capture; then "
                           "re-run warm -- a break recorded only in "
                           "the first pass fired at compile, which "
                           "still shapes every graph replay",
            "error": None}


def _mech_reset():
    """Zero gnf4's dispatch tallies (PREREG-m3 mechanism receipt).

    M3's four arms are selected by env vars, and an env var is a
    REQUEST: GNF4_GEMV_DOTPAD=1 engages dot-pad only if the shape is
    registered AND the part carries >= 160 SMs. An arm whose knob was
    silently ignored matches OFF in step time and -- K6-B measured
    dot-pad token-IDENTICAL at 127 tokens -- in perplexity too, so
    nothing downstream would notice. These tallies record what
    actually dispatched.

    Soft by design: an older gnf4 without the counters yields a None
    receipt, and m3_verdict REFUSES on a missing receipt rather than
    assuming one ([[presence-is-not-usability]]).
    """
    try:
        import fp8_paged_attn
        import nf4_grouped
        # Probe the READERS too, not just the resetters. A gnf4 with
        # one half of the pair would let this return True while
        # _mech_report returns None -- "reset succeeded, receipts
        # unavailable" is the kind of half-truth that surfaces as a
        # confusing REFUSE hours later on a rented box.
        nf4_grouped.dispatch_counts, fp8_paged_attn.compute_counts
        nf4_grouped.reset_dispatch_counts()
        fp8_paged_attn.reset_compute_counts()
        return True
    except (ImportError, AttributeError):
        return False


def _mech_report():
    """Read the tallies back. See _mech_reset for what they mean.

    Under CUDA-graph capture these increment ONCE, at capture:
    replays never re-enter Python. That is the right semantics --
    what was captured is what every replay goes on to execute -- so
    the window deliberately spans warmup AND capture. A window around
    the timed replays alone would read all zeros and look like a
    knob that never engaged.
    """
    try:
        import fp8_paged_attn
        import nf4_grouped
        return {"dispatch": nf4_grouped.dispatch_counts(),
                "compute": fp8_paged_attn.compute_counts(),
                "window": "warmup+capture (graph) or the whole loop "
                          "(eager); graph replays do not re-enter "
                          "Python and cannot increment"}
    except (ImportError, AttributeError):
        return None


def _b1d_stage_a(a, model, runner, sched, kv, ppl_ids=None,
                 ppl_sha=None, probe=None):
    """PREREG-b1d stage A harness: capture smoke + bitwise replay
    identity. Prefill runs through the normal scheduled path; the decode
    loop is then driven manually with static buffers so the 'graph' arm
    can capture ONE step and replay it. Both arms record per-step logits
    hashes and tokens -- equality across arms is the gate; a baked write
    offset (the e4b#227 finding) cannot pass it, because a same-slot KV
    overwrite diverges the continuation within a few steps."""
    import hashlib

    from experts4bit_qlora.engines.paged_attention import set_context

    assert a.batch == 1, "b1d is the single-stream lane"
    assert a.engine == "hybrid" and a.placement_override == "all-vram", \
        "b1d binds to the collapsed all-resident point"
    assert a.amort == "off", \
        "amort-armed runs keep the baseline dispatch path (not capturable)"
    # prefill + a short scheduled warm through the production path
    while (sched.active or sched.queue) and len(runner.decode_rows) < 2:
        if sched.step().is_empty:
            break
    assert len(runner.decode_rows) >= 2, "scheduled warm did not decode"
    rid = next(iter(runner.slot_of))
    slot = runner.slot_of[rid]
    # The replay profiler (post-window, below) advances pos/KV by this
    # many extra appends; reserve their slots here and keep them OUT of
    # the n_steps budget so the timed window is byte-identical to a
    # no-flag run (Bugbot, e4b#275).
    profile_replays = 8 if (a.replay_profile_out and a.b1d_timed
                            and a.b1d_loop == "graph") else 0
    # --ppl-steps scores far past --gen-tokens (1024 vs 128 by
    # default), so capacity must cover whichever window will actually
    # run or the appends index past the pre-ensured blocks -- the same
    # overflow class Bugbot caught on the replay profiler (e4b#275).
    cap_tokens = (a.prompt_len + max(a.gen_tokens, a.ppl_steps) + 8
                  + profile_replays)
    kv.graph_mode_init(seq=slot, upto_tokens=cap_tokens)
    dev = "cuda"
    start_tok = int(runner.tokens[rid][-1])
    in_ids = torch.tensor([[start_tok]], dtype=torch.long,
                          device=dev)
    pos = torch.tensor([[runner.pos_of[rid] - 1]], dtype=torch.long,
                       device=dev)
    runner.ctx.mode = "decode"
    runner.ctx.slots = [slot]
    # Drop the attention timing wrapper for BOTH arms (symmetry): it
    # records torch.cuda.Event(enable_timing=True) per call, and timing
    # cudaEventRecord is ILLEGAL inside stream capture
    # (cudaErrorStreamCaptureInvalidated on the first attention call --
    # hit live on the first stage-A box run). The manual loop needs no
    # attention bracket; the un-wrapped shim is the production path.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from experts4bit_qlora.engines.paged_attention import IMPL_NAME
    _cur = ALL_ATTENTION_FUNCTIONS[IMPL_NAME]
    _orig = getattr(_cur, "_orig", None)
    if _orig is not None:
        # `_orig` is the RAW shim: unwrapping here discards the
        # `dynamo.disable` that --compile-layers applied, so under
        # compile the paged-attention path stops being excluded and
        # inductor re-emits our own fp8 paged-decode triton kernel
        # through its user-kernel path -- where it dies on a
        # loop-carried `m_i` typed fp32 then fp64. PREREG-f1-stageB's
        # B1 arm is defined as "compile owns only the dense layer
        # body", so the disable is re-applied rather than the arm
        # re-scoped (e4b F1 Stage B, first attempt).
        # PREREG-k12 arm 3: --compile-attn-tier deliberately lifts
        # this exclusion. It must be honoured HERE -- this is the b1d
        # graph loop the K12 arms actually run, so re-disabling here
        # would make arm 3 identical to arm 2 while claiming otherwise
        # (review, e4b#289).
        if (getattr(a, "compile_layers", False)
                and not getattr(a, "compile_attn_tier", False)):
            import torch._dynamo as _dyn
            _orig = _dyn.disable(_orig)
        ALL_ATTENTION_FUNCTIONS[IMPL_NAME] = _orig
    prev = set_context(runner.ctx)

    def one_step():
        out = model(input_ids=in_ids, position_ids=pos, use_cache=False)
        lg = out.logits[:, -1]
        tok = lg.argmax(-1)
        in_ids.copy_(tok.reshape(1, 1))
        pos.add_(1)
        return lg

    if a.ppl_steps:
        # PREREG-k8 quality instrument: perplexity measured THROUGH the
        # paged DECODE path, which is the only place the attention
        # compute mode is exercised at all. A teacher-forced forward
        # with use_cache=False -- the obvious way to get perplexity --
        # never calls fp8_paged_decode_attention, so it would return
        # bit-identical numbers for both arms and "prove" quality by
        # construction ([[check-the-result-could-have-failed]]).
        #
        # Every step is TEACHER-FORCED from the corpus, and the KV is
        # first rewound to the prompt boundary to discard the
        # scheduler's warm tokens: those are MODEL-generated, so the
        # two arms could otherwise carry different context and the
        # comparison would not be apples-to-apples. After the rewind
        # both arms see byte-identical context at every step and the
        # ONLY difference is the attention arithmetic.
        assert ppl_ids is not None and ppl_sha is not None, \
            "--ppl-steps needs the corpus threaded in from main()"
        base = runner.pos_of[rid]                 # prompt boundary
        kv.rewind_nosync(slot, a.prompt_len)
        torch.cuda.synchronize()
        route = getattr(a, "ppl_route", "none")
        if route == "replay":
            rec, meta = _route_load(a.ppl_route_file)
            if meta.get("text_sha") and meta["text_sha"] != ppl_sha:
                raise SystemExit(f"routing record is for window {meta['text_sha'][:12]}, "
                                 f"this window is {ppl_sha[:12]}")
            n_armed = _route_install(model, "replay", rec)
            # the prompt's rows were routed by the prefill on THIS path; the
            # scored rows start at the prompt boundary, matching the record
            for layer in rec:
                _ROUTE["consumed"][layer] = a.prompt_len
            _ROUTE["served"] = 0
            _ROUTE["passed"] = 0
            print(f"ROUTE replay armed on {n_armed} routers from position "
                  f"{a.prompt_len} ({len(rec)} recorded layers)", flush=True)
        elif route == "record":
            raise SystemExit("--ppl-route record is for --ppl-oracle full")
        cont = ppl_ids[a.prompt_len:a.prompt_len + a.ppl_steps + 1]
        assert cont.numel() >= a.ppl_steps + 1, (
            f"corpus slice holds {cont.numel()} ids but --ppl-steps "
            f"{a.ppl_steps} needs {a.ppl_steps + 1}; widen "
            "--prompt-span")
        if probe is not None:
            probe.begin_window(a.ppl_steps)
        _mech_reset()
        in_ids.fill_(int(cont[0]))
        # `pos` is the 0-BASED INDEX of the token sitting in in_ids,
        # which this forward is about to append (the harness's own
        # convention: pos = pos_of[rid] - 1 for the last produced
        # token). After the rewind the KV holds indices
        # 0..prompt_len-1, so cont[0] == ids[prompt_len] belongs at
        # prompt_len -- NOT prompt_len - 1, which would run every
        # scored step with RoPE shifted against the stored keys
        # (Bugbot, e4b#278, High).
        pos.fill_(a.prompt_len)
        nll = 0.0
        diff_steps = {int(x) for x in str(getattr(a, "ppl_layer_diff", "") or "").split(",") if x.strip()}
        for t in range(a.ppl_steps):
            cap_paged = {}
            hooks = _layer_diff_install(model, cap_paged) if t in diff_steps else []
            out = model(input_ids=in_ids, position_ids=pos,
                        use_cache=False)
            if hooks:
                for h in hooks:
                    h.remove()
                cap_ref = {}
                _layer_diff_reference(model, ppl_ids[:a.prompt_len + t + 1], cap_ref)
                _layer_diff_report(t, a.prompt_len + t, cap_paged, cap_ref, model, a.out)
            lg = out.logits[:, -1].float()
            nxt = int(cont[t + 1])
            nll += -torch.log_softmax(lg, -1)[0, nxt].item()
            in_ids.fill_(nxt)
            pos.add_(1)
        set_context(prev)
        mean_nll = nll / a.ppl_steps
        rep = {"k8": "ppl", "steps": a.ppl_steps,
               "mean_nll": mean_nll, "ppl": math.exp(mean_nll),
               "prompt_len": a.prompt_len,
               "prompt_offset": a.prompt_offset,
               "ppl_source": a.ppl_source,
               "tokens_scored": a.ppl_steps,
               "text_sha": ppl_sha,
               # os.environ records the REQUEST; mech records the
               # event. PREREG-m3 gates on the latter -- so carry BOTH,
               # and never let the request stand in for the event: every
               # K8 line printed "compute=f32" while the mech tally said
               # fp8 245760 / f32 0, which is how an arm gets
               # mislabelled (P25: it briefly looked as though Gemma-4
               # had been scored through the f32 kernel modes that fail
               # their reference on torch 2.8 / triton 3.4, gnf4#319).
               "attn_compute_requested": os.environ.get("GNF4_ATTN_COMPUTE"),
               "attn_compute": _attn_compute_ran(),
               "mech": _mech_report(),
               "warm_tokens_discarded": base - a.prompt_len,
               "route": (None if getattr(a, "ppl_route", "none") == "none" else
                         {"mode": a.ppl_route, "file": a.ppl_route_file,
                          "rows_served": _ROUTE["served"],
                          "rows_passed": _ROUTE["passed"],
                          "layers": len(_ROUTE["store"])}),
               "router_probe": (probe.report(kv.L)
                                if probe else None),
               "basis": "teacher-forced through the paged decode path "
                        "after rewinding the scheduler's warm tokens; "
                        "identical context in both arms by "
                        "construction (PREREG-k8)"}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
        route_tag = ""
        if rep["route"]:
            route_tag = (f" route=replay served={rep['route']['rows_served']}"
                         f" passed={rep['route']['rows_passed']}")
            _route_clear()
        print(f"K8_PPL steps={a.ppl_steps} nll={mean_nll:.5f} "
              f"ppl={rep['ppl']:.5f} compute={rep['attn_compute']} "
              f"sha={ppl_sha[:12]}{route_tag} out={a.out}", flush=True)
        return

    n_warm = 3
    used = runner.pos_of[rid] + n_warm + 1
    n_steps = min(a.gen_tokens, cap_tokens - profile_replays - used - 2)
    assert n_steps >= 16, f"window too small for stage A ({n_steps})"
    _mech_reset()   # PREREG-m3: window spans warmup + capture
    if getattr(a, "graph_break_census", False):
        _census = _graph_break_census(one_step)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"k13": "graph_break_census",
             "compile_moe_tier": bool(a.compile_moe_tier),
             "compile_attn_tier": bool(a.compile_attn_tier),
             "placement_override": a.placement_override,
             "mech": _mech_report(), **_census}, indent=1))
        n = len(_census.get("breaks") or [])
        print(f"K13_CENSUS breaks={n} "
              f"graph_break_count={_census.get('graph_break_count')} "
              f"err={_census.get('error')} out={a.out}", flush=True)
        set_context(prev)
        return
    # The documented capture recipe: warm on a SIDE stream. cuBLAS/cuDNN
    # bind workspaces per stream, and a first-use allocation landing
    # inside capture invalidates it (cudaErrorStreamCaptureInvalidated
    # at capture_end with no python-level site -- hit live, both before
    # and after the timing-event unwrap; this was the remaining cause).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(n_warm):
            one_step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    if a.s2_verify:
        kk = a.s2_k + 1
        # DEVICE length: warm ran through append_graph_t1, which
        # advances seq_lens only -- _seen is stale here (Bugbot,
        # e4b#241). Same reason base_tok is read from in_ids NOW
        # rather than from the pre-warm token.
        base_seen = kv.seen_device(0, slot)
        base_pos = int(pos.reshape(()).item())
        base_tok = int(in_ids.reshape(()).item())
        # capacity: the oracle runs kk + 4 steps and the verify step
        # re-appends kk, so the window must fit twice over plus slack
        room = cap_tokens - base_seen
        if room < 2 * kk + 8:
            raise SystemExit(
                f"S2 window needs {2 * kk + 8} tokens of paged capacity "
                f"but only {room} remain (cap {cap_tokens}, at "
                f"{base_seen}) -- raise --gen-tokens/--prompt-len or "
                f"lower --s2-k rather than silently overflowing")
        # ---- kk sequential greedy steps (the oracle window) plus 4
        # continuation steps (the post-rewind identity check's expected
        # value -- captured HERE, in the sequential world, before any
        # rewind, so the comparison is between two real worlds)
        seq_toks = []
        for _ in range(kk + 4):
            one_step()
            seq_toks.append(int(in_ids.reshape(()).item()))
        torch.cuda.synchronize()
        cont_expected, seq_toks = seq_toks[kk:], seq_toks[:kk]
        seq_end_seen = kv.seen_device(0, slot)
        # ---- rewind and run ONE verify step over the same window
        kv.rewind(slot, base_seen)
        torch.cuda.synchronize()
        # row 0 is fed the token the oracle window STARTED from, which
        # is the post-warm token, not the pre-warm start_tok
        v_in = torch.tensor([[base_tok] + seq_toks[:-1]], device=dev)
        v_pos = (torch.arange(kk, device=dev) + base_pos).view(1, kk)
        runner.ctx.mode = "verify"
        out = model(input_ids=v_in, position_ids=v_pos, use_cache=False)
        ver_toks = out.logits[0].argmax(-1).tolist()
        torch.cuda.synchronize()
        if a.s2_verify == "parity":
            # S3 gate 2: numeric parity, eager-grouped vs
            # device-grouped, identical routing and identical draft.
            # Both run EAGER here (parity is about the grouping math,
            # not capture); the sequential oracle above already fixed
            # the window.
            from experts4bit_qlora.engines import hot_residency as _hr
            outs = {}
            for mode in ("eager", "device"):
                kv.rewind(slot, base_seen)
                torch.cuda.synchronize()
                _hr.DEVICE_GROUPING[0] = (mode == "device")
                runner.ctx.mode = "verify"
                o = model(input_ids=v_in, position_ids=v_pos,
                          use_cache=False)
                outs[mode] = (o.logits[0].float().cpu(),
                              o.logits[0].argmax(-1).cpu())
                torch.cuda.synchronize()
            _hr.DEVICE_GROUPING[0] = False
            set_context(prev)
            lg_e, tk_e = outs["eager"]
            lg_d, tk_d = outs["device"]
            maxd = (lg_e - lg_d).abs().max().item()
            same = bool(torch.equal(tk_e, tk_d))
            rep = {"s2": "parity", "k": a.s2_k,
                   "max_abs_logit_delta": maxd,
                   "verify_tokens_identical": same,
                   "eager_tokens": tk_e.tolist(),
                   "device_tokens": tk_d.tolist(),
                   "pass": same,
                   "note": "identical greedy verify tokens REQUIRED; "
                           "logit delta recorded (grouped-vs-grouped "
                           "fp order differs across tile shapes, so "
                           "bitwise logits are not claimed)"}
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rep, indent=1))
            print(f"S3_PARITY k={a.s2_k} max|d|={maxd:.3e} "
                  f"tokens_identical={same} PASS={same} out={a.out}",
                  flush=True)
            return
        if a.s2_verify == "gate":
            match = [int(x) == int(y) for x, y in zip(ver_toks, seq_toks)]
            # after rewinding the REJECTED tail (here: keep all), the
            # T=1 continuation must be unchanged -- run 4 more steps in
            # both worlds and compare
            runner.ctx.mode = "decode"
            in_ids.copy_(torch.tensor([[seq_toks[-1]]], device=dev))
            pos.fill_(base_pos + kk)
            cont_a = []
            for _ in range(4):
                one_step()
                cont_a.append(int(in_ids.reshape(()).item()))
            cont_ok = cont_a == cont_expected
            rep = {"s2": "gate", "k": a.s2_k,
                   "base_seen_device": base_seen,
                   "seq_end_seen_device": seq_end_seen,
                   "sequential_tokens": seq_toks,
                   "verify_tokens": ver_toks,
                   "rows_matching": sum(match), "rows_total": kk,
                   "bitwise_identical": all(match),
                   "continuation_expected": cont_expected,
                   "continuation_after_verify": cont_a,
                   "continuation_identical": cont_ok,
                   "gate_pass": all(match) and cont_ok}
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rep, indent=1))
            print(f"S2_GATE k={a.s2_k} match={sum(match)}/{kk} "
                  f"bitwise={all(match)} continuation={cont_ok} "
                  f"PASS={rep['gate_pass']} out={a.out}", flush=True)
            set_context(prev)
            return
        if a.s2_verify == "time" and a.moe_grouping == "eager":
            # the eager-grouped arm of the S3 three-way: this grouping
            # CANNOT capture (unique_consecutive host-size sync), so it
            # is timed as an eager loop -- disclosed basis, comparable
            # as the no-capture grouped reference (Bugbot, e4b#245: the
            # first draft documented this loop and then captured
            # anyway, so the arm failed instead of producing a time;
            # the second draft referenced _hr before importing it and
            # died at cleanup AFTER timing, never writing the receipt)
            from experts4bit_qlora.engines import hot_residency as _hr
            _hr.FORCE_SINGLETON_GROUPS[0] = False
            _hr.DEVICE_GROUPING[0] = False
            kv.rewind(slot, base_seen)
            torch.cuda.synchronize()
            for _ in range(8):
                kv.rewind_nosync(slot, base_seen)
                out = model(input_ids=v_in, position_ids=v_pos,
                            use_cache=False)
                _ = out.logits[0].argmax(-1)
            torch.cuda.synchronize()
            spans = []
            for _c in range(10):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                for _r in range(4):
                    kv.rewind_nosync(slot, base_seen)
                    out = model(input_ids=v_in, position_ids=v_pos,
                                use_cache=False)
                    _ = out.logits[0].argmax(-1)
                e1.record()
                e1.synchronize()
                spans.append(e0.elapsed_time(e1) / 4)
            spans.sort()
            _hr.FORCE_SINGLETON_GROUPS[0] = False
            _hr.DEVICE_GROUPING[0] = False
            set_context(prev)
            rep = {"s2": "time", "k": a.s2_k, "rows_per_step": kk,
                   "moe_grouping": "eager",
                   "timing_basis": "eager loop (not capturable; "
                                   "includes host submission)",
                   "verify_graph_ms": spans[len(spans) // 2],
                   "verify_ms_all_chunks": spans,
                   "past_tokens": base_seen}
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rep, indent=1))
            print(f"S2_TIME k={a.s2_k} rows={kk} grouping=eager "
                  f"verify_eager_ms={rep['verify_graph_ms']:.3f} "
                  f"out={a.out}", flush=True)
            return
        # ---- time: graph the verify step at FIXED position.
        # The documented capture recipe (b1d): warm the EXACT call on a
        # SIDE stream first -- the T=K+1 shapes trigger triton JIT
        # compilation and first-use cuBLAS workspaces, and either
        # landing inside capture invalidates it
        # (cudaErrorStreamCaptureInvalidated; hit live on the first
        # Stage A run -- the gate arm warms its own process, the timing
        # arm must warm its own).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        # grouping per PREREG-s3-grouped-verify: singleton keeps the
        # S2 bound's semantics; device enables the capture-safe grouped
        # path; eager leaves the normal unique_consecutive path in
        # place, which is NOT capturable -- its timing runs eagerly
        from experts4bit_qlora.engines import hot_residency as _hr
        if a.moe_grouping == "device":
            _hr.DEVICE_GROUPING[0] = True
        elif a.moe_grouping == "singleton":
            _hr.FORCE_SINGLETON_GROUPS[0] = True
        # one CHECKED rewind establishes that base_seen is not forward
        # of the true length; everything after uses the sync-free
        # variant, since .item() is illegal under capture
        kv.rewind(slot, base_seen)
        with torch.cuda.stream(side):
            for _ in range(2):
                kv.rewind_nosync(slot, base_seen)
                out = model(input_ids=v_in, position_ids=v_pos,
                            use_cache=False)
                _ = out.logits[0].argmax(-1)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        g = torch.cuda.CUDAGraph()
        kv.rewind_nosync(slot, base_seen)
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            out = model(input_ids=v_in, position_ids=v_pos,
                        use_cache=False)
            _ = out.logits[0].argmax(-1)
        torch.cuda.synchronize()
        spans = []
        for _c in range(10):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _r in range(8):
                kv.rewind_nosync(slot, base_seen)
                g.replay()
            e1.record()
            e1.synchronize()
            spans.append(e0.elapsed_time(e1) / 8)
        spans.sort()
        _hr.FORCE_SINGLETON_GROUPS[0] = False
        _hr.DEVICE_GROUPING[0] = False
        set_context(prev)
        rep = {"s2": "time", "k": a.s2_k, "rows_per_step": kk,
               "moe_grouping": a.moe_grouping,
               "timing_basis": "graph replay",
               "verify_graph_ms": spans[len(spans) // 2],
               "verify_ms_all_chunks": spans,
               "past_tokens": base_seen,
               "basis": "graphed fixed-position verify step; the "
                        "inter-replay kv.rewind (one fill_ per layer, "
                        "~48 launches) is INSIDE the timed span and "
                        "inflates verify_ms slightly -- conservative "
                        "side (PREREG-s2lite Stage A; executor needs "
                        "device-addressed appends -- Stage B)"}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
        print(f"S2_TIME k={a.s2_k} rows={kk} "
              f"verify_graph_ms={rep['verify_graph_ms']:.3f} "
              f"past={base_seen} out={a.out}", flush=True)
        return
    if a.verify_probe:
        # S1 verify-cost probe, second design (Bugbot e4b#239 killed the
        # first): decode attention is T=1 by contract, and the prefill
        # fallback attended over a staging buffer that FLUSH had already
        # popped -- an empty past, underpricing verify and making the GO
        # bound unsound. This probe instead measures PREFILL-CONTINUATION:
        # one untimed staging pass rebuilds a 512-token staged past on
        # this slot (stage() was flushed at real-prefill completion),
        # then R chunks of K+1 rows run through the production prefill
        # path -- causal attention over the staged past + the prefill
        # expert path. That is a REAL, existing, certified-numerics way
        # an S2 executor could verify, so its measured cost upper-bounds
        # the best verify implementation S2 could ship.
        kk = int(a.verify_probe) + 1
        runner.ctx.mode = "prefill"
        past = 512
        st_ids = in_ids.expand(1, past).contiguous()
        st_pos = torch.arange(past, device=dev).view(1, past)
        model(input_ids=st_ids, position_ids=st_pos, use_cache=False)
        torch.cuda.synchronize()
        v_ids = in_ids.expand(1, kk).contiguous()
        v_pos = (torch.arange(kk, device=dev).view(1, kk) + past)

        def verify_step():
            out = model(input_ids=v_ids, position_ids=v_pos,
                        use_cache=False)
            lg = out.logits[:, -1]
            v_pos.add_(kk)
            return lg

        for _ in range(8):
            verify_step()
        torch.cuda.synchronize()
        past_timing_start = past + 8 * kk
        spans = []
        for _c in range(10):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _r in range(4):
                verify_step()
            e1.record()
            e1.synchronize()
            spans.append(e0.elapsed_time(e1) / 4)
        spans.sort()
        past_timing_end = past_timing_start + 40 * kk
        set_context(prev)
        rep = {"verify_probe_k": int(a.verify_probe),
               "probe_path": "prefill-continuation over staged past",
               "rows_per_step": kk,
               "verify_ms": spans[len(spans) // 2],
               "verify_ms_all_chunks": spans,
               "past_at_timing_start": past_timing_start,
               "past_at_timing_end": past_timing_end,
               "basis": "eager chunked-median over an EXECUTABLE verify "
                        "path -- upper bound on best-case S2 verify "
                        "(PREREG-s1-acceptance)"}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
        print(f"S1_VERIFY_PROBE k={a.verify_probe} rows={kk} "
              f"verify_ms={rep['verify_ms']:.3f} "
              f"past={past_timing_start}..{past_timing_end} out={a.out}",
              flush=True)
        return
    if a.b1d_timed:
        # stage C: tokens land in a device log (one tiny D2D copy per
        # step, no sync); the window is timed as a whole
        tok_log = torch.zeros(n_steps, dtype=torch.long, device=dev)
        _frames_before = None
        if a.b1d_loop == "graph":
            torch.cuda.empty_cache()
            g = torch.cuda.CUDAGraph()
            if probe is not None:
                probe.begin_window(1)   # capture traces exactly one step
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                one_step()
            torch.cuda.synchronize()
            # PREREG-f1-stageB refusal 4 counts recompiles inside the
            # TIMED window. Capture is not timed and traces one_step for
            # the first time, so sampling before it would charge the arm
            # for compilation that t0 never sees (Bugbot, e4b#234).
            _frames_before = _dynamo_frame_count()
            t0 = time.perf_counter_ns()
            for i in range(n_steps):
                g.replay()
                tok_log[i].copy_(in_ids.reshape(()))
            torch.cuda.synchronize()
        else:
            _frames_before = _dynamo_frame_count()
            if probe is not None:
                probe.begin_window(n_steps)
            t0 = time.perf_counter_ns()
            for i in range(n_steps):
                one_step()
                tok_log[i].copy_(in_ids.reshape(()))
            torch.cuda.synchronize()
        total_ms = (time.perf_counter_ns() - t0) / 1e6
        if profile_replays:
            # PREREG-sv2: kernel census of the SHIPPED captured replay
            # -- kineto records kernels inside graph replays. Runs
            # AFTER the timed window: replays advance pos/KV, so a
            # pre-window placement shifted the timed tokens off the
            # eager arm's steps AND overran the pre-ensured blocks
            # (Bugbot, e4b#275); the slots these appends land in were
            # reserved via cap_tokens. CUDA activity only (the TR1
            # capacity lesson); the timed loop above stays clean.
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as rp:
                for _ in range(profile_replays):
                    g.replay()
                torch.cuda.synchronize()
            tbl = rp.key_averages().table(
                sort_by="cuda_time_total", row_limit=120)
            hdr = (f"profiled replay steps: {profile_replays} "
                   f"(active window: {profile_replays}/"
                   f"{profile_replays})\n")
            Path(a.replay_profile_out).write_text(hdr + tbl)
            print(f"REPLAY_PROFILE_OUT {a.replay_profile_out}",
                  flush=True)
        set_context(prev)
        _frames_after = _dynamo_frame_count()
        _recompiles = None
        if _frames_before is not None and _frames_after is not None:
            _recompiles = (_frames_after["total"]
                           - _frames_before["total"])
        rep = {
            "b1d_loop": a.b1d_loop, "b1d_timed": True,
            "n_steps": n_steps,
            "mech": _mech_report(),
            "tokens": tok_log.cpu().tolist(),
            "step_ms_clean": total_ms / n_steps,
            "window_ms": total_ms,
            "compile_layers": bool(a.compile_layers),
            "compile_moe_tier": bool(a.compile_moe_tier),
            "compile_attn_tier": bool(a.compile_attn_tier),
            "compile_mode": a.compile_mode if a.compile_layers else None,
            "fuse_qkv": bool(a.fuse_qkv),
            "dyn_limits_env": [os.environ.get("E4B_RECOMPILE_LIMIT"),
                               os.environ.get(
                                   "E4B_ACCUM_RECOMPILE_LIMIT")],
            "dynamo_frames_before": _frames_before,
            "dynamo_frames_after": _frames_after,
            "recompiles_in_window": _recompiles,
            "router_probe": (probe.report(kv.L)
                             if probe else None),
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
        print(f"B1D_TIMED_{a.b1d_loop.upper()} steps={n_steps} "
              f"step={rep['step_ms_clean']:.2f}ms "
              f"compile={rep['compile_mode']} "
              f"recompiles={rep['recompiles_in_window']} "
              f"out={a.out}", flush=True)
        return

    hashes, toks, walls, positions = [], [], [], []
    if a.b1d_loop == "graph":
        # Capture records kernels, it must not EXECUTE them (documented
        # torch.cuda.graph semantics) -- but the gate does not rely on
        # documentation: assert state neutrality at runtime, so a torch
        # that ever executed during capture fails HERE with the cause
        # named instead of downstream as a baffling hash mismatch
        # (Bugbot, e4b#228).
        pos_before = int(pos.reshape(()))
        # The eager pool hoards cached blocks after warmup; the graph's
        # PRIVATE pool then needs fresh cudaMallocs mid-capture, and the
        # allocator's free-and-retry path calls cuStreamSynchronize --
        # instantly invalidating capture (the CUDA_LOG_FILE trace named
        # it). Hand the capture clean headroom first. thread_local mode
        # additionally insulates the capture from stray context probes
        # by other threads (the CPU pool workers).
        torch.cuda.empty_cache()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            lg_static = one_step()
        torch.cuda.synchronize()
        pos_after = int(pos.reshape(()))
        assert pos_after == pos_before, (
            f"capture advanced state (pos {pos_before} -> {pos_after}): "
            f"this torch executes during capture; the replay window is "
            f"misaligned and the arms cannot be compared")
        for _ in range(n_steps):
            t0 = time.perf_counter_ns()
            g.replay()
            torch.cuda.synchronize()
            walls.append(time.perf_counter_ns() - t0)
            hashes.append(hashlib.sha256(
                lg_static.float().cpu().numpy().tobytes()).hexdigest()[:16])
            toks.append(int(in_ids.reshape(())))
            positions.append(int(pos.reshape(())))
    else:
        for _ in range(n_steps):
            t0 = time.perf_counter_ns()
            lg = one_step()
            torch.cuda.synchronize()
            walls.append(time.perf_counter_ns() - t0)
            hashes.append(hashlib.sha256(
                lg.float().cpu().numpy().tobytes()).hexdigest()[:16])
            toks.append(int(in_ids.reshape(())))
            positions.append(int(pos.reshape(())))
    set_context(prev)
    walls_ms = sorted(w / 1e6 for w in walls)
    rep = {
        "b1d_loop": a.b1d_loop,
        "n_steps": n_steps,
        "warm_scheduled": 2, "warm_manual": n_warm,
        "logits_hashes": hashes,
        "tokens": toks,
        # per-step positions make cross-arm alignment CHECKED, never
        # assumed: an off-by-one (any capture-executes semantics, a
        # missed warm step) shows as a position shift, not a mystery
        "positions": positions,
        "step_ms_median_syncd": walls_ms[len(walls_ms) // 2],
        "note": "stage A: per-step sync for hashing inflates the wall; "
                "the H-G bar is stage C's, not this number",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"B1D_{a.b1d_loop.upper()} steps={n_steps} "
          f"median_syncd={rep['step_ms_median_syncd']:.2f}ms "
          f"first_tok={toks[0]} out={a.out}", flush=True)


# --- F1 Stage A (PREREG-f1-elementwise) elementwise attribution ------
# The aten ops whose device work makes up the elementwise block. Kernel
# rows (`void at::native::...`) carry no python stack, so attribution
# necessarily runs on the OP view; step_budget.py owns the kernel-view
# total and the two are reconciled in the verdict.
_EW_OPS = frozenset((
    "aten::copy_", "aten::to", "aten::_to_copy", "aten::mul",
    "aten::add", "aten::add_", "aten::mul_", "aten::div", "aten::div_",
    "aten::rsqrt", "aten::pow", "aten::silu", "aten::softmax",
    "aten::_softmax", "aten::sub", "aten::neg", "aten::cat",
    "aten::index_select", "aten::mean", "aten::sum", "aten::clamp",
    "aten::where", "aten::sigmoid", "aten::type_as", "aten::contiguous",
    # the fp8 KV scale path (abs/amax/gt) and the scatter/fill helpers:
    # 447 us/step on the census receipt, every one of them above the
    # 50 us/step Stage A bar (Bugbot, e4b#231)
    "aten::amax", "aten::abs", "aten::gt", "aten::lt", "aten::fill_",
    "aten::zero_", "aten::index_add_", "aten::index_put_",
    "aten::ones_like", "aten::zeros_like", "aten::masked_fill_",
    "aten::reciprocal", "aten::exp", "aten::maximum", "aten::minimum",
))
# Ops whose device time is deliberately NOT part of the elementwise
# block. Anything with device time that is in neither set is reported
# as `unclassified` rather than silently dropped -- a hand-curated list
# always drifts, so the receipt has to make its own omissions visible.
_NON_EW_OPS = frozenset((
    "aten::mm", "aten::matmul", "aten::linear", "aten::bmm",
    "aten::addmm", "aten::baddbmm", "aten::topk", "aten::sort",
    "aten::argmax", "aten::argsort", "aten::nonzero", "aten::unique2",
    "aten::scaled_dot_product_attention",
))
# Frames that are never the answer: torch internals and this harness.
_FRAME_SKIP = ("/torch/", "site-packages/torch", "torch/nn/modules",
               "torch/autograd", "step_decomp.py", "<built-in",
               "torch/_dynamo", "torch/_inductor", "_python_dispatch",
               "<frozen ", "importlib")


def _self_device_us(evt):
    """Self device time in us across torch versions (>=2.5 renamed
    self_cuda_time_total -> self_device_time_total; the old name warns
    or is absent)."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


class _EwSiteTracer:
    """Count elementwise aten dispatches per python call site.

    PREREG-f1 registered profiler `with_stack` attribution
    (AMENDMENT-f1-tracer supersedes it). On torch 2.13 `with_stack=True` returns EMPTY stacks
    for every event -- key_averages, key_averages(group_by_stack_n=),
    and function_events alike -- so that mechanism yields 100%
    `<no-python-frame>` and attributes nothing. This tracer replaces it
    and depends on nothing but python frames.

    It records COUNTS, not time: device time per op comes from the
    profiler, and each op's time is apportioned across its sites in
    proportion to counts. That is exact when a single op's launches
    cost the same, which is the regime here -- the block averages
    1.21 us per launch, at this GPU's minimum kernel duration, so cost
    tracks launch count rather than tensor size. The receipt records
    counts, apportioned time, and this caveat together."""

    def __init__(self, ops):
        self._ops = ops
        self.counts = {}
        self._mode = None

    def __enter__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        tracer = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                try:
                    name = "aten::" + func._schema.name.split("::")[-1]
                except Exception:                      # noqa: BLE001
                    name = None
                if name in tracer._ops:
                    site = tracer._site()
                    key = (name, site)
                    tracer.counts[key] = tracer.counts.get(key, 0) + 1
                return func(*args, **(kwargs or {}))

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        m, self._mode = self._mode, None
        if m is not None:
            m.__exit__(*exc)
        return False

    @staticmethod
    def _site():
        """Nearest python frame that is OUR code. sys._getframe walking
        is used rather than traceback.extract_stack because this runs on
        every dispatched op."""
        f = sys._getframe(1)
        depth = 0
        while f is not None and depth < 40:
            fn = f.f_code.co_filename
            if not any(sk in fn for sk in _FRAME_SKIP):
                return f"{fn}({f.f_lineno}): {f.f_code.co_name}"
            f = f.f_back
            depth += 1
        return "<no-python-frame>"


def _dynamo_frame_count():
    """Dynamo's cumulative compiled-frame count, or None when dynamo is
    not in use. PREREG-f1-stageB refusal 4 reads this before and after
    the timed window: a recompile inside it means the arm timed
    compilation rather than the fused kernels."""
    try:
        from torch._dynamo.utils import counters
    except Exception:                              # noqa: BLE001
        return None
    ok = counters.get("frames", {}).get("ok", 0)
    total = counters.get("frames", {}).get("total", 0)
    return {"ok": int(ok), "total": int(total)}


def _apportion(by_op, counts):
    """Spread each op's device time over its call sites by launch share."""
    per_op_total = {}
    for (op, _site), c in counts.items():
        per_op_total[op] = per_op_total.get(op, 0) + c
    sites = {}
    for (op, site), c in counts.items():
        tot = per_op_total.get(op, 0)
        if not tot:
            continue
        us = by_op.get(op, {}).get("us", 0.0) * c / tot
        sr = sites.setdefault(site, {"us": 0.0, "calls": 0, "ops": {}})
        sr["us"] += us
        sr["calls"] += c
        sr["ops"][op] = sr["ops"].get(op, 0.0) + us
    return dict(sorted(sites.items(), key=lambda kv: -kv[1]["us"]))


def _ew_attribute(evts, min_us=0.0):
    """Split profiler events into the elementwise block (attributed to
    python call sites), the deliberately-excluded ops, and whatever is
    in NEITHER list. `unclassified` is the important one: it is how a
    missing op announces itself instead of quietly shrinking the block
    the treatment claims to remove."""
    sites, ops, unclassified, attributed = {}, {}, {}, 0.0
    for evt in evts:
        key = getattr(evt, "key", None)
        if not key or not getattr(evt, "count", 0):
            continue
        us = _self_device_us(evt)
        if us <= min_us:
            continue
        if key.startswith("aten::") and key not in _EW_OPS \
                and key not in _NON_EW_OPS:
            u = unclassified.setdefault(key, {"us": 0.0, "calls": 0})
            u["us"] += us
            u["calls"] += evt.count
            continue
        if key not in _EW_OPS:
            continue
        attributed += us
        o = ops.setdefault(key, {"us": 0.0, "calls": 0})
        o["us"] += us
        o["calls"] += evt.count
        site = _py_site(getattr(evt, "stack", None))
        sr = sites.setdefault(site, {"us": 0.0, "calls": 0, "ops": {}})
        sr["us"] += us
        sr["calls"] += evt.count
        sr["ops"][key] = sr["ops"].get(key, 0.0) + us
    return {"attributed_us": attributed,
            "by_site": dict(sorted(sites.items(),
                                   key=lambda kv: -kv[1]["us"])),
            "by_op": dict(sorted(ops.items(),
                                 key=lambda kv: -kv[1]["us"])),
            "unclassified_ops": dict(sorted(unclassified.items(),
                                            key=lambda kv: -kv[1]["us"]))}


def _py_site(stack):
    """First python frame that is neither torch internals nor this
    harness -- i.e. OUR code, which is the only kind of frame a fix can
    act on. Falls back to the outermost python frame, then to a label
    that is obviously not a call site so it cannot be mistaken for one."""
    frames = [f for f in (stack or []) if ".py" in f]
    for f in frames:
        if not any(sk in f for sk in _FRAME_SKIP):
            return f
    return frames[-1] if frames else "<no-python-frame>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--vram-gb", type=float, default=1.2)
    ap.add_argument("--dram-gb", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--ppl-chat", action="store_true",
                    help="build the K8 window INSIDE the tokenizer's chat template: "
                         "a user turn asking for a verbatim continuation, then the "
                         "corpus as the assistant's reply. For chat-only families "
                         "(gpt-oss scores ~2000 ppl on bare wikitext through "
                         "transformers itself) bare text is out of distribution and "
                         "an absolute ppl delta means nothing; in-distribution the "
                         "delta is a quality number again. Both arms hash the same ids.")
    ap.add_argument("--ppl-chat-suffix", default="",
                    help="raw text appended after the template's generation prompt "
                         "(gpt-oss: '<|channel|>final<|message|>' puts the reply in "
                         "the final channel); tokenised WITH special tokens")
    ap.add_argument("--ppl-oracle", default="none",
                    choices=["none", "eager", "full", "upstream", "upstream-full"],
                    help="score the SAME --ppl-steps window through transformers' "
                         "own attention (eager, HF cache, chunked teacher forcing) "
                         "with the paged shim NOT registered: the reference the "
                         "paged path must match on families the shim does not "
                         "yet serve (sliding windows, sinks, per-layer KV)")
    ap.add_argument("--kv-groups", default="auto",
                    help="key scale groups for the fp8 paged cache: 'auto' keeps "
                         "32-wide scales per layer (4/8/16 at head_dim 128/256/512 "
                         "when the kernel unrolls them, else 4); an int broadcasts")
    ap.add_argument("--ppl-route", default="none", choices=["none", "record", "replay"],
                    help="matched routing: 'record' the reference full forward's "
                         "top-k choices per layer and position to --ppl-route-file; "
                         "'replay' them into the paged arm scoring the same window")
    ap.add_argument("--ppl-route-file", default="route.pt")
    ap.add_argument("--ppl-layer-diff", default="",
                    help="comma list of K8 steps at which to diff the paged "
                         "decode step against the chunk-free eager forward per "
                         "layer (attention / dense MLP / experts / router)")
    ap.add_argument("--ppl-chunk", type=int, default=0,
                    help="chunk size for the chunked oracle (0 = the scorer's "
                         "default of 256); two sizes on one window split a "
                         "family's chunked-vs-full floor into routing noise "
                         "versus a chunk-boundary effect")
    ap.add_argument("--ppl-fq", default="none",
                    help="with --ppl-oracle full: apply the fp8 paged kernel's "
                         "roundings inside the full forward; letters from "
                         "q,k,v,p (e.g. kv, qkvp), or none")
    ap.add_argument("--fq-kgroups", type=int, default=4,
                    help="K sub-row scale groups modelled (Fp8PagedKV default 4)")
    ap.add_argument("--fq-vgroups", type=int, default=1)
    ap.add_argument("--fq-layers", default="all", choices=["all", "full", "sliding"],
                    help="apply the roundings on every layer, only the "
                         "full-attention layers, or only the sliding ones")
    ap.add_argument("--fq-from", type=int, default=-1,
                    help="first query row that reads fake-quantised K/V "
                         "(default: prompt_len -- the paged prefill attends "
                         "on bf16, decode reads fp8)")
    ap.add_argument("--ppl-source", default="wikitext",
                    choices=["wikitext", "c4val1"],
                    help="corpus the prompt windows and --ppl-steps score: "
                         "wikitext-2 TEST (the K8 text) or a C4 validation "
                         "shard (00001) disjoint from any calibration shard "
                         "-- the out-of-domain check for a calibrated pack "
                         "whose wikitext delta is 'too good'")
    ap.add_argument("--prompt-offset", type=int, default=0,
                    help="start of the corpus slice the prompt windows are "
                         "cut from (disjoint-window generalization runs)")
    ap.add_argument("--prompt-span", type=int, default=0,
                    help="length of that corpus slice; 0 = to the end. "
                         "Without a bounded span, prompts spread over the "
                         "WHOLE remaining corpus and two offsets overlap")
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--hot-rows", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cpu-us-fixed", type=float, default=None)
    ap.add_argument("--cpu-us-per-row", type=float, default=None)
    ap.add_argument("--profile", default=None,
                    help="expert_profile JSONL for a measured-routing placement")
    ap.add_argument("--profile-out", default=None,
                    help="write this run's decode routing hist as an "
                         "expert_profile JSONL (profile-pass mode)")
    ap.add_argument("--compile-layers", action="store_true",
                    help="torch.compile each decoder layer body; the "
                         "paged-attention fn and the MoE tier forward are "
                         "dynamo-disabled so they graph-break cleanly "
                         "(PREREG-t1-launchpath)")
    ap.add_argument("--layers-attr", default="model.layers",
                    help="dotted path to the decoder-layer list for "
                         "--compile-layers (latent/nested families "
                         "differ, e.g. model.language_model.layers)")
    ap.add_argument("--graph-break-census", action="store_true",
                    help="PREREG-k13 Stage A: run torch._dynamo.explain "
                         "over one decode step and write the graph "
                         "breaks dynamo NAMES (file, line, function, "
                         "its own reason, count, and whether the break "
                         "fires at trace or per step). A census, not a "
                         "treatment: writes the report and returns "
                         "without timing anything.")
    ap.add_argument("--compile-attn-tier", action="store_true",
                    help="PREREG-k12 arm 3 CONTROL: also skip the "
                         "paged-attention dynamo.disable, so BOTH "
                         "exclusions are lifted. Expected to reproduce "
                         "F1 Stage B's failure; if it does not, that is "
                         "a finding. Requires --compile-moe-tier.")
    ap.add_argument("--compile-moe-tier", action="store_true",
                    help="PREREG-k12 Stage A: leave the MoE tier "
                         "forward VISIBLE to dynamo while the "
                         "paged-attention fn stays disabled, so "
                         "inductor can fuse the raw-aten elementwise "
                         "chains the census attributes to that region. "
                         "Requires --placement-override all-vram")
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    help="torch.compile mode for --compile-layers; drop "
                         "to 'default' if cudagraphs misbehave (recorded)")
    ap.add_argument("--replay-profile-out", default=None,
                    help="PREREG-sv2: write a kernel table of 8 "
                         "UNTIMED graph replays (kineto sees inside "
                         "replays) AFTER the timed window, on reserved "
                         "KV slots; b1d graph loop only")
    ap.add_argument("--router-probe", action="store_true",
                    help="PREREG-k10 Stage A: count torch.topk calls "
                         "and record their shapes across the decode "
                         "window; writes the attribution receipt")
    ap.add_argument("--router-sorted", choices=["true", "false"],
                    default=None,
                    help="PREREG-k10 Stage B1: force the sorted= kwarg "
                         "of every torch.topk. 'false' should delete "
                         "bitonicSortKVInPlace; numerics-changing "
                         "(same set, different order), gated by B1")
    ap.add_argument("--ppl-steps", type=int, default=0,
                    help="PREREG-k8: score N teacher-forced tokens "
                         "THROUGH the paged decode path and write the "
                         "perplexity receipt to --out (b1 path; the "
                         "KV is rewound to the prompt boundary first)")
    ap.add_argument("--grouping-parity", action="store_true",
                    help="PREREG-bv3b: per-layer device-vs-eager MoE "
                         "grouping parity on one recorded live decode "
                         "step; writes the parity receipt to --out")
    ap.add_argument("--fuse-qkv", dest="fuse_qkv",
                    action="store_true", default=True,
                    help="fuse q/k/v projections into one matmul per "
                         "attention module. DEFAULT ON since "
                         "RESULTS-f2-tail (PARTIAL ships: +0.120 ms "
                         "under a 0.001 ms A/A, token-identical); "
                         "applied before --compile-layers so dynamo "
                         "traces the fused forward")
    ap.add_argument("--no-fuse-qkv", dest="fuse_qkv",
                    action="store_false",
                    help="rollback for --fuse-qkv (the F2 OFF arm)")
    ap.add_argument("--kv-batched", action="store_true",
                    help="accepted for command compatibility -- batched "
                         "KV append is the DEFAULT since its cert "
                         "(PREREG-g9-kvappend); see --kv-per-seq")
    ap.add_argument("--kv-per-seq", action="store_true",
                    help="run the per-seq KV append path (the kvappend "
                         "cert A arm; the T5 cycle measured this point "
                         "by accident when batched was opt-in)")
    ap.add_argument("--collapse", action="store_true",
                    help="accepted for command compatibility -- the "
                         "all-resident collapse is the DEFAULT since its "
                         "cert (RESULTS-b1c); see --no-collapse")
    ap.add_argument("--no-collapse", action="store_true",
                    help="run the pre-collapse dispatch path (the b1c "
                         "cert C0 arm; fires only on all-VRAM placements "
                         "either way)")
    ap.add_argument("--engine", choices=["hybrid", "pipelined"],
                    default="hybrid",
                    help="PREREG-b1 R1 arm: 'pipelined' bypasses the "
                         "hybrid tier -- enable_pipelined_residency with "
                         "every expert hot (narrowest existing resident "
                         "grouped-NF4 path); no placement, no CPU tier")
    ap.add_argument("--placement-override", choices=["none", "all-vram"],
                    default="none",
                    help="PREREG-b1 R0 arm: after the solver runs, move "
                         "every expert into the VRAM tier, executor "
                         "machinery left intact -- isolates physical "
                         "heterogeneity from orchestration")
    ap.add_argument("--host-brackets", action="store_true",
                    help="T5b Phase A: host-wall brackets + profiler "
                         "regions around each MoE forward and lm_head. "
                         "Timing arms must NOT carry this")
    ap.add_argument("--region-ops-out", default=None,
                    help="JSON of per-region descendant op counts from "
                         "the profiler event tree (needs --host-brackets; "
                         "engages the torch profiler, stacks off)")
    ap.add_argument("--b1d-timed", action="store_true",
                    help="PREREG-b1d stage C: clean timing -- no per-step "
                         "sync/hash; tokens logged to a device buffer and "
                         "read once at the end; wall = whole window / N")
    ap.add_argument("--b1d-loop", choices=["eager", "graph"], default=None,
                    help="PREREG-b1d stage A: after prefill + a short "
                         "scheduled warm, drive the B=1 decode loop "
                         "manually through the graph-shape append path -- "
                         "'graph' captures one step and replays; 'eager' "
                         "runs the identical loop uncaptured (the "
                         "identity reference). Writes per-step logits "
                         "hashes + tokens to --out")
    ap.add_argument("--sync-attr-out", default=None,
                    help="JSON of op counts over the profiler's active "
                         "window, with aten::nonzero attributed to source "
                         "files via stack frames (T5 H1/H2 instrument). "
                         "Implies the torch profiler with with_stack=True "
                         "-- run timing arms WITHOUT this flag")
    ap.add_argument("--s2-verify", choices=["gate", "time", "parity"],
                    default=None,
                    help="PREREG-s2lite Stage A. 'gate': bitwise identity "
                         "-- 17 sequential T=1 greedy steps vs ONE "
                         "verify-mode step fed those same tokens as the "
                         "draft; every verified argmax must equal the "
                         "sequential token, and after rewind the T=1 "
                         "continuation must be unchanged. 'time': graph "
                         "the verify step at fixed position (write "
                         "addresses bake -- legal for same-position "
                         "replay; timing-only) and report verify_ms vs "
                         "the anchor")
    ap.add_argument("--moe-grouping",
                    choices=["singleton", "eager", "device"],
                    default="singleton",
                    help="PREREG-s3-grouped-verify: MoE grouping for the "
                         "--s2-verify arms. singleton = one M=1 group per "
                         "route (reuse disabled; the S2 bound); eager = "
                         "normal unique_consecutive grouping (NOT "
                         "capturable -- its time arm runs an eager timed "
                         "loop); device = capture-safe device grouping "
                         "(gnf4 build_group_tiles_device)")
    ap.add_argument("--s2-k", type=int, default=16,
                    help="draft window for --s2-verify (K; the step runs "
                         "K+1 rows)")
    ap.add_argument("--verify-probe", type=int, default=None,
                    help="PREREG-s1-acceptance: after prefill+warm, time "
                         "R repeats of a (K+1)-row single-seq forward at "
                         "~prompt-len past (chunked-median, eager -- an "
                         "UPPER bound on verify cost; the S1 map uses it "
                         "only where conservatism is sound). Writes "
                         "verify_ms into the b1d-timed report and skips "
                         "the T=1 loop")
    ap.add_argument("--ew-attr-out", default=None,
                    help="attribute the ELEMENTWISE device block to python "
                         "call sites via stack frames (F1 Stage A "
                         "instrument, PREREG-f1-elementwise). Implies the "
                         "torch profiler with with_stack=True -- run "
                         "timing arms WITHOUT this flag")
    ap.add_argument("--torch-profile-out", default=None,
                    help="capture ~12 decode steps under torch.profiler "
                         "and dump the CUDA kernel table (T2/T3 "
                         "attribution: which device kernels own the "
                         "attention and expert buckets, and how many "
                         "launches each)")
    ap.add_argument("--cprofile-out", default=None,
                    help="run the serving loop under cProfile and dump "
                         "the top functions by cumulative time (the G9 "
                         "host-bill attribution instrument)")
    ap.add_argument("--series-out", default=None,
                    help="write the per-step touched-expert series "
                         "(decode-only, all tiers) as gzipped JSON")
    ap.add_argument("--amort-out", default=None,
                    help="write decode-only per-tier unique/activation "
                         "accounting plus the manifest VRAM set")
    ap.add_argument("--dispatch-diet", action="store_true",
                    help="T5: enable the engine's dispatch-algebra diet "
                         "(one sync/layer, cached index algebra); arm B "
                         "of PREREG-t5-dispatch-diet")
    ap.add_argument("--amort", choices=["on", "off"], default="on",
                    help="off = production shape: no per-layer counters, "
                         "no per-layer event syncs; the T5 arms run off "
                         "(--profile-out/--series-out/--amort-out then "
                         "refuse, they have nothing to write)")
    ap.add_argument("--torch-threads", type=int, default=8,
                    help="torch intraop cap while the pool runs (serving "
                         "playbook: 8; the default thrashes pinned workers)")
    ap.add_argument("--out", default="/workspace/g8out/step_decomp.json")
    a = ap.parse_args()
    # E4B_RECOMPILE_LIMIT / E4B_ACCUM_RECOMPILE_LIMIT: set dynamo's
    # budgets UNIFORMLY for this run, whatever the lane. The BV3b
    # parity receipts proved device-vs-eager grouping bitwise, leaving
    # the arms' only uncontrolled difference the graph lane's raised
    # limits: compiled-vs-fallback kernels differ numerically, so a
    # cross-arm identity comparison is confounded unless every arm
    # runs the same compile coverage. Recorded in the timed receipts.
    _rl = os.environ.get("E4B_RECOMPILE_LIMIT")
    _al = os.environ.get("E4B_ACCUM_RECOMPILE_LIMIT")
    if _rl or _al:
        import torch._dynamo as _dyn0
        if _rl:
            _dyn0.config.recompile_limit = max(
                int(_rl), _dyn0.config.recompile_limit)
        if _al:
            _dyn0.config.accumulated_recompile_limit = max(
                int(_al), _dyn0.config.accumulated_recompile_limit)

    from transformers import AutoTokenizer
    from experts4bit_qlora import load_moe_4bit_streaming
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.hot_residency import target_modules
    from experts4bit_qlora.engines.hybrid import enable_hybrid_tier
    from experts4bit_qlora.engines.paged_attention import IMPL_NAME, register
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    from experts4bit_qlora.engines.placement import solve_placement
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler

    torch.manual_seed(1689)
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.ppl_oracle in ("upstream", "upstream-full"):
        assert a.ppl_steps > 0, "--ppl-oracle needs --ppl-steps"
        _upstream_oracle_main(a, tok)
        return
    model, _ = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16,
                                       r=8, alpha=16, quant_type="nf4",
                                       arena=a.arena)
    model.eval()
    mods = target_modules(model)
    L, E = len(mods), mods[0].num_experts
    k = _routed_topk(model.config)
    idx = json.loads(Path(a.arena + ".index.json").read_text())
    bpe = 0
    for seg in idx["segments"]:
        n = 1
        for d in seg["shape_per_expert"]:
            n *= d
        bpe += n * (4 if seg["dtype"] == "F32" else 1)
    torch.set_num_threads(a.torch_threads)
    if a.engine == "pipelined":
        # R1 (PREREG-b1): the narrowest existing resident path. Every
        # expert hot, one code path, no hybrid tier anywhere in the
        # process -- no arena serving tier, no CPU pool, no placement.
        assert a.amort == "off", "--engine pipelined has no amort counters"
        assert a.placement_override == "none", \
            "--placement-override is a hybrid-arm knob"
        assert not a.dispatch_diet, "dispatch_diet is hybrid-only"
        assert not (a.collapse and a.no_collapse), "contradictory flags"
        # T>1 falls back to the reference forward, which cannot run CUDA
        # activations against the host-materialized weights (Bugbot,
        # e4b#223). chunk=1 keeps every forward on the T=1 fast path;
        # KV content is mathematically identical (causal prefix math is
        # chunking-invariant) and the R0==R1 bitwise gate enforces it.
        assert a.chunk == 1, \
            "--engine pipelined requires --chunk 1 (see PREREG-b1 R1 "\
            "mechanics): T>1 prefill would hit the reference forward "\
            "against host-resident weights and die mid-run"
        from experts4bit_qlora.engines.pipelined import (
            enable_pipelined_residency)
        _materialize_from_arena(mods, a.arena)
        man = None
        n = enable_pipelined_residency(
            model, [list(range(E)) for _ in range(L)], device="cuda",
            k_slots=k)
        assert n == L, f"pipelined patched {n}/{L} modules"
    else:
        man = solve_placement(
            n_layers=L, n_experts=E, bytes_per_expert=bpe,
            vram_budget_bytes=int(a.vram_gb * 2**30),
            dram_budget_bytes=int(a.dram_gb * 2**30),
            calibration=json.loads(Path(a.calib).read_text()),
            profile_path=a.profile,
            batch=a.batch, top_k=k,
            cpu_us_fixed=a.cpu_us_fixed, cpu_us_per_row=a.cpu_us_per_row)
        if a.placement_override == "all-vram":
            # R0 (PREREG-b1): physical heterogeneity removed, executor
            # machinery intact -- the solver ran, then every expert is
            # moved into the VRAM tier
            pairs = sorted(tuple(pp) for t in ("vram", "dram", "nvme")
                           for pp in man["tiers"][t])
            man["tiers"] = {"vram": [list(pp) for pp in pairs],
                            "dram": [], "nvme": []}
            man["masses"] = {"vram_frac": 1.0, "dram_frac": 0.0,
                             "nvme_frac": 0.0}
        n = enable_hybrid_tier(model, a.arena, man, hot_rows=a.hot_rows,
                               threads=a.threads, pool=True,
                               dispatch_diet=a.dispatch_diet,
                               collapse_resident=not a.no_collapse)
        assert n == L
    assert not (a.kv_batched and a.kv_per_seq),         "--kv-batched and --kv-per-seq are contradictory"
    states = ([] if a.engine == "pipelined"
               else [m._hot_residency for m in mods])
    amort_on = a.amort == "on"
    if not amort_on:
        for flag in ("profile_out", "series_out", "amort_out"):
            assert not getattr(a, flag), \
                f"--{flag.replace('_', '-')} needs --amort on"
    for st in states:
        st.arm_amortization(amort_on)

    cfg = model.config
    cfg = model.config
    if a.ppl_oracle == "none":
        hkv, hd = _kv_geometry(cfg)
        register(model)
        wrap_attention(IMPL_NAME)
    else:
        # the oracle never builds the paged cache: a family whose KV
        # geometry varies per layer must still be SCORABLE here -- this
        # arm is that family's reference
        hkv = hd = None
    if a.region_ops_out and not a.host_brackets:
        raise SystemExit("--region-ops-out needs --host-brackets")
    if a.host_brackets:
        HOST_BRACKETS[0] = True
        for m in mods:
            m.forward = _wrap_region(m.forward, "moe", "e4b::moe")
        # the sparse-MoE BLOCK (router + experts) as its own region, so
        # router/top-k host cost = moe_block - moe (PREREG-b1)
        _ll = model
        for _part in a.layers_attr.split("."):
            _ll = getattr(_ll, _part)
        n_blk = 0
        for _lyr in _ll:
            _blk = getattr(_lyr, "mlp", None)
            if _blk is not None and hasattr(_blk, "experts"):
                _blk.forward = _wrap_region(_blk.forward, "moe_block",
                                            "e4b::moe_block")
                n_blk += 1
        assert n_blk == len(mods), \
            f"moe_block bracket found {n_blk} blocks for {len(mods)} " \
            f"expert modules -- the region would silently under-count"
        model.lm_head.forward = _wrap_region(
            model.lm_head.forward, "lmhead", "e4b::lmhead")
    if a.fuse_qkv:
        from experts4bit_qlora.engines.qkv_fuse import fuse_qkv
        n_f = fuse_qkv(model)
        if n_f == 0 and "--fuse-qkv" in sys.argv:
            # explicit request must fuse or fail loudly; the ON
            # default degrades gracefully on families without
            # Qwen3MoeAttention (the F1-B2 resolution pattern)
            raise SystemExit("--fuse-qkv matched no Qwen3MoeAttention "
                             "modules -- refusing a vacuous arm")
        print(f"fused q/k/v projections on {n_f} attention modules "
              f"(RESULTS-f2-tail default)", flush=True)
    else:
        # --no-fuse-qkv (non-Qwen families): the glue and router fusions
        # are env-gated and structural; call them so a set flag either
        # engages on matching modules or REFUSES with a sentence. The
        # first non-Qwen sweep ran the fused arm with every flag set and
        # nothing consulted them -- an identical step and no banner.
        from experts4bit_qlora.engines.glue_fuse import fuse_t1_glue
        from experts4bit_qlora.engines.glue_r2 import fuse_t1_glue_r2
        from experts4bit_qlora.engines.router_epilogue import fuse_router_epilogue
        fuse_t1_glue(model)
        fuse_t1_glue_r2(model)
        fuse_router_epilogue(model)
    if a.compile_layers:
        import torch._dynamo as dynamo
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        # clean graph breaks: the paged-attention shim (host-bound KV
        # paging) and the hybrid MoE forward (CPU tier dispatch) must
        # never be traced -- compile owns only the dense layer body
        # PREREG-k12 AMENDMENT: Stage A lists FOUR arms, and arm 3
        # ("both-compiled") is the control that should reproduce F1
        # Stage B's failure -- "if it does NOT fail, F1's exclusion may
        # itself be stale, and that is a finding to record rather than
        # bury". But the prereg's "Instrument required" section named
        # only ONE flag, so arm 3 was not runnable with the registered
        # instrument at all. This flag is named in the amendment rather
        # than smuggled in, which is the discipline that section asks
        # for in its own heading.
        if a.compile_attn_tier:
            assert a.compile_moe_tier, (
                "--compile-attn-tier is the arm-3 CONTROL (BOTH "
                "compiled); on its own it is an arm PREREG-k12 never "
                "registered and whose result nothing would interpret")
            print("K12 arm 3: paged attention NOT dynamo-disabled "
                  "either -- expected to reproduce the F1 Stage B "
                  "failure", flush=True)
        else:
            ALL_ATTENTION_FUNCTIONS[IMPL_NAME] = dynamo.disable(
                ALL_ATTENTION_FUNCTIONS[IMPL_NAME])
        # PREREG-k12: the two exclusions are applied together but for
        # DIFFERENT stated reasons -- the attention shim for host-bound
        # KV paging, the MoE forward for CPU TIER DISPATCH. At
        # --placement-override all-vram (which every certified serving
        # measurement uses) there is no CPU tier to dispatch to, so the
        # MoE half's necessity is an open question. This flag lets an
        # arm answer it WITHOUT touching the attention disable, whose
        # cause is known and specific (F1 Stage B).
        if a.compile_moe_tier:
            assert a.placement_override == "all-vram", (
                "--compile-moe-tier is registered only at the all-vram "
                "point: the exclusion it lifts exists for CPU tier "
                "dispatch, which only all-vram is known to avoid "
                "(PREREG-k12)")
            print("K12: MoE tier NOT dynamo-disabled (attention "
                  + ("also NOT -- arm 3" if a.compile_attn_tier
                     else "still is")
                  + ")", flush=True)
        else:
            for m in mods:
                m.forward = dynamo.disable(m.forward)
        n_c = 0
        layer_list = model
        for part in a.layers_attr.split("."):
            layer_list = getattr(layer_list, part)
        for lyr in layer_list:
            lyr.forward = torch.compile(lyr.forward, mode=a.compile_mode,
                                        dynamic=False)
            n_c += 1
        if "reduce-overhead" in a.compile_mode:
            COMPILE_GRAPH_STEP[0] = True
        # This line is read back out of arm logs, so it must describe
        # the configuration that RAN. It used to say "paged attention
        # + MoE tier dynamo-disabled" unconditionally -- including in
        # the --compile-moe-tier arm, where the MoE tier is precisely
        # NOT disabled, and where it contradicted the K12 line printed
        # a few lines above it.
        _dis = [n for n, off in (("paged attention", not a.compile_attn_tier),
                                 ("MoE tier", not a.compile_moe_tier)) if off]
        print(f"compiled {n_c} layer bodies (mode={a.compile_mode}); "
              f"dynamo-disabled: {' + '.join(_dis) if _dis else 'NOTHING'}; "
              f"graph step marking={COMPILE_GRAPH_STEP[0]}", flush=True)
    kv = None if a.ppl_oracle != "none" else Fp8PagedKV(L, hkv, hd, batch=a.batch,
                    max_tokens_per_seq=a.prompt_len
                    + max(a.gen_tokens, a.ppl_steps) + 8,
                    k_groups=(None if a.kv_groups == "auto" else int(a.kv_groups)),
                    batched_append=not a.kv_per_seq,
                    device="cuda")

    ids, step, prompts, ppl_ids, ppl_sha = _k8_window(a, tok)
    if a.ppl_oracle != "none":
        assert a.ppl_steps > 0, "--ppl-oracle needs --ppl-steps"
        _ppl_oracle_main(a, model, ppl_ids, ppl_sha)
        return

    # ------- timed runner: forward vs drain split, per-regime expert delta
    class TimedRunner(PagedModelRunner):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.decode_rows = []
            self.prefill_rows = []
            # the runner POPS a finished request's tokens (release), so
            # reading runner.tokens after the loop sees an empty dict --
            # which made every generated_tokens record EMPTY and every
            # cross-arm token-identity gate built on it vacuous
            # (t1/t1b/t5 G1; found by the B1 cycle). Capture at
            # generation time instead.
            self.gen_capture = {}
            # PREREG-s1-acceptance: prompt ids, captured at BIND time
            # (the scheduler hands bind() the prompt itself) for the
            # same reason gen_capture exists -- release pops
            # runner.tokens, so a report-time read gets nothing
            # (Bugbot, e4b#239: the first draft did exactly that)
            self.prompt_capture = {}

        def bind(self, rid, slot, prompt):
            self.prompt_capture[rid] = list(map(int, prompt))
            return super().bind(rid, slot, prompt)

        def _amort_snap(self):
            if not states or states[0].amort is None:   # off / pipelined
                return (0, 0)
            return (sum(st.amort["dram_ns"] for st in states),
                    sum(st.amort["gpu_ns"] for st in states))

        @torch.no_grad()
        def run_prefill(self, chunks):
            if COMPILE_GRAPH_STEP[0]:
                torch.compiler.cudagraph_mark_step_begin()
            PROF["mode"] = "prefill"
            d0, g0 = self._amort_snap()
            # decode-only accounting by construction: capture the tier
            # counters, let the prefill run (its own dram/gpu deltas are
            # recorded below, before the rollback), then restore — so
            # prefill chunks interleaved by the scheduler at ANY later
            # step never leak into --amort-out / --profile-out
            # (Bugbot, e4b#189).
            saved = []
            for st in states:
                am = st.amort
                saved.append(None if am is None else
                             {k2: (v.clone() if torch.is_tensor(v)
                                   else list(v) if isinstance(v, list)
                                   else v)
                              for k2, v in am.items()})
            t0 = time.perf_counter_ns()
            out = super().run_prefill(chunks)
            wall = time.perf_counter_ns() - t0
            for _rid, _tk in out.items():
                self.gen_capture.setdefault(_rid, []).append(int(_tk))
            d1, g1 = self._amort_snap()
            for st, am in zip(states, saved):
                if am is not None and st.amort is not None:
                    cur = st.amort
                    for k2, v in am.items():
                        if torch.is_tensor(v):
                            cur[k2].copy_(v)
                        else:
                            # lists were snapshot-copied above, so plain
                            # assignment drops any prefill appends
                            cur[k2] = v
            self.prefill_rows.append(
                {"chunks": len(chunks),
                 "tokens": sum(c[2] for c in chunks),
                 "wall_ns": wall, "dram_ns": d1 - d0, "gpu_ns": g1 - g0})
            PROF["mode"] = "decode"
            return out

        @torch.no_grad()
        def run_decode(self, rids):
            if COMPILE_GRAPH_STEP[0]:
                # the documented remedy for cudagraph replay reuse across
                # steps (T1's crash): declare the step boundary so outputs
                # of the previous replay are not read after overwrite
                torch.compiler.cudagraph_mark_step_begin()
            if not rids:
                return {}
            # duplicated from PagedModelRunner.run_decode by design: the
            # split being measured (forward submission vs drain) lives
            # INSIDE the method, so instrumentation must inline it
            self.ctx.mode = "decode"
            self.ctx.slots = [self.slot_of[r] for r in rids]
            ids_ = torch.tensor([[self.tokens[r][-1]] for r in rids],
                                dtype=torch.long, device=self.device)
            pos = torch.tensor([[self.pos_of[r] - 1] for r in rids],
                               dtype=torch.long, device=self.device)
            from experts4bit_qlora.engines.paged_attention import set_context
            ah0, ac0 = PROF["attn_host_ns"], PROF["attn_calls"]
            rm0 = REGION_PROF["moe"]["decode"]["ns"]
            rb0 = REGION_PROF["moe_block"]["decode"]["ns"]
            rl0 = REGION_PROF["lmhead"]["decode"]["ns"]
            d0, g0 = self._amort_snap()
            prev = set_context(self.ctx)
            t0 = time.perf_counter_ns()
            try:
                out = self.model(input_ids=ids_, position_ids=pos,
                                 use_cache=False)
            finally:
                set_context(prev)
            t_fwd = time.perf_counter_ns() - t0
            t1 = time.perf_counter_ns()
            toks = out.logits[:, -1].argmax(-1).tolist()
            t_drain = time.perf_counter_ns() - t1
            d1, g1 = self._amort_snap()
            self.decode_rows.append(
                {"batch": len(rids), "fwd_ns": t_fwd, "drain_ns": t_drain,
                 "attn_host_ns": PROF["attn_host_ns"] - ah0,
                 "attn_calls": PROF["attn_calls"] - ac0,
                 "moe_ns": REGION_PROF["moe"]["decode"]["ns"] - rm0,
                 "moe_block_ns":
                     REGION_PROF["moe_block"]["decode"]["ns"] - rb0,
                 "lmhead_ns": REGION_PROF["lmhead"]["decode"]["ns"] - rl0,
                 "dram_ns": d1 - d0, "gpu_ns": g1 - g0})
            got = {}
            for rid, tk in zip(rids, toks):
                got[rid] = int(tk)
                self.gen_capture.setdefault(rid, []).append(int(tk))
                self.tokens[rid].append(int(tk))
                self.pos_of[rid] += 1
            return got

    runner = TimedRunner(model, kv, device="cuda")
    sched = ContinuousScheduler(runner=runner, max_seqs=a.batch,
                                kv_slots=a.batch, chunk_tokens=a.chunk,
                                max_prefill_tokens_per_step=a.chunk)
    for p in prompts:
        sched.add_request(p, max_new_tokens=a.gen_tokens)

    tprof = None
    tprof_steps = [0]
    if (a.torch_profile_out or a.sync_attr_out
            or a.region_ops_out or a.ew_attr_out):
        from torch.profiler import (ProfilerActivity, profile, schedule)
        tprof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(skip_first=24, wait=0, warmup=2, active=12,
                              repeat=1),
            record_shapes=True,
            with_stack=bool(a.sync_attr_out))
        tprof.__enter__()
    _ew_tracer = None
    if a.ew_attr_out:
        # AMENDMENT-f1-tracer: torch 2.13's with_stack yields EMPTY
        # stacks, so call sites come from a dispatch-mode tracer that
        # depends only on python frames.
        _ew_tracer = _EwSiteTracer(_EW_OPS)
        _ew_tracer.__enter__()
    prof = None
    if a.cprofile_out:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    if a.grouping_parity:
        _grouping_parity(a, model, runner, sched, kv)
        return
    if a.ppl_steps and not a.b1d_loop:
        raise SystemExit(
            "--ppl-steps scores through the B=1 paged decode loop and "
            "needs --b1d-loop (eager|graph) to reach it. Without that "
            "flag this run silently produces an ORDINARY decode "
            "report, which is exactly the failure the K8 quality gate "
            "exists to avoid -- refusing instead (found live: the K8 "
            "ppl arms wrote default reports and the compose step's "
            "assertion was the only thing that noticed).")
    if a.ppl_steps and a.batch > 1:
        raise SystemExit(
            f"--ppl-steps is a B=1 instrument; got --batch {a.batch}")
    if getattr(a, "graph_break_census", False):
        # The ppl block returns before the census site, so these two
        # together would write a perplexity report and silently skip
        # the census -- an arm that runs, exits 0 and measures nothing,
        # which is exactly how K8's quality arms scored nothing while
        # looking healthy. Refuse instead of choosing one.
        if a.ppl_steps:
            raise SystemExit(
                "--graph-break-census and --ppl-steps are different "
                "instruments; the ppl path returns first, so passing "
                "both would silently skip the census")
        if not a.b1d_loop:
            raise SystemExit(
                "--graph-break-census lives on the b1d stage; pass "
                "--b1d-loop or it never runs")
        if not a.compile_layers:
            raise SystemExit(
                "--graph-break-census with nothing compiled censuses "
                "an untraced region and would report zero breaks for "
                "a reason unrelated to the MoE tier (PREREG-k13)")
        if a.batch > 1:
            # --batch DEFAULTS TO 4, and batch > 1 routes to
            # _bv3_stage, which never reaches the census site and
            # writes a BV3 timing report to --out instead. That is
            # the same silent-skip these guards exist to catch, on
            # the DEFAULT path (review, e4b#290).
            raise SystemExit(
                f"--graph-break-census is a B=1 instrument; got "
                f"--batch {a.batch}. batch > 1 routes to the BV3 "
                "stage, which never runs the census and would write "
                "a timing report to --out instead")
    if a.b1d_loop:
        _ov = (None if a.router_sorted is None
               else a.router_sorted == "true")
        if a.batch > 1:
            _bv3_stage(a, model, runner, sched, kv)
        elif a.router_probe or _ov is not None:
            with _TopkProbe(sorted_override=_ov) as _pr:
                _b1d_stage_a(a, model, runner, sched, kv,
                             ppl_ids=ppl_ids, ppl_sha=ppl_sha,
                             probe=_pr)
        else:
            _b1d_stage_a(a, model, runner, sched, kv,
                         ppl_ids=ppl_ids, ppl_sha=ppl_sha)
        return

    step_walls = []            # decode-ONLY steps: a wall that included a
    while sched.active or sched.queue:   # prefill chunk would smear into
        pf0 = len(runner.prefill_rows)   # sched_py and mis-attribute
        dr0 = len(runner.decode_rows)
        t0 = time.perf_counter_ns()
        if sched.step().is_empty:
            break
        wall = time.perf_counter_ns() - t0
        if len(runner.prefill_rows) == pf0 and len(runner.decode_rows) > dr0:
            step_walls.append(wall)
            if tprof is not None:
                tprof.step()
                tprof_steps[0] += 1
    torch.cuda.synchronize()
    if tprof is not None:
        tprof.__exit__(None, None, None)
        tbl = tprof.key_averages().table(sort_by="cuda_time_total",
                                         row_limit=(400 if a.ew_attr_out
                                                    else 80))
        # the schedule fills its active window only after skip_first(24)
        # + warmup(2) decode steps; label the receipt with the ACTUAL
        # window so a short run cannot masquerade as a full attribution
        active = max(0, min(12, tprof_steps[0] - 24 - 2))
        hdr = (f"profiled decode steps: {tprof_steps[0]} "
               f"(active window: {active}/12)\n")
        if active < 12:
            hdr += ("WARNING: active window INCOMPLETE -- this table "
                    "under-samples and must not be cited as the "
                    "attribution\n")
        if a.torch_profile_out:
            Path(a.torch_profile_out).write_text(hdr + tbl)
            print(f"TORCH_PROFILE_OUT {a.torch_profile_out} "
                  f"active={active}/12", flush=True)
        if a.sync_attr_out:
            # counts over the ACTIVE window; the verdict divides by
            # `active_steps` -- never assume the window filled
            counts = {}
            for evt in tprof.key_averages():
                counts[evt.key] = counts.get(evt.key, 0) + evt.count
            nz = {"engine": 0, "other": 0, "frames": {}}
            for evt in tprof.key_averages(group_by_stack_n=24):
                if evt.key != "aten::nonzero" or not evt.count:
                    continue
                frames = [f for f in (evt.stack or []) if ".py" in f]
                site = next((f for f in frames
                             if "hot_residency.py" in f or "hybrid.py" in f
                             or "nvme_experts.py" in f), None)
                bucket = "engine" if site else "other"
                nz[bucket] += evt.count
                label = site or (frames[0] if frames else "<no-py-frame>")
                nz["frames"][label] = nz["frames"].get(label, 0) + evt.count
            Path(a.sync_attr_out).write_text(json.dumps({
                "active_steps": active,
                "dispatch_diet": bool(a.dispatch_diet),
                "op_counts": {kk: counts.get(kk, 0) for kk in
                              ("aten::nonzero", "aten::copy_", "aten::to",
                               "aten::_to_copy",
                               "aten::index_select", "aten::index_put_",
                               "aten::unique2", "aten::sort",
                               "aten::arange", "aten::item",
                               "aten::_local_scalar_dense",
                               "cudaLaunchKernel", "cudaMemcpyAsync",
                               "cudaStreamSynchronize",
                               "cudaDeviceSynchronize")},
                "nonzero_attr": nz,
            }, indent=1))
            print(f"SYNC_ATTR_OUT {a.sync_attr_out} active={active}/12",
                  flush=True)
        if _ew_tracer is not None:
            _ew_tracer.__exit__(None, None, None)
        if a.ew_attr_out:
            # F1 Stage A: which python call sites own the elementwise
            # device block. Uses the OP view on purpose -- kernel rows
            # carry no python stack -- and reports its own unclassified
            # remainder so a missing op cannot shrink the block silently.
            attr = _ew_attribute(tprof.key_averages())
            if _ew_tracer is not None and _ew_tracer.counts:
                attr["by_site"] = _apportion(attr["by_op"],
                                             _ew_tracer.counts)
                attr["site_method"] = ("dispatch-tracer counts, device "
                                       "time apportioned per op by "
                                       "launch share")
            else:
                attr["site_method"] = "NONE -- tracer recorded no ops"
            unacc = sum(v["us"] for v in
                        attr["unclassified_ops"].values())
            Path(a.ew_attr_out).write_text(json.dumps({
                "active_steps": active,
                "window_complete": active >= 12,
                "attributed_us_total": attr["attributed_us"],
                "attributed_us_per_step": (attr["attributed_us"] / active
                                           if active else None),
                "unclassified_us_per_step": (unacc / active
                                             if active else None),
                "site_method": attr.get("site_method"),
                "by_site": attr["by_site"],
                "by_op": attr["by_op"],
                "unclassified_ops": attr["unclassified_ops"],
            }, indent=1))
            print(f"EW_ATTR_OUT {a.ew_attr_out} active={active}/12 "
                  f"attributed="
                  f"{attr['attributed_us']/max(active,1)/1000:.2f}ms/step "
                  f"unclassified={unacc/max(active,1)/1000:.2f}ms/step",
                  flush=True)
        if a.region_ops_out:
            evs = tprof.profiler.function_events
            regions = {"e4b::moe": {}, "e4b::moe_block": {},
                       "e4b::attn": {}, "e4b::lmhead": {}}
            rcounts = dict.fromkeys(regions, 0)
            rtime = dict.fromkeys(regions, 0.0)

            def _walk(ev, bag):
                for c in ev.cpu_children:
                    if c.name.startswith("aten::"):
                        bag[c.name] = bag.get(c.name, 0) + 1
                    _walk(c, bag)

            for ev in evs:
                if ev.name in regions:
                    rcounts[ev.name] += 1
                    rtime[ev.name] += ev.cpu_time_total
                    _walk(ev, regions[ev.name])
            # a silent no-match must fail loudly, never read as zero
            # (PREREG-t5b): every region must appear at its call rate
            assert rcounts["e4b::moe"] >= L * max(1, active), rcounts
            assert rcounts["e4b::moe_block"] >= L * max(1, active), rcounts
            assert rcounts["e4b::attn"] >= L * max(1, active), rcounts
            assert rcounts["e4b::lmhead"] >= max(1, active), rcounts
            Path(a.region_ops_out).write_text(json.dumps({
                "active_steps": active,
                "layers": L,
                "region_calls": rcounts,
                "region_cpu_ms_total": {k: v / 1e3
                                        for k, v in rtime.items()},
                "region_ops": regions,
            }, indent=1))
            print(f"REGION_OPS_OUT {a.region_ops_out} "
                  f"moe={rcounts['e4b::moe']} attn={rcounts['e4b::attn']} "
                  f"lmhead={rcounts['e4b::lmhead']}", flush=True)
    if prof is not None:
        prof.disable()
        import io
        import pstats
        buf = io.StringIO()
        st_ = pstats.Stats(prof, stream=buf)
        st_.sort_stats("cumulative").print_stats(60)
        Path(a.cprofile_out).write_text(buf.getvalue())
        print(f"CPROFILE_OUT {a.cprofile_out}", flush=True)

    # device-side attention occupancy from the recorded events
    attn_dev = {"prefill": 0.0, "decode": 0.0}
    for mode, e0, e1 in PROF["attn_events"]:
        attn_dev[mode] += e0.elapsed_time(e1)

    dr = runner.decode_rows
    n_full = len(dr)
    n_warm = 4 if a.compile_layers else 0
    n_dropped = 0
    if n_warm and len(dr) > 2 * n_warm:
        dr = dr[n_warm:]
        n_dropped = n_warm
    med = lambda key: statistics.median(r[key] for r in dr) / 1e6
    n_steps = len(dr)
    step_ms = statistics.median(step_walls[-n_steps:]) / 1e6 if dr else 0
    fwd, drain = med("fwd_ns"), med("drain_ns")
    attn_h, dram = med("attn_host_ns"), med("dram_ns")
    gpu_dev = med("gpu_ns")
    other_sub = fwd - attn_h - dram
    sched_py = step_ms - fwd - drain
    rep = {
        "model": a.model, "batch": a.batch, "layers": L,
        "compile_layers": bool(a.compile_layers),
        "compile_moe_tier": bool(a.compile_moe_tier),
        "compile_attn_tier": bool(a.compile_attn_tier),
        "compile_mode": a.compile_mode if a.compile_layers else None,
        "fuse_qkv": bool(a.fuse_qkv),
        "warmup_rows_dropped": n_dropped,
        # the cross-arm void gate: greedy continuations must be
        # token-identical between eager and compiled arms
        "generated_tokens": {str(r): list(map(int, t))
                             for r, t in
                             sorted(runner.gen_capture.items())},
        # PREREG-s1-acceptance: the drafter matches against the FULL
        # visible context. Captured at prefill time -- release pops
        # runner.tokens, so a report-time read would KeyError or hand
        # the drafter empty prompts (Bugbot, e4b#239)
        "prompt_tokens": {str(r): runner.prompt_capture.get(r, [])[
                              :a.prompt_len]
                          for r in sorted(runner.gen_capture)},
        "decode_steps": n_steps,
        "decode_median_ms": {
            "step": step_ms, "forward_submission": fwd, "drain": drain,
            "attention_host": attn_h, "dram_experts_host": dram,
            "other_submission": other_sub,
            "scheduler_python_and_bookkeeping": sched_py,
        },
        "decode_device_ms": {
            "attention_kernels_per_step":
                attn_dev["decode"] / max(1, n_full),
            "gpu_expert_kernels_per_step": gpu_dev,
        },
        "attn_calls_per_step": (statistics.median(r["attn_calls"]
                                                  for r in dr) if dr else 0),
        "attn_host_us_per_call": (attn_h * 1e3 / L if L else 0),
        "prefill": [{"tokens": r["tokens"],
                     "wall_ms": r["wall_ns"] / 1e6,
                     "dram_ms": r["dram_ns"] / 1e6,
                     "gpu_dev_ms": r["gpu_ns"] / 1e6}
                    for r in runner.prefill_rows],
        "prefill_attn_dev_total_ms": attn_dev["prefill"],
    }
    if a.profile_out:
        # mass semantics match load_routing_mass: tokens_routed accumulates
        # raw selection counts; routing_probabilities divides by the layer
        # total and multiplies by top_k, so p_e = count_e / decode_tokens.
        with open(a.profile_out, "w") as f:
            for li, st in enumerate(states):
                hist = st.amort["hist"].cpu().tolist()
                for e, c in enumerate(hist):
                    if c:
                        f.write(json.dumps({"row": "expert", "layer_id": li,
                                            "expert_id": e,
                                            "tokens_routed": int(c)}) + "\n")
        print(f"PROFILE_OUT {a.profile_out}", flush=True)
    if a.series_out:
        import gzip
        ser = []
        for st in states:
            ser.append([u.cpu().tolist() for u in st.amort["series"]])
        n_steps_series = {len(x) for x in ser}
        assert len(n_steps_series) == 1, \
            f"layers disagree on series length: {n_steps_series}"
        with gzip.open(a.series_out, "wt") as f:
            json.dump({"per_layer_series": ser}, f)
        print(f"SERIES_OUT {a.series_out} steps={n_steps_series.pop()}",
              flush=True)
    if a.amort_out:
        per_layer = []
        for st in states:
            am = st.amort
            row = {k2: int(am[k2]) for k2 in
                   ("steps", "acts", "uniq_vram", "uniq_dram",
                    "uniq_nvme", "acts_vram", "acts_dram", "acts_nvme",
                    "dram_steps")}
            row["touch"] = am["touch"].cpu().tolist()
            row["hist"] = am["hist"].cpu().tolist()
            per_layer.append(row)
        Path(a.amort_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.amort_out).write_text(json.dumps({
            "vram_gb": a.vram_gb, "batch": a.batch, "top_k": k,
            "geometry": man["geometry"],
            "manifest_counts": {t: len(p) for t, p in man["tiers"].items()},
            "manifest_vram": man["tiers"]["vram"],
            "decode_steps": n_steps,
            "decode_median_ms": rep["decode_median_ms"],
            "per_layer": per_layer,
        }, indent=1))
        print(f"AMORT_OUT {a.amort_out}", flush=True)
    rep["engine"] = a.engine
    rep["placement_override"] = a.placement_override
    rep["collapse"] = not a.no_collapse
    rep["manifest_counts"] = (None if man is None else
                              {t: len(pp) for t, pp in man["tiers"].items()})
    if a.host_brackets:
        moe_h, lmh_h = med("moe_ns"), med("lmhead_ns")
        blk_h = med("moe_block_ns")
        rep["decode_median_ms"]["moe_host"] = moe_h
        rep["decode_median_ms"]["moe_block_host"] = blk_h
        rep["decode_median_ms"]["router_topk_host"] = blk_h - moe_h
        rep["decode_median_ms"]["lmhead_host"] = lmh_h
        # dram is INSIDE the moe bracket -- never sum them; the residual
        # is what no region owns
        rep["decode_median_ms"]["host_residual"] = (
            fwd - attn_h - moe_h - lmh_h)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    d = rep["decode_median_ms"]
    print(f"DECOMP step={d['step']:.1f}ms  fwd_submit={d['forward_submission']:.1f} "
          f"(attn_host={d['attention_host']:.1f} dram={d['dram_experts_host']:.1f} "
          f"other={d['other_submission']:.1f})  drain={d['drain']:.1f}  "
          f"sched_py={d['scheduler_python_and_bookkeeping']:.1f}", flush=True)
    print(f"DEVICE attn={rep['decode_device_ms']['attention_kernels_per_step']:.2f}ms/step "
          f"gpu_experts={rep['decode_device_ms']['gpu_expert_kernels_per_step']:.2f}ms/step "
          f"attn_host_us_per_call={rep['attn_host_us_per_call']:.0f}",
          flush=True)
    for r in rep["prefill"][:4]:
        print(f"PREFILL tokens={r['tokens']} wall={r['wall_ms']:.0f}ms "
              f"dram={r['dram_ms']:.0f} gpu_dev={r['gpu_dev_ms']:.0f}",
              flush=True)
    print("STEP_DECOMP_DONE", flush=True)


if __name__ == "__main__":
    main()
