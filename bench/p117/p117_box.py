#!/usr/bin/env python3
"""Lane P117's measurement (bench/p117/PREREG-p117.md; e4b#846): does decoding 32 or 64 rows in ONE graph, as
``E4B_PAGED_BUCKETS=auto`` does above 16 slots, cost quality against today's decode in pieces of at most 16 rows,
judged against the 16-row arithmetic's own neutral perturbations?

Lane SC2e (``bench/h2h-2026-10-02/sc2e``) read ``SLOTS_LICENSED(64, auto)``: one 64-row decode graph runs in half the
time of four 16-row replays, and the capacity ceiling rises from 8 to 12 req/s. Above 16 rows the decode step takes the
paths prefill chunks take today -- ``Int4Linear``'s cached bf16 weight above 16 rows, the chained tile table above 256
routed expert rows (64 rows at top-k 8) -- and under load SC2e's wide-bucket arms kept 0.01-0.12 of the 16-slot
server's text against 0.46-0.74 for 64 slots on the default list. Text agreement cannot say which arithmetic is better.
The ruling on #1320: ``E4B_PAGED_BUCKETS=auto`` becomes a default only after this teacher-forced read. Derived from
``bench/p110/p110_box.py`` (P110's method) by named substitutions.

The model is SC2e's served stack: ``PagedServeConfig.from_env()`` + ``build_engine`` with SC1's int4 levers, the three
folds and fused q/k/v, built eager with one tiny slot (its own pool is never used). The box runs its own
teacher-forced paged passes on that engine's model (``parts.runner.model``), each with a fresh ``Fp8PagedKV`` and
``PagedModelRunner``, every one device-grouped (the graph server's grouping) and every decode step through the bucketed
path with ``capture=False``: a padded eager step of the step's bucket, bit-identical to its replay (P109's G = P at 16
rows; G64 below checks it at 64).

Windows: wikitext-2-raw test, joined as K8 joins it; window k from token k * ``--stride`` (3072, so 64 windows fit the
split): ``--prompt`` tokens of prompt and ``--cont`` teacher-forced positions, scored on fp32 log-probs (the true
token's NLL, the argmax, KL against R). One group of ``--group`` windows (64).

Arms, each a pass over the group with its own bucket list (the runner splits a step into pieces of the largest bucket):
- **R**, the reference: buckets 1-16, so every 64-row step runs as four 16-row pieces -- the arithmetic class SC2e's
  s64c served and the #1320 ruling licenses without a quality read.
- **rep**: R again. A floor draw if it is not bit-identical.
- **floor**, arithmetically neutral perturbations of R: **half** (buckets 1-8: eight 8-row pieces), **chunk** (prompts
  prefilled in ``--floor-chunk``-token pieces), **rev** (windows bound to slots in reverse and decoded in reverse order).
- **W32**: buckets 1-32, two 32-row pieces (``Int4Linear`` above 16 rows; 256 routed expert rows).
- **W64**: buckets 1-64, one 64-row piece (the chained tile table above 256 routed rows). **The subject.**
- **W64pad**: the first 48 windows on buckets 1-64: each step padded from 48 to 64 rows with scratch slots.
- **mutant_scale**: W64 with the decode attention's softmax scale halved (P108's mutant). The bar must catch it.
- **G64**: W64's buckets with the graphs CAPTURED; the replay's argmax at every position must equal W64's (FUNCTION).

Engagement per pass: decode attention calls (P108's ``Attention`` patch), the grouping flags in force, and the runner's
bucket statistics (pieces per step, eager steps or replays, padding rows).

``measure()`` takes the model and the windows, so ``tests/test_p117_box.py`` runs it on CPU on a tiny Qwen3-MoE with the
decode attention and the bucket path stood in.

    python p117_box.py --out OUT.json [--windows 64 --group 64 --prompt 512 --cont 128]   (engine from E4B_PAGED_* env)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p108_box  # noqa: E402  (staged at P108's registered bytes: Attention, _score, _kl, _release; imports p97_box)
import p97_box  # noqa: E402  (staged at P97's registered bytes: _sync)

B16, B8, B32, B64 = (1, 2, 4, 8, 16), (1, 2, 4, 8), (1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32, 64)
FLOORS = ("half", "chunk", "rev")
SUBJECTS = ("W32", "W64", "W64pad")
ARMS = ("R", "rep") + FLOORS + SUBJECTS + ("mutant_scale", "G64")
PAD_WINDOWS = 48
ARM_KW = {"R": {"buckets": B16}, "rep": {"buckets": B16}, "half": {"buckets": B8}, "chunk": {"buckets": B16},
          "rev": {"buckets": B16, "reverse": True}, "W32": {"buckets": B32}, "W64": {"buckets": B64},
          "W64pad": {"buckets": B64, "first": PAD_WINDOWS}, "mutant_scale": {"buckets": B64, "mutant": "scale"},
          "G64": {"buckets": B64, "capture": True}}


def windows(tok, n, prompt, cont, stride=3072):
    """The k-th window starts at token k * ``stride`` of wikitext-2-raw test, joined as the K8 corpus joins it."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    out = []
    for k in range(n):
        w = ids[k * stride:k * stride + prompt + cont]
        assert w.numel() == prompt + cont, f"window {k} has {w.numel()} tokens"
        out.append(w.tolist())
    return out


def pieces_of(rows, top):
    """The runner's split (``paged_runner.chunk_rows``): consecutive pieces of at most the largest bucket."""
    return [rows[i:i + top] for i in range(0, len(rows), top)]


def paged_pass(model, ws, P, C, chunk, device, *, buckets, reverse=False, mutant=None, capture=False, stand_in=False):
    """One teacher-forced pass of the group ``ws`` through the bucketed decode path; returns (per-window fp32 log-probs
    [C, V] -- None under ``capture``, whose replays never call the forward -- per-window tokens the runner emitted at
    each decode step (its own argmax, the same buffer a replay or a padded eager step writes), engagement)."""
    from experts4bit_qlora.engines import hot_residency as hr
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry

    cfg = getattr(model.config, "text_config", None) or model.config
    hkv, hd = _kv_geometry(model.config)
    n = len(ws)
    kv = Fp8PagedKV(kv_layers(model, int(cfg.num_hidden_layers)), hkv, hd, batch=n, max_tokens_per_seq=P + C + 16,
                    device=device, scratch_slots=max(buckets))
    runner = PagedModelRunner(model, kv, device=device)
    saved = (hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0])
    fwd = []
    inner = model.forward

    def keep(*a, **kw):
        o = inner(*a, **kw)
        fwd.append(o.logits)
        return o

    order = list(range(n))[::-1] if reverse else list(range(n))
    lps = [[] for _ in ws]
    tk = [[] for _ in ws]
    eng = {"buckets": list(buckets), "reverse": bool(reverse), "capture": bool(capture), "windows": n}
    try:
        hr.DEVICE_GROUPING[0] = True                     # the graph server's grouping (serve_paged sets it above 1 seq)
        hr.FORCE_SINGLETON_GROUPS[0] = False
        eng["graph_status"] = {str(k): v for k, v in
                               runner.enable_decode_graphs(buckets, capture=capture, verbose=False).items()}
        model.forward = keep
        with p108_box.Attention(stand_in, mutant) as att, torch.no_grad():
            for rid in range(n):
                runner.bind(rid, (n - 1 - rid) if reverse else rid, ws[rid][:P])
            for rid in order:
                for s in range(0, P, chunk):
                    fwd.clear()
                    runner.run_prefill([(rid, s, min(chunk, P - s))])
                lp = fwd[-1][0, -1].float().log_softmax(-1).cpu()
                if not capture:
                    lps[rid].append(lp)
                runner.tokens[rid][-1] = ws[rid][P]                       # teacher-forced
            p97_box._sync(device)
            t1 = time.time()
            calls0 = len(att.calls)
            for t in range(C - 1):
                fwd.clear()
                got = runner.run_decode(order)
                for rid in order:
                    tk[rid].append(int(got[rid]))
                if not capture:
                    pcs = pieces_of(order, max(buckets))
                    if len(fwd) != len(pcs):
                        raise RuntimeError(f"step {t}: {len(fwd)} forwards for {len(pcs)} pieces")
                    for piece, lg in zip(pcs, fwd):
                        lg = lg[:, -1].float().log_softmax(-1).cpu()        # a piece's rows first, its padding after
                        for j, rid in enumerate(piece):
                            lps[rid].append(lg[j])
                for rid in order:
                    runner.tokens[rid][-1] = ws[rid][P + t + 1]
            p97_box._sync(device)
            eng["step_ms"] = round(1000 * (time.time() - t1) / max(C - 1, 1), 2)
            eng["decode_calls"] = len(att.calls) - calls0
        eng["grouping_flags_in_pass"] = {"device_grouping": bool(hr.DEVICE_GROUPING[0]),
                                         "force_singleton_groups": bool(hr.FORCE_SINGLETON_GROUPS[0])}
        gs = getattr(runner, "graph_stats", None)
        eng["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) and gs else None
    finally:
        model.forward = inner
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = saved
    del runner, kv
    return (None if capture else [torch.stack(x) for x in lps]), tk, eng


def measure(model, ws_all, *, prompt, cont, chunk, floor_chunk, device, stand_in=False, arms=ARMS):
    from experts4bit_qlora.engines import paged_attention
    P, C = prompt, cont
    paged_attention.register(model)
    per = {a: [] for a in arms if a != "G64"}
    eng = {a: [] for a in arms}
    start = time.time()
    refs, _, e = paged_pass(model, ws_all, P, C, chunk, device, stand_in=stand_in, **ARM_KW["R"])
    eng["R"].append(e)
    ref_am = [r.argmax(-1).tolist() for r in refs]
    rep_identical = None
    function = None

    def add(arm, wi, lp, ref):
        nll, a = p108_box._score(lp, ws_all[wi][P:P + C])
        rec = {"window": wi, "nll": sum(nll) / len(nll), "argmax_agree": sum(int(x == y) for x, y in zip(a, ref_am[wi])) / len(a)}
        if ref is not None:
            kl = p108_box._kl(ref, lp)
            rec["kl"] = sum(kl) / len(kl)
        per[arm].append(rec)

    for i, r in enumerate(refs):
        add("R", i, r, None)
    w64_tk = None
    for arm in arms:
        if arm in ("R", "G64"):
            continue
        kw = dict(ARM_KW[arm])
        first = kw.pop("first", None)
        ws = ws_all[:first] if first else ws_all
        lps, tk, e = paged_pass(model, ws, P, C, floor_chunk if arm == "chunk" else chunk, device, stand_in=stand_in, **kw)
        eng[arm].append(e)
        if arm == "rep":
            rep_identical = all(torch.equal(a, b) for a, b in zip(lps, refs))
        if arm == "W64":
            w64_tk = tk
        for i, x in enumerate(lps):
            add(arm, i, x, refs[i])
        del lps
        p108_box._release(device)
        print(f"P117_ARM {arm} done at {time.time() - start:.0f} s", flush=True)
    if "G64" in arms:
        _, g_tk, e = paged_pass(model, ws_all, P, C, chunk, device, stand_in=stand_in, **ARM_KW["G64"])
        eng["G64"].append(e)
        if w64_tk is not None:                 # the replay's emitted tokens against its padded eager step's, every step
            diff = sum(1 for a, b in zip(g_tk, w64_tk) for x, y in zip(a, b) if x != y)
            function = {"positions": sum(len(a) for a in g_tk), "differ": diff}
        print(f"P117_ARM G64 done at {time.time() - start:.0f} s", flush=True)
    del refs
    p108_box._release(device)
    cfg = getattr(model.config, "text_config", None) or model.config
    return {"windows": len(ws_all), "group": len(ws_all), "pad_windows": PAD_WINDOWS, "prompt": P, "cont": C,
            "chunk": chunk, "floor_chunk": floor_chunk, "layers": int(cfg.num_hidden_layers),
            "rep_identical": rep_identical, "function": function,
            "rehearsal": {"stand_in_attention": bool(stand_in)}, "per_window": per, "engagement": eng,
            "seconds": round(time.time() - start, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--floor-chunk", type=int, default=256)
    ap.add_argument("--stride", type=int, default=3072)
    a = ap.parse_args()
    import transformers

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram" or cfg.max_seqs != 1:
        raise SystemExit(f"REFUSED: the engine is built eager at all-vram with one slot (graphs={cfg.graphs}, "
                         f"placement={cfg.placement}, max_seqs={cfg.max_seqs}); the passes build their own pools")
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    ws = windows(parts.tokenizer, a.windows, a.prompt, a.cont, a.stride)
    rec = measure(model, ws, prompt=a.prompt, cont=a.cont, chunk=a.chunk, floor_chunk=a.floor_chunk, device=cfg.device)
    rec = {"model": cfg.model, "revision": cfg.revision, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "transformers": transformers.__version__, "load_s": round(load_s, 1),
           "stride": a.stride,
           "census": {k: parts.info.get(k) for k in ("moe_layers", "experts", "top_k", "model_type", "int4_expert_layers",
                                                     "int4_attn_projections", "int4_store_kinds", "fuse_qkv_n",
                                                     "fuse_t1_glue_n", "fuse_router_epilogue_n", "levers_env")},
           **rec, "gpu": torch.cuda.get_device_name(0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    open(a.out, "w").write(json.dumps(rec, indent=1))
    pw = rec["per_window"]

    def bias(arm):
        r = {x["window"]: x["nll"] for x in pw["R"]}
        d = [x["nll"] - r[x["window"]] for x in pw[arm]]
        return sum(d) / len(d), sum(abs(v) for v in d) / len(d)

    line = " | ".join(f"{arm} bias {bias(arm)[0]:+.5f} spread {bias(arm)[1]:.5f}" for arm in pw if arm != "R")
    print(f"P117_BOX: windows {rec['windows']} | {line} | rep identical {rec['rep_identical']} | function {rec['function']} "
          f"| load {load_s:.0f}s | mem {rec['max_mem_gb']} GB | {rec['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
