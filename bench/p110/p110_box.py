#!/usr/bin/env python3
"""Lane P110's measurement (bench/p110/PREREG-p110.md; e4b#770): does the arithmetic decode graphs bring to the default
``serve_paged`` server -- device grouping and bucket padding -- cost quality against the eager default's, judged against
the eager default's own arithmetically neutral perturbations?

P109 read the graph server ×5.60 / ×9.02 the eager default and its replay bit-identical to its padded eager step (G ≡ P),
but its tokens left the eager default's within 16 tokens on 7 of 16 rows -- and device grouping alone, no graphs, did the
same. Token agreement cannot say which arithmetic is better. This box scores teacher-forced NLL.

The model is the served stack: ``PagedServeConfig.from_env()`` + ``build_engine`` (graphs off) builds the default server
exactly as P109 did, and the box runs its own teacher-forced paged passes on that engine's model (``parts.runner.model``).
G's arithmetic is read through P, its padded eager step: P109 read G ≡ P bitwise on this model, and a graph replay never
calls the model's forward, where the logits are read.

Windows: wikitext-2-raw test, window k from token k * 4096 (P97's ``wikitext_windows``): ``--prompt`` tokens of prompt
and ``--cont`` teacher-forced positions, scored on fp32 log-probs (the true token's NLL, the argmax, KL against R).
Groups of ``--group`` windows decode together; 12 is not a bucket size, so P pads every step to 16.

Arms, each a fresh ``Fp8PagedKV`` + ``PagedModelRunner`` on the same model, per group:
- **R**, the reference: the eager default. Host grouping (``hot_residency.DEVICE_GROUPING`` off), unpadded, the group
  decoded together, each prompt prefilled in one ``--chunk`` (512).
- **rep**: R again (the first group only). A floor draw if it is not bit-identical.
- **floor**, arithmetically neutral perturbations of R:
  - **half**: the group decoded as two halves (each step two decode calls), a different row count;
  - **chunk**: prompts prefilled in ``--floor-chunk``-token pieces (256);
  - **rev**: windows bound to slots in reverse and decoded in reverse row order.
- **D**: device grouping on (as ``build_engine`` sets it for graphs), unpadded.
- **P**, the subject: device grouping on, every decode step through the bucketed path with ``capture=False``, so the
  group's rows are padded to the next bucket with scratch slots. This is the graph server's arithmetic.
- **mutant_scale**: P with the decode attention's softmax scale halved (P108's mutant). The bar must catch it.

Engagement is counted per arm: the decode attention calls (P108's ``Attention`` patch on ``Fp8PagedKV``), the grouping
flags in force, and for P and the mutant the runner's bucket statistics (every step an eager padded bucket step).

``measure()`` takes the model and the windows, so ``tests/test_p110_box.py`` runs it on CPU on a tiny Qwen3-MoE with the
decode attention stood in (every arm but P, whose bucket path needs the fp8 KV kernels).

    python p110_box.py --out OUT.json [--windows 48 --group 12 --prompt 512 --cont 128]   (engine from E4B_PAGED_* env)
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
import p97_box  # noqa: E402  (staged at P97's registered bytes: wikitext_windows, _sync)

BUCKETS = (1, 2, 4, 8, 16)
FLOORS = ("half", "chunk", "rev")
ARMS = ("R", "rep") + FLOORS + ("D", "P", "mutant_scale")


def paged_pass(model, ws, P, C, chunk, device, *, device_grouping=False, padded=False, halves=False, reverse=False,
               mutant=None, stand_in=False):
    """One teacher-forced paged pass of the group ``ws``; returns (per-window fp32 log-probs [C, V], engagement)."""
    from experts4bit_qlora.engines import hot_residency as hr
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry

    cfg = getattr(model.config, "text_config", None) or model.config
    hkv, hd = _kv_geometry(model.config)
    n = len(ws)
    kv = Fp8PagedKV(kv_layers(model, int(cfg.num_hidden_layers)), hkv, hd, batch=n, max_tokens_per_seq=P + C + 16,
                    device=device, scratch_slots=(max(BUCKETS) if padded else 0))
    runner = PagedModelRunner(model, kv, device=device)
    saved = (hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0])
    last = {}
    inner = model.forward

    def keep(*a, **kw):
        o = inner(*a, **kw)
        last["logits"] = o.logits
        return o

    order = list(range(n))[::-1] if reverse else list(range(n))
    lps = [[] for _ in ws]
    eng = {"device_grouping": bool(device_grouping), "padded": bool(padded), "halves": bool(halves), "reverse": bool(reverse)}
    try:
        hr.DEVICE_GROUPING[0] = bool(device_grouping)
        hr.FORCE_SINGLETON_GROUPS[0] = False
        if padded:
            eng["graph_status"] = {str(k): v for k, v in runner.enable_decode_graphs(BUCKETS, capture=False, verbose=False).items()}
        model.forward = keep
        with p108_box.Attention(stand_in, mutant) as att, torch.no_grad():
            for rid in range(n):
                runner.bind(rid, (n - 1 - rid) if reverse else rid, ws[rid][:P])
            for rid in order:
                for s in range(0, P, chunk):
                    runner.run_prefill([(rid, s, min(chunk, P - s))])
                lps[rid].append(last["logits"][0, -1].float().log_softmax(-1).cpu())
                runner.tokens[rid][-1] = ws[rid][P]                       # teacher-forced
            parts = [order[:n // 2], order[n // 2:]] if halves else [order]
            p97_box._sync(device)
            t1 = time.time()
            for t in range(C - 1):
                for part in parts:
                    runner.run_decode(part)
                    lg = last["logits"][:, -1].float().log_softmax(-1).cpu()   # rows in call order; a bucket's pad after
                    for j, rid in enumerate(part):
                        lps[rid].append(lg[j])
                for rid in order:
                    runner.tokens[rid][-1] = ws[rid][P + t + 1]
            p97_box._sync(device)
            eng["step_ms"] = round(1000 * (time.time() - t1) / max(C - 1, 1), 2)
        eng["decode_calls"] = len(att.calls)
        eng["grouping_flags_in_pass"] = {"device_grouping": bool(hr.DEVICE_GROUPING[0]),
                                         "force_singleton_groups": bool(hr.FORCE_SINGLETON_GROUPS[0])}
        gs = getattr(runner, "graph_stats", None)
        eng["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) and gs else None
    finally:
        model.forward = inner
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = saved
    del runner, kv
    return [torch.stack(x) for x in lps], eng


ARM_KW = {"R": {}, "rep": {}, "half": {"halves": True}, "chunk": {}, "rev": {"reverse": True},
          "D": {"device_grouping": True}, "P": {"device_grouping": True, "padded": True},
          "mutant_scale": {"device_grouping": True, "padded": True, "mutant": "scale"}}


def measure(model, windows, *, prompt, cont, chunk, floor_chunk, group, device, stand_in=False, arms=ARMS):
    from experts4bit_qlora.engines import paged_attention
    P, C = prompt, cont
    paged_attention.register(model)
    per = {a: [] for a in arms}
    eng = {a: [] for a in arms}
    rep_identical = None
    start = time.time()

    def add(arm, wi, lp, ref_lp, cont_tokens, ref_argmax):
        nll, am = p108_box._score(lp, cont_tokens)
        rec = {"window": wi, "nll": sum(nll) / len(nll), "argmax_agree": sum(int(a == b) for a, b in zip(am, ref_argmax)) / len(am)}
        if ref_lp is not None:
            kl = p108_box._kl(ref_lp, lp)
            rec["kl"] = sum(kl) / len(kl)
        per[arm].append(rec)

    groups = 0
    for g0 in range(0, len(windows), group):
        ws = windows[g0:g0 + group]
        groups += 1
        refs, e = paged_pass(model, ws, P, C, chunk, device, stand_in=stand_in, **ARM_KW["R"])
        eng["R"].append(e)
        ref_am = [r.argmax(-1).tolist() for r in refs]
        for i, (w, r) in enumerate(zip(ws, refs)):
            add("R", g0 + i, r, None, w[P:P + C], ref_am[i])
        for arm in arms:
            if arm == "R" or (arm == "rep" and g0):
                continue
            lps, e = paged_pass(model, ws, P, C, floor_chunk if arm == "chunk" else chunk, device, stand_in=stand_in,
                                **ARM_KW[arm])
            eng[arm].append(e)
            if arm == "rep":
                rep_identical = all(torch.equal(a, b) for a, b in zip(lps, refs))
            for i, x in enumerate(lps):
                add(arm, g0 + i, x, refs[i], ws[i][P:P + C], ref_am[i])
            del lps
            p108_box._release(device)
        del refs
        p108_box._release(device)
        mem = torch.cuda.memory_allocated() / 2**30 if str(device).startswith("cuda") else 0.0
        print(f"P110_GROUP {groups} of {-(-len(windows) // group)} done at {time.time() - start:.0f} s, {mem:.2f} GiB allocated",
              flush=True)
    cfg = getattr(model.config, "text_config", None) or model.config
    return {"windows": len(windows), "group": group, "prompt": P, "cont": C, "chunk": chunk, "floor_chunk": floor_chunk,
            "groups": groups, "layers": int(cfg.num_hidden_layers), "rep_identical": rep_identical,
            "rehearsal": {"stand_in_attention": bool(stand_in)}, "per_window": per, "engagement": eng,
            "seconds": round(time.time() - start, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--group", type=int, default=12)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--floor-chunk", type=int, default=256)
    a = ap.parse_args()
    import transformers

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: the subject is the default server built eager at all-vram (graphs={cfg.graphs}, "
                         f"placement={cfg.placement})")
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    windows = p97_box.wikitext_windows(parts.tokenizer, a.windows, a.prompt, a.cont)
    rec = measure(model, windows, prompt=a.prompt, cont=a.cont, chunk=a.chunk, floor_chunk=a.floor_chunk, group=a.group,
                  device=cfg.device)
    rec = {"model": cfg.model, "revision": cfg.revision, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "transformers": transformers.__version__, "load_s": round(load_s, 1),
           "census": {k: parts.info.get(k) for k in ("moe_layers", "experts", "top_k", "model_type", "int4_expert_layers",
                                                     "int4_attn_projections", "graph_status", "grouping")},
           **rec, "gpu": torch.cuda.get_device_name(0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    open(a.out, "w").write(json.dumps(rec, indent=1))
    pw = rec["per_window"]

    def bias(arm):
        d = [x["nll"] - r["nll"] for x, r in zip(pw[arm], pw["R"])]
        return sum(d) / len(d), sum(abs(v) for v in d) / len(d)

    line = " | ".join(f"{arm} bias {bias(arm)[0]:+.5f} spread {bias(arm)[1]:.5f}" for arm in ARMS if arm not in ("R",))
    print(f"P110_BOX: windows {rec['windows']} | {line} | rep identical {rec['rep_identical']} | load {load_s:.0f}s | "
          f"mem {rec['max_mem_gb']} GB | {rec['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
