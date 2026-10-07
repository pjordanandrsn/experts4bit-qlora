#!/usr/bin/env python3
"""p115_quality.py -- lane P115 (bench/p115/PREREG-p115.md; e4b#1313), the QUALITY phase: does the registered B=1 fused
stack cost quality on the default ``serve_paged`` server, judged against the default's own arithmetically neutral
perturbations (P110's floor)?

Two processes, each building the default server with ``PagedServeConfig.from_env()`` + ``build_engine`` (graphs off, as
P110 built it: the passes make their own runners, so the engine's graphs would be unused memory):
- ``--phase off``: the four fusion knobs unset. Scores **R**, the reference, and its floor and mutant, and writes R's
  fp32 log-probs per (text, group) to ``--ref-dir``.
- ``--phase on``: the four knobs as F1 sets them (the three folds only under ``P115_PROVE=1``), so ``build_engine``
  applies the fused stack exactly as the server does. Scores **ON** against R's saved log-probs.

R is the default graph server's arithmetic: device grouping on and every decode step through the bucketed path with
``capture=False`` (P110's P; P109 read the replay bit-identical to it). Arms, per group (``REF_KW`` and its variants):
- **R**; **rep** (R again, first group only: a floor draw if not bit-identical);
- the floor, under R's arithmetic: **half** (the group decoded as two halves, each padded to its own bucket) and
  **chunk** (prompts prefilled in ``--floor-chunk`` pieces);
- **mutant_scale**: R with the decode softmax scale halved (P108's mutant), which the bar must catch;
- **ON** (``--phase on``): R's arithmetic on the fused model.

Texts: wikitext-2-raw test (P97's ``wikitext_windows``) and c4val1 (``c4val1_windows``: K8's c4 validation recipe,
window k from token k * 4096). Scored on fp32 log-probs: the true token's NLL, the argmax, KL against R.

Engagement per pass: P110's (decode attention calls, grouping flags, bucket statistics), plus
- **kernel calls**: every gnf4 glue kernel the folds bind (``KERNELS``) is wrapped in ``int4_b32`` BEFORE
  ``build_engine`` runs -- the folds bind them at patch time -- and counted per pass;
- **forwards** and **qkv_proj calls**, by module hooks.

``measure_phase()`` takes the model and the windows, so ``tests/test_p115_quality_box.py`` runs both phases on CPU on a
tiny Qwen3-MoE with the decode attention stood in and R unpadded (the bucketed path needs the fp8 KV kernels).

    python p115_quality.py --phase off --out quality_off.json --ref-dir work/ref [--windows 48 --cont 128]
    python p115_quality.py --phase on  --out quality_on.json  --ref-dir work/ref [--windows 48 --cont 128]
    python p115_quality.py --self-test
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p108_box  # noqa: E402  (staged at P108's registered bytes: _score, _kl, _release)
import p110_box  # noqa: E402  (staged at P110's registered bytes: paged_pass)
import p97_box  # noqa: E402  (staged at P97's registered bytes: wikitext_windows)

TEXTS = ("wikitext", "c4val1")
KERNELS = ("rmsnorm_rows", "rmsnorm_resid_rows", "scaled_resid_add_rows", "rope_norm_heads", "rope_heads",
           "router_epilogue")
REF_KW = {"device_grouping": True, "padded": True}
OFF_ARMS = ("R", "rep", "half", "chunk", "mutant_scale")
ON_ARMS = ("ON",)
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")


def arm_kw(arm: str, ref_kw: dict) -> dict:
    extra = {"half": {"halves": True}, "mutant_scale": {"mutant": "scale"}}.get(arm, {})
    return {**ref_kw, **extra}


def c4val1_windows(tok, n, prompt, cont):
    """K8's c4val1 corpus (``bench/hybrid-g9/step_decomp.py::_k8_window``, non-chat): the first 2000 documents of
    ``allenai/c4`` ``en/c4-validation.00001-of-00008.json.gz`` joined by blank lines, tokenized as K8 tokenizes it;
    window k starts at token k * 4096, as P97's wikitext windows do."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", data_files={"v": "en/c4-validation.00001-of-00008.json.gz"}, split="v")
    text = "\n\n".join(ds["text"][:2000])
    ids = tok(text, return_tensors="pt").input_ids[0]
    out = []
    for k in range(n):
        a = k * 4096
        w = ids[a:a + prompt + cont]
        assert w.numel() == prompt + cont, f"c4val1 window {k} has {w.numel()} tokens"
        out.append(w.tolist())
    return out


LOADERS = {"wikitext": p97_box.wikitext_windows, "c4val1": c4val1_windows}


def windows_sha(ws) -> str:
    return hashlib.sha256(json.dumps(ws).encode()).hexdigest()


class KernelCounters:
    """Wraps the glue kernels in the ``int4_b32`` module object, so a fold that imports one AFTER ``install()`` binds the
    wrapper. Must run before ``build_engine`` (the folds import at patch time). Each wrapper keeps its kernel's signature
    (``functools.wraps``): glue round 2 licenses GraniteMoe's scaled fold by ``inspect.signature`` of
    ``rmsnorm_resid_rows``, and a bare ``*a, **k`` wrapper would make it refuse."""

    def __init__(self):
        self.counts = {k: 0 for k in KERNELS}
        self.wrapped = []

    def install(self, module=None):
        mod = module if module is not None else importlib.import_module("int4_b32")
        for name in KERNELS:
            orig = getattr(mod, name, None)
            if orig is None or getattr(orig, "_p115_counted", False):
                continue

            def make(_n, _o):
                @functools.wraps(_o)
                def wrapper(*a, **k):
                    self.counts[_n] += 1
                    return _o(*a, **k)
                return wrapper

            wrapper = make(name, orig)
            wrapper._p115_counted = True
            setattr(mod, name, wrapper)
            self.wrapped.append(name)
        return self

    def snapshot(self) -> dict:
        return dict(self.counts)


class ForwardCounter:
    """Counts the model's forward calls and every ``qkv_proj`` module's calls (forward hooks fire on ``module(...)``)."""

    def __init__(self, model):
        self.forwards = 0
        self.qkv_calls = 0
        self.qkv_modules = 0
        model.register_forward_hook(self._fwd)
        for name, m in model.named_modules():
            if name.split(".")[-1] == "qkv_proj":
                m.register_forward_hook(self._qkv)
                self.qkv_modules += 1

    def _fwd(self, *_):
        self.forwards += 1

    def _qkv(self, *_):
        self.qkv_calls += 1

    def snapshot(self) -> dict:
        return {"forwards": self.forwards, "qkv_calls": self.qkv_calls}


def _delta(after: dict, before: dict) -> dict:
    return {k: after[k] - before.get(k, 0) for k in after}


def measure_phase(model, windows: dict, *, phase, prompt, cont, chunk, floor_chunk, group, device, ref_dir,
                  ref_kw=REF_KW, stand_in=False, counters=None, fwd=None, arms=None):
    """``phase`` ``off``: R (saved to ``ref_dir``), rep, half, chunk, mutant_scale. ``on``: ON against the saved R."""
    from experts4bit_qlora.engines import paged_attention
    P, C = prompt, cont
    arms = tuple(arms) if arms is not None else (OFF_ARMS if phase == "off" else ON_ARMS)
    paged_attention.register(model)
    os.makedirs(ref_dir, exist_ok=True)
    per = {t: {a: [] for a in arms} for t in windows}
    eng = {t: {a: [] for a in arms} for t in windows}
    rep_identical = {}
    start = time.time()

    def add(t, arm, wi, lp, ref_lp, cont_tokens, ref_argmax):
        nll, am = p108_box._score(lp, cont_tokens)
        rec = {"window": wi, "nll": sum(nll) / len(nll),
               "argmax_agree": sum(int(a == b) for a, b in zip(am, ref_argmax)) / len(am)}
        if ref_lp is not None:
            kl = p108_box._kl(ref_lp, lp)
            rec["kl"] = sum(kl) / len(kl)
        per[t][arm].append(rec)

    def run(t, arm, ws, ch):
        k0 = counters.snapshot() if counters else {}
        f0 = fwd.snapshot() if fwd else {}
        lps, e = p110_box.paged_pass(model, ws, P, C, ch, device, stand_in=stand_in, **arm_kw(arm, ref_kw))
        e["kernels"] = _delta(counters.snapshot(), k0) if counters else None
        e.update(_delta(fwd.snapshot(), f0) if fwd else {})
        eng[t][arm].append(e)
        return lps

    for t, wins in windows.items():
        for g, g0 in enumerate(range(0, len(wins), group)):
            ws = wins[g0:g0 + group]
            path = os.path.join(ref_dir, f"R_{t}_g{g}.pt")
            if phase == "off":
                refs = run(t, "R", ws, chunk)
                torch.save([r.contiguous() for r in refs], path)
                ref_am = [r.argmax(-1).tolist() for r in refs]
                for i, (w, r) in enumerate(zip(ws, refs)):
                    add(t, "R", g0 + i, r, None, w[P:P + C], ref_am[i])
            else:
                refs = torch.load(path)
                if len(refs) != len(ws):
                    raise SystemExit(f"REFUSED: {path} holds {len(refs)} windows, the group has {len(ws)}")
                ref_am = [r.argmax(-1).tolist() for r in refs]
            for arm in arms:
                if arm == "R" or (arm == "rep" and g):
                    continue
                lps = run(t, arm, ws, floor_chunk if arm == "chunk" else chunk)
                if arm == "rep":
                    rep_identical[t] = all(torch.equal(a, b) for a, b in zip(lps, refs))
                for i, x in enumerate(lps):
                    add(t, arm, g0 + i, x, refs[i], ws[i][P:P + C], ref_am[i])
                del lps
                p108_box._release(device)
            del refs
            p108_box._release(device)
            mem = torch.cuda.memory_allocated() / 2**30 if str(device).startswith("cuda") else 0.0
            print(f"P115_QGROUP {phase} {t} group {g + 1} of {-(-len(wins) // group)} done at {time.time() - start:.0f} s, "
                  f"{mem:.2f} GiB allocated", flush=True)
    cfg = getattr(model.config, "text_config", None) or model.config
    return {"phase": phase, "arms": list(arms), "windows": {t: len(w) for t, w in windows.items()},
            "windows_sha256": {t: windows_sha(w) for t, w in windows.items()}, "group": group, "prompt": P, "cont": C,
            "chunk": chunk, "floor_chunk": floor_chunk, "layers": int(cfg.num_hidden_layers),
            "ref_kw": dict(ref_kw), "rep_identical": rep_identical,
            "counters_wrapped": list(counters.wrapped) if counters else None,
            "qkv_modules": fwd.qkv_modules if fwd else None,
            "rehearsal": {"stand_in_attention": bool(stand_in)}, "per_window": per, "engagement": eng,
            "seconds": round(time.time() - start, 1)}


def self_test() -> int:
    ok = [arm_kw("R", REF_KW) == {"device_grouping": True, "padded": True},
          arm_kw("half", REF_KW) == {"device_grouping": True, "padded": True, "halves": True},
          arm_kw("mutant_scale", REF_KW)["mutant"] == "scale", arm_kw("ON", REF_KW) == REF_KW,
          arm_kw("chunk", REF_KW) == REF_KW, set(LOADERS) == set(TEXTS),
          windows_sha([[1, 2]]) != windows_sha([[1, 3]]), OFF_ARMS == ("R", "rep", "half", "chunk", "mutant_scale")]
    import types
    stub = types.ModuleType("int4_b32_selftest")
    stub.rmsnorm_rows = lambda x: x + 1

    def rmsnorm_resid_rows(x, resid, w, eps, scale=1.0):
        return x, resid
    stub.rmsnorm_resid_rows = rmsnorm_resid_rows
    kc = KernelCounters().install(stub)
    ok.append(stub.rmsnorm_rows(1) == 2 and kc.counts["rmsnorm_rows"] == 1
              and kc.wrapped == ["rmsnorm_rows", "rmsnorm_resid_rows"])
    ok.append("scale" in inspect.signature(stub.rmsnorm_resid_rows).parameters)      # glue round 2's capability probe
    KernelCounters().install(stub)
    ok.append(stub.rmsnorm_rows(1) == 2 and kc.counts["rmsnorm_rows"] == 2)            # never wrapped twice
    print(f"p115_quality self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--phase", choices=("off", "on"))
    ap.add_argument("--out")
    ap.add_argument("--ref-dir")
    ap.add_argument("--texts", default=",".join(TEXTS))
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--group", type=int, default=12)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--floor-chunk", type=int, default=256)
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.phase and a.out and a.ref_dir):
        raise SystemExit("REFUSED: --phase, --out and --ref-dir are required")
    import p115_box
    prove = os.environ.get("P115_PROVE", "0") == "1"
    ok, why = p115_box.arm_env_ok("F1" if a.phase == "on" else "F0", os.environ, prove)
    if not ok:
        raise SystemExit(f"REFUSED: phase {a.phase}: {why}")
    texts = tuple(t for t in a.texts.split(",") if t)
    if not texts or set(texts) - set(TEXTS):
        raise SystemExit(f"REFUSED: --texts {a.texts!r}; expected a subset of {TEXTS}")
    import transformers

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: the quality phase builds the default server eager at all-vram (graphs={cfg.graphs}, "
                         f"placement={cfg.placement})")
    counters = KernelCounters().install()            # BEFORE build_engine: the folds bind the kernels at patch time
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    fwd = ForwardCounter(model)
    windows = {t: LOADERS[t](parts.tokenizer, a.windows, a.prompt, a.cont) for t in texts}
    rec = measure_phase(model, windows, phase=a.phase, prompt=a.prompt, cont=a.cont, chunk=a.chunk,
                        floor_chunk=a.floor_chunk, group=a.group, device=cfg.device, ref_dir=a.ref_dir,
                        counters=counters, fwd=fwd)
    rec = {"model": cfg.model, "revision": cfg.revision, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "prove": prove, "transformers": transformers.__version__,
           "load_s": round(load_s, 1), "fuse_qkv": bool(cfg.fuse_qkv),
           "census": {k: parts.info.get(k) for k in CENSUS_KEYS + ("moe_layers", "experts", "top_k", "model_type",
                                                                    "graph_status", "grouping")},
           **rec, "gpu": torch.cuda.get_device_name(0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    open(a.out, "w").write(json.dumps(rec, indent=1, default=str))
    line = []
    for t in texts:
        pw = rec["per_window"][t]
        ref = {x["window"]: x["nll"] for x in pw.get("R", [])}
        for arm in rec["arms"]:
            if arm == "R" or not pw.get(arm) or not ref:
                continue
            d = [x["nll"] - ref[x["window"]] for x in pw[arm]]
            line.append(f"{t}/{arm} bias {sum(d) / len(d):+.5f}")
    print(f"P115_QUALITY {a.phase}: {' | '.join(line) or 'scored against the saved R'} | census {rec['census']} | "
          f"load {load_s:.0f}s | mem {rec['max_mem_gb']} GB | {rec['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
