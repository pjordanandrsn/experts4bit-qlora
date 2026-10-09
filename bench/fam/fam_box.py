#!/usr/bin/env python3
"""fam_box.py -- lane FAM (bench/fam/PREREG-fam.md; e4b#1362): ONE process per (model, config). A per-family quality
instrument whose neutral floor is drawn on the family itself.

The engine is the shipped default server (``PagedServeConfig.from_env()`` + ``build_engine(cfg)``) built eager at
all-vram, as P115's quality phases build it. The instrument is P115's: ``p115_quality.measure_phase`` (staged at
P115's registered bytes) over ``p110_box.paged_pass``, teacher-forced, scored on fp32 log-probs (the true token's NLL,
the argmax, KL against R). Configs differ ONLY in the four P115 fusion knobs (``CONFIGS``), each named explicitly.

Cells. Every (text, shape, set) is one cell with its own reference:
- texts: wikitext and c4val1 (P115's loaders: window k from token k * 4096);
- shapes: ``1`` (one window per pass: bucket 1, T == 1, the served one-request arithmetic) and ``12`` (Phase C's: 12
  windows per pass, T == 12, padded to bucket 16);
- sets: ``A`` = windows 0-11, ``B`` = 12-23, ``C`` = 24-35 (three disjoint draws; Phase C's wikitext windows are A).

``--config OFF`` (the four knobs ``0``) scores, per cell:
- ``R`` (saved to the cell's reference directory), ``rep`` (first group; a floor draw only if not bit-identical);
- the floor: ``chunk`` (prompts prefilled in 256-token chunks), ``half`` (shape 12 only: two halves of 6, each padded
  to bucket 8), ``split1`` (every decode attention forced to ONE KV split, ``n_split=1``: the same sums in another
  order);
- the mutants: ``mutant_scale`` (P108's: decode softmax scale x0.5, gross) and the graded ``mut090`` (x0.90, the
  gate's resolution test); the ladder ``mut095`` / ``mut098`` on set A only (reported).
``--config ON_*`` scores ``ON`` against the cell's saved R.

``split1`` and the graded mutants wrap the decode attention (``Fp8PagedKV.attention``, and P108's stand-in on CPU)
OUTSIDE the staged P108 / P110 / P115 files, which run at their registered bytes; every wrapped call is counted.

    python fam_box.py --config OFF    --out fam_off.json    --ref-root work/ref     (env: the four knobs as CONFIGS)
    python fam_box.py --config ON_glue --out fam_on_glue.json --ref-root work/ref
    python fam_box.py --self-test
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
_OFF = dict.fromkeys(KNOBS, "0")
CONFIGS = {
    "OFF": dict(_OFF),
    "ON_glue": {**_OFF, "E4B_FUSE_T1_GLUE": "auto"},
    "ON_r2": {**_OFF, "E4B_FUSE_T1_GLUE_R2": "auto"},
    "ON_epi": {**_OFF, "E4B_FUSE_ROUTER_EPI": "auto"},
    "ON_auto": dict.fromkeys(KNOBS, "auto"),
}
#: SC2g's e4b path (``bench/sc2/sc2g_box_g.sh``): native MXFP4 decode for single rows, NF4 kept for batched rows.
#: Only the anchor cell runs it; every other process must leave these unset (the default server's store).
SC2G_ENV = {"E4B_SERVE_EXP_INT4": "1", "E4B_INT4_KEEP_NF4": "1", "E4B_SERVE_ATTN_INT4_CALIB": "0",
            "E4B_CALIB_SOURCE": "c4", "GNF4_TRITON_PREBIND": "1"}
PATHS = ("default", "sc2g")
TEXTS = ("wikitext", "c4val1")
SHAPES = (1, 12)
SETS = {"A": (0, 12), "B": (12, 24), "C": (24, 36)}
FLOOR_ARMS = ("rep", "chunk", "half", "split1")
GRADED = {"mut090": 0.90, "mut095": 0.95, "mut098": 0.98}
GATING_MUTANT = "mut090"                              # must fail every gated cell (PREREG "Resolution")
LADDER_SETS = ("A",)                                  # the ladder, mut095 / mut098, runs on set A only
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
STORE_KEYS = ("int4_expert_layers", "int4_store_kinds", "levers_env", "grouping", "model_type", "moe_layers", "top_k")


def env_ok(config: str, path: str, env) -> tuple[bool, str]:
    """``(ok, why)``: the four fusion knobs carry the config's values, all named; the store levers are SC2g's on the
    anchor path and unset on the default path."""
    if config not in CONFIGS:
        return False, f"unknown config {config!r}"
    if path not in PATHS:
        return False, f"unknown path {path!r}"
    bad = {k: env.get(k) for k, v in CONFIGS[config].items() if (env.get(k) or "").strip().lower() != v}
    if path == "sc2g":
        bad.update({k: env.get(k) for k, v in SC2G_ENV.items() if (env.get(k) or "").strip() != v})
    else:
        bad.update({k: env.get(k) for k in SC2G_ENV if k != "GNF4_TRITON_PREBIND" and (env.get(k) or "").strip()})
    return (not bad), f"{config} on {path} with {bad}"


def cell_key(text: str, shape: int, set_: str) -> str:
    return f"{text}|{shape}|{set_}"


def off_arms(shape: int) -> tuple:
    """The arms ``measure_phase`` scores in the OFF process's first pass over a cell (P115's own arm names)."""
    return ("R", "rep", "chunk") + (("half",) if shape > 1 else ()) + ("mutant_scale",)


def extra_arms(set_: str) -> tuple:
    """The arms scored against the saved R under a wrapper, in the OFF process."""
    return ("split1", GATING_MUTANT) + (("mut095", "mut098") if set_ in LADDER_SETS else ())


class AttnWrap(contextlib.AbstractContextManager):
    """Wraps the decode attention for one arm: ``n_split=1`` (``split1``) or the softmax scale times ``factor`` (the
    graded mutants, as P108's mutant computes its default scale). Installed on ``Fp8PagedKV.attention`` (the real
    kernel, which P108's ``Attention`` captures as its base) and on ``p108_box.stand_in_attention`` (CPU), so it sits
    below P108's counting wrapper. Counts every call it changed."""

    def __init__(self, *, n_split=None, factor=None):
        self.n_split, self.factor, self.calls = n_split, factor, 0

    def _wrap(self, fn):
        me = self

        def attention(kv, layer, q, *a, sm_scale=None, **kw):
            if me.factor is not None:
                sm_scale = (sm_scale if sm_scale is not None else q.shape[-1] ** -0.5) * me.factor
            if me.n_split is not None:
                kw["n_split"] = me.n_split
            me.calls += 1
            return fn(kv, layer, q, *a, sm_scale=sm_scale, **kw)
        return attention

    def __enter__(self):
        import p108_box
        from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
        self._saved = (Fp8PagedKV.attention, p108_box.stand_in_attention)
        Fp8PagedKV.attention = self._wrap(self._saved[0])
        p108_box.stand_in_attention = self._wrap(self._saved[1])
        return self

    def __exit__(self, *exc):
        import p108_box
        from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
        Fp8PagedKV.attention, p108_box.stand_in_attention = self._saved
        return False


def arm_wrap(arm: str) -> AttnWrap:
    if arm == "split1":
        return AttnWrap(n_split=1)
    if arm in GRADED:
        return AttnWrap(factor=GRADED[arm])
    raise ValueError(arm)


def _dispatch():
    """grouped-nf4-gemm's NF4 dispatch tally, or None where the kernel package is absent (CPU tests)."""
    try:
        import nf4_grouped
    except ImportError:
        return None
    return dict(nf4_grouped.dispatch_counts())


def _delta(after, before):
    if after is None or before is None:
        return None
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def shape_order(shapes):
    """The largest group first (Amendment 2). A hybrid's linear-state pool is sized by the first runner on the model
    and frozen by the padded pass; a later runner may bind fewer slots, never more (``LinearStatePool.ensure_slots``)."""
    return tuple(sorted(shapes, reverse=True))


def run_cells(model, windows, *, config, texts, shapes, sets, prompt, cont, chunk, floor_chunk, device, ref_root,
              ref_kw=None, stand_in=False, counters=None, fwd=None):
    """Every cell of one process. ``windows[text]`` holds 36 windows; set S takes ``SETS[S]``. Returns
    ``{cell: {"base": measure_phase record, "extra": {arm: (record, wrapped calls)}, "dispatch": ...}}``."""
    import p115_quality as q
    kw = dict(prompt=prompt, cont=cont, chunk=chunk, floor_chunk=floor_chunk, device=device, counters=counters,
              fwd=fwd, stand_in=stand_in, **({"ref_kw": ref_kw} if ref_kw is not None else {}))
    cells = {}
    for t in texts:
        for s in shape_order(shapes):
            for name in sets:
                a, b = SETS[name]
                ws = {t: windows[t][a:b]}
                ref = os.path.join(ref_root, f"{t}_{s}_{name}")
                d0 = _dispatch()
                if config == "OFF":
                    base = q.measure_phase(model, ws, phase="off", group=s, ref_dir=ref, arms=off_arms(s), **kw)
                    extra = {}
                    for arm in extra_arms(name):
                        with arm_wrap(arm) as w:
                            rec = q.measure_phase(model, ws, phase="on", group=s, ref_dir=ref, arms=(arm,), **kw)
                        extra[arm] = {"record": rec, "wrapped_calls": w.calls}
                else:
                    base = q.measure_phase(model, ws, phase="on", group=s, ref_dir=ref, arms=("ON",), **kw)
                    extra = {}
                cells[cell_key(t, s, name)] = {"base": base, "extra": extra, "dispatch": _delta(_dispatch(), d0)}
                print(f"FAM_CELL {config} {cell_key(t, s, name)} done", flush=True)
    return cells


def main_run(a) -> int:
    ok, why = env_ok(a.config, a.path, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: {why}")
    texts = tuple(x for x in a.texts.split(",") if x)
    shapes = tuple(int(x) for x in a.shapes.split(",") if x)
    sets = tuple(x for x in a.sets.split(",") if x)
    if set(texts) - set(TEXTS) or set(shapes) - set(SHAPES) or set(sets) - set(SETS) or not (texts and shapes and sets):
        raise SystemExit(f"REFUSED: texts {texts} shapes {shapes} sets {sets}")
    import torch
    import transformers

    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: the instrument builds the default server eager at all-vram (graphs={cfg.graphs}, "
                         f"placement={cfg.placement})")
    counters = q.KernelCounters().install()                 # BEFORE build_engine: the folds bind the kernels at patch time
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    fwd = q.ForwardCounter(model)
    n = max(b for _, b in SETS.values())
    windows = {t: q.LOADERS[t](parts.tokenizer, n, a.prompt, a.cont) for t in texts}
    ref_kw = json.loads(a.ref_kw) if a.ref_kw else None
    t1 = time.time()
    cells = run_cells(model, windows, config=a.config, texts=texts, shapes=shapes, sets=sets, prompt=a.prompt,
                      cont=a.cont, chunk=a.chunk, floor_chunk=a.floor_chunk, device=cfg.device, ref_root=a.ref_root,
                      ref_kw=ref_kw, counters=counters, fwd=fwd)
    info = parts.info
    rec = {"config": a.config, "path": a.path, "tag": a.tag, "model": cfg.model, "revision": cfg.revision,
           "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "knobs": {k: os.environ.get(k) for k in KNOBS}, "store_env": {k: os.environ.get(k) for k in SC2G_ENV},
           "transformers": transformers.__version__, "torch": torch.__version__, "load_s": round(load_s, 1),
           "census": {k: info.get(k) for k in CENSUS_KEYS}, "fusion_modes": info.get("fusion_modes"),
           "fusion_report": info.get("fusion_report"),              # Amendment 5: the router epilogue's fp32_upstream
           "store": {k: info.get(k) for k in STORE_KEYS}, "kv": info.get("kv"), "layers": q_layers(model),
           "counters_wrapped": list(counters.wrapped), "qkv_modules": fwd.qkv_modules, "ref_kw": ref_kw,
           "texts": list(texts), "shapes": list(shapes), "sets": list(sets), "prompt": a.prompt, "cont": a.cont,
           "cells": cells, "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
           "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "seconds": round(time.time() - t1, 1),
           "status": "ok"}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(f"FAM_BOX {a.config}/{a.path} census={rec['census']} store={rec['store']['int4_store_kinds']} "
          f"cells={len(cells)} mem={rec['max_mem_gb']} GB {rec['seconds']} s", flush=True)
    return 0


def q_layers(model) -> int:
    cfg = getattr(model.config, "text_config", None) or model.config
    return int(cfg.num_hidden_layers)


def self_test() -> int:
    on = dict(CONFIGS["ON_auto"])
    ok = [set(CONFIGS) == {"OFF", "ON_glue", "ON_r2", "ON_epi", "ON_auto"},
          all(sum(v != "0" for v in c.values()) == 1 for k, c in CONFIGS.items() if k in ("ON_glue", "ON_r2", "ON_epi")),
          CONFIGS["ON_glue"]["E4B_PAGED_FUSE_QKV"] == "0",
          env_ok("OFF", "default", dict(_OFF))[0], env_ok("ON_auto", "default", on)[0],
          env_ok("ON_auto", "default", {k: " AUTO " for k in KNOBS})[0],
          not env_ok("OFF", "default", {})[0],                                   # unset is not the registered 0
          not env_ok("ON_glue", "default", dict(_OFF))[0],
          not env_ok("ON_glue", "default", {**CONFIGS["ON_glue"], "E4B_FUSE_ROUTER_EPI": "auto"})[0],
          not env_ok("OFF", "default", {**_OFF, "E4B_SERVE_EXP_INT4": "1"})[0],  # the default path keeps the default store
          env_ok("OFF", "default", {**_OFF, "GNF4_TRITON_PREBIND": "1"})[0],     # prebind is a default, not a store lever
          env_ok("OFF", "sc2g", {**_OFF, **SC2G_ENV})[0], not env_ok("OFF", "sc2g", dict(_OFF))[0],
          not env_ok("ON_x", "default", on)[0], not env_ok("OFF", "nf4", dict(_OFF))[0],
          off_arms(1) == ("R", "rep", "chunk", "mutant_scale"),
          off_arms(12) == ("R", "rep", "chunk", "half", "mutant_scale"),
          extra_arms("A") == ("split1", "mut090", "mut095", "mut098"), extra_arms("B") == ("split1", "mut090"),
          [SETS[k] for k in "ABC"] == [(0, 12), (12, 24), (24, 36)], cell_key("wikitext", 1, "A") == "wikitext|1|A",
          arm_wrap("split1").n_split == 1 and arm_wrap("split1").factor is None,
          arm_wrap("mut090").factor == 0.90 and arm_wrap("mut090").n_split is None, GATING_MUTANT == "mut090",
          shape_order(SHAPES) == (12, 1) and shape_order((12,)) == (12,)]
    try:
        arm_wrap("R")
        ok.append(False)
    except ValueError:
        ok.append(True)
    seen = {}
    w = AttnWrap(factor=0.5)._wrap(lambda kv, layer, q, *a, sm_scale=None, **kw: seen.update(s=sm_scale, kw=kw))

    class Q:
        shape = (1, 4, 64)
    w(None, 0, Q(), sm_scale=None)
    ok.append(abs(seen["s"] - 0.5 * 64 ** -0.5) < 1e-12 and "n_split" not in seen["kw"])
    w2 = AttnWrap(n_split=1)._wrap(lambda kv, layer, q, *a, sm_scale=None, **kw: seen.update(s=sm_scale, kw=kw))
    w2(None, 0, Q(), sm_scale=0.125)
    ok.append(seen["s"] == 0.125 and seen["kw"] == {"n_split": 1})
    print(f"fam_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--config", choices=tuple(CONFIGS))
    p.add_argument("--path", choices=PATHS, default="default")
    p.add_argument("--tag", default="")
    p.add_argument("--out")
    p.add_argument("--ref-root")
    p.add_argument("--texts", default=",".join(TEXTS))
    p.add_argument("--shapes", default=",".join(str(s) for s in SHAPES))
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--cont", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--floor-chunk", type=int, default=256)
    p.add_argument("--ref-kw", default="", help="JSON override of P115's REF_KW (the A2000 correctness proof only)")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.config and a.out and a.ref_root):
        raise SystemExit("REFUSED: --config, --out and --ref-root are required")
    for d in ("p115", "p110", "p108", "p97"):
        cand = os.path.join(HERE, "..", d)
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    return main_run(a)


if __name__ == "__main__":
    raise SystemExit(main())
