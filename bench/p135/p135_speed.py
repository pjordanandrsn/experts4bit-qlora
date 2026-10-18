# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P135 (Track B1's R-b): what Mixtral's ON_auto folds buy at one row (bench/p135/PREREG-p135.md).

FAM's quality read licensed Mixtral ON_auto at T == 1 (fam-mixtral-4, resolution ``mut090``; bench/fam/PREREG-fam.md
Amendments 5-8), and its registered consequence is that an allow-list pull request needs a speed read of what the
passing knobs buy, by Amendment 4's interleaved method. On Mixtral ON_auto engages three folds (census
``0 / 65 / [32, 32] / 32``): the glue fold's 65 RMSNorms, round 2's 32 decoder layers and 32 attentions, and the router
epilogue's 32 routers. This box is ``bench/fam/fam_speed.py`` with its one swap (the router epilogue) widened to all
three folds:

- **One model, both settings.** The default ``serve_paged`` build with all four fusion knobs at ``auto`` (FAM's ON_auto).
  ``FoldRoute`` puts every fold-patched module's original ``forward`` back (OFF) or keeps the fused one (ON). Each fold
  patches an instance ``forward`` defined in its own module that carries the original as its ``_orig`` default, so the
  swap is exact and touches nothing else. A 47B model does not fit twice on 32 GB.
- **Two blocks** (``a``: OFF first in every pair, ``b``: ON first). Each holds one runner per setting, both alive, each
  with its own KV pool and its bucket-1 graph captured under its own setting. Both are prefilled with the same 512-token
  wikitext prompt, then decode greedily in strict alternation, one step each in lockstep: ``WARM`` warm, ``steps`` timed
  (each ``run_decode``'s synchronised wall) and ``BUSY`` more. Every step's token is recorded.
- **Bucket 1 only.** The read is one row. Mixtral's NF4 weights take about 26.8 GiB of the 5090's 31.4, so each runner
  captures the one bucket it replays (FAM's box captured all five on a 35B model).
- **Each step runs under its own setting.** ``FoldRoute`` is entered before a step's clock starts, so the swap is outside
  the timed wall. A replay calls no Python forward; an eager step would, and the reducer VOIDs on any.

``graphs=False`` decodes eagerly (the CPU tests: bucketed graphs need the fused fp8 KV append, sm_89+).

Usage on the box: ``python p135_speed.py --out speed.json``; ``--self-test`` checks the constants and the swap.
"""
import argparse
import gc
import inspect
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
REVISION = "eba92302a2861cdc0098cc54bc9f17cb2c47eb61"   # FAM Amendment 5's pin, the revision its quality read licensed
SETTINGS = ("OFF", "ON")
BLOCKS = {"a": ("OFF", "ON"), "b": ("ON", "OFF")}
SLOT = {"OFF": 0, "ON": 1}                  # each setting's sequence on its own slot (FAM's rule; Mixtral has no shared pool)
KV_BATCH = 2
WARM, STEPS, BUSY = 5, 256, 32
PROMPT = 512
BUCKETS = (1,)
KNOBS = {"E4B_PAGED_FUSE_QKV": "auto", "E4B_FUSE_T1_GLUE": "auto", "E4B_FUSE_T1_GLUE_R2": "auto", "E4B_FUSE_ROUTER_EPI": "auto"}
#: Knobs that would change what the folds compute; both must stay unset (the decode-only folds FAM's quality read licensed)
OFF_KNOBS = ("E4B_FUSE_PREFILL_GLUE", "E4B_PREFILL_LEAN_DISPATCH")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
FOLD_MODULES = ("experts4bit_qlora.engines.glue_fuse", "experts4bit_qlora.engines.glue_r2",
                "experts4bit_qlora.engines.router_epilogue")


def fused_folds(model):
    """``[(module, fused forward, original forward)]`` for every module a fold patched: an instance ``forward`` defined in
    one of :data:`FOLD_MODULES` that carries its original as the ``_orig`` default."""
    out = []
    for mod in model.modules():
        f = mod.__dict__.get("forward")
        if f is None or getattr(f, "__module__", None) not in FOLD_MODULES:
            continue
        p = inspect.signature(f).parameters.get("_orig")
        if p is not None and p.default is not inspect.Parameter.empty:
            out.append((mod, f, p.default))
    return out


def by_module(folds) -> dict:
    """How many patched forwards each fold module owns: ``{"glue_fuse": 65, ...}``."""
    out = {}
    for _, f, _ in folds:
        k = f.__module__.rsplit(".", 1)[-1]
        out[k] = out.get(k, 0) + 1
    return out


class FoldRoute:
    """``on=False`` puts every fold-patched module's original forward back; ``on=True`` keeps the fused one. The build's
    state (fused) is restored on exit. ``n`` is how many modules the setting touched."""

    def __init__(self, model, on: bool, folds=None):
        self.on, self.folds = bool(on), folds if folds is not None else fused_folds(model)
        self.n = len(self.folds)

    def __enter__(self):
        for mod, fused, orig in self.folds:
            mod.forward = fused if self.on else orig
        return self

    def __exit__(self, *exc):
        for mod, fused, _ in self.folds:
            mod.forward = fused
        return False


class _Clock:
    """``nvidia-smi`` every ``PERIOD`` s while a block runs (P124 Amendment 1; reported, never gated). Records nothing
    where there is no ``nvidia-smi``."""

    PERIOD = 5.0
    QUERY = "clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate"

    def __init__(self):
        self.rows, self._stop = [], threading.Event()

    def _run(self):
        t0 = time.time()
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.SubprocessError):
                return
            line = out.stdout.strip().splitlines()[:1]
            if out.returncode == 0 and line:
                self.rows.append([round(time.time() - t0, 1)] + [x.strip() for x in line[0].split(",")])
            self._stop.wait(self.PERIOD)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=15)
        return False


def _sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class _Grouped:
    """The graph server's device-side expert grouping (``serve_paged`` sets it with decode graphs), restored after."""

    def __enter__(self):
        from experts4bit_qlora.engines import hot_residency as hr
        self.hr, self.saved = hr, (hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0])
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = True, False
        return self

    def __exit__(self, *exc):
        self.hr.DEVICE_GROUPING[0], self.hr.FORCE_SINGLETON_GROUPS[0] = self.saved
        return False


def _pool(model, tokens, scratch, device):
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry
    cfg = getattr(model.config, "text_config", None) or model.config
    hkv, hd = _kv_geometry(model.config)
    return Fp8PagedKV(kv_layers(model, int(cfg.num_hidden_layers)), hkv, hd, batch=KV_BATCH, max_tokens_per_seq=tokens,
                      device=device, scratch_slots=scratch)


def interleaved_block(model, prompt, device, *, block: str, steps: int, buckets=BUCKETS, graphs=True, bulk_kv=True,
                      folds=None, slots=None):
    """One block: a runner per setting, both alive, captured under their settings, prefilled, then ``WARM + steps +
    BUSY`` greedy decode steps in strict alternation (``BLOCKS[block]`` gives each pair's order)."""
    import torch

    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    slots = slots or SLOT
    folds = folds if folds is not None else fused_folds(model)
    first = BLOCKS[block]
    n_steps = WARM + steps + BUSY
    rec = {"block": block, "order": list(first), "steps": steps, "warm": WARM, "busy": BUSY, "graphs": bool(graphs),
           "slots": dict(slots), "arms": {}}
    if str(device).startswith("cuda"):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    run = {}
    try:
        with _Grouped(), torch.no_grad():
            for st in first:                                     # each runner captured under its own setting
                kv = _pool(model, len(prompt) + n_steps + 16, max(buckets) if graphs else 0, device)
                runner = PagedModelRunner(model, kv, device=device, bulk_kv=bulk_kv)
                run[st] = runner
                gs = None
                with FoldRoute(model, st == "ON", folds) as route:
                    if graphs:
                        gs = {str(k): v for k, v in runner.enable_decode_graphs(buckets, capture=True, verbose=False).items()}
                rec["arms"][st] = {"folds": route.n, "graph_status": gs, "step_ms": [], "tokens": []}
            for st in first:
                with FoldRoute(model, st == "ON", folds):
                    run[st].bind(0, slots[st], list(prompt))
                    run[st].run_prefill([(0, 0, len(prompt))])
            with _Clock() as clock:
                for i in range(n_steps):
                    for st in first:                             # strict alternation, one step of each per pair
                        with FoldRoute(model, st == "ON", folds):
                            _sync(device)
                            t0 = time.perf_counter()
                            got = run[st].run_decode([0])
                            _sync(device)
                            ms = (time.perf_counter() - t0) * 1e3
                        arm = rec["arms"][st]
                        if WARM <= i < WARM + steps:
                            arm["step_ms"].append(round(ms, 4))
                        arm["tokens"].append(int(got[0]))
            rec["clock"] = clock.rows
        for st in first:
            gs = getattr(run[st], "graph_stats", None) or {}
            rec["arms"][st]["graph_stats"] = {str(k): dict(v) for k, v in gs.items()}
        cuda = str(device).startswith("cuda")
        rec["memory"] = {"max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
                         "max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1) if cuda else None}
    finally:
        for runner in run.values():
            dis = getattr(runner, "disable_decode_graphs", None)
            if dis is not None and getattr(runner, "_graphs", None) is not None:
                dis()
        del run
    return rec


def main_run(a) -> int:
    bad = {k: os.environ.get(k) for k, v in KNOBS.items() if (os.environ.get(k) or "").strip() != v}
    if bad:
        raise SystemExit(f"REFUSED: the knobs must be {KNOBS} (FAM's ON_auto), got {bad}")
    set_off = {k: os.environ[k] for k in OFF_KNOBS if (os.environ.get(k) or "").strip() not in ("", "0")}
    if set_off:
        raise SystemExit(f"REFUSED: {set_off}: the folds must be the decode-only folds FAM's quality read licensed")
    import torch
    import transformers

    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: the box builds the server eager at all-vram (graphs={cfg.graphs}, "
                         f"placement={cfg.placement}); its runners capture their own graphs")
    if (cfg.model, cfg.revision) != (MODEL, REVISION):
        raise SystemExit(f"REFUSED: {cfg.model}@{cfg.revision} is not the registered {MODEL}@{REVISION}")
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    info = parts.info
    folds = fused_folds(model)
    prompt = q.LOADERS["wikitext"](parts.tokenizer, 1, PROMPT, 1)[0][:PROMPT]
    t1 = time.time()
    blocks = {b: interleaved_block(model, prompt, cfg.device, block=b, steps=a.steps, buckets=BUCKETS,
                                   bulk_kv=cfg.bulk_kv, folds=folds) for b in BLOCKS}
    med = {f"{st}_{b}": round(statistics.median(blocks[b]["arms"][st]["step_ms"]), 4) for b in BLOCKS for st in SETTINGS}
    rec = {"model": cfg.model, "revision": cfg.revision, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "knobs": {k: os.environ.get(k) for k in KNOBS},
           "off_knobs": {k: os.environ.get(k) for k in OFF_KNOBS},
           "census": {k: info.get(k) for k in CENSUS_KEYS}, "fusion_report": info.get("fusion_report"),
           "fusion_modes": info.get("fusion_modes"), "prefill_glue": info.get("prefill_glue"),
           "model_type": info.get("model_type"), "fused_folds": len(folds), "folds_by_module": by_module(folds),
           "buckets": list(BUCKETS), "prompt": PROMPT, "prompt_tokens": len(prompt), "kv_batch": KV_BATCH,
           "blocks": blocks, "medians": med, "transformers": transformers.__version__, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
           "load_s": round(load_s, 1), "seconds": round(time.time() - t1, 1), "status": "ok"}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"P135_SPEED census={rec['census']} folds={rec['folds_by_module']} medians={med} {rec['seconds']} s", flush=True)
    return 0


def self_test() -> int:
    ok = [BLOCKS == {"a": ("OFF", "ON"), "b": ("ON", "OFF")}, SLOT["OFF"] != SLOT["ON"],
          max(SLOT.values()) < KV_BATCH, (WARM, STEPS, BUSY, PROMPT) == (5, 256, 32, 512), BUCKETS == (1,),
          all(v == "auto" for v in KNOBS.values()) and len(KNOBS) == 4, len(REVISION) == 40]

    class M:
        def forward(self, x):
            return ("orig", x)

    plain, other, norm, router = M(), M(), M(), M()           # untouched; patched elsewhere; two fold-patched modules

    def elsewhere(hidden_states, _orig=None):
        return ("other", hidden_states)
    other.forward = elsewhere

    def glue(hidden_states, _m=norm, _orig=M.forward.__get__(norm)):
        return ("glue", hidden_states)
    glue.__module__ = FOLD_MODULES[0]
    norm.forward = glue

    def epi(hidden_states, _m=router, _orig=M.forward.__get__(router)):
        return ("epi", hidden_states)
    epi.__module__ = FOLD_MODULES[2]
    router.forward = epi

    class Model:
        def modules(self):
            return iter([plain, other, norm, router])
    found = fused_folds(Model())
    ok.append(len(found) == 2 and {id(m) for m, _, _ in found} == {id(norm), id(router)})
    ok.append(by_module(found) == {"glue_fuse": 1, "router_epilogue": 1})
    with FoldRoute(Model(), False, found) as r:
        ok.append(r.n == 2 and norm.forward(1) == ("orig", 1) and router.forward(1) == ("orig", 1)
                  and other.forward(1) == ("other", 1))
    ok.append(norm.forward(1) == ("glue", 1) and router.forward(1) == ("epi", 1))   # the build's state is restored
    with FoldRoute(Model(), True, found):
        ok.append(norm.forward(2) == ("glue", 2) and router.forward(2) == ("epi", 2))
    print(f"p135_speed self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out")
    p.add_argument("--steps", type=int, default=STEPS)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.out or a.steps != STEPS:
        raise SystemExit(f"REFUSED: --out is required and --steps is {STEPS}")
    return main_run(a)


if __name__ == "__main__":
    raise SystemExit(main())
