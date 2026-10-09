# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane FAM, Amendment 4: what the router epilogue buys on Qwen3.6 at one row (bench/fam/PREREG-fam.md; e4b#1362).

FAM's quality read licensed Qwen3.6 ON_auto at T == 1, and on this family only the router epilogue engages
(``0 / 0 / [0, 0] / 40``). A default that changes arithmetic should buy something, so this box times it, with P124
Amendment 1's interleaved method at one row:

- **One model, both settings.** The default ``serve_paged`` build with ``E4B_FUSE_ROUTER_EPI=1`` and the other three knobs
  at 0. ``EpiRoute`` puts each fused router's original ``forward`` back (OFF) or keeps the fused one (ON); a 35B model does
  not fit twice on 32 GB.
- **Two blocks** (``a``: OFF first in every pair, ``b``: ON first). Each holds one runner per setting, both alive, each
  with its own KV pool and its bucket graphs captured under its own setting. Both are prefilled with the same 512-token
  wikitext prompt, then decode greedily in strict alternation, one step each in lockstep: ``WARM`` warm, ``steps`` timed
  (each ``run_decode``'s synchronised wall) and ``BUSY`` more. Every step's token is recorded.
- **The hybrid's one linear-state pool.** The model keeps a single per-slot pool that every runner on it shares, so two
  live runners binding slot 0 would overwrite each other's recurrent state on alternate steps. Each runner's KV is built
  for ``KV_BATCH`` sequences, and OFF binds slot 0, ON slot 1 (``SLOT``).
- **Each step runs under its own setting.** ``EpiRoute`` is entered before a step's clock starts, so the swap is outside
  the timed wall. A replay calls no Python forward; an eager step would, and the reducer VOIDs on any.

``graphs=False`` decodes eagerly (the CPU tests: bucketed graphs need the fused fp8 KV append, sm_89+).

Usage on the box: ``python fam_speed.py --out speed_qw36.json [--steps N]``; ``--self-test`` checks the constants.
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

SETTINGS = ("OFF", "ON")
BLOCKS = {"a": ("OFF", "ON"), "b": ("ON", "OFF")}
SLOT = {"OFF": 0, "ON": 1}                  # the hybrid's ONE linear-state pool: each setting's sequence on its own slot
KV_BATCH = 2
WARM, STEPS, BUSY = 5, 256, 32
PROOF_STEPS = 32
PROMPT = 512
KNOBS = {"E4B_PAGED_FUSE_QKV": "0", "E4B_FUSE_T1_GLUE": "0", "E4B_FUSE_T1_GLUE_R2": "0", "E4B_FUSE_ROUTER_EPI": "1"}
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
EPI_MODULE = "experts4bit_qlora.engines.router_epilogue"


def fused_routers(model):
    """``[(module, fused forward, original forward)]`` for every router ``fuse_router_epilogue`` patched: an instance
    ``forward`` defined in the epilogue's module that carries its original as the ``_orig`` default."""
    out = []
    for mod in model.modules():
        f = mod.__dict__.get("forward")
        if f is None or getattr(f, "__module__", None) != EPI_MODULE:
            continue
        p = inspect.signature(f).parameters.get("_orig")
        if p is not None and p.default is not inspect.Parameter.empty:
            out.append((mod, f, p.default))
    return out


class EpiRoute:
    """``on=False`` puts every fused router's original forward back; ``on=True`` keeps the fused one. The build's state
    (fused) is restored on exit. ``n`` is how many routers the setting touched."""

    def __init__(self, model, on: bool, routers=None):
        self.on, self.routers = bool(on), routers if routers is not None else fused_routers(model)
        self.n = len(self.routers)

    def __enter__(self):
        for mod, fused, orig in self.routers:
            mod.forward = fused if self.on else orig
        return self

    def __exit__(self, *exc):
        for mod, fused, _ in self.routers:
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


def interleaved_block(model, prompt, device, *, block: str, steps: int, buckets, graphs=True, bulk_kv=True,
                      routers=None, slots=None):
    """One block: a runner per setting, both alive, captured under their settings, prefilled, then ``WARM + steps +
    BUSY`` greedy decode steps in strict alternation (``BLOCKS[block]`` gives each pair's order). ``slots`` overrides
    ``SLOT`` (the CPU test's shared-slot hazard)."""
    import torch

    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    slots = slots or SLOT
    routers = routers if routers is not None else fused_routers(model)
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
                with EpiRoute(model, st == "ON", routers) as route:
                    if graphs:
                        gs = {str(k): v for k, v in runner.enable_decode_graphs(buckets, capture=True, verbose=False).items()}
                rec["arms"][st] = {"routers": route.n, "graph_status": gs, "step_ms": [], "tokens": []}
            for st in first:
                with EpiRoute(model, st == "ON", routers):
                    run[st].bind(0, slots[st], list(prompt))
                    run[st].run_prefill([(0, 0, len(prompt))])
            with _Clock() as clock:
                for i in range(n_steps):
                    for st in first:                             # strict alternation, one step of each per pair
                        with EpiRoute(model, st == "ON", routers):
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
        raise SystemExit(f"REFUSED: the knobs must be {KNOBS}, got {bad}")
    import torch
    import transformers

    import p115_quality as q
    from experts4bit_qlora.serve_paged import DEFAULT_BUCKETS, PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: the box builds the server eager at all-vram (graphs={cfg.graphs}, "
                         f"placement={cfg.placement}); its runners capture their own graphs")
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    info = parts.info
    routers = fused_routers(model)
    prompt = q.LOADERS["wikitext"](parts.tokenizer, 1, PROMPT, 1)[0][:PROMPT]
    t1 = time.time()
    blocks = {b: interleaved_block(model, prompt, cfg.device, block=b, steps=a.steps, buckets=DEFAULT_BUCKETS,
                                   bulk_kv=cfg.bulk_kv, routers=routers) for b in BLOCKS}
    med = {f"{st}_{b}": round(statistics.median(blocks[b]["arms"][st]["step_ms"]), 4) for b in BLOCKS for st in SETTINGS}
    rec = {"model": cfg.model, "revision": cfg.revision, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "knobs": {k: os.environ.get(k) for k in KNOBS},
           "census": {k: info.get(k) for k in CENSUS_KEYS}, "fusion_report": info.get("fusion_report"),
           "model_type": (info.get("model_type")), "fused_routers": len(routers),
           "buckets": list(DEFAULT_BUCKETS), "prompt": PROMPT, "prompt_tokens": len(prompt), "kv_batch": KV_BATCH,
           "blocks": blocks, "medians": med, "transformers": transformers.__version__, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
           "load_s": round(load_s, 1), "seconds": round(time.time() - t1, 1), "status": "ok"}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"FAM_SPEED census={rec['census']} routers={rec['fused_routers']} medians={med} {rec['seconds']} s", flush=True)
    return 0


def self_test() -> int:
    ok = [BLOCKS == {"a": ("OFF", "ON"), "b": ("ON", "OFF")}, SLOT["OFF"] != SLOT["ON"],
          max(SLOT.values()) < KV_BATCH, (WARM, STEPS, BUSY, PROOF_STEPS, PROMPT) == (5, 256, 32, 32, 512),
          KNOBS["E4B_FUSE_ROUTER_EPI"] == "1" and all(KNOBS[k] == "0" for k in KNOBS if k != "E4B_FUSE_ROUTER_EPI")]

    class M:
        def forward(self, x):
            return ("orig", x)

    plain, other, router = M(), M(), M()                         # untouched; patched elsewhere; patched by the epilogue

    def elsewhere(hidden_states, _orig=None):
        return ("other", hidden_states)
    other.forward = elsewhere

    def fused(hidden_states, _m=router, _orig=M.forward.__get__(router)):
        return ("fused", hidden_states)
    fused.__module__ = EPI_MODULE
    router.forward = fused

    class Model:
        def modules(self):
            return iter([plain, other, router])
    found = fused_routers(Model())
    ok.append(len(found) == 1 and found[0][0] is router and found[0][1] is fused)
    with EpiRoute(Model(), False, found) as r:
        ok.append(r.n == 1 and router.forward(1) == ("orig", 1) and other.forward(1) == ("other", 1))
    ok.append(router.forward(1) == ("fused", 1))                 # the build's state is restored
    with EpiRoute(Model(), True, found):
        ok.append(router.forward(2) == ("fused", 2))
    print(f"fam_speed self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out")
    p.add_argument("--steps", type=int, default=STEPS)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.out or a.steps not in (STEPS, PROOF_STEPS):
        raise SystemExit(f"REFUSED: --out is required and --steps is {STEPS} (the reading) or {PROOF_STEPS} (the proof)")
    return main_run(a)


if __name__ == "__main__":
    raise SystemExit(main())
