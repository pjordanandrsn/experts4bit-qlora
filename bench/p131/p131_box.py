#!/usr/bin/env python3
"""Lane P131's measurement (bench/p131/PREREG-p131.md; e4b#846): does K19 over a split tile table make the served
512-token prefill faster, and is it the same arithmetic?

``E4B_K19_PREFILL_BIG_M`` (e4b#1625, over grouped-nf4-gemm#542): K19's prefill calls above 512 routed rows build two
tables over one ``order`` -- each expert's full ``big_m``-row tiles, then its remainder in 16-row tiles -- and run the
tall tiles at ``block_m=big_m``. The knob is read on every call, so every process captures the first-chunk prefill
graph once per arm and times the three interleaved, as P130 timed P2:
- **m0**: the knob ``0``, the 16-row table (the release's default; the control);
- **m32** and **m64**: the knob ``32`` and ``64``.

One run of this box is one PROCESS of the run's registered order (``--proc``): process 0 is Amendment 2's WARM-UP (it
runs Phase B and a speed pass whose timing is reported only), processes 1..N are the scored speed processes. Each process
builds SC2e's int4 stack as
P117 and P130 build it (eager, one slot, all-vram, no prefill graph: the box makes its own runners), at the release's
prefill defaults (``E4B_FUSE_PREFILL_GLUE`` and ``E4B_PREFILL_LEAN_DISPATCH`` unset: on for ``qwen3_moe``, P130), then:

**Phase A, speed** (every process). A runner made as the server makes its own captures the first-chunk prefill graph at
``T`` = 512 three times, through the server's own ``enable_prefill_graph`` (warm-up, capture, startup check), the knob
set per capture, in ``capture_order(proc)``: m0 m32 m64 rotated by (proc - 1), so over the scored processes every arm is
captured first, second and third (a capture-position effect -- pool placement, alignment -- is not the arm's). Then:
- **FUNCTION:** on every speed window all three graphs replay, and each tall arm's logits and every pool layer's staged
  K and V are compared with m0's by ``torch.equal``; each window's m0 output is digested (sha256) for DETERMINISM;
- **timing:** ``--warm`` untimed rounds, then ``--rounds`` timed rounds over the speed windows. In each round every
  window replays the three graphs in the order m0 m32 m64 rotated by (round + window), each replay between two CUDA
  events. Per graph the record keeps every replay's ms and their median;
- **decode untouched:** one eager forward of a ``--decode-tokens`` = 64-token prompt per arm, with the knob set: 64 x
  top-8 = 512 routed rows, the rows a 64-row decode step routes, which must keep the chained 16-row table;
- **eager (reported, never gated):** ``--eager-windows`` windows through the eager first-chunk forward, wall ms with a
  sync, the three arms rotated, ``--eager-rounds`` rounds;
- **engagement:** every eager forward is counted, and the route, K19's dispatch and the prefill folds are read as deltas
  per capture (P130's counters). Replays run no Python, so they count nothing.

**GPU state** (Amendment 2, reported only): nvidia-smi's SM and memory clocks, power draw, temperature and active clock
event reasons, read at the start of the speed pass, right after the warm rounds and right after the timed rounds (under
load, after a sync, outside every timed pair), and at its end (``gpu_state()``; an unreadable query is recorded as its
error, never raised).

**Phase B, quality** (``--phase-b all``, process 0 only, Amendment 2): P117's instrument as P130 ran it, in ONE process
because the knob is read per call. Teacher-forced paged passes (``p117_box.paged_pass``) over ``--windows`` wikitext
windows decoded together: **R** (m0, buckets 1-16), **rep**, the floor (**half**, **chunk**, **rev**), **mutant_scale**,
then the subjects **S32** and **S64** (R's buckets with the knob set), each scored against R's log-probs held in memory, with
``bitwise`` recording whether every one of its windows' log-probs is ``torch.equal`` to R's. Decode rows here are at
most 16 (128 routed rows), so the knob reaches only the prefill.

``speed()`` and ``quality()`` take the model and the windows, so ``tests/test_p131_box.py`` runs them on CPU with the
runner and the decode attention stood in.

    python p131_box.py --out OUT.json --proc N --phase-b none|all   (engine from E4B_PAGED_*)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p108_box  # noqa: E402  (staged at P108's registered bytes)
import p117_box  # noqa: E402  (staged at P117's registered bytes: windows(), paged_pass(), the bucket lists)
import p130_box  # noqa: E402  (staged at P130's registered bytes: the counters, digests, timers, runner and scoring)

ENV = "E4B_K19_PREFILL_BIG_M"
DEFAULT_ENVS = ("E4B_FUSE_PREFILL_GLUE", "E4B_PREFILL_LEAN_DISPATCH")     # unset: the release's defaults (P130)
ARMS = ("m0", "m32", "m64")
TALL = ("m32", "m64")
KNOB = {"m0": "0", "m32": "32", "m64": "64"}
B16, B8 = p117_box.B16, p117_box.B8
FLOORS = ("half", "chunk", "rev")
OFF_ARMS = ("R", "rep") + FLOORS + ("mutant_scale",)
SUBJECTS = {"S32": "m32", "S64": "m64"}
ARM_KW = {"R": {"buckets": B16}, "rep": {"buckets": B16}, "half": {"buckets": B8}, "chunk": {"buckets": B16},
          "rev": {"buckets": B16, "reverse": True}, "mutant_scale": {"buckets": B16, "mutant": "scale"},
          "S32": {"buckets": B16}, "S64": {"buckets": B16}}


def _knob(arm: str | None) -> None:
    if arm is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = KNOB[arm]


def _rotated(k: int) -> tuple:
    k %= len(ARMS)
    return ARMS[k:] + ARMS[:k]


def capture_order(proc: int) -> tuple:
    """The arms' capture order in process ``proc``: m0 m32 m64 rotated by (proc - 1); the warm-up process 0 is rotated by -1."""
    return _rotated(proc - 1)


def speed(model, prompts, T, device, *, rounds, warm, eager_rounds, eager_windows, decode_tokens=64, bulk_kv=True,
          last_logits=False, runner=None, proc=1) -> dict:
    """Phase A on one process (see the module docstring). ``prompts`` are token lists of at least ``T``."""
    from experts4bit_qlora.engines import hot_residency as hr
    from experts4bit_qlora.engines.paged_runner import PrefillGraphRefused
    runner = runner if runner is not None else p130_box._runner(model, T, device, bulk_kv, last_logits)
    calls = [0]
    inner = runner._prefill_forward

    def counted(*a, **kw):
        calls[0] += 1
        return inner(*a, **kw)

    runner._prefill_forward = counted
    saved = (hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0], os.environ.get(ENV))
    rec = {"T": T, "windows": len(prompts), "rounds": rounds, "warm": warm, "eager_rounds": eager_rounds,
           "eager_windows": eager_windows, "decode_tokens": decode_tokens, "bulk_kv": bool(bulk_kv),
           "last_logits": bool(last_logits), "capture_order": list(capture_order(proc)), "graphs": {}}
    graphs = {}
    ids = [torch.tensor(p[:T], dtype=torch.long)[None] for p in prompts]
    try:
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = True, False     # the graph server's grouping
        for g in capture_order(proc):
            _knob(g)
            c0, calls[0] = p130_box._counters(), 0
            try:
                st = runner.enable_prefill_graph(T)
            except PrefillGraphRefused as e:
                rec["graphs"][g] = {"status": "refused", "why": str(e)[:400], "forwards": calls[0],
                                    "seen": p130_box._delta(c0, p130_box._counters())}
                continue
            graphs[g] = runner._prefill_graph
            rec["graphs"][g] = {"status": st.get("status"), "T": st.get("T"), "pool_mib": st.get("pool_mib"),
                                "forwards": calls[0], "seen": p130_box._delta(c0, p130_box._counters())}
        runner.disable_prefill_graph()                 # the box replays the graphs itself
        # decode untouched: 64 tokens route the rows a 64-row decode step routes; every arm keeps the chained table
        pos = torch.arange(decode_tokens, device=device)[None]
        xd = ids[0][:, :decode_tokens].to(device)
        dec = {}
        for g in ARMS:
            _knob(g)
            c0 = p130_box._counters()
            restore = runner._prefill_scope()
            try:
                with torch.no_grad():
                    runner.ctx.drop(runner._PG_KEY)
                    runner._prefill_forward(xd, pos)
            finally:
                restore()
            dec[g] = p130_box._delta(c0, p130_box._counters())
        rec["decode_check"] = {"tokens": decode_tokens, "seen": dec}
        if "m0" not in graphs:
            return rec                                 # the control's graph refused: nothing to time
        have = tuple(g for g in ARMS if g in graphs)   # a tall arm may have refused: the others are still timed
        fn = {g: {"windows": 0, "differ": []} for g in TALL if g in graphs}
        digests = []
        with torch.no_grad():
            for w, x in enumerate(ids):
                out = {}
                for g in have:
                    pg = graphs[g]
                    pg["ids"].copy_(x)
                    pg["graph"].replay()
                    out[g] = (pg["logits"].clone(), {lay: (k.clone(), v.clone()) for lay, (k, v) in pg["staged"].items()})
                lo, kvo = out["m0"]
                for g in fn:
                    ln, kvn = out[g]
                    bad = [lay for lay in kvo if lay not in kvn or not (torch.equal(kvo[lay][0], kvn[lay][0])
                                                                         and torch.equal(kvo[lay][1], kvn[lay][1]))]
                    same = torch.equal(lo, ln)
                    if not same or bad or sorted(kvo) != sorted(kvn):
                        fn[g]["differ"].append({"window": w, "logits_equal": bool(same), "kv_layers_differ": bad[:8],
                                                "logits_max_abs": float((lo.float() - ln.float()).abs().max())})
                    fn[g]["windows"] += 1
                digests.append(p130_box._digest(lo, kvo))
                del out
        rec["function"] = fn
        rec["digests_m0"] = digests
        ms = {g: [] for g in have}
        for r in range(warm + rounds):
            evs = []
            for w, x in enumerate(ids):
                for g in (a for a in _rotated(r + w) if a in graphs):
                    pg = graphs[g]
                    pg["ids"].copy_(x)
                    s, e = p130_box._pair(device)
                    s.record()
                    pg["graph"].replay()
                    e.record()
                    evs.append((g, s, e))
            p130_box._sync(device)
            if r >= warm:
                for g, s, e in evs:
                    ms[g].append(round(s.elapsed_time(e), 4))
            if r == warm - 1:          # Amendment 2: the clocks the timed rounds start at (under load, after a sync, untimed)
                rec["gpu_after_warm"] = gpu_state()
            if r == warm + rounds - 1:
                rec["gpu_after_timed"] = gpu_state()
        rec["replay_ms"] = ms
        rec["median_ms"] = {g: statistics.median(v) for g, v in ms.items()}
        eager = {g: [] for g in ARMS}
        c0, calls[0] = p130_box._counters(), 0
        pos = torch.arange(T, device=device)[None]
        for r in range(eager_rounds):
            for w, x in enumerate(ids[:eager_windows]):
                for g in _rotated(r + w):
                    _knob(g)
                    restore = runner._prefill_scope()
                    try:
                        with torch.no_grad():
                            runner.ctx.drop(runner._PG_KEY)
                            p130_box._sync(device)
                            t0 = time.perf_counter()
                            runner._prefill_forward(x.to(device), pos)
                            p130_box._sync(device)
                            eager[g].append(round(1000.0 * (time.perf_counter() - t0), 3))
                    finally:
                        restore()
        rec["eager"] = {"forwards": calls[0], "seen": p130_box._delta(c0, p130_box._counters()), "ms": eager,
                        "median_ms": {g: (statistics.median(v) if v else None) for g, v in eager.items()}}
    finally:
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = saved[2]
        runner._prefill_forward = inner
        graphs.clear()
    del runner
    p108_box._release(device)
    return rec


def quality(model, ws, *, prompt, cont, chunk, floor_chunk, device, stand_in=False) -> dict:
    """Phase B in one process: R, rep, the floor and the mutant at m0, then the subjects S32 and S64 with the knob set,
    each scored against R's log-probs held in memory."""
    from experts4bit_qlora.engines import paged_attention
    P, C = prompt, cont
    paged_attention.register(model)
    arms = OFF_ARMS + tuple(SUBJECTS)
    per = {a: [] for a in arms}
    eng = {a: [] for a in arms}
    bitwise = {}
    start = time.time()
    saved = os.environ.get(ENV)
    try:
        _knob("m0")
        refs, e = p130_box._pass(model, ws, P, C, chunk, device, stand_in, **ARM_KW["R"])
        eng["R"].append(e)
        refs = [r.cpu() for r in refs]
        ref_am = [r.argmax(-1).tolist() for r in refs]
        for i, r in enumerate(refs):
            per["R"].append(p130_box._score_rec(i, r, ws[i][P:P + C], ref_am[i], None))
        print(f"P131_ARM R done at {time.time() - start:.0f} s", flush=True)
        rep_identical = None
        for arm in arms[1:]:
            _knob(SUBJECTS.get(arm, "m0"))
            lps, e = p130_box._pass(model, ws, P, C, floor_chunk if arm == "chunk" else chunk, device, stand_in,
                                    **ARM_KW[arm])
            eng[arm].append(e)
            same = all(torch.equal(a.cpu(), b) for a, b in zip(lps, refs))
            if arm == "rep":
                rep_identical = same
            if arm in SUBJECTS:
                bitwise[arm] = same
            for i, x in enumerate(lps):
                per[arm].append(p130_box._score_rec(i, x, ws[i][P:P + C], ref_am[i], refs[i]))
            del lps
            p108_box._release(device)
            print(f"P131_ARM {arm} done at {time.time() - start:.0f} s", flush=True)
        del refs
    finally:
        if saved is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = saved
    p108_box._release(device)
    return {"phase": "all", "windows": len(ws), "prompt": P, "cont": C, "chunk": chunk, "floor_chunk": floor_chunk,
            "rep_identical": rep_identical, "subjects_bitwise": bitwise, "per_window": per, "engagement": eng,
            "rehearsal": {"stand_in_attention": bool(stand_in)}, "seconds": round(time.time() - start, 1)}


GPU_FIELDS = ("clocks.sm", "clocks.mem", "power.draw", "temperature.gpu", "clocks_event_reasons.active")


def gpu_state() -> dict:
    """Amendment 2 (reported only): nvidia-smi's SM / memory clocks (MHz), power (W), temperature (C) and the active clock
    event reasons, now. A field the driver does not know is retried without the event reasons; any failure is recorded as
    its error, never raised: the record must not cost the process."""
    import subprocess
    for fields in (GPU_FIELDS, GPU_FIELDS[:4]):
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
        except Exception as e:  # noqa: BLE001 -- no nvidia-smi on this host (the CPU tests), a hang
            return {"utc": time.time(), "error": f"{type(e).__name__}: {str(e)[:200]}"}
        if r.returncode == 0 and r.stdout.strip():
            vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
            return {"utc": time.time(), **dict(zip(fields, vals))}
    return {"utc": time.time(), "error": f"nvidia-smi rc {r.returncode}: {(r.stdout + r.stderr).strip()[:200]}"}


def _gpu_line(g: dict) -> str:
    if "error" in g:
        return f"unread ({g['error'][:60]})"
    return (f"sm {g.get('clocks.sm')} MHz mem {g.get('clocks.mem')} MHz {g.get('power.draw')} W {g.get('temperature.gpu')} C"
            f" reasons {g.get('clocks_event_reasons.active', '-')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--proc", type=int, required=True)
    ap.add_argument("--phase-b", choices=("none", "all"), default="none")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--speed-windows", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--eager-rounds", type=int, default=2)
    ap.add_argument("--eager-windows", type=int, default=4)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--floor-chunk", type=int, default=256)
    ap.add_argument("--stride", type=int, default=3072)
    a = ap.parse_args()
    import transformers

    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    set_now = {k: os.environ[k] for k in (ENV,) + DEFAULT_ENVS if (os.environ.get(k) or "").strip()}
    if set_now:
        raise SystemExit(f"REFUSED: process {a.proc} runs the release's prefill defaults with {ENV} set per capture "
                         f"by the box; the environment sets {set_now}")
    if a.proc < 0:
        raise SystemExit(f"REFUSED: process {a.proc}: the processes are 0 (the warm-up) and 1..N")
    if a.phase_b == "all" and a.proc != 0:
        raise SystemExit("REFUSED: Phase B runs in process 0 only (Amendment 2: the warm-up)")
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram" or cfg.max_seqs != 1:
        raise SystemExit(f"REFUSED: the engine is built eager at all-vram with one slot (graphs={cfg.graphs}, "
                         f"placement={cfg.placement}, max_seqs={cfg.max_seqs}); the box makes its own runners")
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    paged_attention.register(model)
    load_s = time.time() - t0
    ws = p117_box.windows(parts.tokenizer, a.windows, a.prompt, a.cont, a.stride)
    cfgm = getattr(model.config, "text_config", None) or model.config
    rec = {"proc": a.proc, "phase_b": a.phase_b, "model": cfg.model, "revision": cfg.revision,
           "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "transformers": transformers.__version__, "load_s": round(load_s, 1), "stride": a.stride,
           "layers": int(cfgm.num_hidden_layers), "windows_digest": p130_box.windows_digest(ws),
           "corpus": p130_box.corpus_commits(), "prefill_glue": parts.info.get("prefill_glue"),
           "census": {k: parts.info.get(k) for k in ("moe_layers", "experts", "top_k", "model_type", "int4_expert_layers",
                                                     "int4_attn_projections", "int4_store_kinds", "fusion_report",
                                                     "levers_env")}}
    st = time.time()
    gs0 = gpu_state()
    rec["speed"] = speed(model, [w[:a.prompt] for w in ws[:a.speed_windows]], a.prompt, cfg.device, rounds=a.rounds,
                         warm=a.warm, eager_rounds=a.eager_rounds, eager_windows=a.eager_windows,
                         decode_tokens=a.decode_tokens, bulk_kv=cfg.bulk_kv, last_logits=cfg.last_logits, proc=a.proc)
    rec["gpu_state"] = {"speed_start": gs0, "after_warm": rec["speed"].get("gpu_after_warm"),
                        "after_timed": rec["speed"].get("gpu_after_timed"), "speed_end": gpu_state()}
    print(f"P131_GPU proc {a.proc} | " + " | ".join(f"{k} {_gpu_line(v) if v else '-'}" for k, v in rec["gpu_state"].items()),
          flush=True)
    sp = rec["speed"]
    fn = sp.get("function") or {}
    print(f"P131_SPEED proc {a.proc} | graphs {[(g, v.get('status')) for g, v in sp['graphs'].items()]} "
          f"| median ms {sp.get('median_ms')} | differs {({g: len(v['differ']) for g, v in fn.items()})} "
          f"| eager ms {(sp.get('eager') or {}).get('median_ms')} | {time.time() - st:.0f} s", flush=True)
    if a.phase_b == "all":
        try:                                   # a Phase B failure is recorded; the process's speed record stands
            rec["quality"] = quality(model, ws, prompt=a.prompt, cont=a.cont, chunk=a.chunk, floor_chunk=a.floor_chunk,
                                     device=cfg.device)
        except Exception as e:  # noqa: BLE001 -- the reducer reads it as QUALITY_VOID
            rec["quality"] = {"phase": a.phase_b, "error": f"{type(e).__name__}: {str(e)[:400]}"}
            print(f"P131_QUALITY_ERROR {rec['quality']['error']}", flush=True)
    rec.update(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               max_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)
    with open(a.out, "w") as f:
        f.write(json.dumps(rec, indent=1))
    q = rec.get("quality")
    ql = ""
    if q:
        ql = " | quality " + (f"ERROR {q['error'][:120]}" if "error" in q else f"windows {q['windows']} {q['seconds']} s")
    print(f"P131_BOX proc {a.proc} | load {load_s:.0f}s | mem {rec['max_mem_gb']} GB{ql}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
