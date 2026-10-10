#!/usr/bin/env python3
"""Lane P130's measurement (bench/p130/PREREG-p130.md; e4b#846): do #1583's two opt-in prefill knobs make the served
512-token prefill faster, and does P1's arithmetic cost quality?

- **P1**, ``E4B_FUSE_PREFILL_GLUE=1``: the RMSNorm fold, the residual-and-norm fold and the q/k-norm-and-rotary fold also
  take prefill rows (above 64), with decode's single rounding (``bench/prefill-glue/DESIGN.md``, option B). It changes
  bf16 arithmetic. The knob is read at patch time, so P1 is set per PROCESS.
- **P2**, ``E4B_PREFILL_LEAN_DISPATCH=1``: K19's prefill rows read their token rows themselves (``gather_div=``) and store
  in the caller's order (``scatter=order``), so there is no 4,096-row gather and no unsort. It is bit-identical by
  construction, which #1583's A2000 smoke read with a blindness arm. The knob is read on every call, so one process
  captures one prefill graph with P2 off and one with P2 on, and times them interleaved.

One run of this box is one PROCESS of the run's registered order (``--proc``); ``p130_run.sh`` sets
``E4B_FUSE_PREFILL_GLUE`` per process. Each process builds SC2e's int4 stack as P117 builds it (eager, one slot,
all-vram, no prefill graph: the box makes its own runners), then:

**Phase A, speed** (every process). A runner made as the server makes its own (``bulk_kv`` and ``last_logits`` from
``PagedServeConfig``; device grouping on) captures the first-chunk prefill graph at ``T`` = 512 twice, through the
server's own ``enable_prefill_graph`` (warm-up, capture, startup check): ``p2_off`` with ``E4B_PREFILL_LEAN_DISPATCH=0``
and ``p2_on`` with it at ``1``. Then:
- **FUNCTION (P2):** on every speed window both graphs replay, and their logits and every pool layer's staged K and V
  must be ``torch.equal``. Each window's ``p2_off`` logits and K/V are digested (sha256) for the cross-process
  DETERMINISM gate;
- **timing:** ``--warm`` untimed rounds, then ``--rounds`` timed rounds over the speed windows. In each round every
  window replays both graphs, off-on or on-off alternating with (round + window), each replay between two CUDA events.
  Per graph the record keeps every replay's ms and their median;
- **eager (reported, never gated):** ``--eager-windows`` windows through the eager first-chunk forward (the runner's
  ``_prefill_forward`` in its prefill scope: the path a later chunk takes), wall ms with a sync, the two P2 settings
  interleaved, ``--eager-rounds`` rounds;
- **engagement:** every eager forward is counted (the runner's ``_prefill_forward`` is wrapped), and the route
  (``hot_residency.ROUTE_SEEN``), K19's dispatch (``hot_residency.K19_DISPATCH_SEEN``) and the prefill folds
  (``glue_fuse.PREFILL_FOLD_SEEN``) are read as deltas per capture. Replays run no Python, so they count nothing.

**Phase B, quality** (``--phase-b off`` in process 1 and ``on`` in process 2): P117's instrument by named substitution.
Teacher-forced paged passes (``p117_box.paged_pass``, at P117's registered bytes) over ``--windows`` wikitext windows
decoded together, every pass device-grouped with every decode step a padded eager step of its bucket:
- ``off`` (P1 unset): **R** (buckets 1-16), **rep**, the floor (**half**, **chunk**, **rev**) and **mutant_scale** (R
  with the decode softmax scale halved). R's fp32 log-probs go to ``--ref-dir`` on the box, one file a window, with
  their digests;
- ``on`` (P1 set): the subject **P1**, R's buckets on the P1 model, scored against R's saved log-probs (each file's digest
  checked on load).
Each pass records P117's engagement plus the route and prefill-fold deltas.

``speed()``, ``quality_off()`` and ``quality_on()`` take the model and the windows, so ``tests/test_p130_box.py`` runs
them on CPU with the runner and the decode attention stood in.

    python p130_box.py --out OUT.json --proc N --p1 0|1 --phase-b none|off|on [--ref-dir DIR]   (engine from E4B_PAGED_*)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p108_box  # noqa: E402  (staged at P108's registered bytes: Attention, _score, _kl, _release)
import p117_box  # noqa: E402  (staged at P117's registered bytes: windows(), paged_pass(), the bucket lists)

P1_ENV, P2_ENV = "E4B_FUSE_PREFILL_GLUE", "E4B_PREFILL_LEAN_DISPATCH"
B16, B8 = p117_box.B16, p117_box.B8
FLOORS = ("half", "chunk", "rev")
OFF_ARMS = ("R", "rep") + FLOORS + ("mutant_scale",)
SUBJECT = "P1"
ARM_KW = {"R": {"buckets": B16}, "rep": {"buckets": B16}, "half": {"buckets": B8}, "chunk": {"buckets": B16},
          "rev": {"buckets": B16, "reverse": True}, "mutant_scale": {"buckets": B16, "mutant": "scale"},
          "P1": {"buckets": B16}}
GRAPHS = ("p2_off", "p2_on")


def fold_table(layers: int) -> dict:
    """Prefill-fold calls per prefill forward above 64 rows with P1 on: the input norms and the final norm through the
    RMSNorm fold, the residual-and-post-attention-norm fold once a layer, the q/k-norm-and-rotary fold once a layer."""
    return {"norm": layers + 1, "layer": layers, "attention": layers}


def _counters() -> dict:
    from experts4bit_qlora.engines import glue_fuse
    from experts4bit_qlora.engines import hot_residency as hr
    return {"route": dict(hr.ROUTE_SEEN), "k19": dict(hr.K19_DISPATCH_SEEN), "folds": dict(glue_fuse.PREFILL_FOLD_SEEN)}


def _delta(before: dict, after: dict) -> dict:
    return {name: {k: v - before[name].get(k, 0) for k, v in sorted(after[name].items()) if v != before[name].get(k, 0)}
            for name in after}


def _bytes(t) -> bytes:
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def _digest(logits, staged: dict) -> str:
    h = hashlib.sha256(_bytes(logits))
    for lay in sorted(staged):
        h.update(_bytes(staged[lay][0]))
        h.update(_bytes(staged[lay][1]))
    return h.hexdigest()


class _Wall:
    """A CUDA event's surface on a wall clock: the CPU test's timer."""

    def record(self):
        self.t = time.perf_counter()

    def elapsed_time(self, end) -> float:
        return 1000.0 * (end.t - self.t)


def _pair(device):
    if torch.device(device).type == "cuda":
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return _Wall(), _Wall()


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _p2(value: str | None) -> None:
    if value is None:
        os.environ.pop(P2_ENV, None)
    else:
        os.environ[P2_ENV] = value


def _runner(model, T, device, bulk_kv, last_logits):
    """A one-slot runner made as ``serve_paged.build_engine`` makes the server's (its ``PagedModelRunner`` call)."""
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry
    cfg = getattr(model.config, "text_config", None) or model.config
    hkv, hd = _kv_geometry(model.config)
    kv = Fp8PagedKV(kv_layers(model, int(cfg.num_hidden_layers)), hkv, hd, batch=1, max_tokens_per_seq=T + 16,
                    device=device, scratch_slots=1)
    return PagedModelRunner(model, kv, device=device, bulk_kv=bulk_kv, last_logits=last_logits)


def speed(model, prompts, T, device, *, rounds, warm, eager_rounds, eager_windows, bulk_kv=True, last_logits=False,
          runner=None) -> dict:
    """Phase A on one process (see the module docstring). ``prompts`` are token lists of at least ``T``."""
    from experts4bit_qlora.engines import hot_residency as hr
    from experts4bit_qlora.engines.paged_runner import PrefillGraphRefused
    runner = runner if runner is not None else _runner(model, T, device, bulk_kv, last_logits)
    calls = [0]
    inner = runner._prefill_forward

    def counted(*a, **kw):
        calls[0] += 1
        return inner(*a, **kw)

    runner._prefill_forward = counted
    saved = (hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0], os.environ.get(P2_ENV))
    rec = {"T": T, "windows": len(prompts), "rounds": rounds, "warm": warm, "eager_rounds": eager_rounds,
           "eager_windows": eager_windows, "bulk_kv": bool(bulk_kv), "last_logits": bool(last_logits), "graphs": {}}
    graphs = {}
    ids = [torch.tensor(p[:T], dtype=torch.long)[None] for p in prompts]
    try:
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = True, False     # the graph server's grouping
        for g in GRAPHS:
            _p2("1" if g == "p2_on" else "0")
            c0, calls[0] = _counters(), 0
            try:
                st = runner.enable_prefill_graph(T)
            except PrefillGraphRefused as e:
                rec["graphs"][g] = {"status": "refused", "why": str(e)[:400], "forwards": calls[0],
                                    "seen": _delta(c0, _counters())}
                continue
            graphs[g] = runner._prefill_graph
            rec["graphs"][g] = {"status": st.get("status"), "T": st.get("T"), "pool_mib": st.get("pool_mib"),
                                "forwards": calls[0], "seen": _delta(c0, _counters())}
        runner.disable_prefill_graph()                 # the box replays the two graphs itself
        if "p2_off" not in graphs:
            return rec                                 # the default's graph refused: nothing to time
        have = tuple(g for g in GRAPHS if g in graphs)  # p2_on alone may have refused: p2_off is still timed
        # FUNCTION (P2) and the cross-process digests, before any timing
        fn = {"windows": 0, "differ": []}
        digests = []
        with torch.no_grad():
            for w, x in enumerate(ids):
                out = {}
                for g in have:
                    pg = graphs[g]
                    pg["ids"].copy_(x)
                    pg["graph"].replay()
                    out[g] = (pg["logits"].clone(), {lay: (k.clone(), v.clone()) for lay, (k, v) in pg["staged"].items()})
                lo, kvo = out["p2_off"]
                if "p2_on" in out:
                    ln, kvn = out["p2_on"]
                    bad = [lay for lay in kvo if lay not in kvn or not (torch.equal(kvo[lay][0], kvn[lay][0])
                                                                         and torch.equal(kvo[lay][1], kvn[lay][1]))]
                    same = torch.equal(lo, ln)
                    if not same or bad or sorted(kvo) != sorted(kvn):
                        fn["differ"].append({"window": w, "logits_equal": bool(same), "kv_layers_differ": bad[:8]})
                    fn["windows"] += 1
                digests.append(_digest(lo, kvo))
                del out
        rec["function_p2"] = fn if "p2_on" in have else None
        rec["digests_p2_off"] = digests
        # timing: every captured graph every round, every window, in alternating order
        ms = {g: [] for g in have}
        for r in range(warm + rounds):
            evs = []
            for w, x in enumerate(ids):
                for g in (have if (r + w) % 2 == 0 else have[::-1]):
                    pg = graphs[g]
                    pg["ids"].copy_(x)
                    s, e = _pair(device)
                    s.record()
                    pg["graph"].replay()
                    e.record()
                    evs.append((g, s, e))
            _sync(device)
            if r >= warm:
                for g, s, e in evs:
                    ms[g].append(round(s.elapsed_time(e), 4))
        rec["replay_ms"] = ms
        rec["median_ms"] = {g: statistics.median(v) for g, v in ms.items()}
        # eager first-chunk forwards (reported, never gated)
        eager = {g: [] for g in GRAPHS}
        c0, calls[0] = _counters(), 0
        pos = torch.arange(T, device=device)[None]
        for r in range(eager_rounds):
            for w, x in enumerate(ids[:eager_windows]):
                for g in (GRAPHS if (r + w) % 2 == 0 else GRAPHS[::-1]):
                    _p2("1" if g == "p2_on" else "0")
                    restore = runner._prefill_scope()
                    try:
                        with torch.no_grad():
                            runner.ctx.drop(runner._PG_KEY)
                            _sync(device)
                            t0 = time.perf_counter()
                            runner._prefill_forward(x.to(device), pos)
                            _sync(device)
                            eager[g].append(round(1000.0 * (time.perf_counter() - t0), 3))
                    finally:
                        restore()
        rec["eager"] = {"forwards": calls[0], "seen": _delta(c0, _counters()), "ms": eager,
                        "median_ms": {g: (statistics.median(v) if v else None) for g, v in eager.items()}}
    finally:
        hr.DEVICE_GROUPING[0], hr.FORCE_SINGLETON_GROUPS[0] = saved[0], saved[1]
        _p2(saved[2])
        runner._prefill_forward = inner
        graphs.clear()
    del runner
    p108_box._release(device)
    return rec


def _score_rec(i, lp, cont_tokens, ref_am, ref) -> dict:
    nll, am = p108_box._score(lp, cont_tokens)
    rec = {"window": i, "nll": sum(nll) / len(nll), "argmax_agree": sum(int(x == y) for x, y in zip(am, ref_am)) / len(am)}
    if ref is not None:
        kl = p108_box._kl(ref, lp)
        rec["kl"] = sum(kl) / len(kl)
    return rec


def _pass(model, ws, P, C, chunk, device, stand_in, **kw):
    c0 = _counters()
    lps, _tk, e = p117_box.paged_pass(model, ws, P, C, chunk, device, stand_in=stand_in, **kw)
    e["seen"] = _delta(c0, _counters())
    e["chunk"] = chunk
    return lps, e


def quality_off(model, ws, *, prompt, cont, chunk, floor_chunk, device, ref_dir, stand_in=False) -> dict:
    """Phase B's OFF process: R, rep, the floor and the mutant; R's log-probs saved to ``ref_dir`` with digests."""
    from experts4bit_qlora.engines import paged_attention
    P, C = prompt, cont
    paged_attention.register(model)
    per = {a: [] for a in OFF_ARMS}
    eng = {a: [] for a in OFF_ARMS}
    start = time.time()
    refs, e = _pass(model, ws, P, C, chunk, device, stand_in, **ARM_KW["R"])
    eng["R"].append(e)
    ref_am = [r.argmax(-1).tolist() for r in refs]
    os.makedirs(ref_dir, exist_ok=True)
    digests = []
    for i, r in enumerate(refs):
        torch.save(r, os.path.join(ref_dir, f"r{i:03d}.pt"))
        digests.append(hashlib.sha256(_bytes(r)).hexdigest())
        per["R"].append(_score_rec(i, r, ws[i][P:P + C], ref_am[i], None))
    with open(os.path.join(ref_dir, "digests.json"), "w") as f:
        json.dump(digests, f)
    print(f"P130_ARM R done at {time.time() - start:.0f} s", flush=True)
    rep_identical = None
    for arm in OFF_ARMS[1:]:
        lps, e = _pass(model, ws, P, C, floor_chunk if arm == "chunk" else chunk, device, stand_in, **ARM_KW[arm])
        eng[arm].append(e)
        if arm == "rep":
            rep_identical = all(torch.equal(a, b) for a, b in zip(lps, refs))
        for i, x in enumerate(lps):
            per[arm].append(_score_rec(i, x, ws[i][P:P + C], ref_am[i], refs[i]))
        del lps
        p108_box._release(device)
        print(f"P130_ARM {arm} done at {time.time() - start:.0f} s", flush=True)
    del refs
    p108_box._release(device)
    return {"phase": "off", "windows": len(ws), "prompt": P, "cont": C, "chunk": chunk, "floor_chunk": floor_chunk,
            "rep_identical": rep_identical, "ref_digests": digests, "per_window": per, "engagement": eng,
            "rehearsal": {"stand_in_attention": bool(stand_in)}, "seconds": round(time.time() - start, 1)}


def quality_on(model, ws, *, prompt, cont, chunk, device, ref_dir, stand_in=False) -> dict:
    """Phase B's ON process: the subject P1, scored against R's saved log-probs (each file's digest checked)."""
    from experts4bit_qlora.engines import paged_attention
    P, C = prompt, cont
    paged_attention.register(model)
    start = time.time()
    want = json.load(open(os.path.join(ref_dir, "digests.json")))
    lps, e = _pass(model, ws, P, C, chunk, device, stand_in, **ARM_KW[SUBJECT])
    per, bad = [], []
    for i, x in enumerate(lps):
        r = torch.load(os.path.join(ref_dir, f"r{i:03d}.pt"))
        if i >= len(want) or hashlib.sha256(_bytes(r)).hexdigest() != want[i]:
            bad.append(i)
        per.append(_score_rec(i, x, ws[i][P:P + C], r.argmax(-1).tolist(), r))
        del r
    del lps
    p108_box._release(device)
    print(f"P130_ARM {SUBJECT} done at {time.time() - start:.0f} s", flush=True)
    return {"phase": "on", "windows": len(ws), "prompt": P, "cont": C, "chunk": chunk,
            "ref_ok": not bad and len(want) == len(ws), "ref_bad": bad[:8], "per_window": {SUBJECT: per},
            "engagement": {SUBJECT: [e]}, "rehearsal": {"stand_in_attention": bool(stand_in)},
            "seconds": round(time.time() - start, 1)}


def windows_digest(ws) -> str:
    return hashlib.sha256(json.dumps(ws).encode()).hexdigest()


def corpus_commits(datasets_cache=None, hub_cache=None) -> dict:
    """The wikitext commit(s) this box actually read (Amendment 1): ``datasets`` keys its arrow cache of the config by the
    hub commit it resolved (``Salesforce___wikitext/wikitext-2-raw-v1/<version>/<commit>``), and huggingface_hub keys the
    downloaded files' snapshot by the same commit. Reported beside the fetch gate's ``corpus.main``, never gated."""
    import glob
    if datasets_cache is None:
        import datasets
        datasets_cache = datasets.config.HF_DATASETS_CACHE
    if hub_cache is None:
        from huggingface_hub import constants
        hub_cache = constants.HF_HUB_CACHE
    arrow = sorted({os.path.basename(p) for p in glob.glob(os.path.join(str(datasets_cache), "Salesforce___wikitext",
                                                                         "wikitext-2-raw-v1", "*", "*")) if os.path.isdir(p)})
    snaps = sorted(os.path.basename(p) for p in glob.glob(os.path.join(str(hub_cache), "datasets--Salesforce--wikitext",
                                                                        "snapshots", "*")) if os.path.isdir(p))
    return {"arrow": arrow, "snapshots": snaps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--proc", type=int, required=True)
    ap.add_argument("--p1", choices=("0", "1"), required=True)
    ap.add_argument("--phase-b", choices=("none", "off", "on"), default="none")
    ap.add_argument("--ref-dir", default="/root/p130/work/ref")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--speed-windows", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--eager-rounds", type=int, default=2)
    ap.add_argument("--eager-windows", type=int, default=4)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--floor-chunk", type=int, default=256)
    ap.add_argument("--stride", type=int, default=3072)
    a = ap.parse_args()
    import transformers

    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    p1_env = (os.environ.get(P1_ENV, "") or "").strip() or "0"
    p2_env = (os.environ.get(P2_ENV, "") or "").strip() or "0"
    if p1_env != a.p1 or p2_env != "0":
        raise SystemExit(f"REFUSED: process {a.proc} registers {P1_ENV}={a.p1} and {P2_ENV} unset, the environment has "
                         f"{P1_ENV}={p1_env} {P2_ENV}={p2_env}")
    if (a.phase_b == "off" and a.p1 != "0") or (a.phase_b == "on" and a.p1 != "1"):
        raise SystemExit(f"REFUSED: Phase B {a.phase_b} runs with P1 {'unset' if a.phase_b == 'off' else 'set'}")
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
    rec = {"proc": a.proc, "p1": int(a.p1), "phase_b": a.phase_b, "model": cfg.model, "revision": cfg.revision,
           "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "transformers": transformers.__version__, "load_s": round(load_s, 1), "stride": a.stride,
           "layers": int(cfgm.num_hidden_layers), "windows_digest": windows_digest(ws), "corpus": corpus_commits(),
           "census": {k: parts.info.get(k) for k in ("moe_layers", "experts", "top_k", "model_type", "int4_expert_layers",
                                                     "int4_attn_projections", "int4_store_kinds", "fuse_qkv_n",
                                                     "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n",
                                                     "fusion_report", "levers_env")}}
    st = time.time()
    rec["speed"] = speed(model, [w[:a.prompt] for w in ws[:a.speed_windows]], a.prompt, cfg.device, rounds=a.rounds,
                         warm=a.warm, eager_rounds=a.eager_rounds, eager_windows=a.eager_windows, bulk_kv=cfg.bulk_kv,
                         last_logits=cfg.last_logits)
    sp = rec["speed"]
    fn = sp.get("function_p2")
    print(f"P130_SPEED proc {a.proc} p1 {a.p1} | graphs {[(g, v.get('status')) for g, v in sp['graphs'].items()]} "
          f"| median ms {sp.get('median_ms')} | P2 differs on {len(fn['differ']) if fn else 'n/a'} windows "
          f"| eager ms {(sp.get('eager') or {}).get('median_ms')} | {time.time() - st:.0f} s", flush=True)
    try:                                       # a Phase B failure is recorded; the process's speed record stands
        if a.phase_b == "off":
            rec["quality"] = quality_off(model, ws, prompt=a.prompt, cont=a.cont, chunk=a.chunk,
                                         floor_chunk=a.floor_chunk, device=cfg.device, ref_dir=a.ref_dir)
        elif a.phase_b == "on":
            rec["quality"] = quality_on(model, ws, prompt=a.prompt, cont=a.cont, chunk=a.chunk, device=cfg.device,
                                        ref_dir=a.ref_dir)
    except Exception as e:  # noqa: BLE001 -- the reducer reads it as QUALITY_VOID
        rec["quality"] = {"phase": a.phase_b, "error": f"{type(e).__name__}: {str(e)[:400]}"}
        print(f"P130_QUALITY_ERROR {rec['quality']['error']}", flush=True)
    rec.update(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               max_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)
    with open(a.out, "w") as f:
        f.write(json.dumps(rec, indent=1))
    q = rec.get("quality")
    ql = ""
    if q:
        ql = f" | quality {q['phase']} " + (f"ERROR {q['error'][:120]}" if "error" in q else f"windows {q['windows']} {q['seconds']} s")
    print(f"P130_BOX proc {a.proc} p1 {a.p1} | load {load_s:.0f}s | mem {rec['max_mem_gb']} GB{ql}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
