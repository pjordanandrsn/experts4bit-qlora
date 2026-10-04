"""Host-sync census of serve_paged's prefill forward (e4b, for the E4B_PAGED_PREFILL_GRAPH knob's feasibility).

    python census_prefill_sync.py census OUT.json     # sync sites + runtime-API counts, chunk 1 (no history) and chunk 2
    python census_prefill_sync.py capture OUT.json    # one trial CUDA-graph capture of a 512-token prefill forward

The engine is serve_paged.build_engine under the environment's settings (the defaults unless the driver sets an
arm's env), unmodified. One harness shim: huggingface_hub.snapshot_download returns a local directory as itself, so
the int4 lever reads the local tiny checkpoint. Two instruments, which must agree:
  (1) torch.cuda.set_sync_debug_mode("warn"): every synchronizing op, with its Python stack (site = innermost frame
      outside torch);
  (2) torch.profiler's CUDA runtime events: cudaStreamSynchronize / cudaDeviceSynchronize / cudaMemcpy* counts.
Phases: "forward" is inside the model's __call__ (what a graphed knob captures); "around" is run_prefill's own
bookkeeping (ids/pos, the KV flush and append, the first token's argmax)."""
import collections, json, os, sys, time, traceback, warnings

import torch

MODE, OUT = sys.argv[1], sys.argv[2]
PHASE = ["around"]
SITES = collections.Counter()
STACKS = {}


def _shim_snapshot():
    import huggingface_hub
    orig = huggingface_hub.snapshot_download

    def sd(repo_id, *a, **k):
        return repo_id if os.path.isdir(repo_id) else orig(repo_id, *a, **k)
    huggingface_hub.snapshot_download = sd


def _site(stack):
    keep = [f for f in stack if "/torch/" not in f.filename and "census_prefill_sync" not in f.filename
            and "/warnings.py" not in f.filename]
    if not keep:
        return "?", []
    inner = keep[-1]
    short = lambda f: f"{f.filename.split('site-packages/')[-1].split('/root/')[-1]}:{f.lineno} {f.name}"
    return short(inner), [short(f) for f in keep[-6:]]


def _showwarning(message, category, filename, lineno, file=None, line=None):
    if "synchroniz" not in str(message):
        return
    raw = traceback.extract_stack()[:-1]
    key, chain = _site(raw)
    SITES[(PHASE[0], key)] += 1
    if key == "?":       # no frame outside torch: keep the whole stack and the message
        chain = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.name}" for f in raw[-12:]] + [str(message)[:300]]
    STACKS.setdefault((PHASE[0], key), chain)


def build():
    _shim_snapshot()
    from experts4bit_qlora import serve_paged
    cfg = serve_paged.PagedServeConfig.from_env()
    t0 = time.time()
    parts = serve_paged.build_engine(cfg)
    return serve_paged, parts, round(time.time() - t0, 1)


def phase_hooks(model):
    model.register_forward_pre_hook(lambda m, a: PHASE.__setitem__(0, "forward"))
    model.register_forward_hook(lambda m, a, o: PHASE.__setitem__(0, "around"))


def prompt(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (n,), generator=g).tolist()


def runtime_counts(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    c = collections.Counter()
    for e in prof.events():
        n = e.name
        if n.startswith(("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpy", "cudaEventSynchronize",
                         "cudaStreamWaitEvent", "cudaLaunchKernel", "cuLaunchKernel", "Memcpy")):
            c[n] += 1
        if n in ("aten::_local_scalar_dense", "aten::nonzero", "aten::item", "aten::repeat_interleave",
                 "aten::unique", "aten::_unique2", "aten::bincount", "aten::masked_select", "aten::index"):
            c[n] += 1
    return dict(sorted(c.items()))


def census():
    serve_paged, parts, build_s = build()
    runner = parts.runner
    rec = {"mode": "census", "build_s": build_s, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "prefill_routes": serve_paged.prefill_routes(), "info": parts.info,
           "env": {k: v for k, v in os.environ.items() if k.startswith("E4B_")}}
    phase_hooks(runner.model)
    # warm-up: JIT/autotune and the KV pool's first touch, on both chunk shapes
    for rid, slot in ((90, 2), (91, 3)):
        runner.bind(rid, slot, prompt(1024, rid))
        runner.run_prefill([(rid, 0, 512)])
        runner.run_prefill([(rid, 512, 512)])
    torch.cuda.synchronize()
    warnings.showwarning = _showwarning
    warnings.simplefilter("always")
    out = {}
    for label, rid, slot in (("c1_no_history", 1, 0), ("c2_with_history", 1, 0)):
        if label == "c1_no_history":
            runner.bind(rid, slot, prompt(1024, 7))
            chunk = [(rid, 0, 512)]
        else:
            chunk = [(rid, 512, 512)]
        SITES.clear(); STACKS.clear()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("warn")
        try:
            runner.run_prefill(chunk)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        sites = [{"phase": p, "site": s, "count": n, "chain": STACKS[(p, s)]} for (p, s), n in SITES.most_common()]
        out[label] = {"sites": sites,
                      "forward_syncs": sum(n for (p, _), n in SITES.items() if p == "forward"),
                      "around_syncs": sum(n for (p, _), n in SITES.items() if p == "around")}
    # instrument (2): the same two chunks again on a fresh slot, under the profiler
    runner.bind(2, 1, prompt(1024, 8))
    out["runtime_baseline_noop"] = runtime_counts(lambda: None)      # the profiler's own runtime calls
    out["runtime_c1"] = runtime_counts(lambda: runner.run_prefill([(2, 0, 512)]))
    out["runtime_c2"] = runtime_counts(lambda: runner.run_prefill([(2, 512, 512)]))
    rec.update(out)
    json.dump(rec, open(OUT, "w"), indent=1, default=str)
    for label in ("c1_no_history", "c2_with_history"):
        print(f"CENSUS {label}: forward {out[label]['forward_syncs']} around {out[label]['around_syncs']}", flush=True)
        for s in out[label]["sites"]:
            print(f"  [{s['phase']}] x{s['count']} {s['site']}", flush=True)
    for s in out["c1_no_history"]["sites"] + out["c2_with_history"]["sites"]:
        if s["site"] == "?":
            print("  ? chain: " + " | ".join(s["chain"]), flush=True)
    for k in ("runtime_baseline_noop", "runtime_c1", "runtime_c2"):
        print(f"CENSUS {k}: " + json.dumps(out[k]), flush=True)


def capture():
    serve_paged, parts, build_s = build()
    runner = parts.runner
    from experts4bit_qlora.engines.paged_attention import set_context
    rec = {"mode": "capture", "gpu": torch.cuda.get_device_name(0), "prefill_routes": serve_paged.prefill_routes()}
    runner.bind(1, 0, prompt(512, 7))
    runner.run_prefill([(1, 0, 256)])            # warm (JIT) on a different length, then a fresh slot
    runner.bind(2, 1, prompt(1024, 8))
    ids = torch.tensor(runner.tokens[2][:512], dtype=torch.long, device=runner.device)[None]
    pos = torch.arange(0, 512, device=runner.device)[None]
    runner._mode(True); runner.ctx.mode = "prefill"; runner.ctx.slots = [1]
    prev = set_context(runner.ctx)
    try:
        with torch.no_grad():
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                runner.model(input_ids=ids, position_ids=pos, use_cache=False)   # eager warm on the side stream
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g):
                    runner.model(input_ids=ids, position_ids=pos, use_cache=False)
                rec["captured"] = True
            except Exception as e:  # noqa: BLE001
                rec["captured"] = False
                rec["error"] = repr(e)[:1500]
                rec["where"] = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.name}"
                                for f in traceback.extract_tb(e.__traceback__)][-10:]
    finally:
        set_context(prev); runner.ctx.mode = "decode"; runner._mode(False)
    json.dump(rec, open(OUT, "w"), indent=1, default=str)
    print("CAPTURE " + json.dumps({k: rec.get(k) for k in ("captured", "error")}), flush=True)
    for w in rec.get("where", []):
        print("  at " + w, flush=True)


def replay():
    """Chunk 1: eager references, one capture, replays on new ids compared bit for bit (logits and the staged K/V),
    then a replay after an eager prefill of another length plus allocator churn (the #913 hazard class). Chunk 2: a
    capture over a fixed staged history, replays compared bit for bit."""
    serve_paged, parts, build_s = build()
    runner = parts.runner
    ctx = runner.ctx
    from experts4bit_qlora.engines.paged_attention import set_context
    dev = runner.device
    P = {n: torch.tensor(prompt(1024, n), dtype=torch.long, device=dev) for n in (11, 12, 13)}
    pos1 = torch.arange(0, 512, device=dev)[None]
    pos2 = torch.arange(512, 1024, device=dev)[None]
    rec = {"mode": "replay", "gpu": torch.cuda.get_device_name(0), "prefill_routes": serve_paged.prefill_routes()}

    def staged_last():
        return {f"{k[0]}:{k[1]}": (v[0][-1], v[1][-1]) for k, v in ctx.staging.items()}

    def cmp(logits, staged, ref):
        rl, rs = ref
        d = (logits.float() - rl.float()).abs().max().item()
        st = all(torch.equal(staged[k][0], rs[k][0]) and torch.equal(staged[k][1], rs[k][1]) for k in rs)
        return {"logits_bitwise": bool(torch.equal(logits, rl)), "logits_max_abs": d, "staged_bitwise": bool(st),
                "staged_keys": len(rs)}

    def snap(staging):
        return {k: (list(v[0]), list(v[1])) for k, v in staging.items()}

    def restore(s):
        ctx.staging.clear()
        ctx.staging.update({k: (list(v[0]), list(v[1])) for k, v in s.items()})

    runner._mode(True); ctx.mode = "prefill"
    prev = set_context(ctx)
    try:
        with torch.no_grad():
            # ---- chunk 1 (no history), slot 0
            ctx.slots = [0]
            refs = {}
            for n in (11, 12, 13):
                ctx.staging.clear()
                out = runner.model(input_ids=P[n][:512][None], position_ids=pos1, use_cache=False)
                refs[n] = (out.logits.clone(), {k: (a.clone(), b.clone()) for k, (a, b) in staged_last().items()})
            ids_s = P[11][:512][None].clone()
            ctx.staging.clear()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                runner.model(input_ids=ids_s, position_ids=pos1, use_cache=False)
            torch.cuda.current_stream().wait_stream(s)
            ctx.staging.clear()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out_s = runner.model(input_ids=ids_s, position_ids=pos1, use_cache=False)
            st_s = staged_last()
            ctx.staging.clear()
            r1 = {}
            for n in (12, 11, 13):
                ids_s.copy_(P[n][:512][None]); g.replay(); torch.cuda.synchronize()
                r1[str(n)] = cmp(out_s.logits, st_s, refs[n])
            # hazard: an eager prefill of another length on another slot, then allocator churn, then replay
            ctx.slots = [2]; ctx.staging.clear()
            runner.model(input_ids=P[13][:300][None], position_ids=torch.arange(0, 300, device=dev)[None], use_cache=False)
            ctx.staging.clear(); ctx.slots = [0]
            junk = [torch.full((1 << 18,), 7.0, device=dev) for _ in range(256)]
            del junk
            ids_s.copy_(P[12][:512][None]); g.replay(); torch.cuda.synchronize()
            r1["12_after_other_length_eager_and_churn"] = cmp(out_s.logits, st_s, refs[12])
            rec["c1"] = r1
            # ---- chunk 2 over a fixed staged history (prompt 11's chunk 1), slot 1
            ctx.slots = [1]; ctx.staging.clear()
            runner.model(input_ids=P[11][:512][None], position_ids=pos1, use_cache=False)
            hist = snap(ctx.staging)
            refs2 = {}
            for n in (11, 12):
                restore(hist)
                out = runner.model(input_ids=P[n][512:][None], position_ids=pos2, use_cache=False)
                refs2[n] = (out.logits.clone(), {k: (a.clone(), b.clone()) for k, (a, b) in staged_last().items()})
            ids2 = P[11][512:][None].clone()
            restore(hist)
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                runner.model(input_ids=ids2, position_ids=pos2, use_cache=False)
            torch.cuda.current_stream().wait_stream(s)
            restore(hist)
            g2 = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g2):
                    out2 = runner.model(input_ids=ids2, position_ids=pos2, use_cache=False)
                st2 = staged_last()
                r2 = {"captured": True}
                for n in (12, 11):
                    ids2.copy_(P[n][512:][None]); g2.replay(); torch.cuda.synchronize()
                    r2[str(n)] = cmp(out2.logits, st2, refs2[n])
            except Exception as e:  # noqa: BLE001
                r2 = {"captured": False, "error": repr(e)[:1200]}
            rec["c2"] = r2
    finally:
        set_context(prev); ctx.mode = "decode"; runner._mode(False)
    json.dump(rec, open(OUT, "w"), indent=1, default=str)
    print("REPLAY " + json.dumps({"c1": rec["c1"], "c2": rec["c2"]}), flush=True)


if __name__ == "__main__":
    {"census": census, "capture": capture, "replay": replay}[MODE]()
