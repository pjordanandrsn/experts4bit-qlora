#!/usr/bin/env python3
"""p149_box.py -- lane P149 box Q+S (bench/p149/PREREG-p149.md): ONE process per arm or phase, on Mixtral-8x7B's server.

The engine is the shipped server (``PagedServeConfig.from_env()`` + ``build_engine(cfg)``) at 16 slots of 768 tokens,
graphs on, every fusion knob at its default. The arms differ ONLY in ``E4B_SERVE_ATTN_INT8``:
  S0  unset (the default server);
  S1  ``1`` (engines/int8_attn.py: the attention projections on grouped-nf4-gemm K37's int8 kernels).

``--mode speed`` (term 1, P143 Amendment 3's form verbatim except the arms): P109's W16 and W1 workloads at P109's
registered bytes, SHORT 32 and LONG 160, one untimed warm pass and five timed reps each, the tokens recorded and every
rep digested. ``--q`` (S0a and S1a only) then runs Q in the same process, after every timed rep.

Q (term 2): P147's W1 and W16 windows (sha256-checked against P147's records), 512 prompt + 128 scored tokens each,
through the served decode step with graphs on. A window's first 511 tokens are prefilled; its tokens 511..638 are fed
TEACHER-FORCED at T == 1 (each step's appended argmax overwritten with the window's next token before the next step),
so the step at input position p predicts position p + 1 and every one of P147's 128 scored positions (512..639) is read
at decode. The decode graphs are re-captured for Q with the step also copying its last-position logits into a static
buffer (the timed reps ran before, on the build's own graphs). W1: one window per pass (bucket 1, the M == 1 GEMV);
W16: sixteen per pass (bucket 16, the M == 16 small-M kernel; the last pass's 11 windows pad to 16).

ENGAGEMENT (asserted by the reducer from what this records): per captured bucket, the int8 projections' calls during
that bucket's warm-up and capture, by row count and path; per Q gate, the graph replays and eager steps of its bucket
and the path counts over the gate (prefill rows only on the transient path); S0 carries no int8 module at all.

    python p149_box.py --mode speed --prompts prompts.json --out arm_S0a.json --tag S0a [--q --windows windows.json]
    python p149_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p149_box.py --windows-only --model ID --revision SHA --out windows.json
    python p149_box.py --self-test
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARMS = ("S0", "S1")
KNOB = "E4B_SERVE_ATTN_INT8"
ARM_VALUE = {"S0": "", "S1": "1"}                  # S0: unset, the default server
DEFAULT_KNOBS = ("E4B_SERVE_ATTN_INT4", "E4B_SERVE_ATTN_INT4_CALIB", "E4B_DENSE_GEMV_BW", "E4B_INT8_HEAD_BW",
                 "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI",
                 "GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "E4B_PAGED_DECODE_LOOKAHEAD")
SERVER = {"max_seqs": 16, "max_tokens_per_seq": 768, "buckets": (1, 2, 4, 8, 16), "placement": "all-vram"}
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
PROMPT, CONT = 512, 128
#: P147's gates (bench/p147/receipts/p147-a2000-1/run.json): windows, pass width, and the windows' sha256
Q_SETS = {"W1": (196, 1, "ae47744e4b30fd6ae427d51aab33f596e46336d257402a6e18e294b5f4a96ffe"),
          "W16": (123, 16, "4ae6aa040b7a92c54536c7a53359bdbd0983e4c47143f0eadae3145307177d2d")}
Q_BUCKET = {"W1": 1, "W16": 16}


def arm_env_ok(arm: str, env):
    """``(ok, why)``: the knob carries the arm's value; every other serving knob is unset (its default)."""
    if arm not in ARMS:
        return False, f"unknown arm {arm!r}"
    bad = {}
    if (env.get(KNOB) or "").strip() != ARM_VALUE[arm]:
        bad[KNOB] = env.get(KNOB)
    bad.update({k: env.get(k) for k in DEFAULT_KNOBS if (env.get(k) or "").strip()})
    return (not bad), f"{arm} with {bad}"


def _arm():
    arm = os.environ.get("P149_ARM", "")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: P149_ARM={arm!r}: {why}")
    return arm


def server_refusal(cfg):
    """None for the registered server, else the reason (16 slots x 768 tokens, buckets 1-16, all-vram, graphs on)."""
    got = {"max_seqs": cfg.max_seqs, "max_tokens_per_seq": cfg.max_tokens_per_seq, "buckets": tuple(cfg.buckets),
           "placement": cfg.placement}
    if not cfg.graphs or got != SERVER:
        return f"not the registered server: graphs={cfg.graphs} {got} (want {SERVER})"
    return None


def windows_sha(ws) -> str:
    """P147's window hash (``json.dumps`` of the token lists, default separators)."""
    return hashlib.sha256(json.dumps(ws).encode()).hexdigest()


def window_refusal(name, ws):
    n, _group, sha = Q_SETS[name]
    if len(ws) != n or any(len(w) != PROMPT + CONT for w in ws):
        return f"{name}: {len(ws)} windows, want {n} of {PROMPT + CONT} tokens"
    if windows_sha(ws) != sha:
        return f"{name}: sha256 {windows_sha(ws)} is not P147's {sha}"
    return None


def passes(n, group):
    """``[(start, stop)]`` in passes of ``group`` windows."""
    return [(i, min(i + group, n)) for i in range(0, n, group)]


def heat():
    """The GPU's temperature (C) and SM clock (MHz), read outside every timed window; reported, never ruled."""
    import subprocess
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                                      text=True, timeout=20).strip().splitlines()[0]
        t, c = (int(float(x)) for x in out.split(","))
        return {"temp_c": t, "sm_mhz": c, "t": round(time.time(), 3)}
    except Exception as e:  # noqa: BLE001 -- reported only
        return {"error": f"{type(e).__name__}: {str(e)[:120]}", "t": round(time.time(), 3)}


def nll_argmax(logits, targets):
    """Per row: the true token's NLL (fp32 log-softmax) and the argmax, for one decode step's ``[n, vocab]`` logits."""
    import torch
    lp = torch.log_softmax(logits.float(), dim=-1)
    t = torch.as_tensor(targets, device=logits.device, dtype=torch.long)
    return (-lp.gather(-1, t[:, None]).squeeze(-1)).tolist(), logits.argmax(-1).tolist()


# ---------------------------------------------------------------------------------------------- the int8 records --
def _int8_mods(model):
    try:
        from experts4bit_qlora.engines.int8_attn import Int8Linear
    except ImportError:
        return []
    return [m for m in model.modules() if isinstance(m, Int8Linear)]


def path_counts(mods) -> dict:
    """``{"routed": {rows: calls}, "passed": {rows: calls}}`` summed over the int8 projections (calls, not modules)."""
    out = {"routed": {}, "passed": {}}
    for m in mods:
        for path in ("routed", "passed"):
            for r, c in getattr(m, path).items():
                out[path][str(r)] = out[path].get(str(r), 0) + c
    return out


def counts_delta(after, before) -> dict:
    return {p: {r: c - before[p].get(r, 0) for r, c in after[p].items() if c - before[p].get(r, 0)} for p in after}


class CaptureCensus:
    """Per captured bucket, the int8 projections' calls during its warm-up and capture (Python runs there only). Wraps
    ``PagedModelRunner._capture_bucket`` on the class, before ``build_engine`` captures."""

    def __init__(self):
        self.by_bucket = {}

    def __enter__(self):
        from experts4bit_qlora.engines import paged_runner as pr
        self.cls, self.orig = pr.PagedModelRunner, pr.PagedModelRunner._capture_bucket
        outer = self

        def cap(runner, b, warmup):
            mods = _int8_mods(runner.model)
            before = path_counts(mods)
            try:
                return outer.orig(runner, b, warmup)
            finally:
                outer.by_bucket[str(b)] = counts_delta(path_counts(mods), before)
        self.cls._capture_bucket = cap
        return self

    def __exit__(self, *exc):
        self.cls._capture_bucket = self.orig
        return False


def _common(cfg, parts, arm, load_s, census):
    import torch
    info = parts.info
    return {"arm": arm, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
            "model": cfg.model, "revision": cfg.revision,
            "knobs": {k: os.environ.get(k) for k in (KNOB,) + DEFAULT_KNOBS},
            "torch": torch.__version__, "load_s": round(load_s, 2),
            "fusions": {k: info.get(k) for k in CENSUS_KEYS}, "fusion_modes": info.get("fusion_modes"),
            "model_type": info.get("model_type"), "attn_int8": info.get("attn_int8"),
            "attn_int8_projections": info.get("attn_int8_projections"), "levers_env": info.get("levers_env"),
            "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
            "capture_paths": census.by_bucket, "max_seqs": cfg.max_seqs, "max_tokens_per_seq": cfg.max_tokens_per_seq,
            "buckets": list(cfg.buckets), "int8_modules": len(_int8_mods(parts.runner.model))}


# ------------------------------------------------------------------------------------------------------------ Q --
def _logits_graphs(runner, buckets):
    """Re-capture the decode buckets with the step also copying its last-position logits into ``buf["logits"]``
    (allocated at the first warm-up, outside the capture, so the graph writes a static buffer)."""
    def step(b):
        buf = runner._bufs[b]
        out = runner.model(input_ids=buf["ids"], position_ids=buf["pos"], use_cache=False)
        last = out.logits[:, -1]
        if "logits" not in buf:
            buf["logits"] = last.float().clone()
        buf["logits"].copy_(last.float())
        buf["tok"].copy_(last.argmax(-1))
    runner.disable_decode_graphs()
    runner._padded_step = step
    return runner.enable_decode_graphs(buckets)


class PrefillLogits:
    """Inside the block, every prefill forward's logits are kept (the server's own prefill, the M > 16 path)."""

    def __init__(self, runner):
        self.runner, self.kept = runner, []

    def __enter__(self):
        self.orig = self.runner._prefill_forward

        def fwd(ids, positions):
            out = self.orig(ids, positions)
            self.kept.append(out.logits)
            return out
        self.runner._prefill_forward = fwd
        return self

    def __exit__(self, *exc):
        self.runner._prefill_forward = self.orig
        return False


def prefill_q(runner, ws, base_rid):
    """REPORTED: the same 128 positions teacher-forced through ONE prefill of the window's first 639 tokens."""
    out = []
    for i, w in enumerate(ws):
        rid = base_rid + i
        runner.bind(rid, 0, w[:-1])
        with PrefillLogits(runner) as pl:
            runner.run_prefill([(rid, 0, len(w) - 1)])
        n_, a_ = nll_argmax(pl.kept[-1][0, PROMPT - 1:], w[PROMPT:])   # positions 511..638 predict 512..639
        runner.free_slot(rid)
        out.append({"nll": sum(n_) / CONT, "argmax": a_})
    return out


def q_pass(runner, ws, base_rid, ttft=None):
    """One teacher-forced pass of up to 16 windows: per window, the 128 positions' NLL and argmax, read at decode.
    ``ttft`` (a list) receives the prefill's wall in ms, synchronised (the prefill reads its argmax back)."""
    from experts4bit_qlora.engines.paged_runner import bucket_for
    rids = [base_rid + k for k in range(len(ws))]
    for k, (rid, w) in enumerate(zip(rids, ws)):
        runner.bind(rid, k, w[:PROMPT - 1])
    t0 = time.perf_counter()
    runner.run_prefill([(rid, 0, PROMPT - 1) for rid in rids])
    if ttft is not None:
        ttft.append(round(1000 * (time.perf_counter() - t0), 3))
    for rid, w in zip(rids, ws):
        runner.tokens[rid][-1] = w[PROMPT - 1]                 # forced: position 511's true token, not the argmax
    nll = [[] for _ in ws]
    am = [[] for _ in ws]
    buf = runner._bufs[bucket_for(len(rids), runner._buckets)]
    for j in range(CONT):                                       # input position 511 + j predicts 512 + j
        runner.run_decode(rids)
        n_, a_ = nll_argmax(buf["logits"][:len(rids)], [w[PROMPT + j] for w in ws])
        for k, rid in enumerate(rids):
            nll[k].append(n_[k])
            am[k].append(a_[k])
            runner.tokens[rid][-1] = ws[k][PROMPT + j]          # forced: the next input is the true token
    for rid in rids:
        runner.free_slot(rid)
    return [{"nll": sum(x) / CONT, "argmax": y} for x, y in zip(nll, am)]


def q_main(parts, windows) -> dict:
    """Q on the built engine, after the timed reps: both gates, with their engagement."""
    import torch
    runner = parts.runner
    mods = _int8_mods(runner.model)
    rec = {"recapture": None, "gates": {}}
    with CaptureCensus() as census:
        before = path_counts(mods)
        rec["recapture"] = {"graph_status": {str(k): v for k, v in _logits_graphs(runner, runner._buckets).items()}}
        rec["recapture"]["capture_paths"] = census.by_bucket
        rec["recapture"]["paths"] = counts_delta(path_counts(mods), before)
    for name in ("W1", "W16"):
        ws = windows[name]
        why = window_refusal(name, ws)
        if why:
            raise SystemExit(f"REFUSED: {why}")
        n, group, sha = Q_SETS[name]
        b = Q_BUCKET[name]
        g0 = dict(runner.graph_stats[b])
        c0 = path_counts(mods)
        t0 = time.perf_counter()
        per, ttft = [], ([] if group == 1 else None)
        for i, (s, e) in enumerate(passes(n, group)):
            per += q_pass(runner, ws[s:e], 10_000 + 100 * i, ttft)
        torch.cuda.synchronize()
        g1 = runner.graph_stats[b]
        rec["gates"][name] = {"windows": n, "group": group, "sha256": sha, "bucket": b,
                              "seconds": round(time.perf_counter() - t0, 2),
                              "replays": g1["replays"] - g0["replays"], "eager_steps": g1["eager_steps"] - g0["eager_steps"],
                              "paths": counts_delta(path_counts(mods), c0), "per_window": per}
        if ttft is not None:
            rec["prefill_ttft_ms_511"] = {"n": len(ttft), "median": sorted(ttft)[len(ttft) // 2], "all": ttft}
        print(f"P149_Q {name} windows={n} replays={rec['gates'][name]['replays']} "
              f"eager={rec['gates'][name]['eager_steps']} {rec['gates'][name]['seconds']} s", flush=True)
    t0 = time.perf_counter()
    c0 = path_counts(mods)
    rec["prefill_q"] = {name: prefill_q(runner, windows[name], 50_000 + 1000 * j) for j, name in enumerate(Q_SETS)}
    rec["prefill_q_paths"] = counts_delta(path_counts(mods), c0)
    rec["prefill_q_seconds"] = round(time.perf_counter() - t0, 2)
    return rec


# -------------------------------------------------------------------------------------------------------- speed --
def speed_main(a) -> int:
    import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)
    arm = _arm()
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != p109_box.ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    windows = None
    if a.q:
        if arm and a.tag not in ("S0a", "S1a"):
            raise SystemExit(f"REFUSED: Q runs in S0a and S1a only, not {a.tag}")
        windows = json.load(open(a.windows))
        for name in Q_SETS:
            why = window_refusal(name, windows[name])
            if why:
                raise SystemExit(f"REFUSED: {why}")
    import torch
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    why = server_refusal(cfg)
    if why:
        raise SystemExit(f"REFUSED: {why}")
    t0 = time.perf_counter()
    with CaptureCensus() as census:
        parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    rec = {"mode": "speed", "tag": a.tag, **_common(cfg, parts, arm, load_s, census),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": p109_box._mem(torch), "workloads": {}, "phases": {}, "status": "ok"}
    rec["phases"]["load"] = [round(t0, 3), round(time.perf_counter(), 3)]
    if (arm == "S1") != bool(rec["int8_modules"]):
        raise SystemExit(f"REFUSED: arm {arm} built {rec['int8_modules']} int8 projections")
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last = {}, {}, {}
        tw = time.time()
        heats = {}
        for n in (a.short, a.long):
            p109_box.run_pass(parts, torch, wrows, n)                      # warm, untimed
            ws, ds = [], []
            heats[str(n)] = {"start": heat()}
            for _ in range(a.reps):
                wall, _steps, toks = p109_box.run_pass(parts, torch, wrows, n)
                ws.append(round(wall, 5))
                ds.append(p109_box.digest(toks))
                last[str(n)] = toks
            heats[str(n)]["end"] = heat()
            walls[str(n)], digests[str(n)] = ws, ds
        s = p109_box.slope(walls[str(a.short)], walls[str(a.long)], b, a.short, a.long)
        rec["workloads"][wname] = {"batch": b, "walls": walls, "rep_digests": digests, "tokens": last, "heat": heats,
                                   **s}
        rec["phases"][wname] = [round(tw, 3), round(time.time(), 3)]
        print(f"P149_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} walls={walls}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["paths_after_reps"] = path_counts(_int8_mods(parts.runner.model))
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)                  # the speed record, before Q
    print("P149_ARM " + json.dumps({"arm": arm, "tag": a.tag, "load_s": rec["load_s"], "int8": rec["int8_modules"],
                                    "W16": rec["workloads"]["W16"]["decode_tok_s"],
                                    "W1": rec["workloads"]["W1"]["decode_tok_s"]}, default=str), flush=True)
    if a.q:
        tq = time.time()
        rec["q"] = q_main(parts, windows)
        rec["phases"]["q"] = [round(tq, 3), round(time.time(), 3)]
        rec["q"]["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        json.dump(rec, open(a.out, "w"), indent=1, default=str)
    return 0


# ------------------------------------------------------------------------------------- Amendment 4: T1 and T2 --
T_ORDER = {"T1": "ABBA", "T2": "BAAB"}


class Forcer:
    """Teacher forcing on the served step: each token the runner hands back (the prefill's first, every decode step's)
    is replaced by the stream's, so the runner's own ``copy_`` feeds it into the captured graph's static input buffer at
    the next step. Per step it checks, after the sync the step already makes: the bucket's input buffer is the one
    captured (pointer) and holds the forced tokens, and counts where the step's own argmax differs from the forced."""

    def __init__(self, runner):
        from experts4bit_qlora.engines.paged_runner import bucket_for
        self.r, self.bucket_for = runner, bucket_for
        self.ptr = {b: buf["ids"].data_ptr() for b, buf in runner._bufs.items()}       # taken after capture
        self.orig_pf, self.orig_dec = runner.run_prefill, runner.run_decode
        runner.run_prefill, runner.run_decode = self.prefill, self.decode
        self.arm({})

    def arm(self, streams):
        self.streams = streams
        self.fed = {rid: [] for rid in streams}
        self.counts = {"buf_mismatch": 0, "argmax_mismatch": 0, "steps": 0, "ptr_mismatch": 0}

    def _force(self, got):
        for rid in got:
            s = self.streams.get(rid)
            if s is None:
                continue
            want = s[len(self.fed[rid])]
            self.counts["argmax_mismatch"] += int(got[rid] != want)
            got[rid] = want
            self.r.tokens[rid][-1] = want
            self.fed[rid].append(want)
        return got

    def prefill(self, chunks):
        return self._force(self.orig_pf(chunks))

    def decode(self, rids):
        b = self.bucket_for(len(rids), self.r._buckets)
        buf = self.r._bufs[b]
        want = [self.r.tokens[rid][-1] for rid in rids]
        got = self.orig_dec(rids)
        if buf["ids"].data_ptr() != self.ptr[b]:
            self.counts["ptr_mismatch"] += 1
        seen = buf["ids"][:len(rids), 0].tolist()                       # the graph's own input this step
        self.counts["buf_mismatch"] += sum(int(x != y) for x, y in zip(seen, want))
        self.counts["steps"] += 1
        return self._force(got)

    def restore(self):
        self.r.run_prefill, self.r.run_decode = self.orig_pf, self.orig_dec


def tf_pass(parts, torch, rows, stream, n, forcer):
    """``p109_box.run_pass`` with one line more: each request's forced stream is armed before the scheduler runs."""
    sched = parts.scheduler
    before = len(sched.done)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rids = [sched.add_request(list(r), max_new_tokens=n) for r in rows]
    forcer.arm({rid: s[:n] for rid, s in zip(rids, stream)})
    steps = sched.run_until_idle()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    new = {r.rid: r for r in sched.done[before:]}
    if sorted(new) != sorted(rids) or sched.active or sched.queue:
        raise AssertionError(f"scheduler did not drain: done {sorted(new)} vs added {sorted(rids)}")
    toks = [[int(t) for t in new[r].out] for r in rids]
    fed_ok = all(t == s[:n] for t, s in zip(toks, stream)) and all(len(forcer.fed[r]) == n for r in rids)
    return wall, steps, toks, fed_ok


def route_stats(picks) -> dict:
    """Per layer-step routing over the rows (``picks``: layer -> step -> [(e1, e2) per row]): the distinct experts, the
    per-expert row counts, the 16-row tile count and the largest count, averaged over layer-steps."""
    dist, tiles, mx, hist = [], [], [], {}
    for steps in picks.values():
        for rows in steps:
            cnt = {}
            for pr in rows:
                for e in pr:
                    cnt[e] = cnt.get(e, 0) + 1
            dist.append(len(cnt))
            tiles.append(sum(-(-c // 16) for c in cnt.values()))
            mx.append(max(cnt.values()))
            for c in cnt.values():
                hist[str(c)] = hist.get(str(c), 0) + 1
    n = len(dist) or 1
    return {"layer_steps": len(dist), "distinct_mean": round(sum(dist) / n, 4), "tiles_mean": round(sum(tiles) / n, 4),
            "max_rows_mean": round(sum(mx) / n, 4), "rows_per_expert_histogram": dict(sorted(hist.items(), key=lambda x: int(x[0])))}


def route_replay(parts, rows, streams, long_):
    """EAGER: each row's prompt and its first ``long_ - 1`` stream tokens (the LONG decode steps' inputs) prefilled
    through S0's model; each sparse-MoE block's input routed by the block's own gate (the top ``num_experts_per_tok`` of
    the gate's logits, the experts its softmax ranks first). A pre-hook runs in ``__call__``, whatever a fold patched."""
    import torch
    runner = parts.runner
    mods = [m for m in runner.model.modules() if type(m).__name__.endswith("SparseMoeBlock")]
    cfg = runner.model.config
    if len(mods) != cfg.num_hidden_layers or any(getattr(getattr(m, "gate", None), "weight", None) is None for m in mods):
        raise SystemExit(f"REFUSED: {len(mods)} sparse-MoE blocks with a gate weight, want {cfg.num_hidden_layers}")
    k_top = int(cfg.num_experts_per_tok)
    out = {}
    for name, stream in streams.items():
        got = {i: [] for i in range(len(mods))}
        hooks = [m.register_forward_pre_hook(lambda mod, args, kw=None, _i=i: got[_i].append(args[0].detach()))
                 for i, m in enumerate(mods)]
        picks = {i: [[] for _ in range(long_ - 1)] for i in range(len(mods))}
        try:
            for k, (p, s) in enumerate(zip(rows, stream)):
                for i in got:
                    got[i].clear()
                rid = 90_000 + k
                runner.bind(rid, 0, list(p) + list(s[:long_ - 1]))
                runner.run_prefill([(rid, 0, len(p) + long_ - 1)])
                runner.free_slot(rid)
                for i, m in enumerate(mods):
                    h = got[i][-1].reshape(-1, got[i][-1].shape[-1])[len(p):len(p) + long_ - 1]
                    top = (h.float() @ m.gate.weight.float().T).topk(k_top, dim=-1).indices.tolist()
                    for step, pr in enumerate(top):
                        picks[i][step].append(tuple(sorted(pr)))
        finally:
            for h_ in hooks:
                h_.remove()
        out[name] = route_stats(picks)
    torch.cuda.synchronize()
    return out


def tf_main(a) -> int:
    import p109_box
    arm = _arm()
    if arm != "S0" or a.tag not in T_ORDER:
        raise SystemExit(f"REFUSED: a token-pinned process is S0 at T1 or T2, not {arm} at {a.tag}")
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != p109_box.ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    src = {"A": json.load(open(a.stream_a)), "B": json.load(open(a.stream_b))}
    for s, want in (("A", "S0a"), ("B", "S1a")):
        if src[s].get("tag") != want or src[s].get("prompts_sha256") != pf["prompts_sha256"]:
            raise SystemExit(f"REFUSED: stream {s} is {src[s].get('tag')} on {src[s].get('prompts_sha256')}, not {want}")
    streams = {s: r["workloads"]["W16"]["tokens"][str(a.long)] for s, r in src.items()}
    import torch
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    why = server_refusal(cfg)
    if why:
        raise SystemExit(f"REFUSED: {why}")
    t0 = time.perf_counter()
    with CaptureCensus() as census:
        parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    rec = {"mode": "tf", "tag": a.tag, "order": T_ORDER[a.tag], **_common(cfg, parts, arm, load_s, census),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "streams_identical": streams["A"] == streams["B"], "blocks": [], "phases": {"load": [round(t0, 3)]},
           "status": "ok"}
    forcer = Forcer(parts.runner)
    for s in T_ORDER[a.tag]:
        tb = time.time()
        walls, digests, block = {}, {}, {"stream": s, "reps": a.reps, "heat": {}, "fed_ok": True}
        counts = {"buf_mismatch": 0, "argmax_mismatch": 0, "steps": 0, "ptr_mismatch": 0}
        for n in (a.short, a.long):
            tf_pass(parts, torch, rows, streams[s], n, forcer)                      # warm, untimed
            block["heat"][str(n)] = {"start": heat()}
            ws, ds = [], []
            for _ in range(a.reps):
                wall, _steps, toks, ok = tf_pass(parts, torch, rows, streams[s], n, forcer)
                block["fed_ok"] &= ok
                for k in counts:
                    counts[k] += forcer.counts[k]
                ws.append(round(wall, 5))
                ds.append(p109_box.digest(toks))
            block["heat"][str(n)]["end"] = heat()
            walls[str(n)], digests[str(n)] = ws, ds
        block.update(p109_box.slope(walls[str(a.short)], walls[str(a.long)], len(rows), a.short, a.long))
        block.update(walls=walls, rep_digests=digests, counts=counts, ptr_ok=counts["ptr_mismatch"] == 0,
                     phase=[round(tb, 3), round(time.time(), 3)])
        rec["blocks"].append(block)
        print(f"P149_TF {a.tag} {s} decode_tok_s={block['decode_tok_s']} counts={counts}", flush=True)
    forcer.restore()
    if a.tag == "T1":
        tr = time.time()
        rec["routing"] = route_replay(parts, rows, streams, a.long)
        rec["phases"]["replay"] = [round(tr, 3), round(time.time(), 3)]
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    return 0


# ------------------------------------------------------------------------------------------------------ staging --
def windows_main(a) -> int:
    """P147's W1 and W16 windows, built by the staged loader at its pinned bytes and checked against P147's sha256."""
    import p125_box
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    out = {}
    for name, (n, _group, _sha) in Q_SETS.items():
        ws = p125_box.wikitext_windows(tok, n, PROMPT, CONT, stride=1024)
        why = window_refusal(name, ws)
        if why:
            raise SystemExit(f"REFUSED: {why}")
        out[name] = ws
    json.dump(out, open(a.out, "w"))
    print("P149_WINDOWS " + json.dumps({k: {"windows": len(v), "sha256": windows_sha(v)} for k, v in out.items()}))
    return 0


SANE_PHASE_ARMS = {"S0": ("R",), "S1": ("ON",)}


def sane_main(a) -> int:
    """REPORTED: P143's SANE (Phase B's instrument, ``p115_quality.measure_phase``) on the server built eager, one window
    per pass (T == 1), 12 wikitext windows x 128 positions; S0 scores R and saves its log-probs, S1 scores ON."""
    arm = _arm()
    import torch

    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram" or cfg.max_tokens_per_seq != SERVER["max_tokens_per_seq"]:
        raise SystemExit(f"REFUSED: SANE builds the server eager at all-vram and {SERVER['max_tokens_per_seq']} tokens "
                         f"(graphs={cfg.graphs}, max_tokens_per_seq={cfg.max_tokens_per_seq})")
    counters = q.KernelCounters().install()                          # before build_engine, as Phase B
    t0 = time.time()
    with CaptureCensus() as census:
        parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    fwd = q.ForwardCounter(model)
    windows = {"wikitext": q.LOADERS["wikitext"](parts.tokenizer, a.sane_windows, a.prompt, a.cont)}
    mods = _int8_mods(model)
    c0 = path_counts(mods)
    rec = q.measure_phase(model, windows, phase="off" if arm == "S0" else "on", prompt=a.prompt, cont=a.cont,
                          chunk=a.chunk, floor_chunk=a.chunk, group=1, device=cfg.device, ref_dir=a.ref_dir,
                          counters=counters, fwd=fwd, arms=SANE_PHASE_ARMS[arm])
    rec = {"mode": "sane", **_common(cfg, parts, arm, load_s, census), "paths": counts_delta(path_counts(mods), c0),
           **rec, "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "status": "ok"}
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"P149_SANE {arm} windows={rec.get('windows')} paths={rec['paths']} {rec.get('seconds')} s", flush=True)
    return 0


def prompts_main(a) -> int:
    import p109_box
    return p109_box.prompts_main(a)


# -------------------------------------------------------------------------------------------------------- tests --
def _forcer_cases(torch) -> list:
    """The forcing on a stand-in runner: decode writes the bucket's input buffer from the runner's tokens (as
    ``_run_decode_bucketed`` copies them in) and always returns argmax 7."""
    try:
        import experts4bit_qlora.engines.paged_runner  # noqa: F401  (Forcer reads its bucket_for)
    except ImportError:
        return []

    class R:
        def __init__(self):
            self._buckets = (1, 2)
            self._bufs = {b: {"ids": torch.zeros(b, 1, dtype=torch.long)} for b in self._buckets}
            self.tokens = {1: [5], 2: [5]}

        def run_prefill(self, chunks):
            return {rid: 7 for rid, _s, _t in chunks}

        def run_decode(self, rids):
            self._bufs[len(rids)]["ids"][:len(rids), 0] = torch.tensor([self.tokens[r][-1] for r in rids])
            return {r: 7 for r in rids}
    out = []
    r = R()
    f = Forcer(r)
    f.arm({1: [3, 4], 2: [7, 9]})
    first = r.run_prefill([(1, 0, 1), (2, 0, 1)])
    got = r.run_decode([1, 2])
    out.append(("the forced tokens are what the scheduler sees", first == {1: 3, 2: 7} and got == {1: 4, 2: 9}))
    out.append(("the graph's buffer held the forced inputs; three argmaxes differed",
                f.counts == {"buf_mismatch": 0, "argmax_mismatch": 3, "steps": 1, "ptr_mismatch": 0}
                and f.fed == {1: [3, 4], 2: [7, 9]}))
    r2 = R()
    f2 = Forcer(r2)
    f2.arm({1: [3, 4], 2: [7, 9]})
    r2.run_prefill([(1, 0, 1), (2, 0, 1)])
    r2._bufs[2]["ids"] = torch.zeros(2, 1, dtype=torch.long)          # a rebound buffer: the graph would not see it
    r2.run_decode([1, 2])
    out.append(("a rebound input buffer is caught by its pointer", f2.counts["ptr_mismatch"] == 1))
    f2.restore()
    out.append(("restore puts the runner's own methods back", r2.run_decode([1, 2]) == {1: 7, 2: 7}))
    return out


def self_test() -> int:
    cases = []
    s1 = {KNOB: "1"}
    cases.append(("S1 is the knob alone", arm_env_ok("S1", s1)[0]))
    cases.append(("S0 is the default server", arm_env_ok("S0", {})[0] and not arm_env_ok("S0", s1)[0]))
    cases.append(("another serving knob refuses", not arm_env_ok("S1", {**s1, "E4B_DENSE_GEMV_BW": "1"})[0]
                  and not arm_env_ok("S0", {"E4B_FUSE_T1_GLUE_R2": "0"})[0]))
    cases.append(("the lookahead stays at its default", not arm_env_ok("S0", {"E4B_PAGED_DECODE_LOOKAHEAD": "1"})[0]))
    cases.append(("an unknown arm refuses", not arm_env_ok("S2", s1)[0]))

    class C:
        graphs, max_seqs, max_tokens_per_seq, buckets, placement = True, 16, 768, (1, 2, 4, 8, 16), "all-vram"
    cases.append(("the registered server passes", server_refusal(C) is None))
    C.max_tokens_per_seq = 4096
    cases.append(("4096 tokens per sequence refuses", server_refusal(C) is not None))
    C.max_tokens_per_seq, C.graphs = 768, False
    cases.append(("an eager server refuses", server_refusal(C) is not None))
    cases.append(("W16 is 7 passes of 16 and one of 11", passes(123, 16)[-1] == (112, 123) and len(passes(123, 16)) == 8))
    cases.append(("W1 is 196 passes of one", passes(196, 1)[-1] == (195, 196)))
    cases.append(("the window hash is P147's json sha256", windows_sha([[1, 2]]) == hashlib.sha256(b"[[1, 2]]").hexdigest()))
    cases.append(("a short window set refuses", window_refusal("W1", [[0] * 640]) is not None))
    a = {"routed": {"1": 4, "16": 2}, "passed": {"511": 8}}
    b = {"routed": {"1": 4, "16": 1}, "passed": {}}
    cases.append(("deltas keep only what moved", counts_delta(a, b) == {"routed": {"16": 1}, "passed": {"511": 8}}))
    rs = route_stats({0: [[(0, 1)] * 16, [(0, 1)] * 8 + [(2, 3)] * 8]})
    cases.append(("routing stats per layer-step: distinct, tiles, largest, histogram",
                  (rs["distinct_mean"], rs["tiles_mean"], rs["max_rows_mean"], rs["rows_per_expert_histogram"])
                  == (3.0, 3.0, 12.0, {"8": 4, "16": 2})))
    cases.append(("T1 and T2 rotate the streams", T_ORDER == {"T1": "ABBA", "T2": "BAAB"}))
    try:
        import torch
        cases += _forcer_cases(torch)
    except ImportError:
        pass
    try:
        import torch
        lg = torch.tensor([[0.0, 2.0, 0.0], [5.0, 0.0, 0.0]])
        n_, am = nll_argmax(lg, [1, 2])
        want = [-float(torch.log_softmax(lg, -1)[0, 1]), -float(torch.log_softmax(lg, -1)[1, 2])]
        cases.append(("NLL and argmax per row", am == [1, 0] and all(abs(x - y) < 1e-6 for x, y in zip(n_, want))))
    except ImportError:
        pass
    bad = [c for c, ok in cases if not ok]
    print(f"p149_box self-test: {len(cases) - len(bad)}/{len(cases)} OK" + (f"; FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("speed", "tf", "sane"))
    ap.add_argument("--stream-a")
    ap.add_argument("--stream-b")
    ap.add_argument("--ref-dir")
    ap.add_argument("--sane-windows", type=int, default=12)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--prompts")
    ap.add_argument("--windows")
    ap.add_argument("--out")
    ap.add_argument("--tag")
    ap.add_argument("--q", action="store_true")
    ap.add_argument("--short", type=int, default=32)
    ap.add_argument("--long", type=int, default=160)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--prompts-only", action="store_true")
    ap.add_argument("--windows-only", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--revision")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return prompts_main(a)
    if a.windows_only:
        return windows_main(a)
    if a.mode == "speed":
        return speed_main(a)
    if a.mode == "tf":
        return tf_main(a)
    if a.mode == "sane":
        return sane_main(a)
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    sys.exit(main())
