#!/usr/bin/env python3
"""p127_box.py -- lane P127 (e4b#1313), ONE arm in its own process (bench/p127/PREREG-p127.md).

The engine is built exactly as the shipped server builds it, ``PagedServeConfig.from_env()`` + ``build_engine(cfg)``, at
its defaults with ``E4B_PAGED_MAX_SEQS=16`` (graphs on, all-vram, buckets 1-16). Which e4b and grouped-nf4-gemm it
imports is the runner's install switch; this box records what it imported and refuses another arm's install.

Arms (``P127_ARM``):
  A  before P127: e4b ``a8c01d42`` + grouped-nf4-gemm v0.44.0.
  B  after P127: the launch commit + grouped-nf4-gemm at #527's merge.
  M  B with the router kernel's bf16 weight store rounding TOWARD ZERO (the blindness check; identity pass only).

Observers, installed before the build and never acting on a timed pass:
  engagement  each kernel entry P127 changed is wrapped (``functools.wraps``, so feature detection reads the real
              signature) and its P127 calls counted: router_epilogue with a non-fp32 weights_dtype, rope_norm_qk,
              combine_rows with a residual, gemm_4bit_grouped with gather_div > 1 and with int64 ids. Decode replays run
              no Python, so counts come from the eager warm-ups, the captures and prefill.
  logits      ``PagedModelRunner._padded_step`` is replaced by the same three statements plus one reference: the step's
              ``out.logits[:, -1]``, graph-owned memory a replay overwrites in place (no kernel added).
              ``run_decode`` is wrapped to digest that view for the step's rows AFTER the step's own token read (its
              ``.tolist()`` synchronizes) and BEFORE it returns, so before any later replay can reuse the storage. It
              digests only inside the identity pass; a timed pass does no digest work at all.
  M           ``router_epilogue`` asked for bf16 weights computes them in fp32 and stores them rounded toward zero.

``--prompts-only`` writes ``prompts.json`` (P109's rows). ``--self-test`` checks the pure helpers on CPU.
"""
import argparse
import functools
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time

ROWS, PROMPT, OFFSET = 16, 512, 4096
ARMS = ("A", "B", "M")
WORKLOADS = {"W16": ROWS, "W1": 1}
BUCKETS = (1, 2, 4, 8, 16)
#: ``PagedModelRunner._padded_step`` as both registered e4b commits carry it, byte for byte; the observer replaces it
#: only when it is exactly this (otherwise the box would be measuring a step it did not reproduce).
PADDED_STEP_SRC = '''    def _padded_step(self, b: int) -> None:
        """The step a bucket's graph captures, on its static buffers."""
        buf = self._bufs[b]
        out = self.model(input_ids=buf["ids"], position_ids=buf["pos"], use_cache=False)
        buf["tok"].copy_(out.logits[:, -1].argmax(-1))
'''
COUNTS = ("router_weights_dtype", "rope_norm_qk", "combine_residual", "gather_div", "int64_ids")


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def slope(walls_short, walls_long, batch, short, long_):
    """p37's decode slope from the fastest pass at each length; the median pair beside it (P109's)."""
    import statistics
    dmin = min(walls_long) - min(walls_short)
    dmed = statistics.median(walls_long) - statistics.median(walls_short)
    tok = batch * (long_ - short)
    if dmin <= 0 or dmed <= 0:
        return {"decode_tok_s": None, "decode_tok_s_median": None, "status": "void: LONG not slower than SHORT"}
    return {"decode_tok_s": round(tok / dmin, 2), "decode_tok_s_median": round(tok / dmed, 2),
            "decode_ms_per_step": round(1000 * dmin / (long_ - short), 4), "status": "ok"}


def rtz_bf16(w32, torch):
    """fp32 -> bf16 rounding TOWARD ZERO: the low 16 bits cleared, then an exact cast (M's store)."""
    w32 = w32.to(torch.float32).contiguous()
    return (w32.view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)


class Observers:
    """The engagement counts, M's store and the logits digests; ``identity`` switches the digest work on."""

    def __init__(self, arm):
        self.arm = arm
        self.counts = {k: 0 for k in COUNTS}
        self.identity = False
        self.digests = {}                   # rid -> [sha256 of the step's last-position logits, ...]
        self.steps_seen = 0

    # ---------------------------------------------------------------- the engagement wrappers (and M's store) --
    def wrap_router(self, real, torch):
        @functools.wraps(real)
        def router_epilogue(*a, **kw):
            wd = kw.get("weights_dtype", None)
            if wd is not None and wd != torch.float32:
                self.counts["router_weights_dtype"] += 1
                if self.arm == "M" and wd == torch.bfloat16:
                    first, w, idx = real(*a, **{**kw, "weights_dtype": torch.float32})
                    return first, rtz_bf16(w, torch), idx
            return real(*a, **kw)
        return router_epilogue

    def wrap_count(self, real, key, pred):
        @functools.wraps(real)
        def wrapped(*a, **kw):
            if pred(a, kw):
                self.counts[key] += 1
            return real(*a, **kw)
        return wrapped

    def wrap_gemm(self, real, torch):
        @functools.wraps(real)
        def gemm_4bit_grouped(*a, **kw):
            if int(kw.get("gather_div", 1) or 1) > 1:
                self.counts["gather_div"] += 1
            ids = a[4] if len(a) > 4 else kw.get("expert_ids")
            if torch.is_tensor(ids) and ids.dtype == torch.int64:
                self.counts["int64_ids"] += 1
            return real(*a, **kw)
        return gemm_4bit_grouped

    # ---------------------------------------------------------------------------------- the logits observer --
    def padded_step(self):
        def _padded_step(runner, b: int) -> None:
            buf = runner._bufs[b]
            out = runner.model(input_ids=buf["ids"], position_ids=buf["pos"], use_cache=False)
            lg = out.logits[:, -1]
            buf["tok"].copy_(lg.argmax(-1))
            buf["p127_lg"] = lg                  # graph-owned: a replay of this bucket overwrites it in place
        return _padded_step

    def run_decode(self, real, bucket_for):
        def run_decode(runner, rids):
            got = real(runner, rids)
            if self.identity and rids:
                n = len(rids)
                if runner._graphs is None or n > runner._buckets[-1]:
                    raise RuntimeError("P127 identity pass: decode not on one bucketed graph step")
                lg = runner._bufs[bucket_for(n, runner._buckets)]["p127_lg"][:n]
                rows = lg.contiguous().view(_int16(lg)).cpu()      # [n, V], the bits; read BEFORE any later replay
                for i, rid in enumerate(rids):
                    self.digests.setdefault(rid, []).append(hashlib.sha256(rows[i].numpy().tobytes()).hexdigest())
                self.steps_seen += 1
            return got
        return run_decode


def _int16(t):
    import torch
    return torch.int16 if t.element_size() == 2 else torch.int32


def install_observers(obs, torch):
    """Wrap the kernel entries and the runner, BEFORE the engine is built (every e4b lookup happens at fuse or call
    time, through the module attribute). Returns what was installed."""
    import int4_b32
    import nf4_grouped
    from experts4bit_qlora.engines import paged_runner
    src = inspect.getsource(paged_runner.PagedModelRunner._padded_step)
    if src != PADDED_STEP_SRC:
        raise SystemExit("REFUSED: PagedModelRunner._padded_step is not the registered three statements")
    done = ["router_epilogue", "combine_rows", "gemm_4bit_grouped", "_padded_step", "run_decode"]
    int4_b32.router_epilogue = obs.wrap_router(int4_b32.router_epilogue, torch)
    int4_b32.combine_rows = obs.wrap_count(int4_b32.combine_rows, "combine_residual",
                                           lambda a, kw: kw.get("residual") is not None)
    if hasattr(int4_b32, "rope_norm_qk"):
        int4_b32.rope_norm_qk = obs.wrap_count(int4_b32.rope_norm_qk, "rope_norm_qk", lambda a, kw: True)
        done.append("rope_norm_qk")
    nf4_grouped.gemm_4bit_grouped = obs.wrap_gemm(nf4_grouped.gemm_4bit_grouped, torch)
    paged_runner.PagedModelRunner._padded_step = obs.padded_step()
    paged_runner.PagedModelRunner.run_decode = obs.run_decode(paged_runner.PagedModelRunner.run_decode,
                                                              paged_runner.bucket_for)
    return done


def run_pass(parts, torch, rows, n_tokens):
    """Every row added at once, stepped to idle; exactly n_tokens per row asserted (P109's). Returns the rids too."""
    sched = parts.scheduler
    before = len(sched.done)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rids = [sched.add_request(list(r), max_new_tokens=n_tokens) for r in rows]
    steps = sched.run_until_idle()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    new = {r.rid: r for r in sched.done[before:]}
    if sorted(new) != sorted(rids) or sched.active or sched.queue:
        raise AssertionError(f"scheduler did not drain: done {sorted(new)} vs added {sorted(rids)}")
    for rid, row in zip(rids, rows):
        q = new[rid]
        if len(q.out) != n_tokens or q.prompt_len != len(row):
            raise AssertionError(f"request {rid}: {len(q.out)} tokens from {q.prompt_len}, expected {n_tokens}")
    return wall, steps, [[int(t) for t in new[r].out] for r in rids], rids


def identity_pass(parts, torch, obs, rows, n_tokens):
    """One untimed pass with the logits digests on: each row's tokens and its per-step digests, in row order."""
    obs.identity, obs.digests = True, {}
    try:
        _wall, _steps, toks, rids = run_pass(parts, torch, rows, n_tokens)
    finally:
        obs.identity = False
    return {"tokens": toks, "logits_digests": [obs.digests.get(r, []) for r in rids]}


def _mem(torch):
    free, total = torch.cuda.mem_get_info()
    return {"max_memory_allocated": int(torch.cuda.max_memory_allocated()), "free": int(free), "total": int(total)}


def _smi():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
                                        "--format=csv,noheader"], text=True, timeout=20).strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a: {e!r}"[:200]


def arm_main(a) -> int:
    arm = os.environ.get("P127_ARM", "")
    if arm not in ARMS:
        raise SystemExit(f"REFUSED: P127_ARM={arm!r}, expected one of {ARMS}")
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if digest(rows) != pf["prompts_sha256"] or len(rows) != ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    import torch
    obs = Observers(arm)
    installed = install_observers(obs, torch)
    import experts4bit_qlora
    import int4_b32
    import nf4_grouped
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if not cfg.graphs or cfg.placement != "all-vram" or tuple(cfg.buckets) != BUCKETS:
        raise SystemExit(f"REFUSED: not the subject: graphs={cfg.graphs} placement={cfg.placement} "
                         f"buckets={cfg.buckets}")
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    info = parts.info
    rec = {"arm": arm, "tag": a.tag, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "paths": {"experts4bit_qlora": experts4bit_qlora.__file__, "int4_b32": int4_b32.__file__,
                     "nf4_grouped": nf4_grouped.__file__},
           "model": cfg.model, "revision": cfg.revision, "torch": torch.__version__, "load_s": round(load_s, 2),
           "max_seqs": int(cfg.max_seqs), "buckets": list(cfg.buckets), "graphs": bool(cfg.graphs),
           "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
           "moe_residual": info.get("moe_residual"), "moe_layers": info.get("moe_layers"),
           "installed": installed, "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long,
           "reps": a.reps, "identity_only": bool(a.identity_only), "workloads": {}, "identity": {}, "status": "ok"}
    if not a.identity_only:
        for wname, b in WORKLOADS.items():
            wrows = rows[:b]
            walls, digests, last = {}, {}, {}
            for n in (a.short, a.long):
                run_pass(parts, torch, wrows, n)                             # warm, untimed
                ws, ds = [], []
                for _ in range(a.reps):
                    wall, _steps, toks, _rids = run_pass(parts, torch, wrows, n)
                    ws.append(round(wall, 5))
                    ds.append(digest(toks))
                    last[str(n)] = toks
                walls[str(n)], digests[str(n)] = ws, ds
            s = slope(walls[str(a.short)], walls[str(a.long)], b, a.short, a.long)
            rec["workloads"][wname] = {"batch": b, "walls": walls, "rep_digests": digests, "tokens": last, **s}
            print(f"P127_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} walls={walls}", flush=True)
    for wname, b in WORKLOADS.items():
        rec["identity"][wname] = identity_pass(parts, torch, obs, rows[:b], a.long)
    rec["counts"] = dict(obs.counts)
    rec["identity_steps"] = obs.steps_seen
    rec["graph_stats"] = ({str(k): dict(v) for k, v in parts.runner.graph_stats.items()}
                          if isinstance(getattr(parts.runner, "graph_stats", None), dict) else None)
    rec["mem_after_runs"] = _mem(torch)
    rec["nvidia_smi"] = _smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P127_ARM " + json.dumps({"arm": arm, "tag": a.tag, "load_s": rec["load_s"], "max_seqs": rec["max_seqs"],
                                    "counts": rec["counts"],
                                    "W16": rec["workloads"].get("W16", {}).get("decode_tok_s"),
                                    "W1": rec["workloads"].get("W1", {}).get("decode_tok_s")}), flush=True)
    return 0


def prompts_main(a) -> int:
    """P109's rows, by P109's own function at its registered bytes."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from p109_box import wikitext_rows
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision or None)
    rows = wikitext_rows(tok)
    json.dump({"model": a.model, "revision": a.revision, "rows": rows, "prompts_sha256": digest(rows),
               "offset": OFFSET, "prompt": PROMPT}, open(a.out, "w"))
    print(f"P127_PROMPTS rows={len(rows)} prompt={PROMPT} sha256={digest(rows)}", flush=True)
    return 0


def self_test() -> int:
    import torch
    ok = []
    s = slope([1.0, 1.1], [3.0, 3.2], 16, 32, 160)
    ok.append(s["decode_tok_s"] == round(16 * 128 / 2.0, 2) and s["status"] == "ok")
    ok.append(slope([2.0], [2.0], 1, 32, 160)["status"].startswith("void"))
    w = torch.tensor([0.30000001192092896, -0.123456789, 1.0, 2 ** -7 * 1.99], dtype=torch.float32)
    z = rtz_bf16(w, torch)
    ok.append(z.dtype == torch.bfloat16 and bool((z.float().abs() <= w.abs()).all()))     # toward zero, never away
    ok.append(not torch.equal(z, w.to(torch.bfloat16)))                                 # differs from nearest-even
    obs = Observers("M")

    def real(logits, k, norm, *, select_on_logits=False, bias=None, weights_dtype=torch.float32):
        return logits, (logits[:, :k] * 0.3).to(weights_dtype), None
    wr = obs.wrap_router(real, torch)
    ok.append("weights_dtype" in inspect.signature(wr).parameters)                      # feature detection sees it
    lg = torch.randn(2, 8)
    _f, wm, _i = wr(lg, 4, True, weights_dtype=torch.bfloat16)
    ok.append(obs.counts["router_weights_dtype"] == 1 and torch.equal(wm, rtz_bf16(lg[:, :4] * 0.3, torch)))
    ok.append(PADDED_STEP_SRC.count("\n") == 5 and COUNTS[0] == "router_weights_dtype")
    ok.append(ARMS == ("A", "B", "M") and WORKLOADS == {"W16": 16, "W1": 1} and BUCKETS == (1, 2, 4, 8, 16))
    n = len(ok)
    print(f"p127_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{n} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--identity-only", action="store_true")
    p.add_argument("--model")
    p.add_argument("--revision", default="")
    p.add_argument("--prompts")
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--short", type=int, default=32)
    p.add_argument("--long", type=int, default=160)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return prompts_main(a)
    return arm_main(a)


if __name__ == "__main__":
    sys.exit(main())
