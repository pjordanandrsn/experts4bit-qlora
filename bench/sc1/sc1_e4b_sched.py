#!/usr/bin/env python3
"""sc1_e4b_sched.py -- e4b timed THROUGH its scheduler (lane SC1, experts4bit-qlora#846): the `lic_sched` / `int4_sched`
arms, the e4b TTFT rows, the e4b energy windows, the SAMEPROMPT control on the e4b side, and the proving rental's smoke.

Why this file exists (SC1-PREREG.md "Why this lane" 3): the register's e4b number is `step_decomp.py --b1d-loop
graph --b1d-timed`, a bare `g.replay()` window with no scheduler step, no per-step H2D of ids/pos, no host readback;
vLLM's number is its whole serving loop. The ratios SC1 quotes are taken on THIS arm, which drives
`experts4bit_qlora.serve_paged.build_engine` (the same construction the shipped server uses) through
`ContinuousScheduler.step()` by the same 32->128 slope p37 applies to vLLM. The window arm stays beside it as the
kernel ceiling.

Construction: `PagedServeConfig.from_env()` + `build_engine(cfg)` -> `EngineParts(scheduler, tokenizer, eos_ids, info,
runner)`. The runner sets the engine env (E4B_PAGED_MODEL / REVISION / ARENA / CALIB / PLACEMENT=all-vram / MAX_SEQS=B /
MAX_TOKENS_PER_SEQ / CHUNK_TOKENS / GRAPHS=1 / BUCKETS), the fusion set (E4B_PAGED_FUSE_QKV=1 + the three fold flags
for the registered fused-with-folds set; the folds alone for the unfused set), the lever env (P88's SPEEDENV / LICENV),
the pack env (E4B_INT4_ARTIFACT_DIR + E4B_INT4_EXPECTED_FINGERPRINT) and the four route knobs. Written for the FIXED
`_apply_fusions` (fix/paged-fusions 2719171, landing before registration): `E4B_PAGED_FUSE_QKV=1` TOGETHER with the fold
flags is accepted, `fuse_qkv` applies the folds itself and the census reports the fold counts it captured, never a
literal zero -- the build the round-2 review found refusing that combination is the one this file does NOT target.

Slope (default mode): for N in (SHORT, LONG): one untimed warm pass, then `reps` timed passes of
    torch.cuda.synchronize(); t0; [add_request(row, max_new_tokens=N) for row in rows]; run_until_idle(); synchronize(); wall
with `stop_ids=None` (the scheduler's original contract: the sequence runs to max_new_tokens -- EOS ignored, greedy
argmax in the runner). Every `done` request must have emitted exactly N tokens and have seen exactly 512 prompt tokens
(the engine's own count, `Request.prompt_len`). decode_tok_s = B*(LONG-SHORT)/(min wall_LONG - min wall_SHORT), the
median pair beside it, p37's arithmetic. Receipt in p37's shape with `engine: "e4b-sched"` (+ `census` = EngineParts.info,
`graph_status`, `graph_stats`, scheduler `stats()`, `timed_windows_epoch`, memory, the banners grepped from this
process's own stdout, the route knobs) so the reducer treats it like vLLM's.

Modes:  --ttft        one row (512 or 4096 tokens; B must be 1), max_new_tokens=1, one warm then 3 timed; wall per rep
                      and the scheduler's own Request.ttft beside it; the prefill is chunked at CHUNK_TOKENS (recorded).
        --sameprompt  B copies of row 0 of the prompt file (or a file whose rows are already identical); the receipt says so
                      (`sameprompt.rows_identical`, file sha, effective sha) -- the routing-collapse control.
        --energy      one warm pass of SHORT, then ONE timed pass of --energy-tokens (1024) at B under the runner's
                      nvidia-smi sampler; prints `SC1_ENERGY_WINDOW start_epoch=.. stop_epoch=.. tokens=..` and records it.
        --smoke       B rows synthesised from the tokenizer (no prompt file), --smoke-tokens (16) tokens; prints the census
                      (`SC1SCHED_CENSUS {...}`) and exits 0 only if every lever that is SET reports a nonzero count, every
                      requested graph bucket captured, and every row emitted exactly the tokens asked.
        --selftest    the pure helpers on CPU (no torch).
Env: SC1_ARM, SC1_BATCH, SC1_PROMPTS (not for --smoke), SC1_OUT, SC1_INSTANCE_ID, SC1_LOG (optional; unused -- the
runner redirects); the E4B_PAGED_* / lever / pack / route env above. Stdlib + torch (+ e4b) only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time

SHORT, LONG = 32, 128
PROMPT_LEN = 512
ENGINE = "e4b-sched"
METHOD = "slope(32->128) through ContinuousScheduler.step(); min-of-3 (P20 estimator) and median-of-3 both recorded"
ROUTE_ENV = ("E4B_INT4_GROUPED_SMALLM", "E4B_INT4_LEAN_GLUE", "E4B_NF4_GROUPED_SMALLM", "E4B_MXFP4_GROUPED_SMALLM")
PAGED_ENV = ("E4B_PAGED_MODEL", "E4B_PAGED_REVISION", "E4B_PAGED_ARENA", "E4B_PAGED_CALIB", "E4B_PAGED_PLACEMENT",
             "E4B_PAGED_MAX_SEQS", "E4B_PAGED_MAX_TOKENS_PER_SEQ", "E4B_PAGED_CHUNK_TOKENS", "E4B_PAGED_MAX_PREFILL_TOKENS",
             "E4B_PAGED_GRAPHS", "E4B_PAGED_BUCKETS", "E4B_PAGED_FUSE_QKV", "E4B_PAGED_KV_GROUPS", "E4B_PAGED_TORCH_THREADS")
# a lever that is SET ("1") must show up in the census as a nonzero count (serve_paged's own refusal rule, re-read here)
LEVER_CHECKS = {"E4B_SERVE_EXP_INT4": "int4_expert_layers", "E4B_SERVE_ATTN_INT4": "attn_int4_rtn_projections",
                "E4B_SERVE_ATTN_INT4_CALIB": "attn_int4_calib_projections", "E4B_FUSE_T1_GLUE": "fuse_t1_glue_n",
                "E4B_FUSE_T1_GLUE_R2": "fuse_t1_glue_r2_n", "E4B_FUSE_ROUTER_EPI": "fuse_router_epilogue_n",
                "E4B_PAGED_FUSE_QKV": "fuse_qkv_n"}
BANNERS = {"int4exp_artifact": r"INT4EXP licensed artifact (\S+)", "int4exp_enabled": r"INT4EXP enabled: (\d+) layers",
           "int4exp_calibrated": r"INT4EXP calibrated experts: [^\n]*", "attn_calib": r"ATTNINT4 calibrated: (\d+) projections",
           "attn_rtn": r"ATTNINT4 rtn: (\d+) projections", "fused_qkv": r"fused q/k/v projections on (\d+) attention modules",
           "decode_graph": r"DECODE_GRAPH[^\n]*", "fusions": r"\[serve_paged\] fusions[^\n]*", "ready": r"\[serve_paged\] ready: [^\n]*"}


def harness_hook_loaded(modules=None):
    """The P42 harness hook's path when it is loaded, else None. ``serve_paged.build_engine`` applies the int4 levers itself --
    its ``_apply_levers`` IS the hook's ``_apply_lanes``, at the same point -- so with the hook ALSO loaded they are applied
    twice and the second enable refuses ("matched no attention projections"; sc1a-5090-1, Amendment A4)."""
    m = (sys.modules if modules is None else modules).get("usercustomize")
    f = str(getattr(m, "__file__", "") or "")
    return f if (m is not None and f.endswith("/hook/usercustomize.py")) else None


class Refusal(SystemExit):
    """A validity predicate failed before any number existed: no receipt, exit 2 (p37's asserts)."""


class Tee(io.TextIOBase):
    """stdout + a buffer, so the engine's banners travel in the receipt (the runner's log keeps the stream too)."""

    def __init__(self, real):
        self.real = real
        self.buf = io.StringIO()

    def write(self, s):
        self.real.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self):
        self.real.flush()

    def fileno(self):
        return self.real.fileno()


# ------------------------------------------------------------------------------------------- pure helpers (CPU)
def prompts_digest(prompts) -> str:
    return hashlib.sha256(json.dumps(prompts).encode()).hexdigest()


def load_prompts(path: str, batch: int, prompt_len=PROMPT_LEN, sameprompt: bool = False) -> dict:
    """prompts_b{B}.json with p37's refusals (batch, row length, distinct rows, digest). `sameprompt` accepts the
    distinct B=16 file and replicates row 0, or a file whose rows are already identical (prompts_b16_same.json)."""
    pf = json.load(open(path))
    prompts = pf["prompts"]
    if pf.get("batch") != batch or len(prompts) != batch:
        raise Refusal(f"REFUSED: prompt file batch {pf.get('batch')}/{len(prompts)} != SC1_BATCH {batch}")
    if prompt_len is not None and not all(len(p) == prompt_len for p in prompts):
        raise Refusal(f"REFUSED: prompt_len must be {prompt_len}, got {sorted({len(p) for p in prompts})}")
    file_sha = prompts_digest(prompts)
    if file_sha != pf.get("prompts_sha256"):
        raise Refusal(f"REFUSED: prompt file digest mismatch: computed {file_sha[:12]} != file {str(pf.get('prompts_sha256'))[:12]}")
    distinct = len({tuple(p) for p in prompts}) == batch
    info = {"file": os.path.basename(path), "file_prompts_sha256": file_sha, "rows_sha256": pf.get("rows_sha256"),
            "rows_distinct_in_file": distinct}
    if sameprompt:
        if batch < 2:
            raise Refusal("REFUSED: --sameprompt is a B>1 control (B copies of row 0); at B=1 it is the distinct arm")
        prompts = [list(prompts[0]) for _ in range(batch)]
        info["sameprompt"] = {"source_row": 0, "copies": batch, "rows_identical": True, "file_prompts_sha256": file_sha,
                              "effective_prompts_sha256": prompts_digest(prompts),
                              "source": "replicated here from row 0" if distinct else "file rows already identical"}
    elif not distinct:
        raise Refusal("REFUSED: rows must be distinct prompts (P20's distinct-prompt rule); pass --sameprompt for the control")
    info["prompts_sha256"] = prompts_digest(prompts)
    return {"prompts": prompts, "info": info}


def slope(w_short, w_long, batch: int, short: int = SHORT, long_: int = LONG) -> dict:
    """p37's estimator, byte for byte in its arithmetic; an inverted min-of-3 slope is a refusal, never a number."""
    extra = (long_ - short) * batch
    d_min = min(w_long) - min(w_short)
    d_med = statistics.median(w_long) - statistics.median(w_short)
    out = {"walls_short_s": [round(w, 4) for w in w_short], "walls_long_s": [round(w, 4) for w in w_long],
           "wall_short_s": round(min(w_short), 4), "wall_long_s": round(min(w_long), 4),
           "end_to_end_tok_s_long": round(long_ * batch / min(w_long), 1)}
    if d_min <= 0 or d_med <= 0:
        out.update(decode_tok_s=None, decode_ms_per_step=None, decode_tok_s_median=None, decode_ms_per_step_median=None,
                   slope_note="slope <= 0: LONG was not slower than SHORT -- not a decode reading", status="void")
        return out
    out.update(decode_tok_s=round(extra / d_min, 1), decode_ms_per_step=round(d_min / (long_ - short) * 1e3, 4),
               decode_tok_s_median=round(extra / d_med, 1), decode_ms_per_step_median=round(d_med / (long_ - short) * 1e3, 4))
    return out


def _nonzero(v) -> bool:
    if isinstance(v, (list, tuple)):
        return any(int(x) > 0 for x in v)
    return v is not None and int(v) > 0


def lever_census(info: dict, env: dict) -> dict:
    """Every lever SET in `env` must read a nonzero census count; with graphs on, every bucket must say 'graph'.
    Returns {ok, failures, checked}."""
    failures, checked = [], {}
    for lever, key in LEVER_CHECKS.items():
        if env.get(lever, "0") == "1":
            checked[lever] = info.get(key)
            if not _nonzero(info.get(key)):
                failures.append(f"{lever}=1 but census {key}={info.get(key)!r}")
    if env.get("E4B_PAGED_GRAPHS", "0") == "1":
        gs = info.get("graph_status") or {}
        checked["E4B_PAGED_GRAPHS"] = gs
        want = [int(b) for b in env.get("E4B_PAGED_BUCKETS", "1,2,4,8,16").split(",") if b.strip()]
        for b in want:
            st = gs.get(b, gs.get(str(b)))
            if st != "graph":
                failures.append(f"bucket {b} not captured: {st!r}")
    return {"ok": not failures, "failures": failures, "checked": checked}


def banners(text: str) -> dict:
    out = {}
    for key, rx in BANNERS.items():
        m = re.findall(rx, text)
        out[key] = m if m else None
    return out


def same_stack_fields(info: dict, env: dict) -> dict:
    """The window-vs-sched 'same stack' predicate's inputs (v3 'Fixture'): fuse_qkv_n, int4 layer / projection counts,
    the fold counts, the pack fingerprint -- so the reducer compares them to the window arm's banners."""
    return {"fuse_qkv_n": info.get("fuse_qkv_n"), "int4_expert_layers": info.get("int4_expert_layers"),
            "int4_attn_projections": info.get("int4_attn_projections"), "fuse_t1_glue_n": info.get("fuse_t1_glue_n"),
            "fuse_t1_glue_r2_n": info.get("fuse_t1_glue_r2_n"), "fuse_router_epilogue_n": info.get("fuse_router_epilogue_n"),
            "pack_fingerprint": env.get("E4B_INT4_EXPECTED_FINGERPRINT"), "graph_status": info.get("graph_status")}


def smoke_rows(tok, batch: int, n: int = 64) -> list:
    """B distinct n-token rows from a fixed paragraph (the proof needs no prompt file and no Qwen3 tokenizer)."""
    text = ("The scheduler admits a request, prefills its prompt in chunks, and then decodes one token per step for "
            "every resident sequence; a benchmark that times this loop times what a server does. ") * 12
    ids = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
    if len(ids) < n + 7 * batch:
        raise Refusal(f"REFUSED: smoke text tokenises to {len(ids)} ids, need {n + 7 * batch}")
    rows = [ids[7 * i:7 * i + n] for i in range(batch)]
    if len({tuple(r) for r in rows}) != batch:
        raise Refusal("REFUSED: smoke rows are not distinct")
    return rows


def _raises(exc, fn, *args, **kw) -> bool:
    try:
        fn(*args, **kw)
    except exc:
        return True
    return False


def selftest() -> int:
    """The pure helpers on CPU (no torch): the slope arithmetic, the prompt-file refusals, the SAMEPROMPT replication, the
    lever census, the banner grep, the smoke rows."""
    import tempfile
    checks = []
    s = slope([1.0, 1.1, 1.2], [2.0, 2.2, 2.1], batch=16)
    checks.append(s["decode_tok_s"] == round(16 * 96 / 1.0, 1) and s["decode_ms_per_step"] == round(1.0 / 96 * 1e3, 4))
    checks.append(s["decode_tok_s_median"] == round(16 * 96 / 1.0, 1) and s["end_to_end_tok_s_long"] == round(128 * 16 / 2.0, 1))
    checks.append(slope([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], batch=1)["status"] == "void")
    import types as _t
    checks.append(harness_hook_loaded({"usercustomize": _t.SimpleNamespace(__file__="/root/sc1/hook/usercustomize.py")})
                  == "/root/sc1/hook/usercustomize.py")
    checks.append(harness_hook_loaded({}) is None)
    checks.append(harness_hook_loaded({"usercustomize": _t.SimpleNamespace(__file__="/usr/lib/python3/dist-packages/usercustomize.py")}) is None)
    rows = [[(r * 31 + i) % 1000 for i in range(PROMPT_LEN)] for r in range(16)]
    pf = {"batch": 16, "prompts": rows, "prompts_sha256": prompts_digest(rows), "rows_sha256": ["x"] * 16}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "prompts_b16.json")
        json.dump(pf, open(p, "w"))
        got = load_prompts(p, 16)
        checks.append(got["prompts"] == rows and got["info"]["rows_distinct_in_file"] is True and "sameprompt" not in got["info"])
        same = load_prompts(p, 16, sameprompt=True)
        checks.append(all(r == rows[0] for r in same["prompts"]) and same["info"]["sameprompt"]["rows_identical"] is True)
        checks.append(same["info"]["prompts_sha256"] != same["info"]["file_prompts_sha256"])
        identical = [rows[0]] * 16
        for bad, batch in ((dict(pf, prompts_sha256="0" * 64), 16), (pf, 1),
                           (dict(pf, prompts=identical, prompts_sha256=prompts_digest(identical)), 16)):
            json.dump(bad, open(p, "w"))
            checks.append(_raises(Refusal, load_prompts, p, batch))
        json.dump(dict(pf, prompts=identical, prompts_sha256=prompts_digest(identical)), open(p, "w"))
        checks.append(load_prompts(p, 16, sameprompt=True)["info"]["sameprompt"]["source"] == "file rows already identical")
        json.dump(pf, open(p, "w"))
        checks.append(_raises(Refusal, load_prompts, p, 16, prompt_len=4096))
    info = {"int4_expert_layers": 48, "attn_int4_calib_projections": 192, "fuse_qkv_n": 48, "fuse_t1_glue_n": 48,
            "fuse_t1_glue_r2_n": [48, 48], "fuse_router_epilogue_n": 48, "graph_status": {1: "graph", 16: "graph"}}
    env = {"E4B_SERVE_EXP_INT4": "1", "E4B_SERVE_ATTN_INT4_CALIB": "1", "E4B_PAGED_FUSE_QKV": "1", "E4B_FUSE_T1_GLUE": "1",
           "E4B_FUSE_T1_GLUE_R2": "1", "E4B_FUSE_ROUTER_EPI": "1", "E4B_PAGED_GRAPHS": "1", "E4B_PAGED_BUCKETS": "1,16"}
    checks.append(lever_census(info, env)["ok"])
    bad = lever_census(dict(info, fuse_t1_glue_r2_n=[0, 0], graph_status={1: "graph", 16: "eager: capture failed"}), env)
    checks.append(not bad["ok"] and len(bad["failures"]) == 2)
    checks.append(lever_census({"fuse_qkv_n": 0}, {"E4B_PAGED_FUSE_QKV": "1"})["failures"] == ["E4B_PAGED_FUSE_QKV=1 but census fuse_qkv_n=0"])
    checks.append(lever_census({}, {"E4B_SERVE_EXP_INT4": "0"})["ok"])          # an unset lever is not checked
    b = banners("x\nINT4EXP licensed artifact sha256:abc\nfused q/k/v projections on 48 attention modules\nDECODE_GRAPH bucket=16 captured\n")
    checks.append(b["int4exp_artifact"] == ["sha256:abc"] and b["fused_qkv"] == ["48"]
                  and b["decode_graph"] == ["DECODE_GRAPH bucket=16 captured"] and b["attn_rtn"] is None)

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 251 for c in text]}
    r = smoke_rows(_Tok(), 16)
    checks.append(len(r) == 16 and all(len(x) == 64 for x in r) and len({tuple(x) for x in r}) == 16)
    ss = same_stack_fields(info, {"E4B_INT4_EXPECTED_FINGERPRINT": "sha256:abc"})
    checks.append(ss["fuse_qkv_n"] == 48 and ss["pack_fingerprint"] == "sha256:abc" and ss["fuse_t1_glue_r2_n"] == [48, 48])
    failed = [i for i, ok in enumerate(checks) if not ok]
    if failed:
        print(f"sc1_e4b_sched selftest FAILED cases {failed}")
        return 1
    print(f"sc1_e4b_sched selftest OK ({len(checks)} cases)")
    return 0


# ------------------------------------------------------------------------------------------------ the engine
def _nvidia_smi() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
                                        "--format=csv,noheader"], text=True, timeout=20).strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a: {e!r}"[:200]


def _mem(torch) -> dict:
    free, total = torch.cuda.mem_get_info()
    return {"max_memory_allocated": int(torch.cuda.max_memory_allocated()), "memory_reserved": int(torch.cuda.memory_reserved()),
            "memory_allocated": int(torch.cuda.memory_allocated()), "mem_get_info_free": int(free), "mem_get_info_total": int(total)}


def _dynamo_counters() -> dict:
    try:
        import torch._dynamo.utils as du
        return {k: dict(v) for k, v in du.counters.items() if k in ("frames", "stats", "recompiles")}
    except Exception:  # noqa: BLE001
        return {}


def _graph_stats(runner) -> dict | None:
    gs = getattr(runner, "graph_stats", None)
    return {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None


def run_batch(parts, torch, rows, n_tokens: int, prompt_len=PROMPT_LEN):
    """One timed pass: every row added at once, stepped to idle, exactly n_tokens per row asserted."""
    sched = parts.scheduler
    done_before = len(sched.done)
    torch.cuda.synchronize()
    t_epoch0 = time.time()
    t0 = time.perf_counter()
    rids = [sched.add_request(list(r), max_new_tokens=n_tokens) for r in rows]      # stop_ids=None: EOS ignored by contract
    steps = sched.run_until_idle()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    t_epoch1 = time.time()
    new = sched.done[done_before:]
    by_rid = {r.rid: r for r in new}
    if sorted(by_rid) != sorted(rids) or sched.active or sched.queue:
        raise AssertionError(f"scheduler did not drain: done {sorted(by_rid)} vs added {sorted(rids)}, "
                             f"active {len(sched.active)}, queued {len(sched.queue)}")
    short = [r.rid for r in new if len(r.out) != n_tokens]
    if short:
        raise AssertionError(f"rows {short[:8]} emitted {[len(by_rid[r].out) for r in short[:8]]} tokens, expected {n_tokens} each "
                             "(a row that stopped early is not the registered workload)")
    if prompt_len is not None:
        bad = [r.rid for r in new if r.prompt_len != prompt_len]
        if bad:
            raise AssertionError(f"rows {bad[:8]} saw {[by_rid[r].prompt_len for r in bad[:8]]} prompt tokens, expected {prompt_len}")
    ordered = [by_rid[r] for r in rids]
    return {"wall": wall, "steps": steps, "requests": ordered, "epoch": [t_epoch0, t_epoch1]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttft", action="store_true")
    ap.add_argument("--sameprompt", action="store_true")
    ap.add_argument("--energy", action="store_true")
    ap.add_argument("--energy-tokens", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-tokens", type=int, default=16)
    ap.add_argument("--short", type=int, default=SHORT)
    ap.add_argument("--long", dest="long_", type=int, default=LONG)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    hook = harness_hook_loaded()
    if hook:
        raise Refusal(f"REFUSED: the P42 harness hook is loaded ({hook}) -- serve_paged applies the levers itself; "
                      "run the scheduler arms with PYTHONPATH= (A4)")
    if sum((a.ttft, a.sameprompt, a.energy, a.smoke)) > 1:
        raise Refusal("REFUSED: --ttft / --sameprompt / --energy / --smoke are exclusive")
    batch = int(os.environ["SC1_BATCH"])
    arm = os.environ.get("SC1_ARM") or ("smoke" if a.smoke else None)
    if arm is None:
        raise Refusal("REFUSED: SC1_ARM is unset")
    out_path = os.environ["SC1_OUT"]
    tee = Tee(sys.stdout)
    sys.stdout = tee

    import torch

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.max_seqs != batch:
        raise Refusal(f"REFUSED: E4B_PAGED_MAX_SEQS={cfg.max_seqs} != SC1_BATCH={batch} (capacity is matched per arm)")
    if cfg.placement != "all-vram":
        raise Refusal(f"REFUSED: E4B_PAGED_PLACEMENT={cfg.placement!r}; every registered serving number is all-vram")
    if not cfg.graphs:
        print("WARNING: E4B_PAGED_GRAPHS is not 1 -- an eager engine; the receipt says so", flush=True)
    env = dict(os.environ)
    rec = {"engine": ENGINE, "arm": arm, "e4b_sha": env.get("E4B_SHA"), "gnf4_sha": env.get("GNF4_SHA"), "torch": torch.__version__,
           "model": cfg.model, "revision": cfg.revision or None, "batch": batch, "method": METHOD,
           "prompts": "identical token ids to every other engine (step_decomp._k8_window, wikitext-2 test, 512-token rows)",
           "engine_kwargs": {k: env.get(k) for k in PAGED_ENV if env.get(k) is not None},
           "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items() if k != "token"},
           "generation": {"temperature": 0.0, "greedy": "argmax (PagedModelRunner)", "ignore_eos": True, "stop_ids": None,
                          "min_tokens": "= max_new_tokens (the scheduler's length contract)", "chunk_tokens": cfg.chunk_tokens,
                          "prefill_budget_per_step": cfg.prefill_budget},
           "route_env": {k: env.get(k) for k in ROUTE_ENV}, "pack_fingerprint": env.get("E4B_INT4_EXPECTED_FINGERPRINT"),
           "fusion_set": ("fused q/k/v WITH the env-gated folds (fuse_qkv applies them itself; the registered set)" if cfg.fuse_qkv
                          else "unfused: the three env-gated folds called directly"),
           "vast_instance_id": env.get("SC1_INSTANCE_ID"), "mode": ("ttft" if a.ttft else "energy" if a.energy else "smoke" if a.smoke
                                                                    else "sameprompt" if a.sameprompt else "slope"),
           "status": "ok"}
    try:
        import experts4bit_qlora as e4b
        rec["e4b_version"] = getattr(e4b, "__version__", None)
    except Exception:  # noqa: BLE001
        pass
    dyn0 = _dynamo_counters()
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    rec["load_s"] = round(time.perf_counter() - t0, 1)
    info = parts.info
    info["fingerprint"] = env.get("E4B_INT4_EXPECTED_FINGERPRINT")      # the reducer's census {..., fingerprint?}
    rec["census"] = info
    rec["graph_status"] = info.get("graph_status")
    rec["same_stack"] = same_stack_fields(info, env)
    rec["mem_after_load"] = _mem(torch)
    rec["eos_ids_model"] = sorted(parts.eos_ids)

    def _finish(status="ok"):
        rec["status"] = status if rec.get("status", "ok") == "ok" else rec["status"]
        rec["graph_stats"] = _graph_stats(parts.runner)
        rec["scheduler_stats"] = parts.scheduler.stats()
        rec["mem_after_runs"] = _mem(torch)
        rec["nvidia_smi"] = _nvidia_smi()
        dyn1 = _dynamo_counters()
        rec["dynamo_counters"] = {"before_build": dyn0, "after_runs": dyn1}
        rec["banners"] = banners(tee.buf.getvalue())
        json.dump(rec, open(out_path, "w"), indent=1, default=str)

    if a.smoke:
        rows = smoke_rows(parts.tokenizer, batch)
        rec["prompt_tokens"] = [len(r) for r in rows]
        try:
            r = run_batch(parts, torch, rows, a.smoke_tokens, prompt_len=len(rows[0]))
            rec.update(smoke_wall_s=round(r["wall"], 4), smoke_steps=r["steps"], tokens={str(i): [int(t) for t in q.out] for i, q in enumerate(r["requests"])})
            emitted_ok = True
        except AssertionError as e:
            rec["smoke_error"] = str(e)
            emitted_ok = False
        check = lever_census(info, env)
        rec["lever_census"] = check
        ok = emitted_ok and check["ok"]
        _finish("ok" if ok else "smoke_failed")
        print("SC1SCHED_CENSUS " + json.dumps({"arm": arm, "batch": batch, "ok": ok, "failures": check["failures"],
                                               "smoke_error": rec.get("smoke_error"), "load_s": rec["load_s"],
                                               "census": {k: info.get(k) for k in ("moe_layers", "int4_expert_layers", "int4_attn_projections",
                                                                                   "fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n",
                                                                                   "fuse_router_epilogue_n", "graph_status")}}, default=str), flush=True)
        return 0 if ok else 1

    pf = load_prompts(os.environ["SC1_PROMPTS"], batch, prompt_len=None if a.ttft else PROMPT_LEN, sameprompt=a.sameprompt)
    rows = pf["prompts"]
    rec.update(prompts_sha256=pf["info"]["prompts_sha256"], rows_sha256=pf["info"]["rows_sha256"], prompt_tokens=[len(r) for r in rows],
               prompt_file=pf["info"])
    if a.sameprompt:
        rec["sameprompt"] = pf["info"]["sameprompt"]
    plen = len(rows[0])

    if a.ttft:
        if batch != 1:
            raise Refusal("REFUSED: --ttft measures B=1 (one row, one slot)")
        run_batch(parts, torch, rows, 1, prompt_len=plen)                                       # warm (untimed)
        walls, ttfts, epochs = [], [], []
        for _ in range(3):
            r = run_batch(parts, torch, rows, 1, prompt_len=plen)
            walls.append(r["wall"])
            ttfts.append(r["requests"][0].ttft)
            epochs.append(r["epoch"])
        rec.update(ttft_mode="wall of a max_new_tokens=1 request at add_request (warm, uncached: every request re-prefills)",
                   ttft_prompt_tokens=plen, ttft_walls_s=[round(w, 4) for w in walls], ttft_s_median=round(statistics.median(walls), 4),
                   ttft_s_min=round(min(walls), 4), scheduler_ttft_s=[round(t, 4) for t in ttfts if t is not None],
                   prefill_steps_expected=-(-plen // cfg.chunk_tokens), timed_windows_epoch=epochs,
                   prompt_tokens_engine=[plen], ttft_note="the engine's own clock (Request.ttft, from arrival) beside the wall")
        _finish()
        print("SC1SCHED " + json.dumps({k: rec[k] for k in ("arm", "batch", "ttft_prompt_tokens", "ttft_s_median", "ttft_s_min", "prompts_sha256")}), flush=True)
        return 0

    if a.energy:
        run_batch(parts, torch, rows, a.short)                                                  # warm (untimed)
        r = run_batch(parts, torch, rows, a.energy_tokens)
        rec.update(energy_tokens=a.energy_tokens * batch, energy_wall_s=round(r["wall"], 4), energy_window_epoch=r["epoch"],
                   energy_tok_s=round(a.energy_tokens * batch / r["wall"], 1), energy_steps=r["steps"],
                   prompt_tokens_engine=[q.prompt_len for q in r["requests"]],
                   tokens={str(i): [int(t) for t in q.out] for i, q in enumerate(r["requests"])},
                   energy_note="J/token = the sampler's integral of power.draw.instant over energy_window_epoch / energy_tokens (board power)")
        _finish()
        print(f"SC1_ENERGY_WINDOW start_epoch={r['epoch'][0]:.3f} stop_epoch={r['epoch'][1]:.3f} tokens={a.energy_tokens * batch}", flush=True)
        print("SC1SCHED " + json.dumps({k: rec[k] for k in ("arm", "batch", "energy_tokens", "energy_wall_s", "energy_tok_s", "prompts_sha256")}), flush=True)
        return 0

    walls, epochs, tokens = {}, {}, {}
    for n in (a.short, a.long_):
        run_batch(parts, torch, rows, n)                                                        # warm (untimed)
        ws, es = [], []
        for _ in range(a.reps):
            r = run_batch(parts, torch, rows, n)
            ws.append(r["wall"])
            es.append(r["epoch"])
            tokens[n] = {str(i): [int(t) for t in q.out] for i, q in enumerate(r["requests"])}
            rec["prompt_tokens_engine"] = [q.prompt_len for q in r["requests"]]
        walls[n], epochs[n] = ws, es
    s = slope(walls[a.short], walls[a.long_], batch, a.short, a.long_)
    if s.get("status") == "void":
        rec["status"] = "void"
    rec.update(s)
    rec.update(short=a.short, long=a.long_, reps=a.reps, tokens=tokens[a.long_], tokens_short=tokens[a.short],
               timed_windows_epoch={"short": epochs[a.short], "long": epochs[a.long_]},
               ttft_note="not measured in slope mode; the --ttft arm measures it on this engine")
    _finish()
    print("SC1SCHED " + json.dumps({k: rec.get(k) for k in ("arm", "batch", "decode_tok_s", "decode_ms_per_step", "decode_tok_s_median",
                                                            "end_to_end_tok_s_long", "status", "prompts_sha256")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
