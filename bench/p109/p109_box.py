#!/usr/bin/env python3
"""p109_box.py -- lane P109 (e4b#770), ONE arm in its own process (bench/p109/PREREG-p109.md).

The engine is built exactly as the shipped server builds it, ``PagedServeConfig.from_env()`` + ``build_engine(cfg)``,
from the default configuration (``max_seqs`` 16, ``all-vram``, buckets 1-16, chunk 512, 4096 tokens per sequence).
The runner changes nothing but the arm's own switch, and the box drives ``ContinuousScheduler`` directly.

Arms (``P109_ARM``):
  E  today's default: ``E4B_PAGED_GRAPHS`` unset. Eager decode, and the library's default (host) grouping.
  G  ``E4B_PAGED_GRAPHS=1``: bucketed CUDA-graph decode. ``build_engine`` switches on the batched lane's sync-free
     device grouping before it captures (``serve_paged._batched_graph_grouping``).
  D  eager decode with G's grouping forced before ``build_engine``: ``hot_residency.DEVICE_GROUPING`` on and
     ``FORCE_SINGLETON_GROUPS`` off, as ``_batched_graph_grouping`` sets them. This is the function G's replay must equal
     (P82's identity, here through the server's construction).

Each arm runs two workloads on the same engine:
  W16  the 16 prompt rows added at once;
  W1   row 0 alone.
Per workload and length (SHORT, LONG): one untimed warm pass, then REPS timed passes. Every request runs to
``max_new_tokens`` (``stop_ids`` None, the scheduler's length contract). ``decode_tok_s`` is
``B * (LONG - SHORT) / (min wall_LONG - min wall_SHORT)``, p37's slope: the prefill is in both walls and cancels.
Every timed pass's tokens are digested (rep-to-rep determinism); the last pass's tokens are recorded per row (the
identity axis).

``--prompts-only`` writes ``prompts.json``: 16 rows of PROMPT tokens of wikitext-2-raw test, row k from token k * 4096,
the corpus joined as P97's ``wikitext_windows`` joins it.
``--self-test`` checks the pure helpers on CPU.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROWS, PROMPT, OFFSET = 16, 512, 4096
ARMS = ("E", "G", "D")
WORKLOADS = {"W16": ROWS, "W1": 1}


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def slope(walls_short, walls_long, batch, short, long_):
    """p37's decode slope from the fastest pass at each length; the median pair beside it."""
    import statistics
    dmin = min(walls_long) - min(walls_short)
    dmed = statistics.median(walls_long) - statistics.median(walls_short)
    tok = batch * (long_ - short)
    if dmin <= 0 or dmed <= 0:
        return {"decode_tok_s": None, "decode_tok_s_median": None, "status": "void: LONG not slower than SHORT"}
    return {"decode_tok_s": round(tok / dmin, 2), "decode_tok_s_median": round(tok / dmed, 2),
            "decode_ms_per_step": round(1000 * dmin / (long_ - short), 4), "status": "ok"}


def wikitext_rows(tok, n=ROWS, prompt=PROMPT):
    """Row k is tokens [k * 4096, k * 4096 + prompt) of wikitext-2-raw test, joined as the K8 corpus joins it."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    out = []
    for k in range(n):
        w = ids[k * OFFSET:k * OFFSET + prompt]
        assert w.numel() == prompt, f"row {k} has {w.numel()} tokens"
        out.append([int(t) for t in w.tolist()])
    return out


def run_pass(parts, torch, rows, n_tokens):
    """Every row added at once, stepped to idle; exactly n_tokens per row and the prompt length asserted."""
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
            raise AssertionError(f"request {rid}: {len(q.out)} tokens from {q.prompt_len} prompt tokens, "
                                 f"expected {n_tokens} from {len(row)}")
    return wall, steps, [[int(t) for t in new[r].out] for r in rids]


def _mem(torch):
    free, total = torch.cuda.mem_get_info()
    return {"max_memory_allocated": int(torch.cuda.max_memory_allocated()), "memory_allocated": int(torch.cuda.memory_allocated()),
            "memory_reserved": int(torch.cuda.memory_reserved()), "free": int(free), "total": int(total)}


def _smi():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
                                        "--format=csv,noheader"], text=True, timeout=20).strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a: {e!r}"[:200]


def arm_main(a) -> int:
    arm = os.environ.get("P109_ARM", "")
    if arm not in ARMS:
        raise SystemExit(f"REFUSED: P109_ARM={arm!r}, expected one of {ARMS}")
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if digest(rows) != pf["prompts_sha256"] or len(rows) != ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    import torch
    from experts4bit_qlora.engines import hot_residency as hr
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    want_graphs = arm == "G"
    if cfg.graphs != want_graphs:
        raise SystemExit(f"REFUSED: arm {arm} with E4B_PAGED_GRAPHS -> graphs={cfg.graphs}")
    if (cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, "all-vram", (1, 2, 4, 8, 16)):
        raise SystemExit(f"REFUSED: not the default server: max_seqs={cfg.max_seqs} placement={cfg.placement} "
                         f"buckets={cfg.buckets}")
    if arm == "D":                       # G's grouping, without the graphs
        hr.DEVICE_GROUPING[0] = True
        hr.FORCE_SINGLETON_GROUPS[0] = False
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    info = parts.info
    rec = {"arm": arm, "tag": a.tag, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "model": cfg.model, "revision": cfg.revision, "torch": torch.__version__, "load_s": round(load_s, 2),
           "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items() if k != "token"},
           "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
           "grouping": info.get("grouping"),
           "grouping_flags_at_run": {"device_grouping": bool(hr.DEVICE_GROUPING[0]),
                                     "force_singleton_groups": bool(hr.FORCE_SINGLETON_GROUPS[0])},
           "census": {k: info.get(k) for k in ("moe_layers", "experts", "top_k", "model_type", "int4_expert_layers",
                                               "int4_attn_projections", "kv")},
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": _mem(torch), "workloads": {}, "status": "ok"}
    for wname, b in WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last = {}, {}, {}
        for n in (a.short, a.long):
            run_pass(parts, torch, wrows, n)                                   # warm, untimed
            ws, ds = [], []
            for _ in range(a.reps):
                wall, steps, toks = run_pass(parts, torch, wrows, n)
                ws.append(round(wall, 5))
                ds.append(digest(toks))
                last[str(n)] = toks
            walls[str(n)], digests[str(n)] = ws, ds
        s = slope(walls[str(a.short)], walls[str(a.long)], b, a.short, a.long)
        rec["workloads"][wname] = {"batch": b, "walls": walls, "rep_digests": digests, "tokens": last, **s}
        print(f"P109_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} median={s['decode_tok_s_median']} "
              f"walls={walls}", flush=True)
    rec["graph_stats"] = ({str(k): dict(v) for k, v in parts.runner.graph_stats.items()}
                          if isinstance(getattr(parts.runner, "graph_stats", None), dict) else None)
    rec["mem_after_runs"] = _mem(torch)
    rec["nvidia_smi"] = _smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P109_ARM " + json.dumps({"arm": arm, "tag": a.tag, "load_s": rec["load_s"], "graph_status": rec["graph_status"],
                                    "grouping": rec["grouping_flags_at_run"],
                                    "W16": rec["workloads"]["W16"]["decode_tok_s"], "W1": rec["workloads"]["W1"]["decode_tok_s"],
                                    "peak_gib": round(rec["mem_after_runs"]["max_memory_allocated"] / 2**30, 3)}), flush=True)
    return 0


def prompts_main(a) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision or None)
    rows = wikitext_rows(tok)
    json.dump({"model": a.model, "revision": a.revision, "rows": rows, "prompts_sha256": digest(rows),
               "rows_sha256": [digest(r) for r in rows], "offset": OFFSET, "prompt": PROMPT}, open(a.out, "w"))
    print(f"P109_PROMPTS rows={len(rows)} prompt={PROMPT} sha256={digest(rows)}", flush=True)
    return 0


def self_test() -> int:
    ok = []
    s = slope([1.0, 1.1], [3.0, 3.2], 16, 32, 160)
    ok.append(s["decode_tok_s"] == round(16 * 128 / 2.0, 2) and s["status"] == "ok")
    ok.append(s["decode_tok_s_median"] == round(16 * 128 / (3.1 - 1.05), 2))
    ok.append(slope([2.0], [2.0], 1, 32, 160)["status"].startswith("void"))
    ok.append(digest([[1, 2], [3]]) == digest([[1, 2], [3]]) != digest([[1, 2], [4]]))
    ok.append(WORKLOADS == {"W16": 16, "W1": 1} and ARMS == ("E", "G", "D"))
    n = len(ok)
    print(f"p109_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{n} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
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
