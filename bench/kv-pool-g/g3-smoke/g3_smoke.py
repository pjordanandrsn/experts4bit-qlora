"""bench/kv-pool-g/g3-smoke/g3_smoke.py -- lever G3's GPU smoke (bench/kv-pool-g/DESIGN.md, G3): the shared KV pool at
64 slots on the default decode bucket list, so a 64-row step runs as chained replays of the largest bucket (the
multi-piece path), driven through the real engine (``serve_paged.build_engine``) on one fixed schedule.

As in ``bench/kv-pool-g/g1-smoke/g1_smoke.py``, the engine is driven step by step in this process: request ``i`` is
added before step ``ARRIVE[i]``, the abort lands before step ``ABORT_AT``.

  run --out FILE [--short]          one run on the box (E4B_PAGED_* from the environment)
  compare A.json B.json             rc 0 IDENTICAL, 1 DIFFER, 2 unusable
  verdict --dir D [--out FILE]      the smoke's reading over the box's runs (below)
  --self-test

The box's runs (g3_smoke_run.sh), each at 64 slots on the default bucket list: ``ca_a``, ``sh``, ``ca_b`` (lookahead
off), ``ca_la1``, ``sh_la1`` (lookahead on), and ``sh_short``: shared, with the pool sized from a free-memory reading cut
so the arithmetic gives ``least_rows + (claim_all_rows - least_rows) // 8`` rows (``resolve_kv_pool`` runs unchanged on
that reading), so the pool is physically smaller than its slots can reach and admission waits on its rows.

The reading:
- FLOOR: ``ca_a`` against ``ca_b``. If they differ, INCONCLUSIVE (an identity bar cannot be read), never FAIL.
- G3 identities: ``sh`` against ``ca_a`` and ``sh_la1`` against ``ca_la1``, IDENTICAL token for token and finish for
  finish. Their pools come from the box's real free memory; on this model they hold the claim-all size.
- RUN gates, per run: incremental claims exactly in the shared runs; the decode graphs replayed; a step carried more
  decode rows than the largest captured bucket (the multi-piece path ran); every request finished; at idle no real slot
  holds a row.
- SHORT: ``sh_short``'s pool rows are below the claim-all size, the scheduler's queue head waited on blocks, every
  request finished, no row held at idle. Its tokens are not compared (admission waits, so the batches differ).

PASS needs FLOOR, both identities, every RUN gate and SHORT. No throughput is read.
"""
import argparse
import hashlib
import json
import os
import sys

# 96 requests on 64 slots: long reservations (prompt + max_new up to ~2,000 tokens), so a pool near its floor of 1,024
# tokens a slot cannot hold 64 of them at once. Prompt lengths straddle the block edges and the 512-token chunk.
LENS = [5, 16, 31, 32, 33, 47, 48, 64, 100, 127, 128, 129, 255, 256, 257, 400, 511, 512, 513, 300]
MAX_NEW = [1024, 1536, 1800, 1700, 17, 1200, 1600, 900]
N = 96
ARRIVE = [0] * 64 + [2 * (i - 63) for i in range(64, N)]     # 64 at once, then one every 2 steps
ABORT_RID, ABORT_AT = 10, 60
STOP_EVERY = 4
SLOTS = 64
MAX_STEPS = 400_000

PARAS = [
    "The station sits at the end of a long gravel road, past two cattle guards and a dry creek that floods twice a "
    "year. Its instruments record wind, rain, pressure and the temperature of the soil at three depths, and every "
    "five minutes a small radio sends the readings to a house a mile away, where a computer keeps them.",
    "A paged cache stores each sequence's keys and values in fixed blocks of sixteen tokens. A table maps every "
    "position to a block, so two sequences can share one pool without either knowing where the other's rows live, "
    "and a block freed by a finished request is handed to the next one that needs it.",
    "In the spring the creek runs clear and cold, and the herons come back to the cottonwoods along its banks. By "
    "July it is a string of shallow pools, then cracked mud, and the herons move on to the river. The station "
    "records all of it, a number at a time, without comment.",
    "When a request arrives, the scheduler checks whether its prompt and its longest answer fit in the blocks that are "
    "not yet promised to anyone. If they do, it admits the request and reserves them; if not, the request waits at "
    "the head of the queue until enough blocks come back.",
]


def lens_of(i):
    n = LENS[i % len(LENS)]
    return n, min(MAX_NEW[i % len(MAX_NEW)], 2047 - n)


def text_for(i: int) -> str:
    out, k = [], 0
    while len(out) < 40:
        out.append(f"{k + 1}. {PARAS[(i + k) % len(PARAS)]}")
        k += 1
    return "\n\n".join(out)


def workload(encode, eos_ids, stop_extra):
    reqs = []
    for i in range(N):
        n, m = lens_of(i)
        ids = list(encode(text_for(i)))
        assert len(ids) >= n, f"request {i}: the text encodes to {len(ids)} ids, fewer than {n}"
        stops = set(int(t) for t in eos_ids)
        if i % STOP_EVERY == 1 and stop_extra is not None:
            stops.add(int(stop_extra))
        reqs.append((ids[:n], m, sorted(stops)))
    return reqs


def digest(reqs) -> str:
    h = hashlib.sha256()
    for p, m, s in reqs:
        h.update(json.dumps([p, m, s], separators=(",", ":")).encode())
    h.update(json.dumps([ARRIVE, ABORT_RID, ABORT_AT]).encode())
    return h.hexdigest()


def rows_held(kv) -> int:
    skip = set(getattr(kv, "scratch", ())) | set(getattr(kv, "alias", ()) or ())
    return sum(len(v) for (_, slot), v in kv._rows.items() if slot not in skip)


def drive(sched, reqs, kv=None):
    """Run the schedule to idle. Returns (records by request index, steps, peak rows, most decode rows in one step)."""
    rid_of, step, peak, wide, nxt = {}, 0, 0, 0, 0
    while True:
        while nxt < len(reqs) and ARRIVE[nxt] <= step:
            p, m, s = reqs[nxt]
            rid_of[nxt] = sched.add_request(p, max_new_tokens=m, now=float(step), stop_ids=s)
            nxt += 1
        if step == ABORT_AT and ABORT_RID in rid_of:
            sched.abort(rid_of[ABORT_RID])
        if nxt >= len(reqs) and not sched.queue and not sched.active:
            break
        plan = sched.step()
        wide = max(wide, len(getattr(plan, "decode", ()) or ()))
        step += 1
        if kv is not None:
            peak = max(peak, rows_held(kv))
        if step > MAX_STEPS:
            raise RuntimeError(f"the schedule did not drain in {MAX_STEPS} steps")
    by_rid = {r.rid: r for r in list(sched.done) + list(sched.aborted)}
    recs = []
    for i in range(len(reqs)):
        r = by_rid.get(rid_of.get(i))
        recs.append(None if r is None else {"i": i, "prompt_len": len(reqs[i][0]), "max_new": reqs[i][1],
                                            "out": [int(t) for t in r.out], "finish": r.finish_reason})
    return recs, step, peak, wide


def _short_resolver(sp, sr):
    """``resolve_kv_pool`` on a free-memory reading cut so the arithmetic gives ``least + (full - least) // 8`` rows at
    the configured width: the server's own resolver, fed a smaller number."""
    from dataclasses import replace
    orig = sp.resolve_kv_pool

    def short(cfg, model_config, free):
        topo, setup, _ = sp._sizing_inputs(cfg, model_config)
        s = replace(setup, max_seqs=cfg.max_seqs)
        probe = sr.size_shared_pool(topo, s, free, widths=(cfg.max_seqs,))
        c = probe["candidates"][0]
        target = c["least_rows"] + (c["claim_all_rows"] - c["least_rows"]) // 8
        fp = sr.estimate_serve_footprint(topo, replace(s, kv_pool_rows=target))
        cut = (fp.device_bytes + c["prefill_graph_reserve_bytes"] + sp.int8_head_copy_bytes(cfg, model_config)
               + sr.MAX_SEQS_AUTO_MARGIN_BYTES)
        rec = orig(cfg, model_config, min(int(free), int(cut)))
        rec["smoke_free_reading_cut"] = {"real_free_bytes": int(free), "cut_to_bytes": int(cut), "target_rows": target}
        return rec
    return short


def run(args) -> int:
    from experts4bit_qlora import serve_paged as sp
    from experts4bit_qlora import serve_recipe as sr
    import experts4bit_qlora as e4b
    if args.short:
        sp.resolve_kv_pool = _short_resolver(sp, sr)
    cfg = sp.PagedServeConfig.from_env()
    cfg.validate()
    parts = sp.build_engine(cfg)
    sched, runner, tok = parts.scheduler, parts.runner, parts.tokenizer
    kv = runner.kv
    dot = tok.encode(".", add_special_tokens=False)
    reqs = workload(lambda t: tok.encode(t, add_special_tokens=False), parts.eos_ids, dot[-1] if dot else None)
    recs, steps, peak, wide = drive(sched, reqs, kv)
    gs = {str(k): dict(v) for k, v in (getattr(runner, "graph_stats", None) or {}).items()}
    out = {
        "pool": cfg.kv_pool, "lookahead": bool(cfg.decode_lookahead), "short": bool(args.short),
        "max_seqs": cfg.max_seqs, "max_tokens_per_seq": cfg.max_tokens_per_seq, "buckets": list(cfg.buckets),
        "graphs": bool(cfg.graphs), "bulk_kv": bool(cfg.bulk_kv), "model": cfg.model, "revision": cfg.revision,
        "e4b_version": getattr(e4b, "__version__", None),
        "incremental_claims": bool(getattr(runner, "incremental_claims", False)),
        "pool_rows": kv._n_rows, "claim_all_rows": getattr(kv, "claim_all_rows", None), "scratch_rows": kv.n_scratch,
        "kv_blocks": sched.kv_blocks, "block_tokens": kv.bt, "blocks_per_seq": kv.blocks_per_seq,
        "kv_pool_resolution": cfg.kv_pool_resolution or None,
        "workload_sha256": digest(reqs), "steps": steps, "rows_peak": peak, "rows_end": rows_held(kv),
        "widest_decode_step": wide, "replays": sum(v.get("replays", 0) for v in gs.values()), "graph_stats": gs,
        "scheduler": sched.stats(), "requests": recs,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    done = sum(1 for r in recs if r is not None)
    print(f"G3SMOKE run pool={out['pool']} lookahead={int(out['lookahead'])} short={int(out['short'])} "
          f"slots={cfg.max_seqs} buckets={out['buckets']} pool_rows={out['pool_rows']}/{out['claim_all_rows']} "
          f"steps={steps} requests={done}/{len(recs)} widest_step={wide} replays={out['replays']} rows_peak={peak} "
          f"rows_end={out['rows_end']} waits={(out['scheduler'] or {}).get('kv_blocks_waits')} "
          f"tokens={sum(len(r['out']) for r in recs if r)}")
    return 0


def compare(a: dict, b: dict):
    if a.get("workload_sha256") != b.get("workload_sha256") or not a.get("workload_sha256"):
        return 2, "unusable: the two runs did not serve the same workload"
    ra, rb = a.get("requests") or [], b.get("requests") or []
    if len(ra) != len(rb) or not ra:
        return 2, f"unusable: {len(ra)} against {len(rb)} requests"
    for x, y in zip(ra, rb):
        if x is None or y is None:
            return 2, "unusable: a request did not finish"
        if x["finish"] != y["finish"]:
            return 1, f"DIFFER request {x['i']}: finish {x['finish']} against {y['finish']}"
        if x["out"] != y["out"]:
            k = next((j for j, (p, q) in enumerate(zip(x["out"], y["out"])) if p != q), min(len(x["out"]), len(y["out"])))
            return 1, (f"DIFFER request {x['i']}: token {k} of {len(x['out'])}/{len(y['out'])} "
                       f"({x['out'][k] if k < len(x['out']) else '-'} against {y['out'][k] if k < len(y['out']) else '-'})")
    return 0, f"IDENTICAL {len(ra)} requests, {sum(len(x['out']) for x in ra)} tokens"


RUNS = ("ca_a", "sh", "ca_b", "ca_la1", "sh_la1", "sh_short")


def run_bad(name: str, r: dict):
    if r is None:
        return f"{name}: missing"
    shared = name.startswith("sh")
    if bool(r.get("incremental_claims")) != shared or (r.get("pool") == "shared") != shared:
        return f"{name}: pool {r.get('pool')!r}, incremental_claims {r.get('incremental_claims')}"
    if bool(r.get("short")) != (name == "sh_short"):
        return f"{name}: short is {r.get('short')}"
    if name.endswith("la1") != bool(r.get("lookahead")):
        return f"{name}: lookahead is {r.get('lookahead')}"
    if r.get("max_seqs") != SLOTS:
        return f"{name}: {r.get('max_seqs')} slots, not {SLOTS}"
    if not r.get("graphs") or not r.get("replays"):
        return f"{name}: the decode graphs did not replay"
    if not r.get("buckets") or (r.get("widest_decode_step") or 0) <= max(r["buckets"]):
        return (f"{name}: no step carried more decode rows than the largest bucket "
                f"({r.get('widest_decode_step')} against {max(r.get('buckets') or [0])}): the multi-piece path did not run")
    reqs = r.get("requests") or []
    if not reqs or any(x is None for x in reqs):
        return f"{name}: a request did not finish"
    if r.get("rows_end"):
        return f"{name}: {r['rows_end']} rows still held at idle"
    return None


def verdict(runs: dict) -> dict:
    out = {"runs": {k: (None if runs.get(k) is None else
                        {f: runs[k].get(f) for f in ("steps", "replays", "rows_peak", "pool_rows", "claim_all_rows",
                                                     "widest_decode_step")}) for k in RUNS}}
    bad = [b for b in (run_bad(k, runs.get(k)) for k in RUNS) if b]
    out["run_gates"] = bad or "ok"
    rc_f, line_f = compare(runs.get("ca_a") or {}, runs.get("ca_b") or {})
    out["floor"] = line_f
    ids = {f"{sh}~{ca}": compare(runs.get(sh) or {}, runs.get(ca) or {}) for sh, ca in (("sh", "ca_a"), ("sh_la1", "ca_la1"))}
    out["g3"] = {k: v[1] for k, v in ids.items()}
    t = runs.get("sh_short") or {}
    waits = (t.get("scheduler") or {}).get("kv_blocks_waits") or 0
    short_ok = (run_bad("sh_short", runs.get("sh_short")) is None and waits > 0
                and (t.get("pool_rows") or 0) < (t.get("claim_all_rows") or 0))
    out["short"] = (f"{'ok' if short_ok else 'BAD'}: pool_rows={t.get('pool_rows')} of {t.get('claim_all_rows')}, "
                    f"kv_blocks_waits={waits}")
    if bad or any(rc == 2 for rc, _ in ids.values()) or rc_f == 2:
        out["verdict"] = "FAIL" if bad else "UNREAD"
    elif any(rc == 1 for rc, _ in ids.values()):
        out["verdict"] = "INCONCLUSIVE" if rc_f == 1 else "FAIL"
    elif rc_f == 1:
        out["verdict"] = "INCONCLUSIVE"
    elif not short_ok:
        out["verdict"] = "FAIL"
    else:
        out["verdict"] = "PASS"
    return out


def _fake(pool, la, short=False, outs=None, **kw):
    reqs = [{"i": i, "prompt_len": 4, "max_new": 3, "out": list(o), "finish": "length"}
            for i, o in enumerate(outs or ([1, 2, 3], [4, 5, 6]))]
    r = {"pool": pool, "lookahead": la, "short": short, "graphs": True, "replays": 9, "rows_end": 0, "max_seqs": SLOTS,
         "buckets": [1, 2, 4, 8, 16], "widest_decode_step": 64, "incremental_claims": pool == "shared",
         "workload_sha256": "w", "requests": reqs, "steps": 5, "rows_peak": 8,
         "pool_rows": 5000 if short else 8208, "claim_all_rows": 8208,
         "scheduler": {"kv_blocks_waits": 3 if short else 0}}
    r.update(kw)
    return r


def _all(**over):
    runs = {"ca_a": _fake("claim-all", False), "sh": _fake("shared", False), "ca_b": _fake("claim-all", False),
            "ca_la1": _fake("claim-all", True), "sh_la1": _fake("shared", True), "sh_short": _fake("shared", False, True)}
    runs.update(over)
    return runs


def self_test() -> int:
    cases = [
        ("all identical", _all(), "PASS"),
        ("shared differs, floor holds", _all(sh=_fake("shared", False, outs=([1, 2, 9], [4, 5, 6]))), "FAIL"),
        ("lookahead pair differs", _all(sh_la1=_fake("shared", True, outs=([1, 2, 3], [4, 9, 6]))), "FAIL"),
        ("floor differs", _all(ca_b=_fake("claim-all", False, outs=([1, 2, 3], [4, 5, 7]))), "INCONCLUSIVE"),
        ("no multi-piece step", _all(sh=_fake("shared", False, widest_decode_step=16)), "FAIL"),
        ("not 64 slots", _all(ca_a=_fake("claim-all", False, max_seqs=16)), "FAIL"),
        ("short pool at claim-all size", _all(sh_short=_fake("shared", False, True, pool_rows=8208)), "FAIL"),
        ("short never waited", _all(sh_short=_fake("shared", False, True, scheduler={"kv_blocks_waits": 0})), "FAIL"),
        ("rows held at idle", _all(sh=_fake("shared", False, rows_end=4)), "FAIL"),
        ("shared run not incremental", _all(sh_la1=_fake("shared", True, incremental_claims=False)), "FAIL"),
        ("a run missing", _all(ca_la1=None), "FAIL"),
        ("workload differs", _all(sh=_fake("shared", False, workload_sha256="x")), "UNREAD"),
    ]
    fails = 0
    for name, runs, want in cases:
        got = verdict(runs)["verdict"]
        fails += got != want
        print(f"G3SMOKE self-test {'ok ' if got == want else 'BAD'} {name}: {got} (want {want})")
    assert len(ARRIVE) == N and ARRIVE == sorted(ARRIVE) and ARRIVE.count(0) == SLOTS
    assert all(sum(lens_of(i)) < 2048 for i in range(N))
    # the first 64 requests reserve more than a pool at the floor (1,024 tokens a slot) holds, so sh_short waits
    assert sum(-(-sum(lens_of(i)) // 16) for i in range(SLOTS)) > SLOTS * 1024 // 16 + SLOTS * 1024 // 16 // 8
    print(f"G3SMOKE self-test {'PASS' if not fails else 'FAIL'} ({len(cases) - fails}/{len(cases)})")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("run", "compare", "verdict"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--dir")
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd == "run":
        if not a.out:
            ap.error("run needs --out")
        return run(a)
    if a.cmd == "compare":
        if len(a.files) != 2:
            ap.error("compare takes two files")
        x, y = (json.load(open(f)) for f in a.files)
        rc, line = compare(x, y)
        print(f"G3SMOKE compare {os.path.basename(a.files[0])} {os.path.basename(a.files[1])}: {line}")
        return rc
    if a.cmd == "verdict":
        if not a.dir:
            ap.error("verdict needs --dir")
        runs = {k: (json.load(open(p)) if os.path.exists(p := os.path.join(a.dir, f"g3_{k}.json")) else None) for k in RUNS}
        v = verdict(runs)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(v, f, indent=1)
        for k in ("run_gates", "floor", "g3", "short"):
            print(f"G3SMOKE {k}: {v[k]}")
        print(f"G3SMOKE VERDICT {v['verdict']}")
        return 0 if v["verdict"] == "PASS" else 1
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    sys.exit(main())
