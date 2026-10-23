"""bench/kv-pool-g/g1-smoke/g1_smoke.py -- lever G1's GPU smoke (bench/kv-pool-g/DESIGN.md, "GPU"): one workload on one
fixed schedule, served graphed through the real engine (``serve_paged.build_engine``) under ``E4B_PAGED_KV_POOL=claim-all``
and ``=shared``; every token and finish reason must match.

The engine is driven step by step in this process, never over HTTP, so the schedule (arrival step, abort step, the
order of ``add_request``) is the same in every run by construction: request ``i`` is added before step ``ARRIVE[i]``,
and the abort lands before step ``ABORT_AT``.

  run --out FILE [--tight]          one run on the box (E4B_PAGED_* from the environment, E4B_PAGED_KV_POOL included)
  compare A.json B.json             rc 0 IDENTICAL, 1 DIFFER, 2 unusable
  verdict --dir D [--out FILE]      the smoke's reading over the box's runs (below)
  --self-test

The box's runs (g1_smoke_run.sh): ``ca_la0_a``, ``sh_la0``, ``ca_la0_b`` (lookahead off), ``ca_la1``, ``sh_la1``
(lookahead on), and ``sh_tight`` (shared, the scheduler's reservation cut to ``TIGHT_BLOCKS`` so admission waits).

The reading:
- FLOOR: ``ca_la0_a`` against ``ca_la0_b``, the same pool on the same schedule. If they differ, the engine is not run to
  run reproducible on this schedule and the smoke is INCONCLUSIVE (never FAIL): an identity bar cannot be read.
- G1: ``sh_la0`` against ``ca_la0_a``, and ``sh_la1`` against ``ca_la1``, must be IDENTICAL.
- RUN gates, per run: the decode graphs replayed; ``incremental_claims`` is on exactly in the shared runs; every request
  finished; at idle no real slot holds a row.
- TIGHT: every request of ``sh_tight`` finished, with waits for blocks counted. Its tokens are not compared: the
  admission order differs from claim-all's by design, so the batches differ.
- Reported: rows held at the peak, per run (shared holds fewer by design).

PASS needs FLOOR, both G1 identities, every RUN gate and TIGHT. A G1 or RUN or TIGHT failure is FAIL.
"""
import argparse
import hashlib
import json
import os
import sys

# The workload: 28 requests on 16 slots, so admission queues. Prompt lengths straddle the 16-token block edges (and the
# 512-token chunk); prompt + max_new hits a block edge for several; one request is aborted mid-decode; a quarter carry a
# stop id besides EOS.
LENS = [5, 16, 31, 32, 33, 47, 48, 64, 100, 127, 128, 129, 255, 256, 257, 400, 511, 512, 513, 700, 1000, 15, 17, 1,
        80, 96, 300, 640]
MAX_NEW = [1, 2, 15, 16, 17, 31, 32, 33, 64, 128, 200, 300]
N = len(LENS)
ARRIVE = [0] * 8 + [3 * (i - 7) for i in range(8, N)]       # 8 at once, then one every 3 steps
ABORT_RID, ABORT_AT = 10, 60                                # 128 + 200, arriving at step 9
STOP_EVERY = 4                                               # i % 4 == 1: EOS plus the "." token
TIGHT_BLOCKS = 160                                           # sh_tight's reservation (16-token blocks); >= every need
MAX_STEPS = 200_000

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


def text_for(i: int) -> str:
    """Request i's source text: the four paragraphs in a rotation of their own, numbered, repeated until long enough."""
    out, k = [], 0
    while len(out) < 40:
        p = PARAS[(i + k) % len(PARAS)]
        out.append(f"{k + 1}. {p}")
        k += 1
    return "\n\n".join(out)


def workload(encode, eos_ids, stop_extra):
    """The requests as (prompt ids, max_new, stop ids). ``encode`` maps text to ids; every prompt is cut to its length."""
    reqs = []
    for i, n in enumerate(LENS):
        ids = list(encode(text_for(i)))
        assert len(ids) >= n, f"request {i}: the text encodes to {len(ids)} ids, fewer than {n}"
        stops = set(int(t) for t in eos_ids)
        if i % STOP_EVERY == 1 and stop_extra is not None:
            stops.add(int(stop_extra))
        reqs.append((ids[:n], MAX_NEW[i % len(MAX_NEW)], sorted(stops)))
    return reqs


def digest(reqs) -> str:
    h = hashlib.sha256()
    for p, m, s in reqs:
        h.update(json.dumps([p, m, s], separators=(",", ":")).encode())
    h.update(json.dumps([ARRIVE, ABORT_RID, ABORT_AT]).encode())
    return h.hexdigest()


def rows_held(kv) -> int:
    """Rows real slots hold, over every pool layer (scratch and alias slots excluded)."""
    skip = set(getattr(kv, "scratch", ())) | set(getattr(kv, "alias", ()) or ())
    return sum(len(v) for (_, slot), v in kv._rows.items() if slot not in skip)


def drive(sched, reqs, kv=None):
    """Run the schedule to idle. Returns (records by request index, steps, peak rows)."""
    rid_of, step, peak, nxt = {}, 0, 0, 0
    while True:
        while nxt < len(reqs) and ARRIVE[nxt] <= step:
            p, m, s = reqs[nxt]
            rid_of[nxt] = sched.add_request(p, max_new_tokens=m, now=float(step), stop_ids=s)
            nxt += 1
        if step == ABORT_AT and ABORT_RID in rid_of:
            sched.abort(rid_of[ABORT_RID])
        if nxt >= len(reqs) and not sched.queue and not sched.active:
            break
        sched.step()
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
    return recs, step, peak


def run(args) -> int:
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    import experts4bit_qlora as e4b
    cfg = PagedServeConfig.from_env()
    cfg.validate()
    parts = build_engine(cfg)
    sched, runner, tok = parts.scheduler, parts.runner, parts.tokenizer
    kv = runner.kv
    if args.tight:
        if cfg.kv_pool != "shared":
            print("G1SMOKE refusing: --tight needs E4B_PAGED_KV_POOL=shared", file=sys.stderr)
            return 2
        sched.kv_blocks = TIGHT_BLOCKS
    dot = tok.encode(".", add_special_tokens=False)
    reqs = workload(lambda t: tok.encode(t, add_special_tokens=False), parts.eos_ids, dot[-1] if dot else None)
    recs, steps, peak = drive(sched, reqs, kv)
    gs = {str(k): dict(v) for k, v in (getattr(runner, "graph_stats", None) or {}).items()}
    out = {
        "pool": cfg.kv_pool, "lookahead": bool(cfg.decode_lookahead), "tight": bool(args.tight),
        "max_seqs": cfg.max_seqs, "max_tokens_per_seq": cfg.max_tokens_per_seq, "buckets": list(cfg.buckets),
        "graphs": bool(cfg.graphs), "bulk_kv": bool(cfg.bulk_kv), "model": cfg.model, "revision": cfg.revision,
        "e4b_version": getattr(e4b, "__version__", None),
        "incremental_claims": bool(getattr(runner, "incremental_claims", False)),
        "kv_blocks": sched.kv_blocks, "block_tokens": kv.bt, "blocks_per_seq": kv.blocks_per_seq,
        "workload_sha256": digest(reqs), "steps": steps, "rows_peak": peak, "rows_end": rows_held(kv),
        "replays": sum(v.get("replays", 0) for v in gs.values()), "graph_stats": gs,
        "scheduler": sched.stats(), "requests": recs,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    done = sum(1 for r in recs if r is not None)
    print(f"G1SMOKE run pool={out['pool']} lookahead={int(out['lookahead'])} tight={int(out['tight'])} steps={steps} "
          f"requests={done}/{len(recs)} replays={out['replays']} rows_peak={peak} rows_end={out['rows_end']} "
          f"tokens={sum(len(r['out']) for r in recs if r)}")
    return 0


def compare(a: dict, b: dict):
    """(rc, line): 0 IDENTICAL, 1 DIFFER (first divergence named), 2 unusable."""
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
    n = sum(len(x["out"]) for x in ra)
    return 0, f"IDENTICAL {len(ra)} requests, {n} tokens"


def run_bad(name: str, r: dict):
    """The RUN gates. None when the run passes, else why not."""
    if r is None:
        return f"{name}: missing"
    want_inc = name.startswith("sh_")
    if bool(r.get("incremental_claims")) != want_inc:
        return f"{name}: incremental_claims is {r.get('incremental_claims')}, expected {want_inc}"
    if (r.get("pool") == "shared") != want_inc:
        return f"{name}: pool {r.get('pool')!r}"
    if bool(r.get("tight")) != (name == "sh_tight"):
        return f"{name}: tight is {r.get('tight')}"
    if name.endswith("la1") != bool(r.get("lookahead")) and name != "sh_tight":
        return f"{name}: lookahead is {r.get('lookahead')}"
    if not r.get("graphs") or not r.get("replays"):
        return f"{name}: the decode graphs did not replay (graphs={r.get('graphs')}, replays={r.get('replays')})"
    reqs = r.get("requests") or []
    if not reqs or any(x is None for x in reqs):
        return f"{name}: a request did not finish"
    if r.get("rows_end"):
        return f"{name}: {r['rows_end']} rows still held at idle"
    return None


RUNS = ("ca_la0_a", "sh_la0", "ca_la0_b", "ca_la1", "sh_la1", "sh_tight")


def verdict(runs: dict) -> dict:
    out = {"runs": {k: (None if runs.get(k) is None else {"steps": runs[k].get("steps"), "replays": runs[k].get("replays"),
                                                            "rows_peak": runs[k].get("rows_peak")}) for k in RUNS}}
    bad = [b for b in (run_bad(k, runs.get(k)) for k in RUNS) if b]
    out["run_gates"] = bad or "ok"
    rc_f, line_f = compare(runs.get("ca_la0_a") or {}, runs.get("ca_la0_b") or {})
    out["floor"] = line_f
    ids = {}
    for sh, ca in (("sh_la0", "ca_la0_a"), ("sh_la1", "ca_la1")):
        rc, line = compare(runs.get(sh) or {}, runs.get(ca) or {})
        ids[f"{sh}~{ca}"] = (rc, line)
    out["g1"] = {k: v[1] for k, v in ids.items()}
    t = runs.get("sh_tight") or {}
    waits = ((t.get("scheduler") or {}).get("kv_blocks_waits")) or 0
    tight_ok = run_bad("sh_tight", runs.get("sh_tight")) is None and waits > 0
    out["tight"] = f"{'ok' if tight_ok else 'BAD'}: kv_blocks_waits={waits}, kv_blocks={t.get('kv_blocks')}"
    if bad or any(rc == 2 for rc, _ in ids.values()) or rc_f == 2:
        out["verdict"] = "FAIL" if bad else "UNREAD"
    elif any(rc == 1 for rc, _ in ids.values()):
        out["verdict"] = "INCONCLUSIVE" if rc_f == 1 else "FAIL"
    elif rc_f == 1:
        out["verdict"] = "INCONCLUSIVE"
    elif not tight_ok:
        out["verdict"] = "FAIL"
    else:
        out["verdict"] = "PASS"
    return out


def _fake(pool, la, tight=False, outs=None, **kw):
    reqs = [{"i": i, "prompt_len": 4, "max_new": 3, "out": list(o), "finish": "length"}
            for i, o in enumerate(outs or ([1, 2, 3], [4, 5, 6]))]
    r = {"pool": pool, "lookahead": la, "tight": tight, "graphs": True, "replays": 9, "rows_end": 0,
         "incremental_claims": pool == "shared", "workload_sha256": "w", "requests": reqs, "steps": 5, "rows_peak": 8,
         "scheduler": {"kv_blocks_waits": 3 if tight else 0}, "kv_blocks": 160 if tight else None}
    r.update(kw)
    return r


def _all(**over):
    runs = {"ca_la0_a": _fake("claim-all", False), "sh_la0": _fake("shared", False), "ca_la0_b": _fake("claim-all", False),
            "ca_la1": _fake("claim-all", True), "sh_la1": _fake("shared", True), "sh_tight": _fake("shared", False, True)}
    runs.update(over)
    return runs


def self_test() -> int:
    cases = [
        ("all identical", _all(), "PASS"),
        ("shared differs, floor holds", _all(sh_la0=_fake("shared", False, outs=([1, 2, 9], [4, 5, 6]))), "FAIL"),
        ("lookahead pair differs", _all(sh_la1=_fake("shared", True, outs=([1, 2, 3], [4, 9, 6]))), "FAIL"),
        ("floor differs", _all(ca_la0_b=_fake("claim-all", False, outs=([1, 2, 3], [4, 5, 7]))), "INCONCLUSIVE"),
        ("floor and shared differ", _all(ca_la0_b=_fake("claim-all", False, outs=([1, 2, 7], [4, 5, 6])),
                                         sh_la0=_fake("shared", False, outs=([1, 2, 8], [4, 5, 6]))), "INCONCLUSIVE"),
        ("no replays", _all(sh_la0=_fake("shared", False, replays=0)), "FAIL"),
        ("shared run not incremental", _all(sh_la1=_fake("shared", True, incremental_claims=False)), "FAIL"),
        ("rows held at idle", _all(sh_la0=_fake("shared", False, rows_end=4)), "FAIL"),
        ("tight never waited", _all(sh_tight=_fake("shared", False, True, scheduler={"kv_blocks_waits": 0})), "FAIL"),
        ("a run missing", _all(ca_la1=None), "FAIL"),
        ("workload differs", _all(sh_la0=_fake("shared", False, workload_sha256="x")), "UNREAD"),
        ("finish differs", _all(sh_la0=_fake("shared", False, requests=[
            {"i": 0, "prompt_len": 4, "max_new": 3, "out": [1, 2, 3], "finish": "stop"},
            {"i": 1, "prompt_len": 4, "max_new": 3, "out": [4, 5, 6], "finish": "length"}])), "FAIL"),
        ("a shorter stream", _all(sh_la0=_fake("shared", False, outs=([1, 2], [4, 5, 6]))), "FAIL"),
    ]
    fails = 0
    for name, runs, want in cases:
        got = verdict(runs)["verdict"]
        ok = got == want
        fails += not ok
        print(f"G1SMOKE self-test {'ok ' if ok else 'BAD'} {name}: {got} (want {want})")
    # the workload: arrivals and lengths line up, and some requests end on a block edge
    assert len(ARRIVE) == len(LENS) == N and ARRIVE == sorted(ARRIVE)
    assert any((n + MAX_NEW[i % len(MAX_NEW)]) % 16 == 0 for i, n in enumerate(LENS))
    assert max(n + MAX_NEW[i % len(MAX_NEW)] for i, n in enumerate(LENS)) < 2048
    print(f"G1SMOKE self-test {'PASS' if not fails else 'FAIL'} ({len(cases) - fails}/{len(cases)})")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("run", "compare", "verdict"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--dir")
    ap.add_argument("--tight", action="store_true")
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
        print(f"G1SMOKE compare {os.path.basename(a.files[0])} {os.path.basename(a.files[1])}: {line}")
        return rc
    if a.cmd == "verdict":
        if not a.dir:
            ap.error("verdict needs --dir")
        runs = {}
        for k in RUNS:
            p = os.path.join(a.dir, f"g1_{k}.json")
            runs[k] = json.load(open(p)) if os.path.exists(p) else None
        v = verdict(runs)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(v, f, indent=1)
        for k in ("run_gates", "floor", "g1", "tight"):
            print(f"G1SMOKE {k}: {v[k]}")
        print(f"G1SMOKE VERDICT {v['verdict']}")
        return 0 if v["verdict"] == "PASS" else 1
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    sys.exit(main())
