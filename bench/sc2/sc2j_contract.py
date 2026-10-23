#!/usr/bin/env python3
"""sc2j_contract.py -- lane SC2j's contract arm (bench/sc2/SC2j-PREREG.md, gate 5): at an EXPLICIT 32 slots, the shared
KV pool emits every token and finish reason claim-all emits, on one fixed schedule, on the box's model and stack.

The engine is driven step by step in this process (``serve_paged.build_engine``), never over HTTP, with the G3 GPU
smoke's workload, schedule and drive (``g3_smoke.py``, staged beside this file): request ``i`` is added before step
``ARRIVE[i]``, the abort lands before step ``ABORT_AT``. One engine per process; ``E4B_PAGED_*`` from the environment.

  run --out FILE                     one run (the box sets E4B_PAGED_KV_POOL and E4B_PAGED_MAX_SEQS=32)
  verdict --dir D [--out FILE]       the gate over ca_a, sh, ca_b
  --self-test

The gate (CONTRACT): every run at 32 slots, with incremental claims exactly in ``sh``, every request finished and no
real slot holding a row at idle; ``sh`` never waited on blocks (``kv_blocks_waits`` 0: the schedule covers the claim
timing, not admission under waits); the floor (``ca_a`` against ``ca_b``) IDENTICAL; ``sh`` against ``ca_a`` IDENTICAL.
OK, or VOID with the reason (a floor that differs makes the contract unreadable, so it is VOID too).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g3_smoke as g3  # noqa: E402

SLOTS = 32
RUNS = ("ca_a", "sh", "ca_b")


def run(args) -> int:
    from experts4bit_qlora import serve_paged as sp
    import experts4bit_qlora as e4b
    cfg = sp.PagedServeConfig.from_env()
    cfg.validate()
    parts = sp.build_engine(cfg)
    sched, runner, tok = parts.scheduler, parts.runner, parts.tokenizer
    kv = runner.kv
    dot = tok.encode(".", add_special_tokens=False)
    reqs = g3.workload(lambda t: tok.encode(t, add_special_tokens=False), parts.eos_ids, dot[-1] if dot else None)
    recs, steps, peak, wide = g3.drive(sched, reqs, kv)
    st = sched.stats()
    out = {"pool": cfg.kv_pool, "max_seqs": cfg.max_seqs, "max_tokens_per_seq": cfg.max_tokens_per_seq,
           "buckets": list(cfg.buckets), "e4b_version": getattr(e4b, "__version__", None),
           "incremental_claims": bool(getattr(runner, "incremental_claims", False)),
           "pool_rows": kv._n_rows, "claim_all_rows": getattr(kv, "claim_all_rows", None),
           "kv_blocks": sched.kv_blocks, "kv_blocks_waits": st.get("kv_blocks_waits"),
           "kv_pool_resolution": cfg.kv_pool_resolution or None, "workload_sha256": g3.digest(reqs),
           "steps": steps, "rows_peak": peak, "rows_end": g3.rows_held(kv), "widest_decode_step": wide,
           "requests": recs}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"SC2J_CONTRACT run pool={out['pool']} slots={out['max_seqs']} pool_rows={out['pool_rows']}/"
          f"{out['claim_all_rows']} steps={steps} requests={sum(1 for r in recs if r)}/{len(recs)} "
          f"waits={out['kv_blocks_waits']} rows_end={out['rows_end']}", flush=True)
    return 0


def run_bad(name: str, r):
    if r is None:
        return f"{name}: missing"
    shared = name == "sh"
    if (r.get("pool") == "shared") != shared or bool(r.get("incremental_claims")) != shared:
        return f"{name}: pool {r.get('pool')!r}, incremental_claims {r.get('incremental_claims')}"
    if r.get("max_seqs") != SLOTS:
        return f"{name}: {r.get('max_seqs')} slots, not {SLOTS}"
    reqs = r.get("requests") or []
    if not reqs or any(x is None for x in reqs):
        return f"{name}: a request did not finish"
    if r.get("rows_end"):
        return f"{name}: {r['rows_end']} rows still held at idle"
    if shared and r.get("kv_blocks_waits") != 0:
        return (f"{name}: kv_blocks_waits {r.get('kv_blocks_waits')!r}, not 0 (the contract schedule must never wait, "
                f"and a missing count is not a zero)")
    return None


def verdict(runs: dict) -> dict:
    bad = [b for b in (run_bad(k, runs.get(k)) for k in RUNS) if b]
    rc_f, floor = g3.compare(runs.get("ca_a") or {}, runs.get("ca_b") or {})
    rc_c, contract = g3.compare(runs.get("sh") or {}, runs.get("ca_a") or {})
    why = list(bad)
    if rc_f != 0:
        why.append(f"floor: {floor}")
    if rc_c != 0:
        why.append(f"sh~ca_a: {contract}")
    return {"verdict": "OK" if not why else "VOID", "why": why, "floor": floor, "contract": contract,
            "pool_rows": (runs.get("sh") or {}).get("pool_rows"),
            "claim_all_rows": (runs.get("sh") or {}).get("claim_all_rows")}


def _fake(pool, outs=None, **kw):
    reqs = [{"i": i, "prompt_len": 4, "max_new": 3, "out": list(o), "finish": "length"}
            for i, o in enumerate(outs or ([1, 2, 3], [4, 5, 6]))]
    r = {"pool": pool, "incremental_claims": pool == "shared", "max_seqs": SLOTS, "workload_sha256": "w",
         "requests": reqs, "rows_end": 0, "kv_blocks_waits": 0 if pool == "shared" else None}
    r.update(kw)
    return r


def self_test() -> int:
    def runs(**over):
        r = {"ca_a": _fake("claim-all"), "sh": _fake("shared"), "ca_b": _fake("claim-all")}
        r.update(over)
        return r
    cases = [
        ("identical", runs(), "OK"),
        ("shared differs", runs(sh=_fake("shared", outs=([1, 2, 9], [4, 5, 6]))), "VOID"),
        ("floor differs", runs(ca_b=_fake("claim-all", outs=([1, 2, 3], [4, 5, 7]))), "VOID"),
        ("shared waited", runs(sh=_fake("shared", kv_blocks_waits=3)), "VOID"),
        ("shared wait count missing", runs(sh=_fake("shared", kv_blocks_waits=None)), "VOID"),
        ("wrong width", runs(ca_a=_fake("claim-all", max_seqs=64)), "VOID"),
        ("shared not incremental", runs(sh=_fake("shared", incremental_claims=False)), "VOID"),
        ("rows at idle", runs(sh=_fake("shared", rows_end=2)), "VOID"),
        ("a run missing", runs(ca_b=None), "VOID"),
        ("workload differs", runs(sh=_fake("shared", workload_sha256="x")), "VOID"),
    ]
    fails = 0
    for name, r, want in cases:
        got = verdict(r)["verdict"]
        fails += got != want
        print(f"SC2J_CONTRACT self-test {'ok ' if got == want else 'BAD'} {name}: {got} (want {want})")
    print(f"SC2J_CONTRACT self-test {'PASS' if not fails else 'FAIL'} ({len(cases) - fails}/{len(cases)})")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("run", "verdict"))
    ap.add_argument("--out")
    ap.add_argument("--dir")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd == "run":
        if not a.out:
            ap.error("run needs --out")
        return run(a)
    if a.cmd == "verdict":
        if not a.dir:
            ap.error("verdict needs --dir")
        runs = {k: (json.load(open(p)) if os.path.exists(p := os.path.join(a.dir, f"contract_{k}.json")) else None)
                for k in RUNS}
        v = verdict(runs)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(v, f, indent=1)
        print("SC2J_CONTRACT " + json.dumps({k: v[k] for k in ("verdict", "why", "floor", "contract", "pool_rows",
                                                                "claim_all_rows")}), flush=True)
        return 0 if v["verdict"] == "OK" else 1
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    sys.exit(main())
