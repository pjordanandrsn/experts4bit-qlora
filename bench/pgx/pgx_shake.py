"""PGX shakedown verdict (bench/pgx/PREREG-pgx.md, Amendment 1): did every arm serve at 8 GB, and did each counter fire?

    python bench/pgx/pgx_shake.py --work DIR --out shake_verdict.json
    python bench/pgx/pgx_shake.py --self-test

``DIR`` holds ``<arm>/8/d1/lifetime.json`` with ``P1.json`` and ``P2.json`` for each of S4, S8, V8 and E-hyb, laid out as
the reducer reads a reading (``pgx_reduce.py``). The shakedown is never read: this names what the reading's kit needs.

- **STOP** when any arm did not serve: its lifetime is missing, it records a startup failure, or a pass is missing, has a
  request fault or carries the wrong ``prompts_sha``. The registration: "If any arm cannot serve at its budget: STOP and
  report. That is a finding about the engine, not a retry."
- **SHAKE_FAULTS** when every arm served but a lifetime carries a fault the reducer would VOID it for (a counter dump
  missing, K off the table, the placement proof or profile wrong, a pager give-up). The reading cannot launch until an
  amendment fixes it.
- **SHAKE_OK** otherwise.

Alongside the verdict: each lifetime's wall (server start to stop), peak host RSS and peak VRAM (the reading's guard and
RAM floor are refit from these), and P2 against P1 token identity (reported, as the registration rules it only on a
reading). Exit 0 on SHAKE_OK, 27 otherwise. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import pgx_reduce as red  # noqa: E402

ARMS = ("S4", "S8", "V8", "E-hyb")
BUDGET = 8
REG = {"profile_sha": "0a46edd68005b46db6bb2174542ec948df2843071efce2ecb36099fda04c626a",
       "budget_bytes": {"4": 4 * 10**9, "8": 8 * 10**9, "12": 12 * 10**9},
       "sets": {"warm_c1": "d45648005719bb7ff6979e814f0c9cde5824e85d12b60f5d1cd0e107b838f554"}}


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def shake(work: str) -> dict:
    out = {"verdict": None, "arms": {}, "stop": [], "faults": []}
    for arm in ARMS:
        base = os.path.join(work, arm, str(BUDGET), "d1")
        lt = _load(os.path.join(base, "lifetime.json"))
        row = {"served": False, "faults": [], "wall_s": None, "peak_rss_gib": None, "peak_vram_mib": None,
               "k_registered": None, "k_measured": None, "p2_equals_p1": None}
        out["arms"][arm] = row
        if lt is None:
            out["stop"].append(f"{arm}: no lifetime record")
            continue
        for key in ("wall_s", "peak_rss_gib", "peak_vram_mib", "k_registered", "k_measured"):
            row[key] = lt.get(key)
        if lt.get("startup_failed"):
            out["stop"].append(f"{arm}: the server did not start ({lt['startup_failed']})")
            continue
        recs = {p: _load(os.path.join(base, f"{p}.json")) for p in ("P1", "P2")}
        served = True
        for p, rec in recs.items():
            pf = red.pass_faults(rec, REG["sets"]["warm_c1"])
            if pf:
                served = False
                out["stop"].append(f"{arm} {p}: {'; '.join(pf)}")
        row["served"] = served
        if served:
            a, b = recs["P1"]["requests"], recs["P2"]["requests"]
            row["p2_equals_p1"] = len(a) == len(b) and all(x["text_sha"] == y["text_sha"] for x, y in zip(a, b))
        row["faults"] = red.lifetime_faults(lt, REG)
        out["faults"] += [f"{arm}: {f}" for f in row["faults"]]
    out["verdict"] = "STOP" if out["stop"] else ("SHAKE_FAULTS" if out["faults"] else "SHAKE_OK")
    walls = [r["wall_s"] for r in out["arms"].values() if isinstance(r["wall_s"], (int, float))]
    rss = [r["peak_rss_gib"] for r in out["arms"].values() if isinstance(r["peak_rss_gib"], (int, float))]
    out["fit"] = {"lifetime_wall_max_s": max(walls) if walls else None, "peak_rss_max_gib": max(rss) if rss else None}
    return out


def _fixture(root: str, arm: str, *, lt_over=None, p2_fault=False, drop=None) -> None:
    base = os.path.join(root, arm, str(BUDGET), "d1")
    os.makedirs(base, exist_ok=True)
    lt = {"arm": arm, "budget": BUDGET, "draw": 1, "k_registered": {"S4": 67, "S8": 35, "V8": 35}.get(arm),
          "k_measured": {"S4": 67, "S8": 35, "V8": 35}.get(arm), "counters_ok": True, "pager_giveups": 0,
          "wall_s": 600.0, "peak_rss_gib": 40.0, "peak_vram_mib": 9000}
    if arm == "E-hyb":
        lt.update(vram_tier_bytes=62 * 48 * 2_654_208, nvme_experts=0, profile={"kind": "measured", "sha256": REG["profile_sha"]})
    lt.update(lt_over or {})
    if drop != "lifetime":
        with open(os.path.join(base, "lifetime.json"), "w", encoding="utf-8") as f:
            json.dump(lt, f)
    for p in ("P1", "P2"):
        if drop == p:
            continue
        reqs = [{"id": i, "text_sha": f"t{i}"} for i in range(8)]
        rec = {"prompts_sha": REG["sets"]["warm_c1"], "faults": 1 if (p == "P2" and p2_fault) else 0, "requests": reqs}
        with open(os.path.join(base, f"{p}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f)


def self_test() -> int:
    cases = [
        ("every arm served, every counter fired", {}, "SHAKE_OK"),
        ("V8's server did not start", {"V8": {"lt_over": {"startup_failed": "rc 45"}}}, "STOP"),
        ("S4's P2 has a request fault", {"S4": {"p2_fault": True}}, "STOP"),
        ("E-hyb's P1 record is missing", {"E-hyb": {"drop": "P1"}}, "STOP"),
        ("S8 has no lifetime record", {"S8": {"drop": "lifetime"}}, "STOP"),
        ("S8's routing dump is missing", {"S8": {"lt_over": {"counters_ok": False}}}, "SHAKE_FAULTS"),
        ("V8 gave up once", {"V8": {"lt_over": {"pager_giveups": 1}}}, "SHAKE_FAULTS"),
        ("S4's measured K is off the table", {"S4": {"lt_over": {"k_measured": 66}}}, "SHAKE_FAULTS"),
        ("E-hyb placed an expert on the NVMe tier", {"E-hyb": {"lt_over": {"nvme_experts": 1}}}, "SHAKE_FAULTS"),
        ("E-hyb's VRAM tier is over 8 GB", {"E-hyb": {"lt_over": {"vram_tier_bytes": 8 * 10**9 + 1}}}, "SHAKE_FAULTS"),
        ("E-hyb served uniform-assumed mass", {"E-hyb": {"lt_over": {"profile": {"kind": "uniform-assumed", "sha256": None}}}},
         "SHAKE_FAULTS"),
    ]
    bad = 0
    for name, over, want in cases:
        d = tempfile.mkdtemp()
        try:
            for arm in ARMS:
                _fixture(d, arm, **over.get(arm, {}))
            got = shake(d)["verdict"]
        finally:
            shutil.rmtree(d)
        ok = got == want
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {name}: {got} (want {want})")
    print(f"PGX_SHAKE_SELFTEST {'OK' if not bad else 'FAIL'} {len(cases) - bad}/{len(cases)}")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.work and a.out):
        ap.error("--work and --out, or --self-test")
    v = shake(a.work)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=1, default=str)
    print(f"PGX_SHAKE {v['verdict']}: {len(v['stop'])} stop(s), {len(v['faults'])} fault(s); "
          f"lifetime wall max {v['fit']['lifetime_wall_max_s']} s, peak RSS max {v['fit']['peak_rss_max_gib']} GiB")
    for s in v["stop"] + v["faults"]:
        print(f"  {s}")
    return 0 if v["verdict"] == "SHAKE_OK" else 27


if __name__ == "__main__":
    sys.exit(main())
