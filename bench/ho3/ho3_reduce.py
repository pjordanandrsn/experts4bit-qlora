"""HO3's registered read (bench/ho3/PREREG-ho3.md): HO2's reducer (bench/ho2/ho2_reduce.py) by counted edits. HO2's rule per
row, on a second machine per card (Part A) and on RTX A4000 boxes (Part B, the first host-offload held-out rows). Each family
on a box is its own cohort result (COVERED, MISS or VOID), so a fault in one never reads as the other; the box reads COVERED
only when both are. Each row's plan is compared with the registered one (residency, device and host totals): a difference
is recorded as a finding beside the row and never changes the row's outcome. A row whose experts are on the host also
needs its host peak inside the host estimate (HOST_UNDER otherwise). It computes no band.

    python ho3_reduce.py --dir RUN_DIR --run-id ho3-5090-1 --launch SHA --e4b SHA --gnf4 SHA --loggetta SHA --out verdict_ho3.json
    python ho3_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os

ALPACA = ("tatsu-lab/alpaca", "dce01c9b08f87459cf36a430d809084718273017")
FAMILIES = {"lfm2": ("LiquidAI/LFM2-8B-A1B", "c1c44ff9fc00db3ebf4516970563f5f383d23670"),
            "ernie": ("baidu/ERNIE-4.5-21B-A3B-PT", "87db95487941cb39592ee0abca3b9155a6d19c5c")}
SEQS = (1024, 2048)
STEPS, MICRO_BATCH = 20, 2
CARDS = {"5090": "NVIDIA GeForce RTX 5090", "4090": "NVIDIA GeForce RTX 4090", "a4000": "NVIDIA RTX A4000"}
#: the registered plans (PREREG "Predictions"): Part A is HO2's own plan on both cards (byte-identical across them), Part B the
#: loggetta 0.6.2 plan on a 16 GB RTX A4000, derived before any measurement (off-stack: CPU torch 2.14.1). (residency,
#: plan device total bytes, plan host total bytes or None where not registered)
REGISTERED_PLANS = {
    "5090": {"lfm2_1024": ("device", 9768170758, None), "lfm2_2048": ("device", 9386279379, None),
             "ernie_1024": ("device", 21856438198, None), "ernie_2048": ("device", 20886295683, None)},
    "a4000": {"lfm2_1024": ("device", 9768170758, 3221225472), "lfm2_2048": ("device", 9386279379, 3221225472),
              "ernie_1024": ("host", 9747209139, 15451815936), "ernie_2048": ("host", 8777066624, 15451815936)},
}
REGISTERED_PLANS["4090"] = REGISTERED_PLANS["5090"]
TAGS = tuple(f"{f}_{s}" for f in FAMILIES for s in SEQS)
#: what a row must be; the tests read a real receipt against its own run's values through this
REGISTRATION = {"families": FAMILIES, "data": ALPACA, "format": "text", "steps": STEPS, "micro_batch": MICRO_BATCH, "cards": CARDS,
                "plans": REGISTERED_PLANS}
#: the plan lines a receipt on file would have supplied; with none on file each reads "inferred" (planner.py's defaults)
OVERHEADS = ("allocator reserve", "CUDA context", "process baseline")


def _no_receipts_on_file(receipt):
    lines = (((receipt.get("plan") or {}).get("selected") or {}).get("lines")) or []
    over = [ln for ln in lines if str(ln.get("name", "")).startswith(OVERHEADS)]
    return len(over) == len(OVERHEADS) and all(ln.get("basis") == "inferred" for ln in over)


def _void(cause, rows=None):
    return {"verdict": "VOID", "cause": cause, "rows": rows or []}


def row_of(tag, receipt, card, reg=None):
    """One row from one receipt, or (None, why) when the receipt is not the registered row."""
    reg = reg or REGISTRATION
    steps, mb = reg["steps"], reg["micro_batch"]
    fam, seq = tag.rsplit("_", 1)
    seq = int(seq)
    model_id, rev = reg["families"][fam]
    m, w, d, gpu = receipt.get("model") or {}, receipt.get("workload") or {}, receipt.get("data") or {}, (receipt.get("hardware") or {}).get("gpu") or {}
    opts = d.get("options") or {}
    checks = [
        (receipt.get("status") == "OK", f"status {receipt.get('status')!r}"),
        (receipt.get("backend") == "experts4bit", f"backend {receipt.get('backend')!r}"),
        (m.get("model") == model_id and m.get("revision") == rev, f"model {m.get('model')}@{m.get('revision')}"),
        (w.get("kind", "train") == "train" and w.get("seq_len") == seq and w.get("micro_batch") == mb
         and w.get("grad_accum", 1) == 1 and w.get("steps") == steps, f"workload {({k: w.get(k) for k in ('seq_len', 'micro_batch', 'grad_accum', 'steps')})}"),
        ((d.get("dataset"), opts.get("revision")) == reg["data"] and d.get("format") == reg["format"], f"data {d.get('dataset')}@{opts.get('revision')} {d.get('format')}"),
        (d.get("tokens") == steps * mb * seq and d.get("blocks") == steps * mb, f"tokens {d.get('tokens')} blocks {d.get('blocks')} (an early stop?)"),
        (gpu.get("name") == reg["cards"][card], f"card {gpu.get('name')!r}"),
        (_no_receipts_on_file(receipt), "an overhead line is not the inferred default: the plan learned from receipts on file"),
    ]
    bad = [why for ok, why in checks if not ok]
    if bad:
        return None, f"{tag}: " + "; ".join(bad)
    c = receipt.get("comparison") or {}
    drv, alloc = c.get("device_driver") or {}, c.get("device_allocator") or {}
    vals = (drv.get("estimated"), drv.get("measured"), alloc.get("estimated"), alloc.get("measured"))
    if not all(isinstance(v, int) and v > 0 for v in vals):
        return None, f"{tag}: a plan total, estimate or peak is missing ({vals})"
    plan_total, driver_peak, estimate, allocated_peak = vals
    under, short = driver_peak > plan_total, allocated_peak > estimate
    residency = (receipt.get("setup") or {}).get("expert_residency")
    if residency not in ("device", "host"):                       # fails closed: the host check depends on it (rule 7)
        return None, f"{tag}: the receipt records no expert residency ({residency!r})"
    host = c.get("host") or {}
    host_under = None
    if residency == "host":
        h_est, h_meas = host.get("estimated"), host.get("measured")
        if not (isinstance(h_est, int) and h_est > 0 and isinstance(h_meas, int) and h_meas > 0):
            return None, f"{tag}: a host-residency row needs its host estimate and peak ({h_est}, {h_meas})"
        host_under = h_meas > h_est
    sel = (receipt.get("plan") or {}).get("selected") or {}
    want = (reg.get("plans") or {}).get(card, {}).get(tag)
    got = ((sel.get("setup") or {}).get("expert_residency"), sel.get("device_bytes"), sel.get("host_bytes"))
    plan_diff = None
    if want is not None:
        diffs = [f"{n} registered {w} box {g}" for n, w, g in zip(("residency", "device_bytes", "host_bytes"), want, got)
                 if w is not None and w != g]
        plan_diff = "; ".join(diffs) or None
    return {"tag": tag, "family": fam, "model": model_id, "revision": rev, "seq": seq, "card": reg["cards"][card],
            "setup": receipt.get("setup"), "plan_total_bytes": plan_total, "driver_peak_bytes": driver_peak,
            "estimate_bytes": estimate, "allocated_peak_bytes": allocated_peak,
            "driver_over_plan": driver_peak / plan_total, "allocated_over_estimate": allocated_peak / estimate,
            "outcome": "COVERED" if not (under or short or host_under) else "+".join(
                n for n, f in (("UNDER", under), ("SHORT", short), ("HOST_UNDER", host_under)) if f),
            "residency": residency, "host_estimate_bytes": host.get("estimated"), "host_peak_bytes": host.get("measured"),
            "plan_matches_registration": plan_diff is None, "plan_diff": plan_diff,
            "run_id": receipt.get("run_id")}, ""


def reduce(receipts_by_tag, run_id, pins, skipped=()):
    """``receipts_by_tag``: {tag: [receipt, ...]}; ``pins``: the launch commit and the e4b, gnf4 and loggetta commits the box
    installed; ``skipped``: the rows the runner recorded as skipped for the deadline (its ``SKIPPED train <tag>`` lines). One
    cohort result per family: COVERED, MISS, UNREAD (a row skipped for the deadline, nothing else wrong) or VOID with its
    cause. The box is COVERED only when both are; VOID when either is; UNREAD otherwise when either is."""
    card = run_id.split("-")[1] if run_id.count("-") == 2 else None
    if card not in CARDS:
        return _void(f"unregistered run id {run_id!r}")
    if not all(isinstance(v, str) and len(v) == 40 for v in pins.values()):
        return _void(f"a software pin is not a 40-char commit: {pins}")
    rows, cohorts = [], {}
    for fam in FAMILIES:
        fam_rows, cause, unread = [], None, []
        for tag in (t for t in TAGS if t.startswith(fam + "_")):
            found = receipts_by_tag.get(tag) or []
            if not found and tag in skipped:        # the runner recorded the deadline skip: UNREAD, never VOID or a pass
                unread.append(tag)
                continue
            if len(found) != 1:
                cause = cause or f"{tag}: {len(found)} receipts (not exactly one)"
                continue
            row, why = row_of(tag, found[0], card)
            if row is None:
                cause = cause or why
                continue
            fam_rows.append(row)
        rows.extend(fam_rows)
        if cause:
            cohorts[fam] = {"verdict": "VOID", "cause": cause, "card": CARDS[card]}
        elif unread:
            cohorts[fam] = {"verdict": "UNREAD", "unread": unread, "card": CARDS[card], "n_rows": len(fam_rows)}
        else:
            ok = all(r["outcome"] == "COVERED" for r in fam_rows)
            cohorts[fam] = {"verdict": "COVERED" if ok else "MISS", "card": CARDS[card],
                            "residency": sorted({r["residency"] for r in fam_rows}),
                            "n_covered": sum(r["outcome"] == "COVERED" for r in fam_rows), "n_rows": len(fam_rows)}
    verdicts = [c["verdict"] for c in cohorts.values()]
    box = ("VOID" if "VOID" in verdicts else "UNREAD" if "UNREAD" in verdicts
           else "COVERED" if all(v == "COVERED" for v in verdicts) else "MISS")
    out = {"verdict": box, "run_id": run_id, "card": CARDS[card], "pins": pins, "rows": rows, "cohorts": cohorts,
           "n_covered": sum(r["outcome"] == "COVERED" for r in rows), "n_rows": len(rows),
           "plan_findings": [f"{r['tag']}: {r['plan_diff']}" for r in rows if r.get("plan_diff")], "band_computed": False}
    if box == "VOID":
        out["cause"] = "; ".join(f"{f}: {c['cause']}" for f, c in cohorts.items() if c["verdict"] == "VOID")
    return out


def load_skipped(run_dir):
    """The rows the runner skipped for the deadline: its ``SKIPPED train <tag>`` lines in summary.txt, beside run/."""
    try:
        text = open(os.path.join(run_dir, os.pardir, "summary.txt"), encoding="utf-8").read()
    except OSError:
        return set()
    return {ln.split()[2] for ln in text.splitlines() if ln.startswith("SKIPPED train ") and len(ln.split()) == 3}


def load(run_dir):
    out = {}
    for tag in TAGS:
        out[tag] = []
        for p in sorted(glob.glob(os.path.join(run_dir, f"receipts_{tag}", "*.json"))):
            try:
                out[tag].append(json.load(open(p)))
            except (OSError, ValueError):
                out[tag].append({"status": "UNREADABLE"})
    return out


def _fake(tag, card="5090", driver=20_000_000_000, total=22_000_000_000, alloc=17_000_000_000, est=18_000_000_000,
          residency="device", host_est=None, host_meas=None, plan=None):
    fam, seq = tag.rsplit("_", 1)
    seq = int(seq)
    model_id, rev = FAMILIES[fam]
    return {"status": "OK", "backend": "experts4bit", "run_id": f"x-{tag}", "model": {"model": model_id, "revision": rev},
            "workload": {"kind": "train", "seq_len": seq, "micro_batch": 2, "grad_accum": 1, "steps": 20},
            "data": {"dataset": ALPACA[0], "options": {"revision": ALPACA[1]}, "format": "text", "tokens": 40 * seq, "blocks": 40},
            "hardware": {"gpu": {"name": CARDS[card]}}, "setup": {"expert_residency": residency},
            "plan": {"selected": {"lines": [{"name": n + " (x)", "basis": "inferred"} for n in OVERHEADS] + [{"name": "activations", "basis": "heuristic"}],
                                  **({} if plan is None else {"setup": {"expert_residency": plan[0]}, "device_bytes": plan[1], "host_bytes": plan[2]})}},
            "comparison": {"device_driver": {"estimated": total, "measured": driver}, "device_allocator": {"estimated": est, "measured": alloc},
                           **({} if host_est is None else {"host": {"estimated": host_est, "measured": host_meas}})}}


def self_test():
    pins = {"e4b": "a" * 40, "gnf4": "b" * 40, "loggetta": "c" * 40}
    good = {t: [_fake(t)] for t in TAGS}
    r = reduce(good, "ho3-5090-1", pins)
    assert r["verdict"] == "COVERED" and r["n_covered"] == 4, r
    under = copy.deepcopy(good)
    under["ernie_2048"] = [_fake("ernie_2048", driver=23_000_000_000)]
    r = reduce(under, "ho3-5090-1", pins)
    assert r["verdict"] == "MISS" and [x["outcome"] for x in r["rows"] if x["tag"] == "ernie_2048"] == ["UNDER"], r
    short = copy.deepcopy(good)
    short["lfm2_1024"] = [_fake("lfm2_1024", alloc=19_000_000_000)]
    assert [x["outcome"] for x in reduce(short, "ho3-5090-1", pins)["rows"] if x["tag"] == "lfm2_1024"] == ["SHORT"]
    for name, mutate in (
        ("a missing row", lambda g: g.pop("lfm2_2048")),
        ("two receipts for one row", lambda g: g["lfm2_2048"].append(_fake("lfm2_2048"))),
        ("an early stop", lambda g: g["ernie_1024"][0]["data"].update(tokens=1000)),
        ("another model revision", lambda g: g["ernie_1024"][0]["model"].update(revision="0" * 40)),
        ("another dataset revision", lambda g: g["lfm2_1024"][0]["data"]["options"].update(revision="0" * 40)),
        ("the wrong card", lambda g: g["lfm2_1024"][0]["hardware"]["gpu"].update(name=CARDS["4090"])),
        ("a failed run", lambda g: g["lfm2_1024"][0].update(status="ALARM")),
        ("receipts on file", lambda g: g["lfm2_1024"][0]["plan"]["selected"]["lines"][0].update(basis="measured")),
        ("an overhead line missing", lambda g: g["lfm2_1024"][0]["plan"]["selected"]["lines"].pop(1)),
        ("a missing peak", lambda g: g["lfm2_1024"][0]["comparison"]["device_driver"].update(measured=None)),
    ):
        g = copy.deepcopy(good)
        mutate(g)
        assert reduce(g, "ho3-5090-1", pins)["verdict"] == "VOID", name
    assert reduce(good, "ho3-3090-1", pins)["verdict"] == "VOID"
    assert reduce(good, "ho3-5090-1", {**pins, "loggetta": "LOGGETTA_0_6_2_RELEASE_SHA"})["verdict"] == "VOID"
    four = {t: [_fake(t, card="4090")] for t in TAGS}
    assert reduce(four, "ho3-4090-1", pins)["verdict"] == "COVERED"
    assert reduce(four, "ho3-5090-1", pins)["verdict"] == "VOID"            # a 4090's receipts never read as the 5090 box
    # HO3's own cases. (1) A fault in one family never reads as the other: ERNIE VOID, LFM2 COVERED, the box VOID.
    g = copy.deepcopy(good)
    g["ernie_1024"][0]["data"].update(tokens=1000)
    r = reduce(g, "ho3-5090-1", pins)
    assert r["verdict"] == "VOID" and r["cohorts"]["ernie"]["verdict"] == "VOID" and r["cohorts"]["lfm2"]["verdict"] == "COVERED", r
    # (2) Part B: ERNIE on the host. Its host peak inside the estimate is COVERED; above it is HOST_UNDER (a MISS).
    def a4(tag, **kw):
        res, dev, hst = REGISTERED_PLANS["a4000"][tag]
        hk = {"host_est": 15_000_000_000, "host_meas": 14_000_000_000} if res == "host" else {}
        args = {"card": "a4000", "residency": res, "total": dev, "driver": dev - 10**9, "est": dev - 2 * 10**9,
                "alloc": dev - 3 * 10**9, "plan": (res, dev, hst), **hk, **kw}
        return _fake(tag, **args)
    b = {t: [a4(t)] for t in TAGS}
    r = reduce(b, "ho3-a4000-1", pins)
    assert r["verdict"] == "COVERED" and r["cohorts"]["ernie"]["residency"] == ["host"] and not r["plan_findings"], r
    hb = copy.deepcopy(b)
    hb["ernie_2048"] = [a4("ernie_2048", host_meas=16_000_000_000)]
    r = reduce(hb, "ho3-a4000-1", pins)
    assert r["cohorts"]["ernie"]["verdict"] == "MISS" and r["cohorts"]["lfm2"]["verdict"] == "COVERED", r
    assert [x["outcome"] for x in r["rows"] if x["tag"] == "ernie_2048"] == ["HOST_UNDER"], r
    # (3) A host-residency row without its host numbers voids its cohort only.
    nh = copy.deepcopy(b)
    nh["ernie_1024"][0]["comparison"].pop("host")
    r = reduce(nh, "ho3-a4000-1", pins)
    assert r["cohorts"]["ernie"]["verdict"] == "VOID" and r["cohorts"]["lfm2"]["verdict"] == "COVERED", r
    # (4) A box plan that differs from the registered one is a finding beside the row, never a changed outcome.
    pd = copy.deepcopy(b)
    pd["ernie_1024"] = [a4("ernie_1024", plan=("device", 21_000_000_000, 3_221_225_472))]
    r = reduce(pd, "ho3-a4000-1", pins)
    row = [x for x in r["rows"] if x["tag"] == "ernie_1024"][0]
    assert r["verdict"] == "COVERED" and not row["plan_matches_registration"] and "residency registered host box device" in row["plan_diff"], r
    assert r["plan_findings"] and r["plan_findings"][0].startswith("ernie_1024:"), r
    # (5) Fails closed (bench/common rule 7): a receipt with no recorded residency voids its cohort, never skips the host check.
    nr = copy.deepcopy(b)
    nr["ernie_2048"][0]["setup"] = {}
    r = reduce(nr, "ho3-a4000-1", pins)
    assert r["cohorts"]["ernie"]["verdict"] == "VOID" and "no expert residency" in r["cohorts"]["ernie"]["cause"], r
    # (6) A row the runner skipped for the deadline is UNREAD, not VOID and never a pass; a missing row with no recorded
    # skip is still VOID (a crash or a lost row, rule 7).
    sk = copy.deepcopy(b)
    sk.pop("ernie_2048")
    r = reduce(sk, "ho3-a4000-1", pins, skipped={"ernie_2048"})
    assert r["verdict"] == "UNREAD" and r["cohorts"]["ernie"]["verdict"] == "UNREAD" and r["cohorts"]["lfm2"]["verdict"] == "COVERED", r
    assert reduce(sk, "ho3-a4000-1", pins)["cohorts"]["ernie"]["verdict"] == "VOID"
    sk2 = copy.deepcopy(sk)
    sk2["lfm2_1024"][0].update(status="ALARM")
    assert reduce(sk2, "ho3-a4000-1", pins, skipped={"ernie_2048"})["verdict"] == "VOID"
    # (7) A retry id names its card the same way; an A4000 box never reads as a 4090's.
    assert reduce(b, "ho3-a4000-r3", pins)["verdict"] == "COVERED"
    assert reduce(b, "ho3-4090-1", pins)["verdict"] == "VOID"
    print("ho3 reducer self-test OK (17 HO2 cases + 11 HO3 cases)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dir")
    ap.add_argument("--run-id")
    ap.add_argument("--launch")
    ap.add_argument("--e4b")
    ap.add_argument("--gnf4")
    ap.add_argument("--loggetta")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    v = reduce(load(a.dir), a.run_id, {"launch": a.launch, "e4b": a.e4b, "gnf4": a.gnf4, "loggetta": a.loggetta},
               skipped=load_skipped(a.dir))
    text = json.dumps(v, indent=1, sort_keys=True)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(json.dumps({k: v.get(k) for k in ("verdict", "cause", "n_covered", "n_rows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
