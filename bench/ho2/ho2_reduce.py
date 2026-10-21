"""HO2's registered read (bench/ho2/PREREG-ho2.md): does the planner's own plan cover what its own execution measured, on two
MoE families no planner constant has seen? One box, one card, four rows. Descriptive per row; the box reads COVERED only when
every row is. It computes no band: the rows go to the planner's split as held-out rows, and the band is read from there.

    python ho2_reduce.py --dir RUN_DIR --run-id ho2-5090-1 --e4b SHA --gnf4 SHA --loggetta SHA --out verdict_ho2.json
    python ho2_reduce.py --self-test
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
CARDS = {"5090": "NVIDIA GeForce RTX 5090", "4090": "NVIDIA GeForce RTX 4090"}
TAGS = tuple(f"{f}_{s}" for f in FAMILIES for s in SEQS)
#: the plan lines a receipt on file would have supplied; with none on file each reads "inferred" (planner.py's defaults)
OVERHEADS = ("allocator reserve", "CUDA context", "process baseline")


def _no_receipts_on_file(receipt):
    lines = (((receipt.get("plan") or {}).get("selected") or {}).get("lines")) or []
    over = [ln for ln in lines if str(ln.get("name", "")).startswith(OVERHEADS)]
    return len(over) == len(OVERHEADS) and all(ln.get("basis") == "inferred" for ln in over)


def _void(cause, rows=None):
    return {"verdict": "VOID", "cause": cause, "rows": rows or []}


def row_of(tag, receipt, card):
    """One row from one receipt, or (None, why) when the receipt is not the registered row."""
    fam, seq = tag.rsplit("_", 1)
    seq = int(seq)
    model_id, rev = FAMILIES[fam]
    m, w, d, gpu = receipt.get("model") or {}, receipt.get("workload") or {}, receipt.get("data") or {}, (receipt.get("hardware") or {}).get("gpu") or {}
    opts = d.get("options") or {}
    checks = [
        (receipt.get("status") == "OK", f"status {receipt.get('status')!r}"),
        (receipt.get("backend") == "experts4bit", f"backend {receipt.get('backend')!r}"),
        (m.get("model") == model_id and m.get("revision") == rev, f"model {m.get('model')}@{m.get('revision')}"),
        (w.get("kind", "train") == "train" and w.get("seq_len") == seq and w.get("micro_batch") == MICRO_BATCH
         and w.get("grad_accum", 1) == 1 and w.get("steps") == STEPS, f"workload {({k: w.get(k) for k in ('seq_len', 'micro_batch', 'grad_accum', 'steps')})}"),
        ((d.get("dataset"), opts.get("revision")) == ALPACA and d.get("format") == "text", f"data {d.get('dataset')}@{opts.get('revision')} {d.get('format')}"),
        (d.get("tokens") == STEPS * MICRO_BATCH * seq and d.get("blocks") == STEPS * MICRO_BATCH, f"tokens {d.get('tokens')} blocks {d.get('blocks')} (an early stop?)"),
        (gpu.get("name") == CARDS[card], f"card {gpu.get('name')!r}"),
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
    return {"tag": tag, "family": fam, "model": model_id, "revision": rev, "seq": seq, "card": CARDS[card],
            "setup": receipt.get("setup"), "plan_total_bytes": plan_total, "driver_peak_bytes": driver_peak,
            "estimate_bytes": estimate, "allocated_peak_bytes": allocated_peak,
            "driver_over_plan": driver_peak / plan_total, "allocated_over_estimate": allocated_peak / estimate,
            "outcome": "COVERED" if not (under or short) else "+".join(n for n, f in (("UNDER", under), ("SHORT", short)) if f),
            "run_id": receipt.get("run_id")}, ""


def reduce(receipts_by_tag, run_id, pins):
    """``receipts_by_tag``: {tag: [receipt, ...]}; ``pins``: the e4b, gnf4 and loggetta commits the box installed."""
    card = run_id.split("-")[1] if run_id.count("-") == 2 else None
    if card not in CARDS:
        return _void(f"unregistered run id {run_id!r}")
    if not all(isinstance(v, str) and len(v) == 40 for v in pins.values()):
        return _void(f"a software pin is not a 40-char commit: {pins}")
    rows, missing = [], []
    for tag in TAGS:
        found = receipts_by_tag.get(tag) or []
        if len(found) != 1:
            missing.append(f"{tag}: {len(found)} receipts")
            continue
        row, why = row_of(tag, found[0], card)
        if row is None:
            return _void(why, rows)
        rows.append(row)
    if missing:
        return _void("not every registered row has exactly one receipt: " + "; ".join(missing), rows)
    covered = all(r["outcome"] == "COVERED" for r in rows)
    return {"verdict": "COVERED" if covered else "MISS", "run_id": run_id, "card": CARDS[card], "pins": pins, "rows": rows,
            "n_covered": sum(r["outcome"] == "COVERED" for r in rows), "n_rows": len(rows), "band_computed": False}


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


def _fake(tag, card="5090", driver=20_000_000_000, total=22_000_000_000, alloc=17_000_000_000, est=18_000_000_000):
    fam, seq = tag.rsplit("_", 1)
    seq = int(seq)
    model_id, rev = FAMILIES[fam]
    return {"status": "OK", "backend": "experts4bit", "run_id": f"x-{tag}", "model": {"model": model_id, "revision": rev},
            "workload": {"kind": "train", "seq_len": seq, "micro_batch": 2, "grad_accum": 1, "steps": 20},
            "data": {"dataset": ALPACA[0], "options": {"revision": ALPACA[1]}, "format": "text", "tokens": 40 * seq, "blocks": 40},
            "hardware": {"gpu": {"name": CARDS[card]}}, "setup": {"expert_residency": "device"},
            "plan": {"selected": {"lines": [{"name": n + " (x)", "basis": "inferred"} for n in OVERHEADS] + [{"name": "activations", "basis": "heuristic"}]}},
            "comparison": {"device_driver": {"estimated": total, "measured": driver}, "device_allocator": {"estimated": est, "measured": alloc}}}


def self_test():
    pins = {"e4b": "a" * 40, "gnf4": "b" * 40, "loggetta": "c" * 40}
    good = {t: [_fake(t)] for t in TAGS}
    r = reduce(good, "ho2-5090-1", pins)
    assert r["verdict"] == "COVERED" and r["n_covered"] == 4, r
    under = copy.deepcopy(good)
    under["ernie_2048"] = [_fake("ernie_2048", driver=23_000_000_000)]
    r = reduce(under, "ho2-5090-1", pins)
    assert r["verdict"] == "MISS" and [x["outcome"] for x in r["rows"] if x["tag"] == "ernie_2048"] == ["UNDER"], r
    short = copy.deepcopy(good)
    short["lfm2_1024"] = [_fake("lfm2_1024", alloc=19_000_000_000)]
    assert [x["outcome"] for x in reduce(short, "ho2-5090-1", pins)["rows"] if x["tag"] == "lfm2_1024"] == ["SHORT"]
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
        assert reduce(g, "ho2-5090-1", pins)["verdict"] == "VOID", name
    assert reduce(good, "ho2-3090-1", pins)["verdict"] == "VOID"
    assert reduce(good, "ho2-5090-1", {**pins, "loggetta": "LOGGETTA_0_6_2_RELEASE_SHA"})["verdict"] == "VOID"
    four = {t: [_fake(t, card="4090")] for t in TAGS}
    assert reduce(four, "ho2-4090-1", pins)["verdict"] == "COVERED"
    assert reduce(four, "ho2-5090-1", pins)["verdict"] == "VOID"            # a 4090's receipts never read as the 5090 box
    print("ho2 reducer self-test OK (17 cases)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dir")
    ap.add_argument("--run-id")
    ap.add_argument("--e4b")
    ap.add_argument("--gnf4")
    ap.add_argument("--loggetta")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    v = reduce(load(a.dir), a.run_id, {"e4b": a.e4b, "gnf4": a.gnf4, "loggetta": a.loggetta})
    text = json.dumps(v, indent=1, sort_keys=True)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(json.dumps({k: v.get(k) for k in ("verdict", "cause", "n_covered", "n_rows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
