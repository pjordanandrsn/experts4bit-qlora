#!/usr/bin/env python3
"""Lane P39 reducer: H1 (step-level effect of gnf4#357) and H2 (recorded assignment on a
second box) from the fetched receipts, against the bands in P39-PREREG.md.

    python bench/p39/p39_reduce.py <box1-run-dir> [<box2-run-dir>] [--md RESULTS-p39.md]

Every number is read from a receipt or a log line the box wrote; nothing is recomputed
from memory. A missing receipt is a `host-limited` row, never a zero.
"""
import argparse
import json
import re
from pathlib import Path

H1_B16_BAND = (0.96, 0.99)   # NEW/OLD step_ms_clean at B=16: the registered prediction
H1_B1_BAND = (0.97, 1.03)    # B=1 must not move (the floor)
H1_REFUTE_B16_SLOWER = 1.01  # NEW slower than OLD by >1% refutes


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _log(d: Path, name: str) -> str:
    p = d / "logs" / f"run_{name}.log"
    return p.read_text(errors="ignore") if p.exists() else ""


def _plan_has_r(text: str):
    m = re.search(r"PLAN_HAS_R=(True|False)", text)
    return None if not m else (m.group(1) == "True")


def _counts(text: str):
    """Streamed calibration prints one 'calibrated experts' line per layer chunk; the pack's counts are their SUM."""
    ms = re.findall(r"INT4EXP calibrated experts: (\d+) gptq / (\d+) rtn", text)
    return (sum(int(g) for g, _ in ms), sum(int(r) for _, r in ms)) if ms else None


def _honoured(text: str):
    m = re.search(r"INT4EXP assignment honoured (sha256:[0-9a-f]{64}): (\d+) expert-roles", text)
    return (m.group(1), int(m.group(2))) if m else None


def _ratio(new, old):
    return None if (new is None or old is None or not old) else new / old


def reduce_box1(d: Path) -> dict:
    rows, out = {}, {"box": 1, "arms": {}, "h1": {}}
    for arm in ("nf4_b1", "nf4_b16", "old_b16_build", "old_b16_r1", "new_b16_r1", "new_b1_r1", "old_b1_r1", "old_b16_r2", "new_b16_r2"):
        b = 16 if "_b16" in arm else 1        # "_b1" is a substring of "_b16": test the longer token first
        rec = _load(d / f"e4b_b{b}_{arm}.json")
        log = _log(d, arm)
        has_r = _plan_has_r(log)
        want = None if arm.startswith("nf4") else arm.startswith("new")
        row = {"batch": b, "step_ms_clean": None if not rec else rec.get("step_ms_clean"),
               "aggregate_tok_s": None if not rec else rec.get("aggregate_tok_s"),
               "n_steps": None if not rec else rec.get("n_steps"),
               "plan_has_r": has_r, "kernel_tripwire": ("n/a" if want is None else ("ok" if has_r == want else "VOID: kernel does not match the arm label")),
               "status": ("host-limited" if rec is None else rec.get("status", "ok")),
               "counts": _counts(log)}
        rows[arm] = row
    out["arms"] = rows
    fp = (d / "box1_out" / "FINGERPRINT")
    out["artifact_fingerprint"] = fp.read_text().strip() if fp.exists() else None

    def ms(a):
        r = rows[a]
        return None if r["kernel_tripwire"].startswith("VOID") else r["step_ms_clean"]
    pairs = {"b16_r1": _ratio(ms("new_b16_r1"), ms("old_b16_r1")), "b16_r2": _ratio(ms("new_b16_r2"), ms("old_b16_r2")),
             "b1": _ratio(ms("new_b1_r1"), ms("old_b1_r1"))}
    spread_old = _ratio(ms("old_b16_r2"), ms("old_b16_r1"))
    spread_new = _ratio(ms("new_b16_r2"), ms("new_b16_r1"))
    out["h1"] = {"ratios_new_over_old": pairs, "self_pair_spread": {"old_b16": spread_old, "new_b16": spread_new},
                 "bands": {"b16": H1_B16_BAND, "b1": H1_B1_BAND}}
    b16 = [r for r in (pairs["b16_r1"], pairs["b16_r2"]) if r is not None]
    v = "host-limited"
    if b16:
        spread = max(abs((s or 1) - 1) for s in (spread_old, spread_new) if s is not None) if (spread_old or spread_new) else 0.0
        if any(r > H1_REFUTE_B16_SLOWER for r in b16):
            v = "fail: NEW slower than OLD at B=16 by >1%"
        elif all(H1_B16_BAND[0] <= r <= H1_B16_BAND[1] for r in b16):
            v = "pass: inside the registered 1-4% band"
        elif all(r > H1_B16_BAND[1] for r in b16):
            v = "kernel-level only: <1% at step level"
        elif all(r < H1_B16_BAND[0] for r in b16):
            v = "pass, larger than predicted: >4% -- more than the census attributed"
        else:
            v = "unresolved: repeats disagree"
        if v.startswith("pass") and all(abs(r - 1) <= spread for r in b16):
            v = "unresolved: inside the self-pair spread"
    out["h1"]["b16_verdict"] = v
    r1 = pairs["b1"]
    out["h1"]["b1_verdict"] = ("host-limited" if r1 is None else ("pass: B=1 unchanged (floor)" if H1_B1_BAND[0] <= r1 <= H1_B1_BAND[1] else "fail: B=1 moved >3%"))
    return out


def reduce_box2(d: Path, box1_fp) -> dict:
    out = {"box": 2}
    hon_log = _log(d, "honoured_wikitext")
    out["honoured_banner"] = _honoured(hon_log)
    out["honoured_counts"] = _counts(hon_log)
    out["recipe_counts"] = _counts(_log(d, "recipe_wikitext"))
    summ = (d / "summary.txt").read_text(errors="ignore") if (d / "summary.txt").exists() else ""
    m = re.search(r"ARTIFACT2 (sha256:[0-9a-f]{64}) BOX1 (sha256:[0-9a-f]{64}) (SAME_BYTES|DIFFERENT_BYTES)", summ)
    out["bytes_vs_box1"] = m.group(3) if m else None
    out["gate"] = _load(d / "gate_verdict.json")
    return out


def h2_verdict(b1: dict, b2: dict) -> dict:
    v = {}
    c1 = b1["arms"].get("old_b16_build", {}).get("counts")
    v["a_counts_equal_box1"] = ("pass" if (c1 and b2["honoured_counts"] == c1 and b2["honoured_banner"]) else
                                ("host-limited" if b2["honoured_counts"] is None else "FAIL: #531 did not reproduce the split"))
    n = b2["honoured_banner"][1] if b2["honoured_banner"] else None
    v["b_disagreements"] = (None if n is None else {"n": n, "verdict": "pass" if 2 <= n <= 40 else "outside the predicted 2-40"})
    v["c_bytes_differ"] = {"observed": b2["bytes_vs_box1"], "verdict": ("pass: bytes differ as #531 states" if b2["bytes_vs_box1"] == "DIFFERENT_BYTES"
                                                                        else ("SURPRISE: identical bytes on a second box" if b2["bytes_vs_box1"] == "SAME_BYTES" else "host-limited"))}
    g = b2["gate"] or {}
    v["d_gate"] = g.get("verdict", "host-limited")
    rc = b2["recipe_counts"]
    v["optional_recipe_counts_differ"] = (None if rc is None else ("pass: recipe flips on this box too" if rc != c1 else "recipe reproduced box 1's counts -- no flip here"))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("box1")
    ap.add_argument("box2", nargs="?")
    ap.add_argument("--md")
    a = ap.parse_args()
    b1 = reduce_box1(Path(a.box1) / "p39" if (Path(a.box1) / "p39").exists() else Path(a.box1))
    res = {"h1": b1}
    if a.box2:
        b2 = reduce_box2(Path(a.box2) / "p39" if (Path(a.box2) / "p39").exists() else Path(a.box2), b1["artifact_fingerprint"])
        res["h2"] = b2
        res["h2_verdict"] = h2_verdict(b1, b2)
    print(json.dumps(res, indent=1, default=str))
    if a.md:
        lines = ["# P39 results (reduced from receipts; bands from P39-PREREG.md)", "",
                 "## H1 -- step-level effect of gnf4#357 (box 1)", "",
                 "| arm | B | step_ms_clean | tok/s | kernel tripwire |", "|---|---:|---:|---:|---|"]
        for k, r in b1["arms"].items():
            lines.append(f"| {k} | {r['batch']} | {r['step_ms_clean']} | {r['aggregate_tok_s']} | {r['kernel_tripwire']} |")
        h = b1["h1"]
        lines += ["", f"NEW/OLD step time: B=16 r1 {h['ratios_new_over_old']['b16_r1']}, r2 {h['ratios_new_over_old']['b16_r2']}; B=1 {h['ratios_new_over_old']['b1']}",
                  f"self-pair spread: {h['self_pair_spread']}", "", f"**B=16: {h['b16_verdict']}**  ", f"**B=1: {h['b1_verdict']}**", ""]
        if a.box2:
            lines += ["## H2 -- the recorded assignment on a second box (box 2)", "", "```", json.dumps(res["h2_verdict"], indent=1, default=str), "```", ""]
        Path(a.md).write_text("\n".join(lines) + "\n")
        print(f"wrote {a.md}")


if __name__ == "__main__":
    main()
