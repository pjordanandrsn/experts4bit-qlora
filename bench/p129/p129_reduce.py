# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P129's reducer: Amendment 1's verdict re-derived from its records (``records/a1``).

- ``runs.jsonl``: one line per (config, seed) run -- 30 training losses and the held-out loss at steps 0 and 30.
- ``counts.json``: launches and Python calls per training step, today's path against the fused one, and the dequantize's equality.
- ``projection.json``: the fused projection in isolation against the three ``LoRALinear`` modules (reported).
- ``grads_step1.json``: the first step's gradients, each variant against the baseline (reported).

``python bench/p129/p129_reduce.py [records dir] [--md out.md]``; ``--selftest`` runs hand-built cases through every rung."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SEEDS = (211, 223, 227, 229, 233)
FLOOR = ("F1", "F2", "F3")
BACKSTOP = 0.005                 # TC1's held-out bar
LAUNCH_CUT_MIN = 50              # Phase 1's counts gate, unchanged by Amendment 1
PY_CUT_MIN_PCT = 3.0


def divergence(v, b):
    """(D_traj, D_held) of run ``v`` from the baseline ``b`` with the same seed."""
    if len(v["loss"]) != len(b["loss"]):
        raise ValueError(f"{v['cfg']} seed {v['seed']}: {len(v['loss'])} steps against {len(b['loss'])}")
    return max(abs(x - y) for x, y in zip(v["loss"], b["loss"])), abs(v["held30"] - b["held30"])


def verdict(runs, counts):
    """Amendment 1's rule. Returns (verdict, facts)."""
    by = {(r["cfg"], r["seed"]): r for r in runs}
    missing = [(c, s) for s in SEEDS for c in ("B",) + FLOOR + ("FUSED",) if (c, s) not in by or by[(c, s)].get("error")]
    if missing:
        return "VOID", {"missing": missing}
    floor = [(c, s) + divergence(by[(c, s)], by[("B", s)]) for s in SEEDS for c in FLOOR]
    fused = [(s,) + divergence(by[("FUSED", s)], by[("B", s)]) for s in SEEDS]
    f = {"floor": floor, "fused": fused,
         "floor_worst_traj": max(x[2] for x in floor), "floor_worst_held": max(x[3] for x in floor),
         "fused_max_traj": max(x[1] for x in fused), "fused_max_held": max(x[2] for x in fused)}
    f["e2e_held"] = (f["fused_max_traj"] <= f["floor_worst_traj"] and f["fused_max_held"] <= f["floor_worst_held"]
                     and all(x[2] <= BACKSTOP for x in fused))
    cut = counts["ref_launches"] - counts["fused_launches"]
    py = 100.0 * (counts["ref_python_calls"] - counts["fused_python_calls"]) / counts["ref_python_calls"]
    f.update(launch_cut=cut, python_cut_pct=py, dequant_bitwise=bool(counts.get("dequant_bitwise")))
    f["counts_held"] = cut >= LAUNCH_CUT_MIN and py >= PY_CUT_MIN_PCT
    if not f["counts_held"]:
        return "NO_GAIN", f
    return ("PASS" if f["e2e_held"] else "FAIL"), f


def render(v, f, counts, proj=None, grads=None) -> str:
    out = [f"# P129 Amendment 1, re-derived: **{v}**", ""]
    if v == "VOID":
        return "\n".join(out + [f"missing or failed runs: {f['missing']}"]) + "\n"
    out += ["| measure | fused, worst of 5 | floor, worst of 15 |", "|---|---|---|",
            f"| `D_traj` | {f['fused_max_traj']:.5f} | {f['floor_worst_traj']:.5f} |",
            f"| `D_held` | {f['fused_max_held']:.5f} | {f['floor_worst_held']:.5f} |", "",
            f"- end-to-end gate: {'HELD' if f['e2e_held'] else 'MISSED'} (backstop: every fused `D_held` <= {BACKSTOP})",
            f"- counts: launches {counts['ref_launches']} -> {counts['fused_launches']} ({-f['launch_cut']:+d}), Python calls "
            f"{counts['ref_python_calls']} -> {counts['fused_python_calls']} ({-f['python_cut_pct']:+.2f} %): "
            f"{'HELD' if f['counts_held'] else 'MISSED'}; dequant bitwise: {f['dequant_bitwise']}", "",
            "| draw | `D_traj` | `D_held` |", "|---|---|---|"]
    out += [f"| {c} {s} | {t:.5f} | {h:.5f} |" for c, s, t, h in f["floor"]]
    out += [f"| FUSED {s} | {t:.5f} | {h:.5f} |" for s, t, h in f["fused"]]
    if proj:
        out += ["", "Reported -- the fused projection in isolation (relative difference, bar, within):"]
        out += [f"- {r['name']}: {r['rel']:.2e} (bar {r['bar']:.2e}): {r['within']}" for r in proj]
    if grads:
        out += ["", "Reported -- the first step's worst relative gradient difference from the baseline, per draw:"]
        out += [f"- {g['cfg']} {g['seed']}: {g['worst_rel']:.4f} ({g['worst_name']})" for g in grads]
    return "\n".join(out) + "\n"


def _selftest() -> int:
    def run(cfg, s, d=0.0, h=0.0):
        return {"cfg": cfg, "seed": s, "loss": [8.0 + d] * 30, "held30": 8.0 + h}
    counts = {"ref_launches": 785, "fused_launches": 675, "ref_python_calls": 16440, "fused_python_calls": 14466, "dequant_bitwise": True}
    base = [run("B", s) for s in SEEDS] + [run(c, s, 0.003, 0.0007) for s in SEEDS for c in FLOOR]
    cases = [
        ("pass", base + [run("FUSED", s, 0.002, 0.0005) for s in SEEDS], counts, "PASS"),
        ("traj over the floor", base + [run("FUSED", s, 0.004, 0.0005) for s in SEEDS], counts, "FAIL"),
        ("held over the floor", base + [run("FUSED", s, 0.002, 0.0009) for s in SEEDS], counts, "FAIL"),
        ("counts missed", base + [run("FUSED", s, 0.002, 0.0005) for s in SEEDS], dict(counts, fused_launches=760), "NO_GAIN"),
        ("a run missing", base + [run("FUSED", s, 0.002, 0.0005) for s in SEEDS[:4]], counts, "VOID"),
    ]
    big = [run("B", s) for s in SEEDS] + [run(c, s, 0.003, 0.01) for s in SEEDS for c in FLOOR]
    cases.append(("backstop", big + [run("FUSED", s, 0.002, 0.006) for s in SEEDS], counts, "FAIL"))
    for name, runs, c, want in cases:
        got, _ = verdict(runs, c)
        assert got == want, (name, got, want)
    print(f"P129 REDUCE SELFTEST OK cases={len(cases)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(Path(__file__).parent / "records" / "a1"))
    ap.add_argument("--md")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    d = Path(a.dir)
    runs = [json.loads(line) for line in (d / "runs.jsonl").read_text().splitlines() if line.strip()]
    counts = json.loads((d / "counts.json").read_text())
    proj = json.loads((d / "projection.json").read_text()) if (d / "projection.json").exists() else None
    grads = json.loads((d / "grads_step1.json").read_text()) if (d / "grads_step1.json").exists() else None
    v, f = verdict(runs, counts)
    md = render(v, f, counts, proj, grads)
    if a.md:
        Path(a.md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
