"""P114's reducer and rule (bench/p114/PREREG-p114.md): the two energy harnesses, three passes each, on one rented card.

    python p114_reduce.py --self-test
    python p114_reduce.py <work-dir> <out.json>

`<work-dir>` holds `runs/run_proj_{1,2,3}.txt` (bench_energy.py), `runs/run_excl_{1,2,3}.txt` (bench_energy_excluded.py),
each pass's `runs/pmon_<tag>.txt` and `runs/pre_<tag>.txt`, and `versions.txt`. Every number the rule reads is parsed from
those files; the harnesses' own "vs native" / "vs batch=64" columns are cross-checked, never trusted alone.

The rule, fixed before the data (the PREREG's "Rule"):
  * VOID if: a pass file is missing; a pass ran on another card than CARD; a cell is missing, an ERROR row or NaN; the
    native arm is absent in a workload; versions.txt is not bitsandbytes 0.50.2 on torch 2.8.0; a pass's process monitor
    shows any process but the pass's own, or the card held a compute process before a pass started.
  * NOISY if, over the three passes, any of these spreads ((max - min) / median) exceeds SPREAD (0.10): the matmul_4bit
    ratio of a workload, the dequant ratio of a workload, or Part B's J/token ratio of a batch against batch 64.
  * READ otherwise. The reading is the median over the passes of each ratio.
Exit 0 on READ, 3 on NOISY or VOID.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import statistics
import sys

CARD = "NVIDIA GeForce RTX 5090"
BNB = "0.50.2"
TORCH = "2.8.0"
SPREAD = 0.10
WORKLOADS = ("decode", "prefill", "train")
PATHS = ("native-bf16", "before(dequant)", "after(matmul_4bit)")
BATCHES = (64, 256, 1024, 4096)
NUM = r"(nan|[-+]?\d+(?:\.\d+)?)"
PROJ_ROW = re.compile(rf"^\s*(native-bf16|before\(dequant\)|after\(matmul_4bit\))\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM} uJ"
                      rf"\s*\|\s*{NUM} uJ\s*\|\s*{NUM}x\s*$")
PROJ_ERR = re.compile(r"^\s*(native-bf16|before\(dequant\)|after\(matmul_4bit\))\s*\|\s*ERROR")
WL_HEAD = re.compile(r"^--- .*\((decode|prefill|train)\) ---\s*$")
EXCL_ROW = re.compile(rf"^\s*(64|256|1024|4096)\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM} uJ\s*\|\s*{NUM} uJ\s*\|\s*{NUM}x\s*$")
ALLOC = re.compile(r"^\s*allocate (.+?)\s*: (OK|OOM)")
IDLE = re.compile(rf"idle (?:power: )?{NUM} W")


def _f(s: str) -> float:
    return float("nan") if s == "nan" else float(s)


def parse_proj(text: str) -> dict:
    """{'device': .., 'idle': W, 'cells': {workload: {path: {ops_s, watts, j_op, printed}}}, 'errors': [..]}."""
    out = {"device": None, "idle": None, "cells": {w: {} for w in WORKLOADS}, "errors": []}
    wl = None
    for line in text.splitlines():
        if line.startswith("device:"):
            out["device"] = line.split("device:", 1)[1].split("|")[0].strip()
        m = IDLE.search(line)
        if m and out["idle"] is None and line.startswith("idle"):
            out["idle"] = _f(m.group(1))
        m = WL_HEAD.match(line)
        if m:
            wl = m.group(1)
            continue
        m = PROJ_ERR.match(line)
        if m and wl:
            out["errors"].append(f"{wl} {m.group(1)}: {line.strip()[:80]}")
            continue
        m = PROJ_ROW.match(line)
        if m and wl:
            out["cells"][wl][m.group(1)] = {"ops_s": _f(m.group(2)), "watts": _f(m.group(3)), "j_op": _f(m.group(4)),
                                            "printed": _f(m.group(6))}
    return out


def parse_excl(text: str) -> dict:
    out = {"device": None, "idle": None, "alloc": [], "rows": {}}
    for line in text.splitlines():
        if line.startswith("device:"):
            out["device"] = line.split("device:", 1)[1].strip()
        m = ALLOC.match(line)
        if m:
            out["alloc"].append((m.group(1), m.group(2)))
        if "(idle" in line:
            m = IDLE.search(line)
            if m:
                out["idle"] = _f(m.group(1))
        m = EXCL_ROW.match(line)
        if m:
            out["rows"][int(m.group(1))] = {"tok_s": _f(m.group(2)), "watts": _f(m.group(3)), "j_tok": _f(m.group(4)),
                                           "printed": _f(m.group(6))}
    return out


def pmon_pids(text: str) -> tuple[set, bool]:
    """(distinct (pid, command) pairs in the monitor's rows, whether the monitor produced rows at all)."""
    pids, rows = set(), False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            rows = True
            if parts[1] != "-":
                pids.add((parts[1], parts[-1]))
    return pids, rows


def spread(vals: list[float]) -> float:
    med = statistics.median(vals)
    return (max(vals) - min(vals)) / med if med else float("inf")


def reduce(passes: dict) -> dict:
    """passes = {'proj': [text x3], 'excl': [text x3], 'pmon': {tag: text}, 'pre': {tag: text}, 'versions': text}."""
    void, notes = [], []
    proj = [parse_proj(t) for t in passes["proj"]]
    excl = [parse_excl(t) for t in passes["excl"]]
    if len(proj) != 3 or len(excl) != 3:
        void.append(f"expected 3 + 3 passes, read {len(proj)} + {len(excl)}")
    for i, p in enumerate(proj + excl):
        dev = (p["device"] or "").split("|")[0].strip()
        if dev != CARD:
            void.append(f"pass {i + 1}: card is {dev!r}, the lane registers {CARD!r}")
    v = passes.get("versions", "")
    if f"bitsandbytes {BNB}" not in v:
        void.append(f"versions.txt does not record bitsandbytes {BNB}")
    if f"torch {TORCH}" not in v:
        void.append(f"versions.txt does not record torch {TORCH}")
    for tag, text in sorted(passes.get("pmon", {}).items()):
        pids, rows = pmon_pids(text)
        if not rows:
            notes.append(f"{tag}: the process monitor produced no rows (unchecked)")
        elif len(pids) > 1 or any("python" not in cmd for _pid, cmd in pids):
            void.append(f"{tag}: another process used the card during the pass: {sorted(pids)}")
    for tag, text in sorted(passes.get("pre", {}).items()):
        apps = [ln for ln in text.splitlines()[1:] if ln.strip() and "No running" not in ln]
        if apps:
            void.append(f"{tag}: the card held a compute process before the pass: {apps[:3]}")
    ratios = {"matmul_4bit": {w: [] for w in WORKLOADS}, "dequant": {w: [] for w in WORKLOADS}}
    detail = {w: {p: [] for p in PATHS} for w in WORKLOADS}
    for i, p in enumerate(proj):
        for e in p["errors"]:
            void.append(f"proj pass {i + 1}: {e}")
        for w in WORKLOADS:
            cells = p["cells"][w]
            if "native-bf16" not in cells:
                void.append(f"proj pass {i + 1} {w}: no native arm")
                continue
            base = cells["native-bf16"]["j_op"]
            for path in PATHS:
                c = cells.get(path)
                if c is None or any(math.isnan(c[k]) for k in ("ops_s", "watts", "j_op")):
                    void.append(f"proj pass {i + 1} {w} {path}: missing or NaN")
                    continue
                detail[w][path].append(c)
                r = c["j_op"] / base
                if abs(r - c["printed"]) > 0.006:
                    void.append(f"proj pass {i + 1} {w} {path}: ratio {r:.3f} disagrees with the printed {c['printed']:.2f}")
                if path == "after(matmul_4bit)":
                    ratios["matmul_4bit"][w].append(r)
                elif path == "before(dequant)":
                    ratios["dequant"][w].append(r)
    partb = {n: [] for n in BATCHES}
    partb_rows = {n: [] for n in BATCHES}
    for i, p in enumerate(excl):
        if 64 not in p["rows"]:
            void.append(f"excl pass {i + 1}: no batch-64 row")
            continue
        base = p["rows"][64]["j_tok"]
        for n in BATCHES:
            c = p["rows"].get(n)
            if c is None or any(math.isnan(c[k]) for k in ("tok_s", "watts", "j_tok")):
                void.append(f"excl pass {i + 1} batch {n}: missing or NaN")
                continue
            partb[n].append(c["j_tok"] / base)
            partb_rows[n].append(c)
    out = {"card": CARD, "void": void, "notes": notes, "rule": {"spread": SPREAD},
           "idle_w": {"proj": [p["idle"] for p in proj], "excl": [p["idle"] for p in excl]},
           "part_a": excl[0]["alloc"] if excl else []}
    noisy = []
    if not void:
        med = {}
        for arm in ("matmul_4bit", "dequant"):
            med[arm] = {}
            for w in WORKLOADS:
                vals = ratios[arm][w]
                med[arm][w] = statistics.median(vals)
                s = spread(vals)
                out.setdefault("spreads", {})[f"{arm} {w}"] = s
                if s > SPREAD:
                    noisy.append(f"{arm} {w}: passes {['%.3f' % x for x in vals]} spread {s:.3f} > {SPREAD}")
        med["partb"] = {}
        for n in BATCHES:
            vals = partb[n]
            med["partb"][n] = statistics.median(vals)
            s = spread(vals)
            out.setdefault("spreads", {})[f"partb {n}"] = s
            if s > SPREAD:
                noisy.append(f"partb batch {n}: passes {['%.3f' % x for x in vals]} spread {s:.3f} > {SPREAD}")
        out["median"] = med
        out["by_pass"] = {"matmul_4bit": ratios["matmul_4bit"], "dequant": ratios["dequant"], "partb": partb}
        out["cells"] = {w: {p: {k: statistics.median(c[k] for c in detail[w][p]) for k in ("ops_s", "watts", "j_op")}
                            for p in PATHS} for w in WORKLOADS}
        out["partb_cells"] = {n: {k: statistics.median(c[k] for c in partb_rows[n]) for k in ("tok_s", "watts", "j_tok")}
                              for n in BATCHES}
        out["headline"] = {"matmul_4bit_range": [min(med["matmul_4bit"].values()), max(med["matmul_4bit"].values())],
                           "dequant_range": [min(med["dequant"].values()), max(med["dequant"].values())],
                           "partb_4096_vs_64": med["partb"][4096]}
    out["noisy"] = noisy
    out["verdict"] = "VOID" if void else ("NOISY" if noisy else "READ")
    return out


def load(work: pathlib.Path) -> dict:
    runs = work / "runs"
    rd = lambda p: p.read_text(errors="replace") if p.is_file() else ""  # noqa: E731
    return {"proj": [rd(runs / f"run_proj_{i}.txt") for i in (1, 2, 3) if (runs / f"run_proj_{i}.txt").is_file()],
            "excl": [rd(runs / f"run_excl_{i}.txt") for i in (1, 2, 3) if (runs / f"run_excl_{i}.txt").is_file()],
            "pmon": {p.stem[len("pmon_"):]: rd(p) for p in sorted(runs.glob("pmon_*.txt"))},
            "pre": {p.stem[len("pre_"):]: rd(p) for p in sorted(runs.glob("pre_*.txt"))},
            "versions": rd(work / "versions.txt")}


# ---------------------------------------------------------------------------------------------------- self-test --

def _proj_text(card: str, r_mm: dict, r_dq: dict, *, error: str | None = None, nan: str | None = None) -> str:
    """A bench_energy.py stdout in its exact print format, with the given ratios against a native J/op of 1000 uJ."""
    lines = [f"device: {card} | bf16 | gate_up [2048,2048]", "idle power: 40.0 W", ""]
    heads = {"decode": "--- fwd  M=1   (decode) ---", "prefill": "--- fwd  M=512 (prefill) ---",
             "train": "--- fwd+bwd M=32 (train) ---"}
    for w in WORKLOADS:
        lines += [heads[w], f"{'path':>20} | {'ops/s':>9} | {'power W':>8} | {'J/op (tot)':>11} | {'J/op (dyn)':>11} | "
                  f"{'vs native':>9}"]
        for name, r in (("native-bf16", 1.0), ("before(dequant)", r_dq[w]), ("after(matmul_4bit)", r_mm[w])):
            if error == f"{w} {name}":
                lines.append(f"{name:>20} | ERROR: RuntimeError: boom")
                continue
            watts, j = 100.0, 1000.0 * r
            ops = watts / (j * 1e-6)
            jw = "nan" if nan == f"{w} {name}" else f"{j:>8.2f}"
            lines.append(f"{name:>20} | {ops:>9.0f} | {watts:>8.1f} | {jw} uJ | {j * 0.6:>8.2f} uJ | {r:>8.2f}x")
        lines.append("")
    return "\n".join(lines)


def _excl_text(card: str, r: dict) -> str:
    lines = [f"device: {card}", "", "=== Part A: memory wall on a 12 GB card (full OLMoE-1B-7B) ===",
             "  allocate bf16 experts (12.9 GB)    : OK (12.9 GB resident)",
             "  allocate 4-bit experts (3.2 GB)    : OK (3.2 GB resident)", "",
             "=== Part B: tokens-per-joule of the fused 4-bit MoE forward vs batch (utilization) ===",
             "  (idle 40.0 W subtracted for dynamic; total shown too)",
             f"{'batch (tok)':>12} | {'tok/s':>10} | {'power W':>8} | {'J/tok (tot)':>12} | {'J/tok (dyn)':>12} | "
             f"{'vs batch=64':>11}"]
    for n in BATCHES:
        j = 2000.0 * r[n]
        lines.append(f"{n:>12} | {100.0 / (j * 1e-6):>10.0f} | {100.0:>8.1f} | {j:>9.3f} uJ | {j * 0.6:>9.3f} uJ | "
                     f"{r[n]:>10.2f}x")
    return "\n".join(lines)


def _passes(card=CARD, mm=None, dq=None, pb=None, jitter=(1.0, 1.0, 1.0), **kw) -> dict:
    mm = mm or {"decode": 1.00, "prefill": 1.30, "train": 1.80}
    dq = dq or {"decode": 2.40, "prefill": 1.30, "train": 1.50}
    pb = pb or {64: 1.00, 256: 0.60, 1024: 0.35, 4096: 0.25}
    proj = [_proj_text(card, {w: v * j for w, v in mm.items()}, dq, **kw) for j in jitter]
    excl = [_excl_text(card, pb) for _ in jitter]
    own = "# gpu pid type sm mem enc dec command\n# Idx # C/G % % % % name\n    0   4242   C    45    3    -    -   python\n"
    return {"proj": proj, "excl": excl, "pmon": {f"proj_{i}": own for i in (1, 2, 3)},
            "pre": {f"proj_{i}": "0 %, 1 MiB, 40.00 W, 35\n" for i in (1, 2, 3)},
            "versions": f"bitsandbytes {BNB}\ntorch {TORCH}+cu128 cuda 12.8\n"}


def self_test() -> int:
    foreign = _passes()
    foreign["pmon"]["proj_2"] += "    0   999   C    30    2    -    -   ollama\n"
    held = _passes()
    held["pre"]["proj_1"] += "999, ollama, 4000 MiB\n"
    wrong_bnb = _passes()
    wrong_bnb["versions"] = f"bitsandbytes 0.50.0.dev0\ntorch {TORCH}\n"
    two = _passes()
    two["proj"] = two["proj"][:2]
    cases = [
        ("a stable reading reads", _passes(), "READ"),
        ("a drifting train cell is NOISY", _passes(jitter=(1.0, 1.15, 0.9)), "NOISY"),
        ("the wrong card is VOID", _passes(card="NVIDIA RTX A2000 12GB"), "VOID"),
        ("an ERROR row is VOID", _passes(error="train after(matmul_4bit)"), "VOID"),
        ("a NaN cell is VOID", _passes(nan="decode before(dequant)"), "VOID"),
        ("a foreign process during a pass is VOID", foreign, "VOID"),
        ("a compute process before a pass is VOID", held, "VOID"),
        ("the wrong bitsandbytes is VOID", wrong_bnb, "VOID"),
        ("two passes are VOID, not a crash", two, "VOID"),
        ("a 4-bit saving reads, whatever its sign", _passes(mm={"decode": 0.80, "prefill": 0.95, "train": 1.10}), "READ"),
    ]
    bad = 0
    for name, passes, want in cases:
        got = reduce(passes)["verdict"]
        ok = got == want
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {name}: {got} (want {want})")
    r = reduce(_passes())
    assert math.isclose(r["headline"]["matmul_4bit_range"][0], 1.0, abs_tol=0.01), r["headline"]
    assert math.isclose(r["headline"]["matmul_4bit_range"][1], 1.8, abs_tol=0.01), r["headline"]
    assert math.isclose(r["headline"]["partb_4096_vs_64"], 0.25, abs_tol=0.005), r["headline"]
    print(f"P114 rule self-test: {len(cases) - bad}/{len(cases)} cases; the parsers read the harnesses' print formats")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if argv[1:] == ["--self-test"]:
        return self_test()
    if len(argv) != 3:
        print(__doc__)
        return 2
    out = reduce(load(pathlib.Path(argv[1])))
    pathlib.Path(argv[2]).write_text(json.dumps(out, indent=1, default=str))
    for n in out["notes"]:
        print(f"P114 note: {n}")
    if "median" in out:
        mm, dq, pb = out["median"]["matmul_4bit"], out["median"]["dequant"], out["median"]["partb"]
        print("P114 matmul_4bit / native J/op (median of 3): " + ", ".join(f"{w} {mm[w]:.3f}" for w in WORKLOADS))
        print("P114 dequant / native J/op (median of 3): " + ", ".join(f"{w} {dq[w]:.3f}" for w in WORKLOADS))
        print("P114 Part B J/token vs batch 64 (median of 3): " + ", ".join(f"{n} {pb[n]:.3f}" for n in BATCHES))
        print("P114 spreads: " + ", ".join(f"{k} {v:.3f}" for k, v in out["spreads"].items()))
    for v in out["void"] + out["noisy"]:
        print(f"P114 reason: {v}")
    print(f"P114_VERDICT {out['verdict']} on {CARD}")
    return 0 if out["verdict"] == "READ" else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
