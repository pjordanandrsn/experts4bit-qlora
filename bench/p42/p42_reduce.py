#!/usr/bin/env python3
"""Lane P42 reducer: turn the four kernel censuses into one attribution table.

    python bench/p42/p42_reduce.py <run-dir> [--md RESULTS-p42.md]

`<run-dir>` is the fetched `…/p42` directory: `logs/census_<arm>.txt` written by
step_decomp's `--replay-profile-out`, and `e4b_b<B>_<arm>.json` carrying the timed
`step_ms_clean` the census is attributed against.

Two numbers per arm are the point:

* **ms/step per kernel** -- the census times 8 graph replays, so a kernel's self-CUDA
  total divided by 8 is its cost in one decode step. Percentages inside the census are
  percentages OF THE CENSUS, which is not the same denominator as the step.
* **accounted fraction** -- summed self-CUDA over 8, against the timed `step_ms_clean`.
  A census that accounts for half the step is not an attribution, it is a hint, and the
  reducer says which it is rather than letting a table imply the stronger claim. Above
  100 % is not an error either: self-CUDA sums across streams, so overlapping kernels
  can total more wall time than the step takes. Either way the fraction is printed and
  the reader is told which regime they are in.

Nothing is recomputed from memory: an arm with no census is a missing row, never a zero.
"""
import argparse
import json
import re
from pathlib import Path

ARMS = ["nf4_b1", "nf4_b16", "int4_b1", "int4_b16"]
REPLAYS = 8          # step_decomp's profiled-replay count; re-read from the header
_UNITS = {"us": 1e-3, "ms": 1.0, "s": 1e3}   # -> milliseconds


def _dur_ms(tok: str):
    m = re.fullmatch(r"([0-9.]+)(us|ms|s)", tok.strip())
    return None if not m else float(m.group(1)) * _UNITS[m.group(2)]


def parse_census(text: str):
    """(replays, [{name, self_ms, calls}]) from a torch profiler key_averages table.

    Rows are split on runs of 2+ spaces and read from the RIGHT: the Name column is the
    only one that can itself contain spaces, and every kernel name does.
    """
    replays = None
    m = re.search(r"profiled replay steps:\s*(\d+)", text)
    if m:
        replays = int(m.group(1))
    rows = []
    for line in text.splitlines():
        if not line.strip() or set(line.strip()) <= set("- "):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 11:
            continue
        name, tail = parts[0], parts[1:]
        try:
            calls = int(tail[-1].replace(",", ""))
        except ValueError:
            continue                      # the header row, and the summary lines below it
        self_ms = _dur_ms(tail[-5])
        if self_ms is None:
            continue
        rows.append({"name": name, "self_ms": self_ms, "calls": calls})
    return replays, rows


def arm_rows(d: Path, arm: str):
    p = d / "logs" / f"census_{arm}.txt"
    if not p.exists():
        return None
    replays, rows = parse_census(p.read_text(errors="ignore"))
    replays = replays or REPLAYS
    for r in rows:
        r["ms_per_step"] = r["self_ms"] / replays
        r["calls_per_step"] = r["calls"] / replays
    return {"replays": replays, "rows": sorted(rows, key=lambda r: -r["ms_per_step"])}


def step_ms(d: Path, arm: str):
    b = 16 if arm.endswith("_b16") else 1
    p = d / f"e4b_b{b}_{arm}.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["step_ms_clean"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def reduce_run(d: Path) -> dict:
    out = {"run_dir": str(d), "arms": {}}
    for arm in ARMS:
        cen = arm_rows(d, arm)
        ms = step_ms(d, arm)
        if cen is None:
            out["arms"][arm] = {"census": "MISSING", "step_ms_clean": ms}
            continue
        acc = sum(r["ms_per_step"] for r in cen["rows"])
        out["arms"][arm] = {
            "step_ms_clean": ms,
            "replays": cen["replays"],
            "accounted_ms_per_step": acc,
            "accounted_fraction": (acc / ms) if ms else None,
            "top": cen["rows"][:25],
        }
    return out


def to_md(rep: dict) -> str:
    lines = ["# P42 — where the B=16 step goes", "",
             "Per-kernel cost is self-CUDA over the profiled replays, so the unit is "
             "**milliseconds of one decode step**. The accounted fraction is that sum "
             "against the arm's timed `step_ms_clean`.", ""]
    for arm, a in rep["arms"].items():
        lines.append(f"## {arm}")
        if a.get("census") == "MISSING":
            lines += ["", f"census MISSING (step_ms_clean {a.get('step_ms_clean')})", ""]
            continue
        frac = a["accounted_fraction"]
        note = ""
        if frac and frac > 1.02:
            note = ("  \n**Over 100 %: self-CUDA sums across streams, so these kernels "
                    "overlap. Read the rows as relative weight, not as a budget.**")
        elif frac and frac < 0.9:
            note = ("  \n**Under 90 %: this is a hint, not an attribution -- most of the "
                    "step is outside the census.**")
        lines += ["", f"step {a['step_ms_clean']:.3f} ms · accounted "
                      f"{a['accounted_ms_per_step']:.3f} ms"
                      + (f" ({frac * 100:.1f} %)" if frac else "") + note, "",
                  "| kernel | ms/step | % of step | calls/step |",
                  "|---|---|---|---|"]
        for r in a["top"]:
            pct = (r["ms_per_step"] / a["step_ms_clean"] * 100) if a["step_ms_clean"] else None
            lines.append(f"| `{r['name'][:58]}` | {r['ms_per_step']:.3f} | "
                         + (f"{pct:.2f}" if pct is not None else "—")
                         + f" | {r['calls_per_step']:.0f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rep = reduce_run(Path(a.run_dir))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    md = to_md(rep)
    if a.md:
        Path(a.md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
