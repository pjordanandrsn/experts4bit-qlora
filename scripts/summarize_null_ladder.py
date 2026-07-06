"""Controller-side summarizer for the per-example ∅ ladder (debts D1 + D2).

Reads null_eval job dirs (result.json + result_rows.jsonl), then reports with example-PAIRED
statistics (the eval set is fixed, so every mode/placement pair shares examples):

- the ∅ ladder per placement, mean ± SE;
- D1: paired deltas for every mode pair at resident placement (mean, sd, SE, |t|) — settles
  whether the ladder's fine ordering and the G/coverage numbers exceed sampling error;
- D2: resident-vs-offload paired delta per mode (the serve-side placement certificate);
- the eval-determinism repeat null (same mode, two jobs, per-example max |d|);
- S9 check: G_int8 (∅_nf4 − ∅_int8) against its paired SE.

Controller-only; never writes into job dirs. Usage:
    python scripts/summarize_null_ladder.py --jobs-root runs/jobs --out runs/results/null_ladder_per_example.md
"""

import argparse
import glob
import json
import math
import os
import sys

MODES = ("fp4", "nf4", "int8", "fp8", "bf16", "fp16")


def load_jobs(jobs_root):
    out = {}
    for d in sorted(glob.glob(os.path.join(jobs_root, "null_olmoe_*_perexample*"))):
        rp = os.path.join(d, "result.json")
        rows_p = os.path.join(d, "result_rows.jsonl")
        if not (os.path.exists(rp) and os.path.exists(rows_p)):
            continue
        res = json.load(open(rp, encoding="utf-8"))
        rows = [json.loads(line) for line in open(rows_p, encoding="utf-8")]
        losses = {r["example_index"]: r["loss"] for r in rows if not r.get("is_nan")}
        out[os.path.basename(d)] = {"result": res, "losses": losses}
    return out


def paired(a, b):
    """Paired stats for two {example_index: loss} maps (shared keys only)."""
    ks = sorted(set(a) & set(b))
    ds = [a[k] - b[k] for k in ks]
    n = len(ds)
    if n < 2:
        return {"n": n, "mean": None, "sd": None, "se": None, "t": None}
    mean = sum(ds) / n
    var = sum((d - mean) ** 2 for d in ds) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": (mean / se if se > 0 else float("inf"))}


def mean_se(losses):
    vs = list(losses.values())
    n = len(vs)
    m = sum(vs) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vs) / (n - 1))
    return m, sd / math.sqrt(n), n


def job_key(mode, placement, rep=None):
    return f"null_olmoe_{mode}_{placement}_perexample" + (f"_rep{rep}" if rep else "")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # output has non-ASCII; Windows pipes default to the locale codepage
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    jobs = load_jobs(args.jobs_root)
    if not jobs:
        print(f"no null_eval jobs under {args.jobs_root}", file=sys.stderr)
        return 1
    gpus = sorted({j["result"].get("gpu_name") for j in jobs.values()})
    hosts = sorted({j["result"].get("hostname", j["result"].get("pod_id", "?")) for j in jobs.values()})

    lines = ["# Per-example ∅ ladder (debts D1 + D2)", ""]
    lines.append(f"- jobs: {len(jobs)} | GPU(s): {gpus} | host(s): {hosts}")
    if len(gpus) > 1:
        lines.append("- **WARNING: more than one GPU architecture — cross-mode deltas are "
                     "evaluator-confounded (see TRAIN_PLACEMENT_CERTIFICATE.md T5c). Rerun on one pod.**")
    lines.append("")

    lines.append("## Ladder (mean ± SE over examples)")
    lines.append("")
    lines.append("| mode | resident | offload |")
    lines.append("|---|---|---|")
    for m in MODES:
        cells = []
        for pl in ("resident", "offload"):
            j = jobs.get(job_key(m, pl))
            if j:
                mu, se, n = mean_se(j["losses"])
                cells.append(f"{mu:.4f} ± {se:.4f} (n={n})")
            else:
                cells.append("—")
        lines.append(f"| {m} | {cells[0]} | {cells[1]} |")
    lines.append("")

    lines.append("## D1 — paired mode deltas, resident (rowmode − colmode)")
    lines.append("")
    lines.append("| pair | mean d | sd | SE | |t| |")
    lines.append("|---|---|---|---|---|")
    res_jobs = {m: jobs.get(job_key(m, "resident")) for m in MODES}
    for i, m1 in enumerate(MODES):
        for m2 in MODES[i + 1:]:
            if res_jobs[m1] and res_jobs[m2]:
                p = paired(res_jobs[m1]["losses"], res_jobs[m2]["losses"])
                lines.append(f"| {m1} − {m2} | {p['mean']:+.4f} | {p['sd']:.4f} | {p['se']:.4f} | {abs(p['t']):.2f} |")
    lines.append("")

    lines.append("## D2 — placement delta per mode (resident − offload, example-paired)")
    lines.append("")
    lines.append("| mode | mean d | SE | |t| | verdict |")
    lines.append("|---|---|---|---|---|")
    for m in MODES:
        jr, jo = jobs.get(job_key(m, "resident")), jobs.get(job_key(m, "offload"))
        if jr and jo:
            p = paired(jr["losses"], jo["losses"])
            verdict = "identical" if p["sd"] == 0 else ("clean (|t|<2)" if abs(p["t"]) < 2 else "**GAP**")
            lines.append(f"| {m} | {p['mean']:+.6f} | {p['se']:.6f} | {abs(p['t']):.2f} | {verdict} |")
    lines.append("")

    rep1, rep2 = jobs.get(job_key("int8", "resident")), jobs.get(job_key("int8", "resident", rep=2))
    if rep1 and rep2:
        ks = sorted(set(rep1["losses"]) & set(rep2["losses"]))
        ds = [abs(rep1["losses"][k] - rep2["losses"][k]) for k in ks]
        lines.append("## Eval-determinism repeat (int8-resident, run twice)")
        lines.append("")
        lines.append(f"- per-example max |d| = {max(ds):.3e}, mean |d| = {sum(ds)/len(ds):.3e}, "
                     f"bitwise-identical examples: {sum(1 for d in ds if d == 0)}/{len(ds)}")
        lines.append("")

    if res_jobs.get("nf4") and res_jobs.get("int8") and res_jobs.get("bf16"):
        g_int8 = paired(res_jobs["nf4"]["losses"], res_jobs["int8"]["losses"])
        g_total = paired(res_jobs["nf4"]["losses"], res_jobs["bf16"]["losses"])
        lines.append("## S9 check — G_int8 against paired SE")
        lines.append("")
        lines.append(f"- G_int8 (nf4−int8) = {g_int8['mean']:+.4f} ± {g_int8['se']:.4f} (|t| = {abs(g_int8['t']):.2f})")
        lines.append(f"- G_total (nf4−bf16) = {g_total['mean']:+.4f} ± {g_total['se']:.4f} (|t| = {abs(g_total['t']):.2f})")
        if g_total["mean"]:
            lines.append(f"- coverage = {100 * g_int8['mean'] / g_total['mean']:.0f}%")
        s9 = abs(g_int8["t"]) < 2
        lines.append(f"- **S9 {'FIRES — G_int8 indistinguishable from 0; precision program drops to screening' if s9 else 'clear'}**")
        lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
