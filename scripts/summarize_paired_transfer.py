"""Paired-transfer summarizer — the review's own analysis, made reproducible (order O-2).

Marginal cell standard deviations are the wrong ruler for a same-adapter paired design: the
seed-to-seed level shifts cancel inside a pair. This script emits, for every train row that
shares >= 2 query modes across seeds:

- per-seed SAME-ADAPTER query-pair deltas L(->qA) - L(->qB), with sign counts, paired sd,
  and paired t;
- seed-paired CROSS-ADAPTER contrasts per query column (train row A vs train row B on the
  same query mode, same seed), same statistics.

Reads runpod query-job results (runs/query_jobs/*/result.json). Output is deterministic
(sorted keys throughout) — the first run on the shipped runs/query_jobs is committed as the
golden test fixture (tests/fixtures/paired_transfer_golden.md), so the external review's
numbers stay reproducible forever.

Usage:
    python scripts/summarize_paired_transfer.py --query-jobs runs/query_jobs \\
        --out-md runs/results/paired_transfer.md --out-csv runs/results/paired_transfer.csv
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

JOB_RE = re.compile(r"query_olmoe_train-(\w+)-(resident|offload)-seed(\d+)_query-(\w+)-(resident|offload)")


def load(query_jobs_root):
    rows = {}
    for p in sorted(glob.glob(os.path.join(query_jobs_root, "query_olmoe_*", "result.json"))):
        r = json.load(open(p))
        m = JOB_RE.match(r.get("job_id", ""))
        if not m or r.get("status") != "pass":
            continue
        tsch, tpl, seed, qsch, qpl = m.groups()
        train_row = f"{tsch}-{tpl}"
        query_col = f"{qsch}-{qpl}"
        rows[(train_row, query_col, int(seed))] = r["eval_loss_with_adapter"]
    return rows


def paired_stats(deltas):
    n = len(deltas)
    mean = sum(deltas) / n
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "t": None, "pos": sum(1 for d in deltas if d > 0)}
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    return {"n": n, "mean": mean, "sd": sd, "t": t, "pos": sum(1 for d in deltas if d > 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-jobs", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    rows = load(args.query_jobs)
    if not rows:
        print(f"no query jobs under {args.query_jobs}", file=sys.stderr)
        return 1
    train_rows = sorted({k[0] for k in rows})
    query_cols = sorted({k[1] for k in rows})
    seeds = sorted({k[2] for k in rows})

    md = ["# Paired transfer statistics (same-adapter and seed-paired contrasts)", ""]
    md.append(f"- source: query jobs, train rows {train_rows}, query columns {query_cols}, seeds {seeds}")
    md.append("- ruler: PAIRED deltas (marginal cell sds are the wrong ruler for a paired design)")
    csv_rows = []

    md.append("")
    md.append("## Same-adapter query-pair deltas, per seed")
    md.append("")
    md.append("| train row | pair | per-seed deltas | mean | paired sd | t | same-sign |")
    md.append("|---|---|---|---|---|---|---|")
    for tr in train_rows:
        for i, qa in enumerate(query_cols):
            for qb in query_cols[i + 1:]:
                ds = []
                for s in seeds:
                    a, b = rows.get((tr, qa, s)), rows.get((tr, qb, s))
                    if a is not None and b is not None:
                        ds.append(a - b)
                if len(ds) < 2:
                    continue
                st = paired_stats(ds)
                md.append(f"| {tr} | L(→{qa}) − L(→{qb}) | " + " / ".join(f"{d:+.4f}" for d in ds)
                          + f" | {st['mean']:+.4f} | {st['sd']:.4f} | {st['t']:.2f} | {st['pos']}/{st['n']} |")
                csv_rows.append({"kind": "same_adapter_pair", "train_row": tr, "contrast": f"{qa}-vs-{qb}",
                                 "deltas": ";".join(f"{d:+.6f}" for d in ds),
                                 "mean": f"{st['mean']:+.6f}", "paired_sd": f"{st['sd']:.6f}",
                                 "t": f"{st['t']:.3f}", "n_pos": st["pos"], "n": st["n"]})

    md.append("")
    md.append("## Seed-paired cross-adapter contrasts, per query column")
    md.append("")
    md.append("| query col | contrast | per-seed deltas | mean | paired sd | t | same-sign |")
    md.append("|---|---|---|---|---|---|---|")
    for qc in query_cols:
        for i, ta in enumerate(train_rows):
            for tb in train_rows[i + 1:]:
                ds = []
                for s in seeds:
                    a, b = rows.get((ta, qc, s)), rows.get((tb, qc, s))
                    if a is not None and b is not None:
                        ds.append(a - b)
                if len(ds) < 2:
                    continue
                st = paired_stats(ds)
                md.append(f"| {qc} | {ta} − {tb} | " + " / ".join(f"{d:+.4f}" for d in ds)
                          + f" | {st['mean']:+.4f} | {st['sd']:.4f} | {st['t']:.2f} | {st['pos']}/{st['n']} |")
                csv_rows.append({"kind": "cross_adapter_contrast", "train_row": f"{ta}-vs-{tb}", "contrast": qc,
                                 "deltas": ";".join(f"{d:+.6f}" for d in ds),
                                 "mean": f"{st['mean']:+.6f}", "paired_sd": f"{st['sd']:.6f}",
                                 "t": f"{st['t']:.3f}", "n_pos": st["pos"], "n": st["n"]})

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "train_row", "contrast", "deltas", "mean",
                                          "paired_sd", "t", "n_pos", "n"])
        w.writeheader()
        w.writerows(csv_rows)
    print("\n".join(md))
    print(f"\nwrote {args.out_md} and {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
