"""Summarize a train/query mode matrix JSONL into Markdown + CSV.

Emits: the train-x-query eval-loss table, the delta-vs-no-adapter-baseline table, the query-cost
table (peak GPU / decode tok/s where measured), best-per-train-mode and best-per-query-mode
tables, and the same-mode / upward / downward / offload-transfer / symmetry observations. All
observations are phrased as observed-in-this-run; this is a storage-mode portability test, not a
benchmark, and it does not prove universal adapter compatibility.

Example:
    python scripts/summarize_train_query_matrix.py \\
      --input runs/olmoe_mode_adapters/query_matrix.jsonl \\
      --out-md runs/olmoe_mode_adapters/query_matrix.md \\
      --out-csv runs/olmoe_mode_adapters/query_matrix.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mode_matrix_common import (  # noqa: E402
    best_per,
    matrix_table,
    read_jsonl,
    transfer_summary,
    write_csv,
)


def render_markdown(rows) -> str:
    passed = [r for r in rows if r["status"] == "pass"]
    failed = [r for r in rows if r["status"] != "pass"]
    hosts = sorted({str(r.get("host")) for r in rows})
    gpus = sorted({str(r.get("gpu")) for r in rows})
    models = sorted({str(r.get("base_model")) for r in rows})

    md = ["# Train/query storage-mode matrix", ""]
    md += [f"- base model: {', '.join(models)}",
           f"- host: {', '.join(hosts)} | GPU: {', '.join(gpus)}",
           f"- legs: {len(passed)} pass / {len(failed)} fail-or-skip",
           "",
           "Read as a storage-mode portability test observed on this host/model/dataset/run — "
           "not a benchmark, not a universal compatibility claim.",
           ""]

    md += ["## Held-out eval loss (with adapter)", "",
           matrix_table(passed, "eval_loss_with_adapter"), ""]
    md += ["## Delta vs same-query-mode no-adapter baseline (negative = adapter helps)", "",
           matrix_table(passed, "delta_vs_base_query_mode"), ""]
    md += ["## Query cost (peak GPU GB; decode tok/s not measured this pass)", "",
           matrix_table(passed, "peak_gpu_query_gb", nd=2), ""]

    for group, title in (("train_mode_label", "Best query mode per train mode"),
                         ("query_mode_label", "Best train mode per query mode")):
        md += [f"## {title}", ""]
        for g, r in sorted(best_per(passed, group).items()):
            other = r["query_mode_label"] if group == "train_mode_label" else r["train_mode_label"]
            md.append(f"- `{g}` -> `{other}`: eval {r['eval_loss_with_adapter']:.4f}")
        md.append("")

    md += ["## Transfer observations (this run)", ""]
    md += [f"- {line}" for line in transfer_summary(passed)]
    if failed:
        md += ["", "## Failed / skipped legs", ""]
        md += [f"- {r['train_mode_label']} -> {r['query_mode_label']}: {r['status'].upper()} — "
               f"{r.get('skip_or_fail_reason')}" for r in failed]
    md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit(f"no rows in {args.input}")
    md = render_markdown(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(md)
    write_csv(args.out_csv, rows)
    print(md)
    print(f"wrote {args.out_md} and {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
