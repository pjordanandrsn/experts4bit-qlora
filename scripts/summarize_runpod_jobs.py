"""Controller-only aggregation for the distributed validation runs (no GPU required).

Reads ``<jobs-root>/*/result.json`` (job-local files written by workers via
``runpod_claim_and_run.py``), ignores jobs still running (no result.json yet), keeps failed and
skipped jobs visible in a failure table, and writes the aggregate artifacts — the ONLY place
aggregates are ever written; workers never touch them:

    <results-root>/olmoe_repeat_training_all.jsonl / .csv
    <results-root>/olmoe_repeat_decode_all.jsonl   / .csv
    <results-root>/olmoe_portability_all.jsonl     / .csv
    <results-root>/summary.md    (the six tables, incl. the claim-status table)

Claim statuses are computed by explicit rules printed next to each claim — Stable / Candidate /
Host-specific / Needs repeat / Not claimed — and everything is phrased as observed on the
recorded hosts. Safe to rerun at any time.
"""

import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mode_matrix_common import matrix_table, write_csv  # noqa: E402


def _mode(r):
    return f"{r.get('storage_mode')}-{'offload' if r.get('offload') else 'resident'}"


def _fmt(v, nd=4):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None, None
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, min(vals), max(vals))


def load_results(jobs_root):
    rows = []
    for p in sorted(glob.glob(os.path.join(jobs_root, "*", "result.json"))):
        try:
            rows.append(json.load(open(p)))
        except Exception as e:
            rows.append({"job_id": os.path.basename(os.path.dirname(p)), "status": "fail",
                         "fail_or_skip_reason": f"unreadable result.json: {e}"})
    return rows


def training_tables(train_rows):
    md = ["## Training repeats: mode x seed", ""]
    ok = [r for r in train_rows if r["status"] == "pass"]
    seeds = sorted({r.get("seed") for r in ok})
    modes = sorted({_mode(r) for r in ok})
    md += ["| mode | " + " | ".join(f"seed {s}" for s in seeds) + " |",
           "|---" * (len(seeds) + 1) + "|"]
    by = {(_mode(r), r.get("seed")): r for r in ok}
    for m in modes:
        cells = []
        for s in seeds:
            r = by.get((m, s))
            cells.append("-" if r is None else
                         f"best {_fmt(r.get('train_eval_best'))} / final {_fmt(r.get('train_eval_after'))} / "
                         f"{_fmt(r.get('peak_train_gpu_gb'), 2)} GB / {_fmt(r.get('train_s_per_step'), 1)} s/step")
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    md += ["", "## Training aggregate (across seeds)", "",
           "| mode | n | best eval mean ± std [min, max] | peak GB mean | s/step mean |",
           "|---|---|---|---|---|"]
    for m in modes:
        rs = [r for r in ok if _mode(r) == m]
        mean, std, lo, hi = _mean_std([r.get("train_eval_best") for r in rs])
        pk, _, _, _ = _mean_std([r.get("peak_train_gpu_gb") for r in rs])
        sp, _, _, _ = _mean_std([r.get("train_s_per_step") for r in rs])
        md.append(f"| {m} | {len(rs)} | {_fmt(mean)} ± {_fmt(std)} [{_fmt(lo)}, {_fmt(hi)}] "
                  f"| {_fmt(pk, 2)} | {_fmt(sp, 1)} |")
    return md


def decode_table(decode_rows):
    md = ["## Decode repeats (resident; N measured samples after 1 discarded warmup)", "",
          "| mode | samples | tok/s mean ± std [min, max] | peak GB |", "|---|---|---|---|"]
    for r in sorted([r for r in decode_rows if r["status"] == "pass"], key=_mode):
        md.append(f"| {_mode(r)} | {r.get('samples')} | {_fmt(r.get('tok_s_mean'), 2)} ± "
                  f"{_fmt(r.get('tok_s_std'), 2)} [{_fmt(r.get('tok_s_min'), 2)}, {_fmt(r.get('tok_s_max'), 2)}] "
                  f"| {_fmt(r.get('peak_gpu_gb'), 2)} |")
    return md


def _portability_rows_for_matrix(query_rows):
    out = []
    for r in query_rows:
        if r.get("train_storage_mode") is None:
            continue
        variant = "offload" if r.get("train_offload") else "resident"
        out.append(dict(r,
                        train_mode_label=f"{r['train_storage_mode']}-{variant}-s{r.get('train_seed')}",
                        query_mode_label=f"{r.get('query_storage_mode')}-resident"))
    return out


def claims_table(train_rows, decode_rows):
    """Explicit, printed rules; statuses are observations on the recorded hosts, not warranties."""
    ok = [r for r in train_rows if r["status"] == "pass"]
    seeds = sorted({r.get("seed") for r in ok})
    by = {(_mode(r), r.get("seed")): r for r in ok}
    dec = {_mode(r): r for r in decode_rows if r["status"] == "pass"}
    md = ["## Claim status (rule shown per claim; all host-specific observations)", "",
          "| claim | evidence | rule | status |", "|---|---|---|---|"]

    def add(claim, evidence, rule, status):
        md.append(f"| {claim} | {evidence} | {rule} | **{status}** |")

    # int8-offload best training eval
    wins = 0
    for s in seeds:
        target = by.get(("int8-offload", s))
        others = [by[k] for k in by if k[1] == s and k[0] != "int8-offload"]
        if target and others and all(
            target.get("train_eval_best") is not None and o.get("train_eval_best") is not None
            and target["train_eval_best"] < o["train_eval_best"] for o in others
        ):
            wins += 1
    n = len(seeds)
    status = ("Stable (host-specific)" if n >= 3 and wins == n else
              "Candidate" if wins >= max(2, n - 1) and n >= 2 else
              "Needs repeat" if n < 3 else "Not claimed")
    add("int8-offload posts the best training eval",
        f"best-eval wins in {wins}/{n} seeds", "win in all seeds vs every other mode", status)

    # fp4 resident decode faster than nf4
    f, nn = dec.get("fp4-resident"), dec.get("nf4-resident")
    if f and nn and f.get("tok_s_mean") is not None:
        sep = (f["tok_s_mean"] - f.get("tok_s_std", 0)) > (nn["tok_s_mean"] + nn.get("tok_s_std", 0))
        higher = f["tok_s_mean"] > nn["tok_s_mean"]
        status = "Stable (host-specific)" if sep else ("Candidate" if higher else "Not claimed")
        add("fp4 resident decode faster than nf4 (this host/path)",
            f"fp4 {f['tok_s_mean']:.2f}±{f.get('tok_s_std', 0):.2f} vs nf4 "
            f"{nn['tok_s_mean']:.2f}±{nn.get('tok_s_std', 0):.2f} tok/s",
            "means separated by >1 std each", status)
    else:
        add("fp4 resident decode faster than nf4", "decode repeats not complete", "-", "Needs repeat")

    # offload memory-floor collapse + resident width scaling
    def peaks(mode):
        vals = [by[k].get("peak_train_gpu_gb") for k in by if k[0] == mode]
        m, _, _, _ = _mean_std(vals)
        return m

    n4r, n4o, i8r, i8o = peaks("nf4-resident"), peaks("nf4-offload"), peaks("int8-resident"), peaks("int8-offload")
    if None not in (n4r, n4o, i8r, i8o):
        res_delta, off_delta = i8r - n4r, i8o - n4o
        ratio = off_delta / res_delta if res_delta else None
        add("offload collapses the storage-width memory difference",
            f"resident delta {res_delta:.2f} GB vs offload delta {off_delta:.2f} GB (ratio {_fmt(ratio, 2)})",
            "offload delta < 0.25x resident delta",
            "Stable (host-specific)" if ratio is not None and ratio < 0.25 else "Candidate")
        add("resident training memory scales with storage width",
            f"int8 {i8r:.2f} GB vs nf4 {n4r:.2f} GB resident",
            "int8 resident peak > 1.4x nf4 resident peak",
            "Stable (host-specific)" if i8r > 1.4 * n4r else "Candidate")
    else:
        add("offload memory-floor / width-scaling claims", "training repeats not complete", "-", "Needs repeat")

    # fidelity ordering before training
    order_ok = order_n = 0
    for s in seeds:
        i8 = by.get(("int8-resident", s)) or by.get(("int8-offload", s))
        n4 = by.get(("nf4-resident", s)) or by.get(("nf4-offload", s))
        if i8 and n4 and i8.get("train_eval_before") is not None and n4.get("train_eval_before") is not None:
            order_n += 1
            order_ok += i8["train_eval_before"] < n4["train_eval_before"]
    add("BEFORE-training eval tracks fidelity ordering (int8 < nf4)",
        f"holds in {order_ok}/{order_n} seed-matched pairs", "holds in all pairs",
        "Stable (host-specific)" if order_n >= 3 and order_ok == order_n else
        ("Candidate" if order_ok and order_ok == order_n else "Needs repeat"))
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-root", required=True)
    ap.add_argument("--results-root", required=True)
    args = ap.parse_args()

    rows = load_results(args.jobs_root)
    train = [r for r in rows if r.get("job_type") == "train"]
    decode = [r for r in rows if r.get("job_type") == "decode"]
    query = [r for r in rows if r.get("job_type") == "query"]
    failed = [r for r in rows if r.get("status") != "pass"]

    os.makedirs(args.results_root, exist_ok=True)
    for name, subset in (("olmoe_repeat_training_all", train), ("olmoe_repeat_decode_all", decode),
                         ("olmoe_portability_all", query)):
        with open(os.path.join(args.results_root, f"{name}.jsonl"), "w") as f:
            for r in subset:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        write_csv(os.path.join(args.results_root, f"{name}.csv"), subset)

    md = ["# Distributed validation summary (controller aggregate)", "",
          f"jobs with results: {len(rows)} (train {len(train)}, decode {len(decode)}, query {len(query)}) "
          f"| non-pass: {len(failed)}",
          "", "Observed on the recorded hosts/pods — not universal claims.", ""]
    md += training_tables(train) + [""]
    md += decode_table(decode) + [""]
    pq = [r for r in _portability_rows_for_matrix(query) if r["status"] == "pass"]
    md += ["## Portability: eval with adapter (train row x query mode)", "",
           matrix_table(pq, "eval_loss_with_adapter") if pq else "*no query jobs completed yet*", ""]
    md += ["## Portability: delta vs query-mode no-adapter baseline", "",
           matrix_table(pq, "delta_vs_base_query_mode") if pq else "*no query jobs completed yet*", ""]
    md += claims_table(train, decode) + [""]
    if failed:
        md += ["## Failed / skipped / unreadable jobs", ""]
        md += [f"- {r.get('job_id')}: {r.get('status')} — {r.get('fail_or_skip_reason')}" for r in failed]
        md.append("")
    out_md = os.path.join(args.results_root, "summary.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote aggregates under {args.results_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
