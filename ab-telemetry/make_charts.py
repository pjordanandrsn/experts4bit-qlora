#!/usr/bin/env python3
"""Render the expert-offload A/B charts from two METRICS_JSONL files (offload off vs on).

Usage: make_charts.py OFF.jsonl ON.jsonl OUTDIR
Produces loss_curve.png, vram.png, throughput.png in OUTDIR and prints a markdown perf table.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OFF_C, ON_C = "#d9534f", "#0275d8"


def load(path):
    cfg, summ, steps, evals = {}, {}, [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") == "config":
                cfg = r
            elif r.get("event") == "summary":
                summ = r
            elif "train/loss" in r:
                steps.append(r)
            elif "eval/held_out_loss" in r:
                evals.append(r)
    return dict(cfg=cfg, summ=summ, steps=steps, evals=evals)


def rolling(xs, ys, k=5):
    out = []
    for i in range(len(ys)):
        lo = max(0, i - k + 1)
        out.append(sum(ys[lo : i + 1]) / (i - lo + 1))
    return xs, out


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def median_step_time(d):
    """Median per-step time, excluding eval steps (mid-loop eval inflates those)."""
    eval_steps = {r["step"] for r in d["evals"] if r["step"] > 0}
    times = [r["perf/s_per_step"] for r in d["steps"] if r["step"] not in eval_steps]
    return median(times)


def main():
    off, on, outdir = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    model = off["cfg"].get("model", "MoE").split("/")[-1]

    # 1. Loss: train EMA (line) + held-out eval (dashed w/ markers), off vs on overlaid.
    plt.figure(figsize=(7, 4.5))
    for d, label, c in [(off, "offload off", OFF_C), (on, "offload on", ON_C)]:
        xs = [r["step"] for r in d["steps"]]
        ys = [r["train/ema"] for r in d["steps"]]
        if xs:
            plt.plot(xs, ys, color=c, lw=1.8, label=f"{label} — train EMA")
        ex = [r["step"] for r in d["evals"]]
        ey = [r["eval/held_out_loss"] for r in d["evals"]]
        if ex:
            plt.plot(ex, ey, color=c, marker="o", ls="--", lw=1.2, alpha=0.75, label=f"{label} — held-out")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"{model} QLoRA — training loss, expert offload off vs on")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/loss_curve.png", dpi=130)
    plt.close()

    # 2. VRAM: allocated over step + peak dashed line, off vs on.
    plt.figure(figsize=(7, 4.5))
    for d, label, c in [(off, "offload off", OFF_C), (on, "offload on", ON_C)]:
        xs = [r["step"] for r in d["steps"]]
        ys = [r["mem/gpu_alloc_gb"] for r in d["steps"]]
        if xs:
            plt.plot(xs, ys, color=c, lw=1.6, label=f"{label} — allocated")
        peak = d["summ"].get("peak_gpu_gb")
        if peak:
            plt.axhline(peak, color=c, ls=":", lw=1.2, alpha=0.8, label=f"{label} — peak {peak:.2f} GB")
    plt.xlabel("step")
    plt.ylabel("GPU memory (GB)")
    plt.title(f"{model} QLoRA — GPU memory, expert offload off vs on")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/vram.png", dpi=130)
    plt.close()

    # 3. Throughput: s/step (5-step rolling mean), off vs on. Eval steps are excluded —
    #    the mid-loop held-out eval inflates that step's wall time and smears the mean.
    plt.figure(figsize=(7, 4.5))
    for d, label, c in [(off, "offload off", OFF_C), (on, "offload on", ON_C)]:
        # exclude the eval step AND the step after it — the eval + best-checkpoint save
        # runs between those two timer marks, so it lands on step+1's wall time.
        ev = {r["step"] for r in d["evals"] if r["step"] > 0}
        eval_steps = ev | {s + 1 for s in ev}
        pts = [(r["step"], r["perf/s_per_step"]) for r in d["steps"] if r["step"] not in eval_steps]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            rx, ry = rolling(xs, ys)
            plt.plot(rx, ry, color=c, lw=1.8, label=f"{label} (median {median_step_time(d):.0f}s/step)")
    plt.xlabel("step (eval steps excluded)")
    plt.ylabel("s / step (5-step rolling mean)")
    plt.title(f"{model} QLoRA — training step time, expert offload off vs on")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/throughput.png", dpi=130)
    plt.close()

    # Markdown perf table from summaries.
    def row(tag, d):
        s = d["summ"]
        return (
            f"| {tag} | {s.get('loaded_gpu_gb', float('nan')):.2f} | {s.get('peak_gpu_gb', float('nan')):.2f} "
            f"| {median_step_time(d):.1f} | {s.get('eval_before', float('nan')):.4f} "
            f"| {s.get('eval_after', float('nan')):.4f} | {s.get('eval_delta', float('nan')):+.4f} |"
        )

    print(f"\nPerf table ({model}):\n")
    print("| config | loaded GPU (GB) | peak GPU (GB) | median s/step | eval BEFORE | eval AFTER | delta |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    print(row("offload off", off))
    print(row("offload on", on))
    off_t, on_t = median_step_time(off), median_step_time(on)
    if off_t == off_t and on_t == on_t and off_t > 0:
        print(f"\nthroughput cost of offload: {(on_t / off_t - 1) * 100:+.1f}% s/step")
    print(f"\nCharts written to {outdir}/ : loss_curve.png, vram.png, throughput.png")


if __name__ == "__main__":
    main()
