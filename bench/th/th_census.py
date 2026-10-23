#!/usr/bin/env python
"""T-H: the training kernel census, T-B's deciding instrument (Map 1; bench/train-capture/DESIGN-TB.md section 4). EXPLORE tier:
a design input, never a reading.

One profiled optimizer step of the fused training path (Qwen3-30B-A3B at the field recipe's shape: rows of 512 tokens, batch 1,
4 micro-batches a step, r 16, fp32 adapters) on one RTX 5090. Each launch is classified by kernel family and phase, and the
host time it cost is attributed to it. The share each T-B option could remove decides between them by the rule fixed in the
note, before any data.

Everything below the CPU line is fixed here BEFORE a profile exists:
  FAMILY_PATTERNS   the five-family classifier (first match wins); anything else is "unclassified"
  attribute()       each kernel joined to its launch (the runtime call with its correlation id); the launch's phase by time
                    containment in record_function ranges (recompute > backward > forward > optimizer); its host time = the
                    gap on its thread since the previous launch or the phase start, minus the host syncs inside that gap
  decide()          DESIGN-TB's rule on those shares, FAILING CLOSED (bench/common rule 7): no launches, a missing phase, fewer
                    than 99 % of kernels joined to a launch, or unclassified above 10 % of host launch time reads VOID
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

FAMILY_PATTERNS = (                       # first match wins; the order is part of the rule
    ("moe_expert", ("nf4", "gnf4", "_grouped", "dgrad", "lora_delta", "dequant", "grouped_mm", "scatter_combine", "swiglu_rows",
                    "combine_rows")),
    ("attention", ("flash", "fmha", "sdpa", "attention", "attn")),
    ("norm_rope", ("rms", "rope", "rotary", "layer_norm", "layernorm")),
    ("dense_gemm", ("gemm", "gemv", "cutlass", "xmma", "cublas", "matmul", "magma", "splitk", "sm80_", "sm90_", "sm100_",
                    "sm120_", "ampere_")),
    ("glue", ("elementwise", "vectorized", "unrolled", "copy", "fill", "gather", "scatter", "index", "cat", "reduce", "sum",
              "softmax", "where", "mul", "add", "topk", "sort", "radix", "scan", "multi_tensor", "adam", "cast", "embedding",
              "nll", "cross_entropy", "arange", "masked", "clamp", "div", "neg", "exp", "pow", "sqrt", "silu", "sigmoid",
              "memset", "memcpy", "triu", "tril", "compare", "bitwise", "max", "min", "mean", "norm_kernel")),
)
FAMILIES = tuple(f for f, _ in FAMILY_PATTERNS) + ("unclassified",)
PHASES = ("forward", "recompute", "backward", "optimizer")
B_FAMILIES = ("attention", "dense_gemm", "norm_rope")
LAUNCH_NAMES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx", "cudaLaunchCooperativeKernel",
                "cudaMemsetAsync", "cudaMemcpyAsync", "cudaGraphLaunch")
SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy", "cuStreamSynchronize")
SHARE_BAR, UNCLASSIFIED_MAX, MATCH_MIN = 0.40, 0.10, 0.99


def classify(kernel_name: str) -> str:
    n = kernel_name.lower()
    for fam, pats in FAMILY_PATTERNS:
        if any(p in n for p in pats):
            return fam
    return "unclassified"


def phase_of(t_ns: int, ranges: list[tuple[str, int, int]]) -> str | None:
    """The phase whose record_function range contains t; recompute wins over backward (it nests inside it in time)."""
    hit = [name for name, s, e in ranges if s <= t_ns <= e]
    for ph in ("recompute", "backward", "forward", "optimizer"):
        if ph in hit:
            return ph
    return None


def attribute(launches, kernels, syncs, ranges):
    """launches: [{corr, thread, start, end}] (runtime calls, ns); kernels: [{corr, name, dur}]; syncs: [{thread, start, end}];
    ranges: [(phase, start, end)]. Returns the census table and the join ratio."""
    by_corr = {}
    for ln in launches:
        by_corr.setdefault(ln["corr"], ln)
    table = {ph: {f: {"launches": 0, "host_ns": 0, "device_ns": 0} for f in FAMILIES} for ph in PHASES}
    host_of = {}
    per_thread = {}
    for ln in sorted(launches, key=lambda x: x["start"]):
        per_thread.setdefault(ln["thread"], []).append(ln)
    for th, lns in per_thread.items():
        prev_end = None
        for ln in lns:
            ph = phase_of(ln["start"], ranges)
            ph_start = min((s for name, s, e in ranges if name == ph and s <= ln["start"] <= e), default=ln["start"])
            gap_start = ln["start"] if prev_end is None else max(prev_end, ph_start)
            waited = sum(min(sy["end"], ln["start"]) - max(sy["start"], gap_start) for sy in syncs
                         if sy["thread"] == th and sy["end"] > gap_start and sy["start"] < ln["start"])
            host_of[ln["corr"]] = max(0, ln["end"] - gap_start - max(0, waited))
            prev_end = ln["end"]
    joined = 0
    for k in kernels:
        ln = by_corr.get(k["corr"])
        if ln is None:
            continue
        ph = phase_of(ln["start"], ranges)
        if ph is None:
            continue
        joined += 1
        cell = table[ph][classify(k["name"])]
        cell["launches"] += 1
        cell["device_ns"] += int(k["dur"])
        cell["host_ns"] += host_of.pop(k["corr"], 0)          # a launch's host time counts once, with its first kernel
    return table, (joined / len(kernels) if kernels else 0.0)


def decide(table, join_ratio):
    """DESIGN-TB section 4's rule, failing closed. Shares are of total host launch time in the step."""
    tot = sum(c["host_ns"] for ph in table.values() for c in ph.values())
    why = []
    if tot <= 0:
        why.append("no host launch time")
    for ph in ("forward", "backward"):
        if sum(c["launches"] for c in table.get(ph, {}).values()) == 0:
            why.append(f"no {ph} launches")
    if join_ratio < MATCH_MIN:
        why.append(f"only {join_ratio:.3f} of kernels joined to a launch (< {MATCH_MIN})")
    if why:
        return {"verdict": "VOID", "why": why}
    fam_host = {f: sum(table[ph][f]["host_ns"] for ph in PHASES) for f in FAMILIES}
    shares = {"A": 1.0,
              "B": sum(table["forward"][f]["host_ns"] for f in B_FAMILIES) / tot,
              "C": fam_host["glue"] / tot,
              "unclassified": fam_host["unclassified"] / tot,
              "moe_expert": fam_host["moe_expert"] / tot}
    out = {"shares": {k: round(v, 4) for k, v in shares.items()}, "host_launch_ms": round(tot / 1e6, 2)}
    if shares["unclassified"] > UNCLASSIFIED_MAX:
        return {**out, "verdict": "VOID", "why": [f"unclassified {shares['unclassified']:.3f} > {UNCLASSIFIED_MAX}: amend the patterns"]}
    if shares["B"] >= SHARE_BAR:
        return {**out, "verdict": "B", "why": [f"B's forward share {shares['B']:.3f} >= {SHARE_BAR}"]}
    if shares["C"] >= SHARE_BAR:
        return {**out, "verdict": "C", "why": [f"glue share {shares['C']:.3f} >= {SHARE_BAR}"]}
    return {**out, "verdict": "A", "why": [f"neither B ({shares['B']:.3f}) nor C ({shares['C']:.3f}) reaches {SHARE_BAR}: "
                                           "only whole-step capture reaches the rest; its prerequisites get their own design"]}


def single_thread_mops(n=5_000_000):
    """The box's single-thread rate: a pure-Python counting loop, million iterations a second (median of 3)."""
    rates = []
    for _ in range(3):
        t = time.perf_counter()
        i = 0
        while i < n:
            i += 1
        rates.append(n / (time.perf_counter() - t) / 1e6)
    return round(sorted(rates)[1], 2)


def self_test() -> int:
    names = {"_gemm_nf4_grouped": "moe_expert", "_dgrad_nf4_grouped": "moe_expert", "flash_fwd_kernel": "attention",
             "_rms_norm_fwd": "norm_rope", "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16>": "dense_gemm",
             "void at::native::vectorized_elementwise_kernel<4>": "glue", "multi_tensor_apply_kernel": "glue",
             "some_new_kernel_xyz": "unclassified", "dequant_groups_kernel": "moe_expert"}
    for n, f in names.items():
        assert classify(n) == f, (n, classify(n), f)
    ms = 1_000_000
    ranges = [("forward", 0, 10 * ms), ("backward", 10 * ms, 30 * ms), ("recompute", 12 * ms, 15 * ms),
              ("optimizer", 30 * ms, 32 * ms)]
    launches = [{"corr": 1, "thread": 1, "start": 1 * ms, "end": 1 * ms + 10_000},
                {"corr": 2, "thread": 1, "start": 4 * ms, "end": 4 * ms + 10_000},
                {"corr": 3, "thread": 2, "start": 13 * ms, "end": 13 * ms + 10_000},
                {"corr": 4, "thread": 2, "start": 20 * ms, "end": 20 * ms + 10_000},
                {"corr": 5, "thread": 1, "start": 31 * ms, "end": 31 * ms + 10_000}]
    kernels = [{"corr": 1, "name": "flash_fwd_kernel", "dur": 100}, {"corr": 2, "name": "_gemm_nf4_grouped", "dur": 500},
               {"corr": 3, "name": "flash_fwd_kernel", "dur": 100}, {"corr": 4, "name": "vectorized_elementwise_kernel", "dur": 50},
               {"corr": 5, "name": "multi_tensor_apply_kernel", "dur": 80}]
    syncs = [{"thread": 1, "start": 2 * ms, "end": 3 * ms}]          # a 1 ms host sync between launches 1 and 2
    table, jr = attribute(launches, kernels, syncs, ranges)
    assert jr == 1.0
    assert table["forward"]["attention"]["launches"] == 1 and table["recompute"]["attention"]["launches"] == 1
    assert table["backward"]["glue"]["launches"] == 1 and table["optimizer"]["glue"]["launches"] == 1
    gap2 = table["forward"]["moe_expert"]["host_ns"]
    assert gap2 == (4 * ms + 10_000) - (1 * ms + 10_000) - 1 * ms, gap2       # the sync's 1 ms is not launch cost
    opt = table["optimizer"]["glue"]["host_ns"]
    assert opt == (31 * ms + 10_000) - 30 * ms, opt                            # the gap starts at the phase, not at backward
    d = decide(table, jr)
    assert d["verdict"] in ("A", "B", "C"), d
    # the rule: each branch, then each fail-closed case with its evidence removed
    def t(fwd_b, glue, unc, other):
        tb = {ph: {f: {"launches": 0, "host_ns": 0, "device_ns": 0} for f in FAMILIES} for ph in PHASES}
        tb["forward"]["attention"].update(launches=1, host_ns=fwd_b)
        tb["backward"]["glue"].update(launches=1, host_ns=glue)
        tb["backward"]["unclassified"].update(launches=1, host_ns=unc)
        tb["backward"]["moe_expert"].update(launches=1, host_ns=other)
        return tb
    assert decide(t(50, 10, 1, 39), 1.0)["verdict"] == "B"
    assert decide(t(10, 50, 1, 39), 1.0)["verdict"] == "C"
    assert decide(t(45, 45, 1, 9), 1.0)["verdict"] == "B"                     # both qualify: B first (no kernel change)
    assert decide(t(10, 10, 5, 75), 1.0)["verdict"] == "A"
    assert decide(t(10, 10, 20, 60), 1.0)["verdict"] == "VOID"                # unclassified 0.2 > 0.1
    assert decide(t(50, 10, 1, 39), 0.95)["verdict"] == "VOID"                # under 99 % of kernels joined
    empty = {ph: {f: {"launches": 0, "host_ns": 0, "device_ns": 0} for f in FAMILIES} for ph in PHASES}
    assert decide(empty, 1.0)["verdict"] == "VOID"                            # no evidence at all
    nofwd = t(0, 50, 1, 49)
    nofwd["forward"]["attention"]["launches"] = 0
    assert decide(nofwd, 1.0)["verdict"] == "VOID"                            # a missing phase
    assert single_thread_mops(200_000) > 0
    print("th self-test: classifier, attribution, rule and 4 fail-closed cases passed")
    return 0


# ---------------------------------------------------------------------------------------------------------------- the box


def _ladder():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moegen"))
    import ladder
    return ladder


def raw_events(prof):
    """The profiler's raw Kineto events as plain dicts (name, kind, start/end ns, thread, correlation id)."""
    out = []
    for e in prof.profiler.kineto_results.events():
        dev = str(e.device_type())
        s = int(e.start_ns())
        out.append({"name": e.name(), "cuda": "CUDA" in dev, "start": s, "end": s + int(e.duration_ns()),
                    "corr": int(e.correlation_id()), "thread": int(e.start_thread_id())})
    return out


def census_from(events):
    ranges = [(e["name"].split(":", 1)[1], e["start"], e["end"]) for e in events
              if not e["cuda"] and e["name"].startswith("phase:")]
    launches = [{"corr": e["corr"], "thread": e["thread"], "start": e["start"], "end": e["end"]} for e in events
                if not e["cuda"] and e["name"] in LAUNCH_NAMES]
    syncs = [{"thread": e["thread"], "start": e["start"], "end": e["end"]} for e in events
             if not e["cuda"] and e["name"] in SYNC_NAMES]
    kernels = [{"corr": e["corr"], "name": e["name"], "dur": e["end"] - e["start"]} for e in events if e["cuda"]]
    table, jr = attribute(launches, kernels, syncs, ranges)
    return table, jr, {"ranges": len(ranges), "launch_calls": len(launches), "syncs": len(syncs), "kernels": len(kernels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--adapter-dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="th_census.json")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    ladder = _ladder()
    import torch
    from transformers import AutoTokenizer
    from experts4bit_qlora import load_moe_4bit_streaming, verify_moe_4bit
    from experts4bit_qlora.lora import (add_attention_lora, quantize_attention_projections_4bit, trainable_lora_param_ids,
                                        ExpertsLoRA)
    from experts4bit_qlora.engines.fast import enable_fast_train
    from torch.profiler import ProfilerActivity, profile, record_function

    mops = single_thread_mops()
    ladder.prewarm(a.model, a.revision)
    model, cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, a.r, a.alpha, offload=False, revision=a.revision)
    model.to("cuda")
    verify_moe_4bit(model, strict=True)
    adt = torch.float32 if a.adapter_dtype == "fp32" else torch.bfloat16
    for m in model.modules():
        if isinstance(m, ExpertsLoRA):
            for n, p in m.named_parameters(recurse=False):
                if "lora" in n:
                    p.data = p.data.to(adt)
    quantize_attention_projections_4bit(model)
    add_attention_lora(model, a.r, a.alpha, adt)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    ex_ids, at_ids = trainable_lora_param_ids(model)
    train = [p for p in model.parameters() if id(p) in ex_ids or id(p) in at_ids]
    for p in model.parameters():
        p.requires_grad_(id(p) in ex_ids or id(p) in at_ids)
    n_patched = enable_fast_train(model, verbose=False, dgrad=True)
    # the checkpoint recompute: a decoder layer called during the backward is marked "phase:recompute"
    have_gt = hasattr(torch._C, "_current_graph_task_id")
    layers = getattr(getattr(model, "model", model), "layers", [])
    for layer in layers:
        f = layer.forward

        def g(*args, _f=f, **kw):
            if have_gt and torch._C._current_graph_task_id() != -1:
                with record_function("phase:recompute"):
                    return _f(*args, **kw)
            return _f(*args, **kw)
        layer.forward = g
    opt = torch.optim.AdamW(train, lr=1e-5)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    batches = ladder.alpaca_batches(tok, a.seq, 1, a.accum * (a.warmup + 1))

    def step(bs, mark):
        for b in bs:
            ids = b.to("cuda")
            with record_function("phase:forward") if mark else _null():
                loss = model(input_ids=ids, labels=ids).loss / a.accum
            with record_function("phase:backward") if mark else _null():
                loss.backward()
        with record_function("phase:optimizer") if mark else _null():
            opt.step()
            opt.zero_grad(set_to_none=True)

    for i in range(a.warmup):
        step(batches[i * a.accum:(i + 1) * a.accum], False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t = time.perf_counter()
        step(batches[a.warmup * a.accum:(a.warmup + 1) * a.accum], True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
    events = raw_events(prof)
    table, jr, counts = census_from(events)
    verdict = decide(table, jr)
    import importlib.metadata as md
    out = {"tier": "EXPLORE (T-H census, Map 1): a design input, never a reading", "model": a.model, "seq": a.seq,
           "accum": a.accum, "r": a.r, "alpha": a.alpha, "adapter_dtype": a.adapter_dtype, "n_patched": n_patched,
           "recompute_marked": have_gt and bool(layers), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "versions": {"e4b": md.version("experts4bit-qlora"), "gnf4": md.version("grouped-nf4-gemm")},
           "cpu": (open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t")
                   if os.path.exists("/proc/cpuinfo") else None),
           "single_thread_mops": mops, "profiled_wall_s": round(wall, 4), "join_ratio": round(jr, 4), "counts": counts,
           "table": table, "decision": verdict, "rule": {"share_bar": SHARE_BAR, "unclassified_max": UNCLASSIFIED_MAX,
                                                        "match_min": MATCH_MIN, "b_families": B_FAMILIES}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    unc = sorted({e["name"][:120] for e in events if e["cuda"] and classify(e["name"]) == "unclassified"})[:40]
    Path(a.out).with_suffix(".unclassified.txt").write_text("\n".join(unc) + "\n")
    print(f"[th] {out['cpu']} {mops} Mops; wall {wall:.3f}s; join {jr:.4f}; {counts}", flush=True)
    print(f"[th] DECISION {json.dumps(verdict)}", flush=True)
    return 0


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    sys.exit(main())
