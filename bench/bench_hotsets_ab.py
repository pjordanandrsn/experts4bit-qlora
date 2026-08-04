"""bench_hotsets_ab.py — does an INFORMED hot set beat a by-index one on this model?

`docs/RESIDENCY-ENGINES.md` calls the hot-set choice "the single largest lever": informed
sets worth +37.1% on DeepSeek-V4-Flash/A5000 and +120% on gpt-oss-20b K=8, with by-index
sets statistically identical to pure streaming. It also warns the gain is a property of
the HOST. This driver answers the question per (model, host) instead of assuming it.

Three configs plus a control, run INTERLEAVED (A,B,C,A,B,C,...) rather than blocked, so
drift and any co-tenant load on a shared card hit every config equally:

  stream     hot_set = ()        every routed expert streams from host
  index      hot_set = 0..K-1    the naive pick
  informed   hot_set = top-K by routed-token count, per layer
  unwrapped  informed, with the ExpertsLoRA wrappers stripped — a control for whether
             delegating through the wrapper costs anything at decode

WHY THIS REPORTS BYTES AS WELL AS TOKENS/S. Coverage predicts READ REDUCTION, not speed;
the two only agree when reads bind. Printing the engine's own traffic counters next to
the timings is what distinguishes "the dial does nothing here" from "the dial worked and
the bottleneck is elsewhere" — which are the same number and different conclusions.

Measured on granite-3.0-1b-a400m/A2000 2026-08-04, the total gather column turned out to
be INVARIANT across configs: a hot set does not reduce bytes copied, it moves them from
the cold (PCIe) column to the hot (device-to-device) column, because every routed expert
is copied into a slot whether or not it was already resident. See
`_PipelinedResidency.__init__`'s note on the accepted re-copy inefficiency.

Env: MODEL, HOT_K (default = the model's top-k), N_TOK, REPS, OUT.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time

import torch

MODEL = os.environ.get("MODEL", "ibm-granite/granite-3.0-1b-a400m-instruct")
DEV = os.environ.get("DEVICE", "cuda")
N_TOK = int(os.environ.get("N_TOK", "128"))
REPS = int(os.environ.get("REPS", "11"))
OUT = os.environ.get("OUT", "")
PROMPT = ("Explain, in careful detail, how a mixture-of-experts transformer routes each "
          "token to a subset of its experts, and why that sparsity makes offloading "
          "expert weights to host memory practical:")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def mann_whitney(a, b):
    """Two-sided U test, normal approximation. Inline so the driver needs no scipy.

    N=11 per config with a 10-40% spread cannot resolve a single-digit effect, and a
    median-vs-median percentage invites reading noise as a result — this run's own
    3-rep pilot reported -6.3% where 11 reps gave +0.7%, sign included.
    """
    n1, n2 = len(a), len(b)
    allv = sorted(a + b)
    def rank(x):
        i = allv.index(x)
        j = len(allv) - 1 - allv[::-1].index(x)
        return (i + j) / 2 + 1
    u1 = sum(rank(x) for x in a) - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    z = (u - n1 * n2 / 2) / math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return u, math.erfc(abs(z) / math.sqrt(2))


def timed_decode(model, tok, n_tokens):
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEV)
    out = model(input_ids=ids, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_tokens):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    return n_tokens / (time.time() - t0)


def main() -> int:
    from transformers import AutoTokenizer

    from experts4bit_qlora.hot_residency import dispatched_modules, target_modules
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import ExpertsLoRA
    from experts4bit_qlora.pipelined import (disable_pipelined_residency,
                                             enable_pipelined_residency)

    log(f"loading {MODEL} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model, _ = load_moe_4bit_streaming(MODEL, DEV, torch.bfloat16, 8, 16,
                                       offload=False, prefetch=False, quant_type="nf4")
    model.eval()
    cfg = model.config
    n_exp = getattr(cfg, "num_local_experts", None) or cfg.num_experts
    k_slots = int(getattr(cfg, "num_experts_per_tok", None) or cfg.num_experts_per_token)
    hot_k = int(os.environ.get("HOT_K", k_slots))
    mods = target_modules(model)
    log(f"{len(mods)} MoE layers, {n_exp} experts, top-{k_slots}, HOT_K={hot_k}")

    def traffic():
        t = [m._pipelined.traffic() for m in target_modules(model) if hasattr(m, "_pipelined")]
        return (sum(x["cold_pcie_bytes"] for x in t), sum(x["hot_d2d_bytes"] for x in t))

    # -- calibration ---------------------------------------------------------------
    # Hook `dispatched_modules`, NOT `target_modules`. The latter returns the frozen
    # bases, which are not called until an engine is attached — and this pass runs
    # before that by construction, so those hooks would count nothing, `topk` of zeros
    # would return 0..K-1, and `informed` would silently become a second copy of
    # `index`. That is what this driver measured on its first run.
    counts = [torch.zeros(n_exp, dtype=torch.long) for _ in mods]

    def _mk(slot):
        def _hook(_m, args, kwargs):
            idx = next((a for a in list(args) + list(kwargs.values())
                        if isinstance(a, torch.Tensor) and not a.is_floating_point()), None)
            if idx is not None:
                counts[slot] += torch.bincount(idx.reshape(-1).long().cpu(), minlength=n_exp)
        return _hook

    handles = [m.register_forward_pre_hook(_mk(i), with_kwargs=True)
               for i, m in enumerate(dispatched_modules(model))]
    log("calibration pass ...")
    with torch.no_grad():
        timed_decode(model, tok, 48)
    for h in handles:
        h.remove()
    if sum(int(c.sum()) for c in counts) == 0:
        raise RuntimeError(
            "calibration counted ZERO routed selections — hooks landed on modules nothing "
            "dispatched. `informed` would silently equal `index` and this A/B would report "
            "a confident null. Use experts4bit_qlora.dispatched_modules().")

    informed, cov_inf, cov_idx = [], [], []
    for c in counts:
        informed.append(torch.topk(c, hot_k).indices.sort().values.tolist())
        tot = max(int(c.sum()), 1)
        cov_inf.append(float(c[informed[-1]].sum()) / tot)
        cov_idx.append(float(c[:hot_k].sum()) / tot)
    COV_INF, COV_IDX = sum(cov_inf) / len(cov_inf), sum(cov_idx) / len(cov_idx)
    uniform = hot_k / n_exp
    log(f"coverage: informed {COV_INF:.1%}, index {COV_IDX:.1%}, uniform {uniform:.1%} "
        f"({COV_INF/uniform:.2f}x uniform — "
        f"{'no skew, expect a null' if COV_INF < 1.25 * uniform else 'skew present'})")

    configs = {"stream": [[] for _ in mods],
               "index": [list(range(hot_k)) for _ in mods],
               "informed": informed}
    res, byts = {k: [] for k in configs}, {}
    for rep in range(REPS):
        for name, hs in configs.items():
            assert enable_pipelined_residency(
                model, hs, device=DEV, k_slots=k_slots) == len(mods)
            with torch.no_grad():
                timed_decode(model, tok, 24)               # warm-up, discarded
                c0, h0 = traffic()
                tps = timed_decode(model, tok, N_TOK)
                c1, h1 = traffic()
            res[name].append(tps)
            byts[name] = {"cold": (c1 - c0) / N_TOK, "hot_d2d": (h1 - h0) / N_TOK}
            log(f"  rep{rep} {name:9s} {tps:6.2f} tok/s  cold {(c1-c0)/N_TOK/1e6:6.1f} "
                f"hot_d2d {(h1-h0)/N_TOK/1e6:6.1f} MB/tok")
            disable_pipelined_residency(model)

    log("control: stripping ExpertsLoRA wrappers (the pre-0.9.2 workaround) ...")
    for parent in list(model.modules()):
        for cn, child in list(parent.named_children()):
            if isinstance(child, ExpertsLoRA):
                setattr(parent, cn, child.base)
    unwrapped = []
    for _ in range(REPS):
        assert enable_pipelined_residency(
            model, informed, device=DEV, k_slots=k_slots) == len(mods)
        with torch.no_grad():
            timed_decode(model, tok, 24)
            unwrapped.append(timed_decode(model, tok, N_TOK))
        disable_pipelined_residency(model)

    series = dict(res, unwrapped=unwrapped)
    out = {
        "model": MODEL, "device": DEV, "n_experts": n_exp, "top_k": k_slots,
        "hot_k": hot_k, "layers": len(mods), "n_tokens": N_TOK, "reps": REPS,
        "coverage": {"informed": COV_INF, "index": COV_IDX, "uniform": uniform},
        "tok_s": {k: {"median": statistics.median(v), "min": min(v), "max": max(v),
                      "all": v} for k, v in series.items()},
        "bytes_per_token": byts,
        "total_gather_bytes_per_token": {k: v["cold"] + v["hot_d2d"] for k, v in byts.items()},
        "comparisons": {},
        "note": ("Coverage predicts reads, not speed. Compare the total_gather column "
                 "across configs before reading any timing difference: if it is flat, a "
                 "hot set is relocating bytes (PCIe -> device-to-device), not removing "
                 "them, and cannot pay regardless of coverage."),
    }
    for a, b in [("index", "informed"), ("stream", "index"), ("stream", "informed"),
                 ("informed", "unwrapped")]:
        _, p = mann_whitney(series[a], series[b])
        ma, mb = statistics.median(series[a]), statistics.median(series[b])
        out["comparisons"][f"{a}->{b}"] = {
            "delta_pct": 100 * (mb - ma) / ma, "p": p,
            "verdict": "significant" if p < 0.05 else "not distinguishable from noise"}

    print("\n" + json.dumps(out, indent=1))
    if OUT:
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nreceipt -> {OUT}")
    for k, v in out["comparisons"].items():
        print(f"{k:22s} {v['delta_pct']:+6.1f}%  p={v['p']:.3f}  {v['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
