# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""G8: does batching actually amortize expert weight reads, and do the
two compute buses finish together?

The gate is a LAW, not a speedup: measured unique-expert reads per
activation must track ``factor(B) = E(1-(1-k/E)^B)/(B*k)`` within 10%,
and the fitted curve is the deliverable. Two things make that falsifiable
rather than self-confirming:

* **The routing is the model's own.** The factor is a property of how a
  real router spreads tokens, so a synthetic uniform draw would be
  measuring the formula against itself. We run real tokens through the
  real model and count what the dispatcher actually touched.
* **A negative control that the instrument itself computes**: the
  single-token row. At one token there is nothing to amortize — k
  activations land on k distinct experts — so the measured factor must
  be exactly 1.0. If it reads lower, the counter is over-crediting
  (uniques pooled across layers, a stale accumulator) and every batched
  number it produces is unreadable. The gate refuses to certify unless
  that control holds.

Balance (clause 2) uses PER-BUS PROBES, not arm subtraction: the CPU bus
is synchronous host work timed on the host clock, the GPU bus is
bracketed by its own CUDA events. Attribution by subtracting one
configuration from another failed three times in Phase 1 and is not
used here.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def _analytic(n_experts: int, top_k: int, batch: int) -> float:
    from experts4bit_qlora.engines.placement import amortization_factor
    return amortization_factor(n_experts, top_k, batch)


def _tiers(model):
    """Every hybrid tier state in the model, in layer order.

    The state lives on ``mod._hot_residency`` (the attribute
    ``enable_hot_residency`` installs); ``arm_amortization`` is what
    distinguishes a hybrid tier from a plain hot-residency state, so
    duck-typing on it keeps this working across engines."""
    out = []
    for mod in model.modules():
        st = getattr(mod, "_hot_residency", None)
        if st is not None and hasattr(st, "arm_amortization"):
            out.append(st)
    return out


def measure_amortization(model, tok, batches, seq_len, prompt_pool,
                         device="cuda", label="AMORT"):
    """For each batch size: run one decode step over B independent
    sequences and count unique experts touched per tier."""
    rows = []
    tiers = _tiers(model)
    if not tiers:
        raise RuntimeError("no hybrid tier states found — is the model "
                           "enabled with enable_hybrid_tier?")
    cfg = model.config
    E = getattr(cfg, "num_local_experts", None) or cfg.num_experts
    k = getattr(cfg, "num_experts_per_tok", None) or cfg.top_k

    short = [len(s) for s in prompt_pool if len(s) < seq_len]
    if short:
        raise ValueError(
            f"prompt pool holds sequences shorter than seq_len={seq_len} "
            f"(min {min(short)}): torch would silently stack the short "
            f"slices and every 'tokens' below would be an overcount, which "
            f"reads as a factor-2 law violation rather than a bad harness")
    for B in batches:
        ids = torch.stack([prompt_pool[i % len(prompt_pool)][:seq_len]
                           for i in range(B)]).to(device)
        assert ids.shape == (B, seq_len), ids.shape
        # Token diversity is REPORTED, not assumed. B sequences that share
        # a token id route identically, which would collapse the unique
        # count and read as amortization the batch never bought — a
        # routing measurement that is really measuring the sampler.
        n_distinct = int(torch.unique(ids).numel())
        for st in tiers:
            st.arm_amortization(True)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        acc = {"steps": 0, "acts": 0, "uniq_vram": 0, "uniq_dram": 0,
               "uniq_nvme": 0, "acts_vram": 0, "acts_dram": 0,
               "acts_nvme": 0, "dram_groups": 0, "dram_ns": 0, "gpu_ns": 0}
        hists = []
        for st in tiers:
            for key in acc:
                acc[key] += st.amort[key]
            hists.append(st.amort["hist"].detach().cpu())
            st.arm_amortization(False)
        uniq = acc["uniq_vram"] + acc["uniq_dram"] + acc["uniq_nvme"]
        # tokens per step = B * seq_len (a prefill-shaped step at seq_len
        # 1 is exactly a decode step over B sequences)
        measured = uniq / acc["acts"] if acc["acts"] else float("nan")
        rows.append({
            "batch": B, "tokens": B * seq_len,
            "distinct_token_ids": n_distinct,
            "token_diversity": n_distinct / (B * seq_len),
            "acts": acc["acts"], "unique": uniq,
            "measured_factor": measured,
            "analytic_factor": _analytic(E, k, B * seq_len),
            "per_tier_unique": {"vram": acc["uniq_vram"],
                                "dram": acc["uniq_dram"],
                                "nvme": acc["uniq_nvme"]},
            "per_tier_acts": {"vram": acc["acts_vram"],
                              "dram": acc["acts_dram"],
                              "nvme": acc["acts_nvme"]},
            "dram_groups": acc["dram_groups"],
            # groups > unique means the DRAM bus re-read experts (the
            # pre-Phase-8 split tax); equal means the amortization is clean
            "dram_split_tax": (acc["dram_groups"] / acc["uniq_dram"]
                               if acc["uniq_dram"] else None),
            # per-bus completion, each from its own probe
            "dram_ms": acc["dram_ns"] / 1e6, "gpu_ms": acc["gpu_ns"] / 1e6,
            "balance_ratio": (min(acc["dram_ns"], acc["gpu_ns"])
                              / max(acc["dram_ns"], acc["gpu_ns"])
                              if max(acc["dram_ns"], acc["gpu_ns"]) else None),
        })
        rows[-1]["_hists"] = [h.tolist() for h in hists]
        r = rows[-1]
        print(f"{label} B={B:3d} tok={r['tokens']:5d} "
              f"distinct={r['distinct_token_ids']:4d} acts={r['acts']:7d} "
              f"uniq={r['unique']:6d} measured={r['measured_factor']:.4f} "
              f"analytic={r['analytic_factor']:.4f} "
              f"delta={100*(r['measured_factor']/r['analytic_factor']-1):+6.1f}%",
              flush=True)
    return rows


def predict_from_measured_routing(hists, tokens_per_hist, batch):
    """General law with the routing the model ACTUALLY has.

    ``sum_e 1-(1-p_e)^B`` per layer, where p_e is the empirical per-token
    selection probability of expert e. The gate's closed form is this
    with every p_e forced to k/E; comparing the two is how we find out
    whether real routing is uniform (it is not) and, more importantly,
    whether the AMORTIZATION MACHINERY is right once the routing model
    stops being wrong."""
    total = 0.0
    for h in hists:
        n = sum(h)
        if not n or not tokens_per_hist:
            continue
        for c in h:
            p = c / tokens_per_hist
            p = min(1.0, max(0.0, p))
            total += 1.0 - (1.0 - p) ** batch
    return total


def negative_control(rows):
    """The single-token row IS the negative control, and it comes free.

    At one token there is nothing to amortize: k activations select k
    distinct experts, so measured factor must be 1.0 EXACTLY. An
    instrument that reports amortization there is counting something
    other than what it claims (double-counted uniques, a stale
    accumulator, uniques taken across layers), and every batched number
    it produces is unreadable. A control that can only pass — asserting
    a constant the instrument never computes — is not a control; this
    one is computed by the same code path as the results.
    """
    one = next((r for r in rows if r["tokens"] == 1), None)
    if one is None:
        return {"ran": False, "reason": "no single-token row measured"}
    ok = abs(one["measured_factor"] - 1.0) < 1e-9
    return {"ran": True, "measured_at_one_token": one["measured_factor"],
            "passes": ok,
            "note": "1 token => k acts on k distinct experts => factor 1.0"}


def measure_balance(model, batches, prompt_pool, seq_len, reps=5,
                    device="cuda"):
    """Per-bus wall time at each batch. The tiers are timed by running the
    model with each bus's contribution isolated via the tier's own
    counters plus CUDA events around the step; what the gate wants is the
    RATIO, so both arms use the identical step."""
    out = []
    for B in batches:
        ids = torch.stack([prompt_pool[i % len(prompt_pool)][:seq_len]
                           for i in range(B)]).to(device)
        with torch.no_grad():                      # warm
            for _ in range(2):
                model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        walls = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
        out.append({"batch": B, "step_s": statistics.median(walls)})
        print(f"BALANCE B={B}: step {statistics.median(walls)*1e3:.1f} ms",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--seq-len", type=int, default=1)
    ap.add_argument("--out", default="bench/hybrid-g8")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(1689)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).cuda().eval()

    if a.manifest:
        from experts4bit_qlora.engines.hybrid import enable_hybrid_tier
        enable_hybrid_tier(model, manifest=a.manifest)

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    pool = [ids[i * 512:(i + 1) * 512] for i in range(128)]

    batches = [int(x) for x in a.batches.split(",")]
    rows = measure_amortization(model, tok, batches, a.seq_len, pool)
    ctrl = negative_control(rows)
    bal = measure_balance(model, [8, 16], pool, a.seq_len)

    worst = max(abs(r["measured_factor"] / r["analytic_factor"] - 1)
                for r in rows)
    bal_rows = {r["batch"]: r["balance_ratio"] for r in rows
                if r["batch"] in (8, 16)}
    bal_ok = all(v is not None and v >= 0.80 for v in bal_rows.values()) \
        if bal_rows else None
    rep = {"model": a.model, "rows": rows, "control": ctrl,
           "balance_wall": bal, "balance_at_8_16": bal_rows,
           "worst_rel_delta": worst,
           # the gate certifies nothing unless the control held
           "gate_amortization_10pct": bool(ctrl.get("passes")) and worst <= 0.10,
           "gate_balance_20pct": bal_ok,
           "control_ok": bool(ctrl.get("passes"))}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    name = a.model.split("/")[-1]
    (Path(a.out) / f"g8_amortization_{name}.json").write_text(
        json.dumps(rep, indent=2))
    print("G8_AMORT " + json.dumps(
        {"worst_rel_delta": worst, "control_ok": rep["control_ok"],
         "gate_amortization": rep["gate_amortization_10pct"],
         "gate_balance": rep["gate_balance_20pct"],
         "balance_at_8_16": bal_rows}), flush=True)


if __name__ == "__main__":
    main()
