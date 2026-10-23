#!/usr/bin/env python
"""The training path's deterministic mode: one base-only self-pair per arm (EXPLORE: a design input, never a reading).

After tb-x-1 found eager training not bitwise equal to itself, a source read named two nondeterministic ops: the fused
training forward's token gather (fixed by ``E4B_TRAIN_DET_GATHER=1``, e4b#2054) and the attention backward (SDPA's
flash / mem-efficient / cuDNN backwards). One arm per process (``--arm``), so no setting leaks between arms:

  A   E4B_TRAIN_DET_GATHER=1 only
  B   A + math SDPA (flash, mem-efficient and cuDNN SDPA off)
  C   A + torch.use_deterministic_algorithms(True, warn_only=False) around loss.backward() ONLY (the forward and its
      loss run with the mode off: NLLLoss raises under it); CUBLAS_WORKSPACE_CONFIG=:4096:8 from process start (the
      runner sets it); a strict-mode RAISE is caught and named, and the arm reads VOID with that name
  Cp  C + torch.utils.deterministic.fill_uninitialized_memory = False

The step is the field recipe's shape (Qwen3-30B-A3B, 4 micro-batches of 512 tokens, r 16, fp32 adapters, the fused
path). Two blocks of ``--reps`` steps in one process; every step's trainable gradients are compared bitwise to block
1's first step. Per arm: SELF_IDENTICAL or NOT (with the first differing tensor and its max abs diff); the SDPA
backend actually run inside the step (the profiler's aten op names); a recompute check on one decoder layer (its
outputs recomputed in the backward must equal the forward's bitwise: a checkpoint replaying a different kernel under
strict mode would otherwise give gradients for another forward); "does not have a deterministic implementation"
warnings; step wall, device time and reserved memory.

decide() fails closed (bench/common rule 7): no engagement of the knob, a missing measurement, a recompute mismatch or
a strict-mode raise read VOID; SELF_IDENTICAL needs every compared step bitwise equal.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

ARMS = ("A", "B", "C", "Cp")
SDPA_OPS = {"flash": "_scaled_dot_product_flash_attention", "efficient": "_scaled_dot_product_efficient_attention",
            "cudnn": "_scaled_dot_product_cudnn_attention", "math": "_scaled_dot_product_attention_math"}


def backends_from_names(names):
    """The SDPA backends a profiled step ran, from its aten op names (forward or backward)."""
    found = {}
    for n in names:
        for b, op in SDPA_OPS.items():
            if op in n:
                found[b] = found.get(b, 0) + 1
    return found


def decide(r: dict) -> dict:
    """Fails closed: every count must equal its expected value exactly (absence or a shortfall is VOID, never a pass)."""
    void = []
    for k in ("det_gather_calls", "compared_steps", "expected_compared", "backends", "recompute_checked",
              "expected_recompute", "step_wall_ms", "recompute_split_bad"):
        if r.get(k) is None or (k in ("backends", "step_wall_ms") and not r.get(k)):
            void.append(f"{k} missing")
    if r.get("raised"):
        void.append(f"strict mode raised: {r['raised']}")
    if not void:
        if r["det_gather_calls"] <= 0:
            void.append("E4B_TRAIN_DET_GATHER did not engage")
        if r["compared_steps"] != r["expected_compared"] or r["expected_compared"] <= 0:
            void.append(f"compared {r['compared_steps']} steps, expected {r['expected_compared']}")
        if r["recompute_checked"] != r["expected_recompute"] or r["expected_recompute"] <= 0:
            void.append(f"recompute checked on {r['recompute_checked']} micro-batches, expected {r['expected_recompute']}")
        if r["recompute_split_bad"]:
            void.append(f"forward/recompute hook split not (accum, accum): {r['recompute_split_bad'][:3]}")
        if not r["recompute_equal"]:
            void.append("a recomputed layer output differs from its forward: the checkpoint replayed another computation")
    if void:
        return {"verdict": "VOID", "why": void}
    if r["mismatches"]:
        return {"verdict": "NOT_SELF_IDENTICAL", "why": r["mismatches"][:3]}
    return {"verdict": "SELF_IDENTICAL", "why": []}


def self_test() -> int:
    assert backends_from_names(["aten::_scaled_dot_product_flash_attention", "aten::_scaled_dot_product_flash_attention_backward",
                                "aten::mm"]) == {"flash": 2}
    assert backends_from_names(["aten::_scaled_dot_product_attention_math", "aten::bmm"]) == {"math": 1}
    ok = {"det_gather_calls": 8, "compared_steps": 5, "expected_compared": 5, "backends": {"flash": 4},
          "recompute_checked": 24, "expected_recompute": 24, "recompute_split_bad": [], "recompute_equal": True,
          "step_wall_ms": [1.0], "mismatches": [], "raised": None}
    assert decide(ok)["verdict"] == "SELF_IDENTICAL"
    assert decide({**ok, "mismatches": ["x: 1e-5"]})["verdict"] == "NOT_SELF_IDENTICAL"
    n = 0
    for k in ("det_gather_calls", "compared_steps", "expected_compared", "backends", "recompute_checked",
              "expected_recompute", "step_wall_ms", "recompute_split_bad"):
        bad = dict(ok)
        bad[k] = None
        assert decide(bad)["verdict"] == "VOID", k                       # each input removed
        n += 1
    cases = [{"det_gather_calls": 0}, {"raised": "aten::nll_loss"}, {"recompute_equal": False}, {"backends": {}},
             {"step_wall_ms": []},
             {"compared_steps": 0},                                        # nothing compared (review-desk, fail-open 1)
             {"compared_steps": 4},                                        # a shortfall
             {"recompute_checked": 0},                                     # the hook never split (fail-open 2)
             {"recompute_checked": 0, "recompute_equal": False},
             {"recompute_checked": 20},                                    # a shortfall
             {"recompute_split_bad": [(8, 0)]},                            # every output landed in fwd
             {"expected_compared": 0, "compared_steps": 0}, {"expected_recompute": 0, "recompute_checked": 0}]
    for c in cases:
        assert decide({**ok, **c})["verdict"] == "VOID", c
        n += 1
    print(f"det self-test: the backend reader, 2 verdicts and {n} fail-closed cases passed")
    return 0


def gather_probe(tokens=4096, hidden=2048, k=8, reps=20, out="det_gather_probe.json"):
    """RECORD ONLY (no verdict): on the card, how many distinct dL/dx the OLD index_select backward and the NEW
    _GatherTokenRows backward give over ``reps`` identical calls at a field-sized gather (4096 tokens x 2048, top-8), in
    bf16 and fp32. Shows whether the old path visibly varies at this size; the self-pair arms are the check."""
    import hashlib
    import torch
    from experts4bit_qlora.engines.batched import _GatherTokenRows
    res = {}
    for dtype in (torch.bfloat16, torch.float32):
        g = torch.Generator(device="cuda").manual_seed(0)
        flat = torch.randint(0, 128, (tokens * k,), device="cuda", generator=g)
        order = torch.argsort(flat, stable=True)
        hs0 = torch.randn(tokens, hidden, device="cuda", generator=g).to(dtype)
        gout = torch.randn(tokens * k, hidden, device="cuda", generator=g).to(dtype)
        for name in ("old_index_select", "new_gather_token_rows"):
            digests = set()
            for _ in range(reps):
                hs = hs0.clone().requires_grad_(True)
                o = (_GatherTokenRows.apply(hs, order, k) if name.startswith("new")
                     else hs.index_select(0, torch.div(order, k, rounding_mode="floor")))
                o.backward(gout)
                digests.add(hashlib.sha256(hs.grad.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16])
            res[f"{name}_{str(dtype).split('.')[-1]}"] = len(digests)
    Path(out).write_text(json.dumps({"tokens": tokens, "hidden": hidden, "k": k, "reps": reps,
                                     "distinct_dLdx": res}, indent=1))
    print(f"[det] gather probe (record only, {tokens}x{hidden}, k={k}, {reps} reps): distinct dL/dx {res}", flush=True)
    return 0


def _ladder():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moegen"))
    import ladder
    return ladder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--gather-probe", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--layer", type=int, default=24, help="the decoder layer whose recompute is checked")
    ap.add_argument("--out", default="det_selfpair.json")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.gather_probe:
        return gather_probe(out=a.out)
    assert a.arm, "--arm is required"
    assert os.environ.get("E4B_TRAIN_DET_GATHER") == "1", "the runner sets E4B_TRAIN_DET_GATHER=1 for every arm"
    strict = a.arm in ("C", "Cp")
    assert (os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8") == strict, "CUBLAS_WORKSPACE_CONFIG only for C, Cp"
    ladder = _ladder()
    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoTokenizer
    from experts4bit_qlora import load_moe_4bit_streaming, verify_moe_4bit
    from experts4bit_qlora.engines import fast
    from experts4bit_qlora.lora import (add_attention_lora, quantize_attention_projections_4bit, trainable_lora_param_ids,
                                        ExpertsLoRA)

    if a.arm == "B":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    import torch.utils.deterministic
    if a.arm == "Cp":
        torch.utils.deterministic.fill_uninitialized_memory = False
    warnings.simplefilter("always")
    caught = []
    _show = warnings.showwarning

    def _record(message, category, filename, lineno, file=None, line=None):
        caught.append(f"{category.__name__}: {str(message)[:300]}")
        _show(message, category, filename, lineno, file, line)
    warnings.showwarning = _record

    ladder.prewarm(a.model, a.revision)
    model, cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, a.r, a.alpha, offload=False, revision=a.revision)
    model.to("cuda")
    verify_moe_4bit(model, strict=True)
    for m in model.modules():
        if isinstance(m, ExpertsLoRA):
            for n, p in m.named_parameters(recurse=False):
                if "lora" in n:
                    p.data = p.data.to(torch.float32)
    quantize_attention_projections_4bit(model)
    add_attention_lora(model, a.r, a.alpha, torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    ex_ids, at_ids = trainable_lora_param_ids(model)
    named = [(n, p) for n, p in model.named_parameters() if id(p) in ex_ids or id(p) in at_ids]
    for p in model.parameters():
        p.requires_grad_(id(p) in ex_ids or id(p) in at_ids)
    n_patched = fast.enable_fast_train(model, verbose=False, dgrad=True)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    batches = [b.to("cuda") for b in ladder.alpaca_batches(tok, a.seq, 1, a.accum)]

    # the recompute check: layer `a.layer`'s outputs in the forward vs in the backward's recompute, per micro-batch
    have_gt = hasattr(torch._C, "_current_graph_task_id")
    seen = {"fwd": [], "re": []}

    def hook(mod, args, out):
        o = out[0] if isinstance(out, tuple) else out
        key = "re" if (have_gt and torch._C._current_graph_task_id() != -1) else "fwd"
        seen[key].append(o.detach().to("cpu", copy=True))
    model.model.layers[a.layer].register_forward_hook(hook)

    raised = []

    def step():
        for _, p in named:
            p.grad = None
        assert not torch.are_deterministic_algorithms_enabled(), "deterministic mode left on from a previous step"
        for b in batches:
            loss = model(input_ids=b, labels=b).loss / a.accum
            if strict:
                torch.use_deterministic_algorithms(True, warn_only=False)
            try:
                loss.backward()
            except RuntimeError as e:
                raised.append(str(e).splitlines()[0][:300])
                raise
            finally:
                if strict:
                    torch.use_deterministic_algorithms(False)

    out = {"tier": "EXPLORE (the training path's deterministic mode, self-pair): a design input, never a reading",
           "arm": a.arm, "seq": a.seq, "accum": a.accum, "reps": a.reps, "layer_checked": a.layer, "n_patched": n_patched,
           "gpu": torch.cuda.get_device_name(), "sm": "sm_%d%d" % torch.cuda.get_device_capability(),
           "torch": torch.__version__, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           "fill_uninitialized_memory": bool(torch.utils.deterministic.fill_uninitialized_memory)}
    rec = {"det_gather_calls": None, "compared_steps": 0, "expected_compared": 2 * a.reps - 1, "backends": None,
           "recompute_checked": 0, "expected_recompute": 2 * a.reps * a.accum, "recompute_split_bad": [],
           "recompute_equal": True, "step_wall_ms": [], "mismatches": [], "raised": None, "warnings": []}
    try:
        for _ in range(2):                        # warm: Triton compiles, plan memos, the allocator
            step()
        torch.cuda.synchronize()
        g0 = fast.FAST_TRAIN_STATS["det_gather"]
        ref = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for block in range(2):
            for i in range(a.reps):
                seen["fwd"].clear()
                seen["re"].clear()
                torch.cuda.synchronize()
                t = time.perf_counter()
                step()
                torch.cuda.synchronize()
                rec["step_wall_ms"].append((time.perf_counter() - t) * 1e3)
                grads = [(n, p.grad) for n, p in named]
                if ref is None:
                    ref = [(n, None if g is None else g.detach().to("cpu", copy=True)) for n, g in grads]
                else:
                    rec["compared_steps"] += 1
                    for (n, r0), (_, g) in zip(ref, grads):
                        if (r0 is None) != (g is None) or (r0 is not None and not torch.equal(r0, g.cpu())):
                            d = None if r0 is None or g is None else float((r0.float() - g.float().cpu()).abs().max())
                            rec["mismatches"].append(f"block {block} rep {i}: {n}: max abs diff {d}")
                            break
                if len(seen["fwd"]) == len(seen["re"]) == a.accum:
                    rec["recompute_checked"] += a.accum
                    if not all(torch.equal(f, r) for f, r in zip(seen["fwd"], seen["re"])):
                        rec["recompute_equal"] = False
                else:                                     # recorded, never skipped: decide() reads it as VOID
                    rec["recompute_split_bad"].append((len(seen["fwd"]), len(seen["re"])))
        rec["reserved_peak"] = torch.cuda.max_memory_reserved()
        rec["det_gather_calls"] = fast.FAST_TRAIN_STATS["det_gather"] - g0
        names, dev = [], 0.0
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        for e in prof.profiler.kineto_results.events():
            if "CUDA" in str(e.device_type()):
                dev += e.duration_ns() / 1e6
            else:
                names.append(e.name())
        rec["backends"] = backends_from_names(names)
        rec["step_device_ms"] = round(dev, 1)
    except RuntimeError as e:
        rec["raised"] = raised[-1] if raised else str(e).splitlines()[0][:300]
    rec["warnings"] = sorted({w for w in caught if "determinis" in w.lower()})
    rec["step_wall_ms_median"] = round(statistics.median(rec["step_wall_ms"]), 1) if rec["step_wall_ms"] else None
    out.update(rec)
    out["decision"] = decide(rec)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"[det] arm {a.arm}: {out['decision']['verdict']} {out['decision']['why']}; backends {rec['backends']}; "
          f"recompute {rec['recompute_checked']} checked, equal {rec['recompute_equal']}; step {rec['step_wall_ms_median']} ms, "
          f"device {rec.get('step_device_ms')} ms, reserved {rec.get('reserved_peak', 0) / 2**30:.2f} GiB; "
          f"det_gather {rec['det_gather_calls']}; warnings {len(rec['warnings'])}", flush=True)
    for w in rec["warnings"]:
        print(f"[det] warning: {w}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
