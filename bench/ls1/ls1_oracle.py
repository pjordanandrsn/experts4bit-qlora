#!/usr/bin/env python
"""LS1 explore screen (Map 1 T-A, e4b#945): what the last host sync of a fused MoE layer pass costs a training step.

The fused grouped path reads its per-expert counts back once per grouped forward (`engines/fast.py::_group_by_expert`, the
launch grid's host sizes). Under gradient checkpointing a layer runs that forward twice per step: the forward and its
recompute in the backward. This script bounds what removing those reads could buy, before anyone builds a device grid
(the maintainer's 2026-10-04 scoping on #945: measure the bound before building).

One process, one load, ONE fixed batch and lr = 0, so every timed step routes identically. The arms differ only in how the
grouping's host lists are obtained:

  fused      the shipped path: one read per grouped forward (two per layer pass under checkpointing)
  oracle_rc  the recompute reuses its own forward's lists (LIFO within the step): one read per layer pass
  oracle     every grouped forward replays lists recorded from an identical earlier step: no read at all (the bound)
  legacy13   E4B_GROUPING=legacy GNF4_PINNED_RING=0: the pre-#946 path (13 syncs per pass), the calibration

The oracle arms are a measuring device, never a training path: they are only valid because routing is fixed. Each one is
CHECKED in an untimed step after its timed steps, where the real grouping runs beside the oracle and every list must match.
A mismatch marks the arm INVALID. Arms run interleaved in mirrored order (A B C D D C B A) and each is reported as the mean
of its two readings. This is the explore tier: a screen, never a reading. A registered paired read follows only if
oracle / fused is resolvably below 0.97 (the #945 decision rule).
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

LADDER_DIR = Path(__file__).resolve().parents[1] / "moegen"


def _ladder():
    """The ladder's helpers (batches, counters, prewarm, unpatch_rmsnorm, profile), imported on the box only: it needs a
    Unix `resource` module, and the self-test does not."""
    sys.path.insert(0, str(LADDER_DIR))
    import ladder
    return ladder


ARMS = {
    "fused": {},
    "oracle_rc": {},
    "oracle": {},
    "legacy13": {"E4B_GROUPING": "legacy", "GNF4_PINNED_RING": "0"},
}


class GroupingOracle:
    """Stands in for `fast._group_by_expert` in this script only.

    mode "real": pass through.  "record": pass through and append each call's lists to the tape.
    "replay": return the tape's lists in call order, no device read.  "rc": the first `n_fwd` calls of a step read and push,
    every later call pops (the non-reentrant checkpoint recomputes the layers in reverse order).
    "check": run the real grouping AND the oracle the arm used, count list mismatches (an untimed validation step).
    """

    def __init__(self, real):
        self.real = real
        self.mode = "real"
        self.check_of = None              # "replay" or "rc" while checking
        self.tape: list[tuple[list[int], list[int]]] = []
        self.n_fwd = 0
        self.mismatch = 0
        self.begin_step()

    def begin_step(self):
        self.pos = 0
        self.stack: list[tuple[list[int], list[int]]] = []
        self.calls = 0

    def _oracle(self, kind):
        if kind == "replay":
            if self.pos >= len(self.tape):
                raise RuntimeError(f"oracle tape exhausted at call {self.pos} (tape {len(self.tape)})")
            s, e = self.tape[self.pos]
            self.pos += 1
            return list(s), list(e)
        if self.calls <= self.n_fwd:      # rc: this call is a forward (calls was incremented first)
            return None
        if not self.stack:
            raise RuntimeError("oracle_rc: a recompute call with nothing pushed")
        s, e = self.stack.pop()
        return list(s), list(e)

    def __call__(self, flat, E):
        self.calls += 1
        if self.mode == "real":
            return self.real(flat, E)
        if self.mode == "record":
            s, e = self.real(flat, E)
            self.tape.append((list(s), list(e)))
            return s, e
        if self.mode == "replay":
            return self._oracle("replay")
        if self.mode == "rc":
            got = self._oracle("rc")
            if got is None:
                s, e = self.real(flat, E)
                self.stack.append((list(s), list(e)))
                return s, e
            return got
        if self.mode == "check":
            want = self.real(flat, E)
            if self.check_of == "rc":
                got = self._oracle("rc")
                if got is None:
                    self.stack.append((list(want[0]), list(want[1])))
                    return want
            else:
                got = self._oracle("replay")
            if (list(got[0]), list(got[1])) != (list(want[0]), list(want[1])):
                self.mismatch += 1
            return want
        raise ValueError(self.mode)


def self_test() -> int:
    """CPU-only: the oracle's record/replay/rc/check bookkeeping against a fake grouping (no torch needed)."""
    calls = []

    def real(flat, E):
        calls.append(flat)
        return [len(flat)], [flat[0]]

    o = GroupingOracle(real)
    o.n_fwd = 3
    seq = [[1, 2], [3], [4, 5, 6]]                       # a forward: layers 0, 1, 2
    rec = seq + list(reversed(seq))                    # then the recompute in reverse
    o.mode = "record"
    o.begin_step()
    for x in rec:
        o(x, 8)
    assert len(o.tape) == 6 and len(calls) == 6
    o.mode = "replay"
    o.begin_step()
    calls.clear()
    out = [o(x, 8) for x in rec]
    assert not calls and out == [([len(x)], [x[0]]) for x in rec], out          # zero real reads
    o.mode = "rc"
    o.begin_step()
    out = [o(x, 8) for x in rec]
    assert len(calls) == 3 and out == [([len(x)], [x[0]]) for x in rec], (calls, out)   # one read per layer pass
    for kind in ("replay", "rc"):
        o.mode, o.check_of, o.mismatch = "check", kind, 0
        o.begin_step()
        for x in rec:
            o(x, 8)
        assert o.mismatch == 0, kind
    o.mode, o.check_of, o.mismatch = "check", "replay", 0
    o.begin_step()
    for x in [[9]] + rec[1:]:                          # a routing change is caught
        o(x, 8)
    assert o.mismatch == 1
    o.mode = "replay"
    o.begin_step()
    for x in rec:
        o(x, 8)
    try:
        o([1], 8)
        raise AssertionError("an exhausted tape must refuse")
    except RuntimeError:
        pass
    print("ls1 self-test: 6 checks passed")
    return 0


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--arms", default="fused,oracle_rc,oracle,legacy13")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--adapter-dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--attn4", type=int, default=1)
    ap.add_argument("--attn-lora", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--profile", type=int, default=1)
    ap.add_argument("--out", default="ls1_oracle.json")
    return ap.parse_args()


def run_arm(name, model, train, batch, oracle, torch, a):
    dev = "cuda"
    opt = torch.optim.AdamW(train, lr=0.0, weight_decay=0.0)      # lr = 0: the weights, and so the routing, never move
    ids = batch.to(dev)

    def step():
        oracle.begin_step()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.detach()

    mode = {"oracle": "replay", "oracle_rc": "rc"}.get(name, "real")
    oracle.mode = mode
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    for i in range(a.warmup + a.steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        lo = step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        losses.append(float(lo))
        if i >= a.warmup:
            times.append(dt)
        print(f"[ls1] {name} step {i} {dt:.3f}s loss {float(lo):.5f}", flush=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode(1)
        try:
            step()
        finally:
            torch.cuda.set_sync_debug_mode(0)
    syncs = sum(1 for x in w if "synchroniz" in str(x.message).lower())
    check = None
    if mode in ("replay", "rc"):
        oracle.mode, oracle.check_of, oracle.mismatch = "check", mode, 0
        step()
        check = {"mismatch": oracle.mismatch, "valid": oracle.mismatch == 0}
    oracle.mode = mode
    prof = _ladder().profile_step(lambda _b: step(), None, torch) if a.profile else None
    oracle.mode = "real"
    med = statistics.median(times)
    return {"arm": name, "s_per_step": round(med, 4), "s_per_step_all": [round(x, 4) for x in times],
            "losses": [round(x, 5) for x in losses], "syncs_one_step": syncs, "check": check, "profile": prof,
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}


def main():
    a = parse()
    if a.self_test:
        return self_test()
    ladder = _ladder()
    arms =[x.strip() for x in a.arms.split(",") if x.strip()]
    for x in arms:
        if x not in ARMS:
            raise SystemExit(f"unknown arm {x!r}")
    base_env = {k: os.environ.get(k) for v in ARMS.values() for k in v}
    import torch
    from transformers import AutoTokenizer
    from experts4bit_qlora import load_moe_4bit_streaming, verify_moe_4bit
    from experts4bit_qlora.lora import (add_attention_lora, quantize_attention_projections_4bit, trainable_lora_param_ids,
                                        ExpertsLoRA)
    from experts4bit_qlora.engines import fast as ef
    from experts4bit_qlora.engines.fast import enable_fast_train, disable_fast_train

    oracle = GroupingOracle(ef._group_by_expert)
    ef._group_by_expert = oracle                       # module global: the patched forwards look it up at call time
    adt = torch.float32 if a.adapter_dtype == "fp32" else torch.bfloat16
    ladder.prewarm(a.model, a.revision)
    model, cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, a.r, a.alpha, offload=False, revision=a.revision)
    model.to("cuda")
    verify_moe_4bit(model, strict=True)
    for m in model.modules():
        if isinstance(m, ExpertsLoRA):
            for n, p in m.named_parameters(recurse=False):
                if "lora" in n:
                    p.data = p.data.to(adt)
    if a.attn4:
        quantize_attention_projections_4bit(model)
    if a.attn_lora:
        add_attention_lora(model, a.r, a.alpha, adt)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    ex_ids, at_ids = trainable_lora_param_ids(model)
    train = [p for p in model.parameters() if id(p) in ex_ids or id(p) in at_ids]
    for p in model.parameters():
        p.requires_grad_(id(p) in ex_ids or id(p) in at_ids)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    batch = ladder.alpaca_batches(tok, a.seq, a.mb, 1)[0]     # ONE batch, every step: routing is fixed
    import importlib.metadata as md
    out = {"model": a.model, "revision": a.revision, "seq": a.seq, "mb": a.mb, "steps": a.steps, "warmup": a.warmup,
           "adapter_dtype": a.adapter_dtype, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "versions": {"e4b": md.version("experts4bit-qlora"), "gnf4": md.version("grouped-nf4-gemm")},
           "cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": ")
           if os.path.exists("/proc/cpuinfo") else None, "arms": []}
    # the tape: one untimed step on the shipped path, recording every grouped forward's lists
    n = enable_fast_train(model, verbose=False, dgrad=True)
    oracle.mode = "record"
    oracle.tape.clear()
    oracle.begin_step()
    ids = batch.to("cuda")
    model(input_ids=ids, labels=ids).loss.backward()
    model.zero_grad(set_to_none=True)
    oracle.mode = "real"
    out["n_patched"], out["tape_calls"] = n, len(oracle.tape)
    oracle.n_fwd = len(oracle.tape) // 2
    out["rc_shape_ok"] = len(oracle.tape) == 2 * n           # one forward + one recompute per patched layer
    print(f"[ls1] patched {n} expert modules; tape {len(oracle.tape)} grouped forwards; rc shape ok {out['rc_shape_ok']}",
          flush=True)
    order = arms + [x + "#2" for x in reversed(arms)]
    for arm in order:
        name = arm.split("#")[0]
        for k, v in base_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k, v in ARMS[name].items():
            os.environ[k] = v
        disable_fast_train(model)
        ladder.unpatch_rmsnorm(model)
        enable_fast_train(model, verbose=False, dgrad=True)
        if name == "oracle_rc" and not out["rc_shape_ok"]:
            out["arms"].append({"arm": arm, "skipped": "tape is not one forward + one recompute per layer"})
            continue
        rec = run_arm(name, model, train, batch, oracle, torch, a)
        rec["arm"], rec["env"], rec["counters"] = arm, ARMS[name], _ladder().counters()
        out["arms"].append(rec)
        print(f"[ls1] {arm}: {rec['s_per_step']:.4f} s/step, syncs {rec['syncs_one_step']}, check {rec['check']}", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    by = {}
    for r in out["arms"]:
        if "s_per_step" in r:
            by.setdefault(r["arm"].split("#")[0], []).append(r)
    summ = {k: {"s_per_step_mean": round(sum(r["s_per_step"] for r in v) / len(v), 4),
                "readings": [r["s_per_step"] for r in v], "syncs_one_step": v[0]["syncs_one_step"],
                "valid": all((r["check"] or {}).get("valid", True) for r in v),
                "device_busy_frac": [(r["profile"] or {}).get("device_busy_frac") for r in v]} for k, v in by.items()}
    if "fused" in summ:
        f = summ["fused"]["s_per_step_mean"]
        for k in summ:
            summ[k]["ratio_to_fused"] = round(summ[k]["s_per_step_mean"] / f, 4)
    out["summary"] = summ
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in summ.items():
        print(f"[ls1] SUMMARY {k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
