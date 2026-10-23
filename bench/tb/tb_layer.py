#!/usr/bin/env python
"""T-B option C, the one-layer prototype bench (EXPLORE: a design input, never a reading; Map 1, e4b#2043 section 5).

Four arms on one Qwen3-30B-A3B decoder layer (the real layer, loaded and patched as the field recipe trains it: fp32
adapters, r 16, the fused training path, checkpointing on) and on the whole model's micro-batches:

  base     NF4_QLORA_FUSED_PAD=0 E4B_TRAIN_LEAN_COMBINE=0
  t2       the lean combine (e4b engines/combine_lean.py)
  t1       the fused pad (grouped-nf4-gemm lora_pad.py)
  t1t2     both

Layer: the layer's real inputs (captured from the model's forward on a fixed batch), then forward + checkpoint recompute +
backward with a fixed output gradient, synced, in mirrored blocks (base t2 t1 t1t2 t1t2 t1 t2 base). Per arm: the median
synced wall, the device busy time (the profiler's kernel durations, median of 3 profiled reps), the CUDA-event span,
launches, peak reserved and allocated memory, and every gradient (the input's and each trainable parameter's) against the
base's FIRST block, bitwise.

Step: the same 4 fixed micro-batches through the whole model (forward + backward, no optimizer: the optimizer does not see
the knobs and is timed once at the end), mirrored the same way; every trainable gradient against base's, bitwise. The
MEASURED step decides. The projection from the layer (e4b#2043's form, host and device split per arm from the profile) is
reported beside it as a check on the instrument:
  projected = base step - [min(192 x host saved a layer, base step wall - base step device) + 192 x device saved a layer]
where a layer's host saved = its wall saved - its device saved. Where measured and projected differ by more than the
self-pair spread (the largest gap between one arm's two mirrored blocks), that is flagged: a finding about the instrument.

decide() is DESIGN-TB section 5's kill criteria, fixed before the box, FAILING CLOSED (bench/common rule 7):
  wall      the layer's median wall falls by >= 20 % against base (judged on the fast-CPU host; the host is named)
  device    the layer's device busy time rises by <= 5 %
  reserved  peak reserved memory rises by <= 0.5 GB (layer and step)
  gradients every gradient bitwise equal to base's (layer and step)
  VOID      a knob that did not engage (its counter did not move) or one that engaged when off, or base not bitwise
            equal to itself across its two blocks (no reference to hold the arms to), or a missing measurement
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ARMS = {"base": {"NF4_QLORA_FUSED_PAD": "0", "E4B_TRAIN_LEAN_COMBINE": "0"},
        "t2": {"NF4_QLORA_FUSED_PAD": "0", "E4B_TRAIN_LEAN_COMBINE": "1"},
        "t1": {"NF4_QLORA_FUSED_PAD": "1", "E4B_TRAIN_LEAN_COMBINE": "0"},
        "t1t2": {"NF4_QLORA_FUSED_PAD": "1", "E4B_TRAIN_LEAN_COMBINE": "1"}}
ORDER = ("base", "t2", "t1", "t1t2", "t1t2", "t1", "t2", "base")
WALL_FALL_MIN, DEVICE_RISE_MAX, RESERVED_RISE_MAX = 0.20, 0.05, 0.5e9
LAYER_CALLS_PER_STEP = 48 * 4


def decide(arm: str, m: dict, base: dict) -> dict:
    """The kill criteria for one arm against base. m / base: {"layer": {...}, "step": {...}, "engaged": {...}}."""
    void, fail = [], []
    need = ("wall_ms", "device_ms", "reserved_peak", "grads_equal_base")
    for part in ("layer", "step"):
        for side, d in (("arm", m), ("base", base)):
            for k in need if part == "layer" else ("wall_ms", "reserved_peak", "grads_equal_base"):
                if d.get(part, {}).get(k) is None:
                    void.append(f"{side} {part} {k} missing")
    if void:
        return {"arm": arm, "verdict": "VOID", "why": void}
    want = ARMS[arm]
    for knob, counter in (("NF4_QLORA_FUSED_PAD", "fused_pad_calls"), ("E4B_TRAIN_LEAN_COMBINE", "lean_combine_calls")):
        n = m["engaged"].get(counter)
        if n is None:
            void.append(f"{counter} not recorded")
        elif want[knob] == "1" and n <= 0:
            void.append(f"{knob}=1 but {counter} did not move: the knob did not engage")
        elif want[knob] == "0" and n != 0:
            void.append(f"{knob}=0 but {counter} moved by {n}")
    if not base["layer"]["grads_equal_base"] or not base["step"]["grads_equal_base"]:
        void.append("base is not bitwise equal to itself across its two blocks: no reference to hold the arms to")
    if void:
        return {"arm": arm, "verdict": "VOID", "why": void}
    if arm == "base":
        return {"arm": arm, "verdict": "REFERENCE", "why": []}
    L, B = m["layer"], base["layer"]
    fall = (B["wall_ms"] - L["wall_ms"]) / B["wall_ms"]          # difference first: the edges compare exactly
    rise = (L["device_ms"] - B["device_ms"]) / B["device_ms"]
    res = {"layer_wall_fall": round(fall, 4), "layer_device_rise": round(rise, 4),
           "layer_reserved_rise_bytes": L["reserved_peak"] - B["reserved_peak"],
           "step_reserved_rise_bytes": m["step"]["reserved_peak"] - base["step"]["reserved_peak"],
           "step_wall_fall": round(1 - m["step"]["wall_ms"] / base["step"]["wall_ms"], 4)}
    if fall < WALL_FALL_MIN:
        fail.append(f"layer wall fell {fall:.3f} < {WALL_FALL_MIN}")
    if rise > DEVICE_RISE_MAX:
        fail.append(f"layer device time rose {rise:.3f} > {DEVICE_RISE_MAX}")
    for part in ("layer", "step"):
        r = res[f"{part}_reserved_rise_bytes"]
        if r > RESERVED_RISE_MAX:
            fail.append(f"{part} reserved rose {r / 1e9:.3f} GB > {RESERVED_RISE_MAX / 1e9} GB")
        if not m[part]["grads_equal_base"]:
            fail.append(f"{part} gradients differ from base: {m[part].get('grads_first_diff')}")
    return {"arm": arm, "verdict": "FAIL" if fail else "PASS", "why": fail, **res}


def project(b: dict, s: dict, opt_ms: float, spread: float) -> dict:
    """The arm's measured step and its projection from the layer (e4b#2043's form), and whether they disagree beyond the
    self-pair spread. b / s: the base's and the arm's summaries (layer wall/device, step wall/device, all ms)."""
    exposed = b["step"]["wall_ms"] - b["step"]["device_ms"]
    dev_saved = b["layer"]["device_ms"] - s["layer"]["device_ms"]
    host_saved = (b["layer"]["wall_ms"] - s["layer"]["wall_ms"]) - dev_saved
    proj = b["step"]["wall_ms"] - (min(LAYER_CALLS_PER_STEP * host_saved, exposed) + LAYER_CALLS_PER_STEP * dev_saved)
    measured = s["step"]["wall_ms"] + opt_ms
    gap = measured - (proj + opt_ms)
    return {"measured_step_ms": round(measured, 1), "projected_step_ms": round(proj + opt_ms, 1),
            "layer_host_saved_ms": round(host_saved, 4), "layer_device_saved_ms": round(dev_saved, 4),
            "projection_gap_ms": round(gap, 1), "projection_disagrees": abs(gap) > spread}


def self_test() -> int:
    def arm(wall, dev, res, ge=True, sw=1000.0, sres=0, sge=True, fp=0, lc=0):
        return {"layer": {"wall_ms": wall, "device_ms": dev, "reserved_peak": res, "grads_equal_base": ge},
                "step": {"wall_ms": sw, "reserved_peak": sres, "grads_equal_base": sge},
                "engaged": {"fused_pad_calls": fp, "lean_combine_calls": lc}}
    base = arm(100.0, 50.0, 10 * 10**9)
    assert decide("base", base, base)["verdict"] == "REFERENCE"
    assert decide("t1t2", arm(75, 50, 10 * 10**9, fp=5, lc=5), base)["verdict"] == "PASS"
    assert decide("t1t2", arm(80, 52.5, 10 * 10**9 + 5 * 10**8, fp=5, lc=5), base)["verdict"] == "PASS"   # every edge exactly
    assert decide("t2", arm(95, 50, 10 * 10**9, lc=5), base)["verdict"] == "FAIL"                       # wall short of 20 %
    assert decide("t1", arm(70, 53, 10 * 10**9, fp=5), base)["verdict"] == "FAIL"                       # device +6 %
    assert decide("t1", arm(70, 50, 11 * 10**9, fp=5), base)["verdict"] == "FAIL"                       # reserved +1 GB
    assert decide("t1", arm(70, 50, 10 * 10**9, ge=False, fp=5), base)["verdict"] == "FAIL"             # layer gradients
    assert decide("t1", arm(70, 50, 10 * 10**9, sge=False, fp=5), base)["verdict"] == "FAIL"            # step gradients
    assert decide("t1", arm(70, 50, 10 * 10**9, sres=10**9, fp=5), base)["verdict"] == "FAIL"           # step reserved
    # fail closed
    assert decide("t1", arm(70, 50, 10 * 10**9, fp=0), base)["verdict"] == "VOID"                       # knob did not engage
    assert decide("t2", arm(70, 50, 10 * 10**9, fp=3, lc=5), base)["verdict"] == "VOID"                 # off knob engaged
    assert decide("t1", arm(70, 50, 10 * 10**9, fp=5), arm(100, 50, 10**10, ge=False))["verdict"] == "VOID"   # base unstable
    missing = arm(70, 50, 10 * 10**9, fp=5)
    missing["layer"]["device_ms"] = None
    assert decide("t1", missing, base)["verdict"] == "VOID"
    nocount = arm(70, 50, 10 * 10**9, fp=5)
    del nocount["engaged"]["lean_combine_calls"]
    assert decide("t1", nocount, base)["verdict"] == "VOID"
    assert ORDER == tuple(list(ORDER[:4]) + list(reversed(ORDER[:4]))) and set(ORDER) == set(ARMS)
    # the projection: base step 3000 ms with 2000 ms device (1000 ms exposed); a layer saving 2 ms host and 0.5 ms device
    pb = {"layer": {"wall_ms": 20.0, "device_ms": 10.0}, "step": {"wall_ms": 3000.0, "device_ms": 2000.0}}
    ps = {"layer": {"wall_ms": 17.5, "device_ms": 9.5}, "step": {"wall_ms": 2530.0, "device_ms": 1900.0}}
    pr = project(pb, ps, 100.0, 50.0)
    assert pr["layer_host_saved_ms"] == 2.0 and pr["layer_device_saved_ms"] == 0.5, pr
    assert pr["projected_step_ms"] == 3000 - (192 * 2.0 + 192 * 0.5) + 100, pr          # host 384 ms < 1000 exposed
    assert pr["measured_step_ms"] == 2630 and not pr["projection_disagrees"], pr        # gap 10 ms <= 50
    big = {"layer": {"wall_ms": 10.0, "device_ms": 10.0}, "step": {"wall_ms": 2500.0, "device_ms": 2000.0}}
    pr2 = project(pb, big, 100.0, 50.0)                                                  # host 10 ms a layer: capped at 1000; measured 400 ms off
    assert pr2["projected_step_ms"] == 3000 - 1000 + 100 and pr2["projection_disagrees"], pr2
    print("tb self-test: 9 verdict cases, 5 fail-closed cases, the mirrored order and the projection passed")
    return 0


# ---------------------------------------------------------------------------------------------------------------- the box


def _ladder():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moegen"))
    import ladder
    return ladder


def _set_arm(name):
    for k, v in ARMS[name].items():
        os.environ[k] = v


def _counters():
    import nf4_qlora
    from experts4bit_qlora.engines import fast
    s = fast.COMBINE_STATS
    return {"fused_pad_calls": int(nf4_qlora.FUSED_PAD_STATS["calls"]), "lean_combine_calls": int(s["lean_fwd"] + s["lean_bwd"])}


def _grads_cpu(tensors):
    return [None if t is None else t.detach().to("cpu", copy=True) for t in tensors]


def _first_diff(names, ref, got):
    import torch
    for n, a, b in zip(names, ref, got):
        if (a is None) != (b is None):
            return f"{n}: one side has no gradient"
        if a is not None and not torch.equal(a, b.cpu() if b.is_cuda else b):
            return f"{n}: max abs diff {float((a.float() - b.float().cpu()).abs().max()):.3e}"
    return None


def _device_ms(prof):
    """Device busy time: the sum of the profiled kernels', memsets' and copies' durations (ms), and their count."""
    tot, n = 0, 0
    for e in prof.profiler.kineto_results.events():
        if "CUDA" in str(e.device_type()) and not e.name().startswith(("phase:", "ProfilerStep")):
            tot += int(e.duration_ns())
            n += 1
    return tot / 1e6, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--step-reps", type=int, default=2)
    ap.add_argument("--out", default="tb_layer.json")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    for v in ARMS["base"]:
        os.environ[v] = "0"
    ladder = _ladder()
    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoTokenizer
    from experts4bit_qlora import load_moe_4bit_streaming, verify_moe_4bit
    from experts4bit_qlora.lora import (add_attention_lora, quantize_attention_projections_4bit, trainable_lora_param_ids,
                                        ExpertsLoRA)
    from experts4bit_qlora.engines.fast import enable_fast_train

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
    n_patched = enable_fast_train(model, verbose=False, dgrad=True)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    batches = [b.to("cuda") for b in ladder.alpaca_batches(tok, a.seq, 1, a.accum)]
    layers = model.model.layers
    assert len(layers) * a.accum == LAYER_CALLS_PER_STEP, (len(layers), a.accum)
    layer = layers[a.layer]

    # the layer's real inputs, captured from the model's forward on the first fixed micro-batch
    cap = {}

    def pre(mod, args, kwargs):
        if "args" not in cap:
            cap["args"], cap["kwargs"] = args, dict(kwargs)
    h = layer.register_forward_pre_hook(pre, with_kwargs=True)
    with torch.no_grad():
        model(input_ids=batches[0])
    h.remove()
    hs_key = "hidden_states" if "hidden_states" in cap["kwargs"] else None
    x0 = (cap["kwargs"]["hidden_states"] if hs_key else cap["args"][0]).detach().clone()
    rest_args = () if hs_key else tuple(cap["args"][1:])
    kw = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in cap["kwargs"].items() if k != "hidden_states"}
    gen = torch.Generator(device="cuda").manual_seed(0)
    gout = torch.randn(x0.shape, generator=gen, device="cuda", dtype=torch.float32).to(x0.dtype)
    lnames = [n for n, p in layer.named_parameters() if p.requires_grad]
    lparams = [p for n, p in layer.named_parameters() if p.requires_grad]

    def layer_rep():
        for p in lparams:
            p.grad = None
        x = x0.clone().requires_grad_(True)
        out = layer(x, *rest_args, **kw)
        out = out[0] if isinstance(out, tuple) else out
        out.backward(gout)
        return x

    def step_rep():
        for _, p in named:
            p.grad = None
        for b in batches:
            (model(input_ids=b, labels=b).loss / a.accum).backward()

    res = {arm: {"layer": {"walls": [], "device": [], "events": [], "launches": [], "reserved": [], "allocated": [],
                           "grads_equal_base": True, "grads_first_diff": None},
                 "step": {"walls": [], "block_medians": [], "device": [], "reserved": [], "grads_equal_base": True,
                          "grads_first_diff": None},
                 "engaged": {"fused_pad_calls": 0, "lean_combine_calls": 0}} for arm in ARMS}
    ref_layer = ref_step = None

    # warm every arm once (Triton compiles, the plan memos), outside the measurement
    for arm in ARMS:
        _set_arm(arm)
        for _ in range(2):
            layer_rep()
        step_rep()
    torch.cuda.synchronize()

    for bi, arm in enumerate(ORDER):
        _set_arm(arm)
        c0 = _counters()
        # layer: the timed reps
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        R = res[arm]["layer"]
        for i in range(a.reps):
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t = time.perf_counter()
            e0.record()
            x = layer_rep()
            e1.record()
            torch.cuda.synchronize()
            R["walls"].append((time.perf_counter() - t) * 1e3)
            R["events"].append(e0.elapsed_time(e1))
            if i == 0:
                got = _grads_cpu([x.grad] + [p.grad for p in lparams])
                if ref_layer is None:
                    ref_layer = got
                else:
                    d = _first_diff(["input"] + lnames, ref_layer, got)
                    if d:
                        R["grads_equal_base"], R["grads_first_diff"] = False, R["grads_first_diff"] or f"block {bi}: {d}"
        R["reserved"].append(torch.cuda.max_memory_reserved())
        R["allocated"].append(torch.cuda.max_memory_allocated())
        for _ in range(3):
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                layer_rep()
                torch.cuda.synchronize()
            dms, n = _device_ms(prof)
            R["device"].append(dms)
            R["launches"].append(n)
        # step
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        S = res[arm]["step"]
        for i in range(a.step_reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            step_rep()
            torch.cuda.synchronize()
            S["walls"].append((time.perf_counter() - t) * 1e3)
            if i == 0:
                if ref_step is None:
                    ref_step = _grads_cpu([p.grad for _, p in named])
                else:
                    d = _first_diff([n for n, _ in named], ref_step, [p.grad for _, p in named])
                    if d:
                        S["grads_equal_base"], S["grads_first_diff"] = False, S["grads_first_diff"] or f"block {bi}: {d}"
        S["reserved"].append(torch.cuda.max_memory_reserved())
        S["block_medians"].append(statistics.median(S["walls"][-a.step_reps:]))
        if bi < len(ARMS):                                  # one profiled step per arm: its device busy time
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                step_rep()
                torch.cuda.synchronize()
            S["device"].append(_device_ms(prof)[0])
        c1 = _counters()
        for k in c1:
            res[arm]["engaged"][k] += c1[k] - c0[k]
        print(f"[tb] block {bi} {arm}: layer {statistics.median(R['walls'][-a.reps:]):.2f} ms "
              f"device {statistics.median(R['device'][-3:]):.2f} ms; step {statistics.median(S['walls'][-a.step_reps:]):.0f} ms; "
              f"engaged {res[arm]['engaged']}", flush=True)

    # the optimizer, once, at the end (it does not see the knobs)
    _set_arm("base")
    step_rep()
    opt = torch.optim.AdamW([p for _, p in named], lr=1e-5)
    opt.step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    opt.step()
    torch.cuda.synchronize()
    opt_ms = (time.perf_counter() - t) * 1e3

    summary = {}
    for arm, d in res.items():
        L, S = d["layer"], d["step"]
        summary[arm] = {"layer": {"wall_ms": round(statistics.median(L["walls"]), 3), "device_ms": round(statistics.median(L["device"]), 3),
                                  "event_ms": round(statistics.median(L["events"]), 3), "launches": int(statistics.median(L["launches"])),
                                  "reserved_peak": max(L["reserved"]), "allocated_peak": max(L["allocated"]),
                                  "grads_equal_base": L["grads_equal_base"], "grads_first_diff": L["grads_first_diff"]},
                        "step": {"wall_ms": round(statistics.median(S["walls"]), 1), "reserved_peak": max(S["reserved"]),
                                 "device_ms": round(statistics.median(S["device"]), 1) if S["device"] else None,
                                 "self_pair_ms": round(abs(S["block_medians"][0] - S["block_medians"][-1]), 1),
                                 "grads_equal_base": S["grads_equal_base"], "grads_first_diff": S["grads_first_diff"]},
                        "engaged": d["engaged"]}
    b = summary["base"]
    spread = max(s["step"]["self_pair_ms"] for s in summary.values())
    exposed = b["step"]["wall_ms"] - b["step"]["device_ms"]                 # the base step's wall not covered by device work
    for arm, s in summary.items():
        s["step"].update(project(b, s, opt_ms, spread))
    verdicts = {arm: decide(arm, summary[arm], b) for arm in ARMS}
    import importlib.metadata as md
    cpu = (open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t") if os.path.exists("/proc/cpuinfo") else None)
    out = {"tier": "EXPLORE (T-B option C prototype, Map 1): a design input, never a reading", "model": a.model, "seq": a.seq,
           "accum": a.accum, "r": a.r, "alpha": a.alpha, "layer": a.layer, "reps": a.reps, "step_reps": a.step_reps,
           "order": ORDER, "n_patched": n_patched, "gpu": torch.cuda.get_device_name(), "cpu": cpu, "torch": torch.__version__,
           "versions": {"e4b": md.version("experts4bit-qlora"), "gnf4": md.version("grouped-nf4-gemm")},
           "optimizer_ms": round(opt_ms, 1), "self_pair_spread_ms": round(spread, 1), "base_step_exposed_ms": round(exposed, 1),
           "summary": summary, "verdicts": verdicts, "raw": res,
           "rule": {"wall_fall_min": WALL_FALL_MIN, "device_rise_max": DEVICE_RISE_MAX, "reserved_rise_max": RESERVED_RISE_MAX,
                    "judged_on": "the fast-CPU host (the 7950X); any other host is reported beside it"}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for arm in ARMS:
        s, v = summary[arm], verdicts[arm]
        print(f"[tb] {arm}: layer wall {s['layer']['wall_ms']} ms device {s['layer']['device_ms']} ms launches "
              f"{s['layer']['launches']} reserved {s['layer']['reserved_peak'] / 2**30:.2f} GiB; step {s['step']['measured_step_ms']} ms "
              f"(projected {s['step']['projected_step_ms']}, gap {s['step']['projection_gap_ms']}"
              f"{' DISAGREES beyond the self-pair spread' if s['step']['projection_disagrees'] else ''}) -> {v['verdict']} {v['why']}",
              flush=True)
    print(f"[tb] self-pair spread {spread:.1f} ms; base step exposed (wall - device) {exposed:.1f} ms", flush=True)
    print(f"[tb] host {cpu}; gpu {out['gpu']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
