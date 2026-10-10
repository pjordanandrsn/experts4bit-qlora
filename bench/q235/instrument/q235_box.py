# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's step: one full-depth QLoRA optimizer step with the records bench/q235/PREREG-q235.md registers.

The same script runs the registered box, the gate-1 rehearsal (a depth-truncated checkpoint, ``--device cpu`` or the RTX A2000) and
the rehearsal's planted errors (``--plant routing`` / ``--plant epilogue``, at layer floor(n/2), Amendment 1). ``--repeat`` is the
fresh-process repeat of step 0: it loads, runs the step-0 forward and records its loss, and stops.

Records (one JSON): host facts; versions; the load (timed); frozen-base sha256 after load and after the step (every frozen parameter and
buffer, host homes included); the in-process spread (two forward-backward passes, no optimizer step); the step's loss, per-adapter
gradient checks (finite, nonzero count, the global norm) and optimizer delta (every trained tensor changed); device memory (allocated
and reserved peaks); host memory (peak RSS); bytes moved host-to-device by the expert offload (``E4B_OFFLOAD_STATS=1``: its per-copy
counters, on for the step) and NVMe bytes (``/proc/diskstats``) around the load and around the step; the step's synchronized elapsed
times; and each decoder layer's output on the step-0 forward, for the fidelity read against the layer-streamed bf16 reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time

import torch

QKV = ("q_proj", "k_proj", "v_proj")


def diskstats():
    """Sectors read and written per block device (512 B each), from /proc/diskstats; empty where there is none."""
    out = {}
    try:
        for line in open("/proc/diskstats"):
            f = line.split()
            if len(f) >= 10 and not f[2].startswith(("loop", "ram")):
                out[f[2]] = (int(f[5]) * 512, int(f[9]) * 512)
    except OSError:
        pass
    return out


def disk_delta(a, b):
    return {d: {"read": b[d][0] - a[d][0], "written": b[d][1] - a[d][1]} for d in b if d in a and b[d] != a[d]}


def host_facts():
    facts = {"platform": platform.platform(), "cpu": platform.processor() or "", "python": platform.python_version()}
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                facts["cpu"] = line.split(":", 1)[1].strip()
                break
        mem = {k: int(v.split()[0]) * 1024 for k, v in (ln.split(":", 1) for ln in open("/proc/meminfo")) if k in ("MemTotal", "MemAvailable")}
        facts.update({"host_ram_total": mem.get("MemTotal"), "host_ram_available": mem.get("MemAvailable")})
    except OSError:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        facts.update({"gpu": p.name, "gpu_memory": p.total_memory, "capability": f"{p.major}.{p.minor}"})
    return facts


def versions():
    import importlib.metadata as m
    out = {"torch": torch.__version__}
    for d in ("transformers", "bitsandbytes", "grouped-nf4-gemm", "experts4bit-qlora"):
        try:
            out[d] = m.version(d)
        except m.PackageNotFoundError:
            out[d] = None
    return out


def frozen_hashes(model):
    """sha256 of every frozen tensor in the model's state dict, by name. The expert offload's state-dict hook puts each evicted
    tensor's host home in place of its 0-element placeholder, so the hashes cover the frozen bytes wherever they live."""
    trained = {n for n, p in model.named_parameters() if p.requires_grad}
    out = {}
    for n, t in model.state_dict().items():
        if n in trained or not isinstance(t, torch.Tensor) or t.is_meta:
            continue
        h = hashlib.sha256()
        flat = t.detach().reshape(-1)
        step = 1 << 26
        for s in range(0, flat.numel(), step):
            h.update(flat[s:s + step].to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
        out[n] = h.hexdigest()
    return out


def adapter_params(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def plant(model, kind):
    """Amendment 1's planted errors, at layer floor(n / 2), on the model under test. Returns the planted layer index."""
    layers = model.model.layers
    i = len(layers) // 2
    mlp = layers[i].mlp
    if kind == "routing":
        E = mlp.gate.weight.shape[0] if hasattr(mlp.gate, "weight") else mlp.experts.num_experts

        def shift(mod, args, kwargs):
            # experts(hidden, top_k_index, top_k_weights): every routed token goes to the next expert at its own weight
            a = list(args)
            if len(a) >= 2:
                a[1] = (a[1] + 1) % E
            elif "top_k_index" in kwargs:
                kwargs["top_k_index"] = (kwargs["top_k_index"] + 1) % E
            return tuple(a), kwargs

        mlp.experts.register_forward_pre_hook(shift, with_kwargs=True)
    elif kind == "epilogue":
        base = getattr(mlp.experts, "base", mlp.experts)
        base.act_fn = torch.nn.Identity()                           # gate * up: the activation dropped
    else:
        raise ValueError(f"unknown plant {kind!r}")
    return i


def build(path, device, r, alpha, adapter_dtype, enabler):
    from experts4bit_qlora import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
    torch.manual_seed(0)
    offload = device == "cuda"
    model, _ = load_moe_4bit_streaming(path, device, torch.bfloat16, r, alpha, offload=offload, pin=offload, prefetch=False,
                                       quant_type="nf4")
    quantize_attention_projections_4bit(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    add_attention_lora(model, r, alpha, {"bf16": torch.bfloat16, "fp32": torch.float32}[adapter_dtype])
    rec = {"offload": offload}
    if enabler == "fast":
        from experts4bit_qlora import enable_fast_train
        import experts4bit_qlora.engines.fast as fast
        n = enable_fast_train(model)
        assert n > 0, "enable_fast_train patched nothing"
        rec.update({"patched": n, "absmax_dq": fast.FAST_TRAIN_STATS.get("absmax_dq")})
    model.train()
    return model, rec


def run(a):
    out = {"argv": vars(a), "host": host_facts(), "versions": versions(), "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dev = a.device
    cuda = dev == "cuda"
    tokens = torch.load(a.tokens)
    out["tokens_sha256"] = {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in tokens.items()}
    d0, t0 = diskstats(), time.time()
    model, rec = build(a.checkpoint, dev, a.r, a.alpha, a.adapter_dtype, a.enabler)
    out["load"] = {**rec, "seconds": round(time.time() - t0, 1), "nvme": disk_delta(d0, diskstats())}
    planted = plant(model, a.plant) if a.plant else None
    out["plant"] = {"kind": a.plant, "layer": planted}
    ids, mask, labels = (tokens[k].to(dev) for k in ("input_ids", "attention_mask", "labels"))

    layer_out = [None] * len(model.model.layers)
    topk = [None] * len(model.model.layers)

    def keep(i):
        def h(m, args, o):
            if layer_out[i] is None:
                layer_out[i] = (o[0] if isinstance(o, tuple) else o).detach().to("cpu").contiguous()
        return h

    def route(i):
        def h(m, args, o):
            if topk[i] is None:
                topk[i] = o[2].detach().to("cpu")
        return h
    hooks = [lay.register_forward_hook(keep(i)) for i, lay in enumerate(model.model.layers)]
    hooks += [lay.mlp.gate.register_forward_hook(route(i)) for i, lay in enumerate(model.model.layers)]
    out["_topk"] = topk

    def sync():
        if cuda:
            torch.cuda.synchronize()

    if a.repeat:
        with torch.enable_grad():
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        sync()
        out["repeat"] = {"loss": float(loss)}
        for h in hooks:
            h.remove()
        return out, layer_out

    t_h = time.time()
    out["frozen_sha256_after_load"] = frozen_hashes(model)
    out["frozen_hash_seconds"] = round(time.time() - t_h, 1)
    params = adapter_params(model)

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        sync()
        t = time.time()
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        sync()
        tf = time.time()
        loss.backward()
        sync()
        tb = time.time()
        return float(loss), {n: p.grad.detach().float().clone() for n, p in params.items() if p.grad is not None}, tf - t, tb - tf

    l0, g0, _, _ = fwd_bwd()                                   # step-0 forward: the layer outputs are recorded here
    for h in hooks:
        h.remove()
    l1, g1, _, _ = fwd_bwd()
    spread = max(((g1[n] - g0[n]).abs().max() / g0[n].abs().max().clamp_min(1e-30)).item() for n in g0)
    out["in_process"] = {"loss_0": l0, "loss_1": l1, "loss_equal": l0 == l1, "grad_maxrel": spread}

    if cuda:
        import bitsandbytes as bnb
        opt, opt_name = bnb.optim.AdamW8bit(list(params.values()), lr=a.lr, weight_decay=a.weight_decay), "bitsandbytes AdamW8bit"
    else:                                                       # the CPU rehearsal: mechanics only, not the registered optimizer
        opt, opt_name = torch.optim.AdamW(list(params.values()), lr=a.lr, weight_decay=a.weight_decay), "torch AdamW (CPU rehearsal)"
    before = {n: p.detach().float().clone() for n, p in params.items()}
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    os.environ["E4B_OFFLOAD_STATS"] = "1"
    from experts4bit_qlora.engines import offload as off
    off.reset_offload_stats()
    d1 = diskstats()
    t_step = time.time()
    loss, grads, t_fwd, t_bwd = fwd_bwd()
    sync()
    t_o = time.time()
    opt.step()
    sync()
    t_opt = time.time() - t_o
    out["step"] = {
        "loss": loss, "forward_s": round(t_fwd, 3), "backward_s": round(t_bwd, 3), "optimizer_s": round(t_opt, 3),
        "elapsed_s": round(time.time() - t_step, 3),
        "h2d_bytes_offload": int(sum(c[2] for c in off._stats().copies)), "h2d_copies_offload": len(off._stats().copies),
        "nvme": disk_delta(d1, diskstats()),
        "max_memory_allocated": torch.cuda.max_memory_allocated() if cuda else None,
        "max_memory_reserved": torch.cuda.max_memory_reserved() if cuda else None,
        "host_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        "lr": a.lr, "weight_decay": a.weight_decay, "optimizer": opt_name, "instrumented": "E4B_OFFLOAD_STATS=1 during the step",
    }
    os.environ["E4B_OFFLOAD_STATS"] = "0"
    gstats = {n: {"finite": bool(torch.isfinite(g).all()), "nonzero": int((g != 0).sum()), "numel": g.numel()} for n, g in grads.items()}
    gnorm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads.values())))
    delta = {n: float((p.detach().float() - before[n]).abs().max()) for n, p in params.items()}
    # LoRA's B starts at zero, so at step 0 every A's gradient is exactly zero (it flows through B) and only weight decay can move
    # an A (below bf16's resolution at lr 2e-4, wd 0.001). The checks are therefore by factor: every B nonzero and changed, every A
    # finite and exactly zero -- a nonzero A gradient at step 0 would mean B was not zero-initialized.
    is_a = {n: ".lora_A" in n or n.endswith("lora_A") or "_lora_A" in n for n in gstats}
    zero = sorted(n for n, v in gstats.items() if v["nonzero"] == 0)
    unchanged = sorted(n for n, v in delta.items() if v == 0)
    b_names = [n for n in gstats if not is_a[n]]
    a_names = [n for n in gstats if is_a[n]]
    out["gradients"] = {"global_norm": gnorm, "all_finite": all(v["finite"] for v in gstats.values()), "tensors": len(gstats),
                        "lora_B": len(b_names), "lora_A": len(a_names),
                        "every_B_nonzero": all(gstats[n]["nonzero"] > 0 for n in b_names),
                        "every_A_zero_at_step0": all(gstats[n]["nonzero"] == 0 for n in a_names),
                        "zero_tensors": zero, "missing": sorted(set(params) - set(grads))}
    out["optimizer_delta"] = {"every_B_changed": all(delta[n] > 0 for n in b_names), "A_changed": sum(delta[n] > 0 for n in a_names),
                              "unchanged_tensors": unchanged, "min_max_abs_delta_B": min(delta[n] for n in b_names)}
    after = frozen_hashes(model)
    out["frozen_sha256_after_step_equal"] = after == out["frozen_sha256_after_load"]
    out["frozen_tensors"] = len(after)
    out["frozen_sha256_after_load"] = hashlib.sha256(json.dumps(out["frozen_sha256_after_load"], sort_keys=True).encode()).hexdigest()
    return out, layer_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("tokens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--enabler", default="fast", choices=("fast", "reference"))
    ap.add_argument("--adapter-dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--plant", choices=("routing", "epilogue"), default=None)
    ap.add_argument("--repeat", action="store_true")
    a = ap.parse_args()
    if a.device == "cpu" and a.enabler == "fast":
        ap.error("the fused training kernels need CUDA; use --enabler reference on the CPU")
    out, layer_out = run(a)
    os.makedirs(a.out, exist_ok=True)
    sha = []
    for i, t in enumerate(layer_out):
        if t is None:
            continue
        torch.save(t, os.path.join(a.out, f"layer{i:03d}.pt"))
        sha.append(hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())
    out["layer_sha256"] = sha
    topk = out.pop("_topk")
    torch.save({"router_topk": topk}, os.path.join(a.out, "nf4.pt"))
    loss = out["repeat"]["loss"] if a.repeat else out["in_process"]["loss_0"]
    json.dump({"loss": loss}, open(os.path.join(a.out, "nf4.json"), "w"))
    json.dump(out, open(os.path.join(a.out, "box.json"), "w"), indent=1, default=str)
    brief = {k: out[k] for k in ("plant", "in_process", "gradients", "optimizer_delta", "frozen_sha256_after_step_equal") if k in out}
    if "step" in out:
        brief["step"] = {k: out["step"][k] for k in ("loss", "elapsed_s", "h2d_bytes_offload", "max_memory_allocated", "host_peak_rss")}
    if "repeat" in out:
        brief["repeat"] = out["repeat"]
    print(json.dumps(brief, default=str))


if __name__ == "__main__":
    main()
