# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's step: one full-depth QLoRA optimizer step with the records bench/q235/PREREG-q235.md registers.

The same script runs the registered box, the gate-1 rehearsal (a depth-truncated checkpoint, ``--device cpu`` or the RTX A2000) and
the rehearsal's planted errors (``--plant routing`` / ``--plant epilogue``, at layer floor(n/2), Amendment 1). ``--repeat`` is the
fresh-process repeat of step 0: it loads, runs the step-0 forward and records its loss, and stops. ``--repeat --plants`` (Amendment 3)
then runs each planted error in the same process after that clean forward (the null): plant, forward, unplant, each checked against
the null and structurally unplanted; their records go to ``plant_routing/`` and ``plant_epilogue/`` under ``--out``. The fp32 copies
behind the gradient and delta checks live on the host (Amendment 4), so the device peak is the training step's own.

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
import sys
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


EXPERT_STACK_TENSORS = ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")


def frozen_hashes(model):
    """sha256 of every frozen tensor in the model's state dict, by name. The expert offload's state-dict hook puts each evicted
    tensor's host home in place of its 0-element placeholder, so the hashes cover the frozen bytes wherever they live.

    STRICT (the bands assume the base does not move): a frozen tensor that reads as 0 elements (a placeholder the hook did not
    replace) or a 4-bit expert stack whose packed storage or absmax is absent from the hashes raises, and the box fails."""
    from experts4bit_qlora import ExpertsNbit
    trained = {n for n, p in model.named_parameters() if p.requires_grad}
    out = {}
    for n, t in model.state_dict().items():
        if n in trained or not isinstance(t, torch.Tensor) or t.is_meta:
            continue
        if t.numel() == 0:
            raise RuntimeError(f"frozen tensor {n} reads as 0 elements: a placeholder, not the frozen bytes")
        h = hashlib.sha256()
        flat = t.detach().reshape(-1)
        step = 1 << 26
        for s in range(0, flat.numel(), step):
            h.update(flat[s:s + step].to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
        out[n] = h.hexdigest()
    stacks = [n for n, m in model.named_modules() if isinstance(m, ExpertsNbit) and int(getattr(m, "bits", 4)) < 16]
    missing = [f"{n}.{k}" for n in stacks for k in EXPERT_STACK_TENSORS          # an absmax may be stored double-quantized
               if not any(h == f"{n}.{k}" or (k.endswith("absmax") and h.startswith(f"{n}.{k}")) for h in out)]
    if not stacks or missing:
        raise RuntimeError(f"the frozen hashes miss expert storage: {len(stacks)} 4-bit stacks, missing {missing[:4]}")
    return out


STRICT_RULES = ("step0_all_finite", "step0_every_B_nonzero", "step0_every_A_zero", "step0_every_B_changed", "step1_all_finite",
                "step1_every_A_nonzero", "step1_every_B_nonzero", "step1_every_A_changed", "step1_every_B_changed")


def strict_failures(out, repeat=False):
    """The STRICT checks a box record fails, by name: Amendment 2's rules and the frozen base across both steps. Empty = none.
    Fails CLOSED: outside --repeat (step 0's forward only, which records neither), a rule or the frozen-base equality that is
    absent or not exactly True is a failure, so a record that stopped early never reads as clean."""
    if repeat:
        return []
    rules = out.get("rules") or {}
    bad = [k for k in STRICT_RULES if rules.get(k) is not True]
    bad += [k for k, v in rules.items() if k not in STRICT_RULES and v is not True]
    if out.get("frozen_sha256_after_both_steps_equal") is not True:
        bad.append("frozen_sha256_after_both_steps_equal")
    return bad


def host_memory():
    """This process's host memory from ``/proc/self/status``, in bytes (Amendment 4; reported): the peak and current resident set
    and its anonymous, file-backed and shared parts. File-backed pages (the checkpoint's mmap) are reclaimable; the anonymous and
    shared ones (pinned expert homes, the check copies) are what the host-RAM floor covers. Empty where the file does not exist."""
    keys = ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem")
    try:
        with open("/proc/self/status") as f:
            return {k: int(v.split()[0]) * 1024 for k, v in (ln.split(":", 1) for ln in f if ":" in ln) if k in keys}
    except OSError:
        return {}


def host_copy(t):
    """An fp32 copy of ``t`` on the host, never a view of it (Amendment 4). The box's gradient and delta checks hold their copies of the
    adapters (``g0``, ``g1``, ``before``, ``grads``) here rather than on the device, so the step's device peak measures the training
    step alone. The tensor leaves the device at its own dtype and is widened on the host, so no fp32 temporary lands on the device."""
    return t.detach().to("cpu", copy=True).float()


def adapter_params(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def plant(model, kind):
    """Amendment 1's planted errors, at layer floor(n / 2), on the model under test. Returns the planted layer index and a callable
    that removes the plant again (Amendment 3's in-process plants)."""
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

        undo = mlp.experts.register_forward_pre_hook(shift, with_kwargs=True).remove
    elif kind == "epilogue":
        base = getattr(mlp.experts, "base", mlp.experts)
        act = base.act_fn
        base.act_fn = torch.nn.Identity()                           # gate * up: the activation dropped

        def undo():
            base.act_fn = act
    else:
        raise ValueError(f"unknown plant {kind!r}")
    return i, undo


def plant_surface(model, i):
    """What a plant may touch at layer i, by identity: every hook and instance-level forward on the layer's MLP, its router and
    its experts (and the experts' base), and the experts' activation. Amendment 3 asserts it equal before and after each in-process
    plant, so one plant cannot leak into the next forward."""
    mlp = model.model.layers[i].mlp
    mods = {"mlp": mlp, "gate": mlp.gate, "experts": mlp.experts, "base": getattr(mlp.experts, "base", mlp.experts)}
    out = {}
    for name, m in mods.items():
        out[name] = {k: tuple((h_id, id(fn)) for h_id, fn in getattr(m, k).items())
                     for k in ("_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_forward_hooks", "_forward_hooks_with_kwargs")
                     if hasattr(m, k)}
        out[name]["forward"] = id(m.__dict__["forward"]) if "forward" in m.__dict__ else None
        out[name]["act_fn"] = id(getattr(m, "act_fn", None))
    return out


class LoadProfile:
    """Where the load's time goes, for gate 2's premise (the time-left fit assumes the rented host is no slower than the rehearsal at
    whatever bounds the load). For the load's duration it wraps, and then restores:
    - ``read``: every shard handle's ``get_tensor`` (the loader's ``safe_open``), with the bytes it returned;
    - ``assemble``: the MoE conventions' ``fuse_experts`` / ``stack_experts`` (the per-expert rows fused into stacks, on the host
      under offload);
    - ``quantize``: ``Experts4bit.from_float`` / ``ExpertsNbit.from_float``;
    - ``offload``: ``enable_expert_offload`` (the quantized stacks copied to their pinned host homes).
    Each wrapped call synchronizes CUDA at its boundaries, so asynchronous device work is charged to the call that issued it.
    ``other`` is the rest of the load: the fused stacks' host-to-device copy, the non-expert weights and the bookkeeping."""

    def __init__(self, cuda):
        self.cuda = cuda
        self.s = {k: 0.0 for k in ("read", "assemble", "quantize", "offload")}
        self.n = dict.fromkeys(self.s, 0)
        self.read_bytes = 0
        self._undo = []

    def _timed(self, key, fn):
        def wrap(*a, **k):
            if self.cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            if self.cuda:
                torch.cuda.synchronize()
            self.s[key] += time.perf_counter() - t
            self.n[key] += 1
            if key == "read" and isinstance(out, torch.Tensor):
                self.read_bytes += out.numel() * out.element_size()
            return out
        return wrap

    def _patch(self, obj, name, new):
        self._undo.append((obj, name, obj.__dict__[name] if isinstance(obj, type) else getattr(obj, name)))
        setattr(obj, name, new)

    def __enter__(self):
        import experts4bit_qlora.arch.moe_conventions as conv
        import experts4bit_qlora.loader as loader
        from experts4bit_qlora import Experts4bit, ExpertsNbit
        prof, real_open = self, loader.safe_open

        class _Handle:
            def __init__(self, h):
                self._h = h

            def get_tensor(self, name):
                return prof._timed("read", self._h.get_tensor)(name)

            def __enter__(self):
                self._h.__enter__()
                return self

            def __exit__(self, *a):
                return self._h.__exit__(*a)

            def __getattr__(self, name):
                return getattr(self._h, name)

        self._patch(loader, "safe_open", lambda *a, **k: _Handle(real_open(*a, **k)))
        for name in ("fuse_experts", "stack_experts"):
            self._patch(conv, name, self._timed("assemble", getattr(conv, name)))
        self._patch(loader, "enable_expert_offload", self._timed("offload", loader.enable_expert_offload))
        for cls in {Experts4bit, ExpertsNbit}:
            if "from_float" in cls.__dict__:
                self._patch(cls, "from_float", classmethod(self._timed("quantize", cls.__dict__["from_float"].__func__)))
        return self

    def __exit__(self, *a):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)
        return False

    def record(self, total_s):
        rec = {k: round(v, 2) for k, v in self.s.items()}
        rec.update(calls=self.n, read_bytes=self.read_bytes, total_s=round(total_s, 2),
                   other_s=round(total_s - sum(self.s.values()), 2))
        rec["read_gb_s"] = round(self.read_bytes / self.s["read"] / 1e9, 3) if self.s["read"] else None
        return rec


def build(path, device, r, alpha, adapter_dtype, enabler, profile=None):
    from experts4bit_qlora import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
    torch.manual_seed(0)
    offload = device == "cuda"
    t0 = time.perf_counter()
    if profile is not None:
        with profile:
            model, _ = load_moe_4bit_streaming(path, device, torch.bfloat16, r, alpha, offload=offload, pin=offload, prefetch=False,
                                               quant_type="nf4")
    else:
        model, _ = load_moe_4bit_streaming(path, device, torch.bfloat16, r, alpha, offload=offload, pin=offload, prefetch=False,
                                           quant_type="nf4")
    load_s = time.perf_counter() - t0
    quantize_attention_projections_4bit(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    add_attention_lora(model, r, alpha, {"bf16": torch.bfloat16, "fp32": torch.float32}[adapter_dtype])
    rec = {"offload": offload}
    if profile is not None:
        rec["profile"] = profile.record(load_s)
    if enabler == "fast":
        from experts4bit_qlora import enable_fast_train
        import experts4bit_qlora.engines.fast as fast
        n = enable_fast_train(model)
        assert n > 0, "enable_fast_train patched nothing"
        rec.update({"patched": n, "absmax_dq": fast.FAST_TRAIN_STATS.get("absmax_dq")})
    model.train()
    return model, rec


def capture(model):
    """Forward hooks that keep each decoder layer's output and each router's top-k from the FIRST forward after they are set
    (gradient checkpointing's recompute calls the layers again in the backward). Returns the two lists and the hook handles."""
    n = len(model.model.layers)
    layer_out, topk = [None] * n, [None] * n

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
    return layer_out, topk, hooks


def _sha(t):
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def in_process_plants(model, forward, null, mask=None):
    """Amendment 3: after the repeat's clean step-0 forward (the NULL, ``null``: its layer outputs), each planted error in turn,
    in this process: plant, forward, unplant. Each plant's record carries its layer outputs and router top-k, its loss, each layer
    before the planted one's relative difference from the null over the real positions (``mask``; the verdict applies Amendment 3's
    tolerance) beside whether they are equal bit for bit (reported), and whether the planted layer differs from it (the plant engaged).
    The unplant is checked structurally: the planted layer's hooks, instance forwards and activation must equal their snapshot from
    before the plant, or this raises (a harness error) -- the routing plant cannot leak into the epilogue forward."""
    out = {}
    for kind in ("routing", "epilogue"):
        i = len(model.model.layers) // 2
        before = plant_surface(model, i)
        layer, undo = plant(model, kind)
        assert layer == i
        lo, tk, hooks = capture(model)
        try:
            t_f = time.perf_counter()
            loss = forward()
            forward_s = time.perf_counter() - t_f
        finally:
            for h in hooks:
                h.remove()
            undo()
        if plant_surface(model, i) != before:
            raise RuntimeError(f"the {kind} plant did not come off layer {i}: its hooks or forwards differ from before it")
        sha, sha_null = [_sha(t) for t in lo], [_sha(t) for t in null]
        m = mask.bool().cpu() if mask is not None else None
        rel = []
        for a, b in zip(lo[:i], null[:i]):
            a, b = (a.float()[m], b.float()[m]) if m is not None else (a.float(), b.float())
            rel.append(float((a - b).norm() / b.norm().clamp_min(1e-30)))
        out[kind] = {"layer": i, "loss": loss, "forward_s": round(forward_s, 3), "layer_sha256": sha,
                     "pre_plant_rel_diff": rel, "pre_plant_max_rel_diff": max(rel, default=0.0),
                     "pre_plant_equal_null": sha[:i] == sha_null[:i], "planted_layer_differs": sha[i] != sha_null[i],
                     "_layers": lo, "_topk": tk}
    return out


def run(a):
    out = {"argv": vars(a), "host": host_facts(), "versions": versions(), "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dev = a.device
    cuda = dev == "cuda"
    tokens = torch.load(a.tokens, weights_only=True)
    out["tokens_sha256"] = {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in tokens.items()}
    d0, t0 = diskstats(), time.time()
    model, rec = build(a.checkpoint, dev, a.r, a.alpha, a.adapter_dtype, a.enabler, profile=LoadProfile(cuda))
    out["load"] = {**rec, "seconds": round(time.time() - t0, 1), "nvme": disk_delta(d0, diskstats()), "host_memory": host_memory()}
    planted = plant(model, a.plant)[0] if a.plant else None
    out["plant"] = {"kind": a.plant, "layer": planted}
    ids, mask, labels = (tokens[k].to(dev) for k in ("input_ids", "attention_mask", "labels"))
    layer_out, topk, hooks = capture(model)
    out["_topk"] = topk

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def forward():
        with torch.enable_grad():                                  # the step's own forward: gradient checkpointing as trained
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        sync()
        return float(loss.detach())

    if a.repeat:
        t_f = time.perf_counter()
        out["repeat"] = {"loss": forward(), "forward_s": round(time.perf_counter() - t_f, 3)}
        for h in hooks:
            h.remove()
        if a.plants:
            out["_plants"] = in_process_plants(model, forward, layer_out, tokens["attention_mask"])
            out["plants"] = {k: {f: v[f] for f in v if not f.startswith("_")} for k, v in out["_plants"].items()}
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
        grads = {n: host_copy(p.grad) for n, p in params.items() if p.grad is not None}
        return float(loss), grads, tf - t, tb - tf, time.time() - tb

    l0, g0, *_ = fwd_bwd()                                     # step-0 forward: the layer outputs are recorded here
    for h in hooks:
        h.remove()
    l1, g1, *_ = fwd_bwd()
    spread = max(((g1[n] - g0[n]).abs().max() / g0[n].abs().max().clamp_min(1e-30)).item() for n in g0)
    out["in_process"] = {"loss_0": l0, "loss_1": l1, "loss_equal": l0 == l1, "grad_maxrel": spread}

    if cuda:
        import bitsandbytes as bnb
        opt, opt_name = bnb.optim.AdamW8bit(list(params.values()), lr=a.lr, weight_decay=a.weight_decay), "bitsandbytes AdamW8bit"
    else:                                                       # the CPU rehearsal: mechanics only, not the registered optimizer
        opt, opt_name = torch.optim.AdamW(list(params.values()), lr=a.lr, weight_decay=a.weight_decay), "torch AdamW (CPU rehearsal)"
    from experts4bit_qlora.engines import offload as off
    is_a = {n: n.endswith("lora_A") for n in params}
    a_names = [n for n in params if is_a[n]]
    b_names = [n for n in params if not is_a[n]]

    def one_step():
        """One forward-backward and optimizer step at the registered shape, instrumented: its record, gradients and deltas."""
        before = {n: host_copy(p) for n, p in params.items()}
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        os.environ["E4B_OFFLOAD_STATS"] = "1"
        off.reset_offload_stats()
        d1 = diskstats()
        t_step = time.time()
        loss, grads, t_fwd, t_bwd, t_copy = fwd_bwd()
        sync()
        t_o = time.time()
        opt.step()
        sync()
        t_opt = time.time() - t_o
        rec = {
            "loss": loss, "forward_s": round(t_fwd, 3), "backward_s": round(t_bwd, 3), "optimizer_s": round(t_opt, 3),
            "elapsed_s": round(time.time() - t_step, 3), "grads_host_copy_s": round(t_copy, 3),   # inside elapsed_s (Amendment 4)
            "h2d_bytes_offload": int(sum(c[2] for c in off._stats().copies)), "h2d_copies_offload": len(off._stats().copies),
            "nvme": disk_delta(d1, diskstats()),
            "max_memory_allocated": torch.cuda.max_memory_allocated() if cuda else None,
            "max_memory_reserved": torch.cuda.max_memory_reserved() if cuda else None,
            "host_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
            "host_memory": host_memory(),                            # after opt.step: g0, g1, before and grads all alive
            "lr": a.lr, "weight_decay": a.weight_decay, "optimizer": opt_name, "instrumented": "E4B_OFFLOAD_STATS=1 during the step",
        }
        os.environ["E4B_OFFLOAD_STATS"] = "0"
        nonzero = {n: int((g != 0).sum()) for n, g in grads.items()}
        delta = {n: float((host_copy(p) - before[n]).abs().max()) for n, p in params.items()}
        rec["gradients"] = {
            "global_norm": float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads.values()))),
            "all_finite": all(bool(torch.isfinite(g).all()) for g in grads.values()), "tensors": len(grads),
            "lora_A": len(a_names), "lora_B": len(b_names), "missing": sorted(set(params) - set(grads)),
            "every_B_nonzero": all(nonzero.get(n, 0) > 0 for n in b_names),
            "every_A_nonzero": all(nonzero.get(n, 0) > 0 for n in a_names),
            "every_A_zero": all(nonzero.get(n, 0) == 0 for n in a_names),
            "zero_tensors": sorted(n for n, v in nonzero.items() if v == 0)}
        rec["optimizer_delta"] = {
            "every_B_changed": all(delta[n] > 0 for n in b_names), "every_A_changed": all(delta[n] > 0 for n in a_names),
            "A_changed": sum(delta[n] > 0 for n in a_names), "unchanged_tensors": sorted(n for n, v in delta.items() if v == 0)}
        return rec

    # Amendment 2. Step 0: B starts at zero, so every A's gradient is exactly zero (it flows through B) -- a nonzero one means B was
    # not zero-initialized (VOID, a harness error); B's gradients are nonzero and every B changes. Step 1, after step 0 has moved B:
    # every A's gradient is nonzero and every A changes (a broken A path cannot pass), and B's rules repeat. The frozen base's hashes
    # hold across both steps.
    s0 = one_step()
    s1 = one_step()
    out["step"], out["step1"] = s0, s1
    out["rules"] = {
        "step0_all_finite": s0["gradients"]["all_finite"], "step0_every_B_nonzero": s0["gradients"]["every_B_nonzero"],
        "step0_every_A_zero": s0["gradients"]["every_A_zero"], "step0_every_B_changed": s0["optimizer_delta"]["every_B_changed"],
        "step1_all_finite": s1["gradients"]["all_finite"], "step1_every_A_nonzero": s1["gradients"]["every_A_nonzero"],
        "step1_every_B_nonzero": s1["gradients"]["every_B_nonzero"], "step1_every_A_changed": s1["optimizer_delta"]["every_A_changed"],
        "step1_every_B_changed": s1["optimizer_delta"]["every_B_changed"]}
    after = frozen_hashes(model)
    out["frozen_sha256_after_both_steps_equal"] = after == out["frozen_sha256_after_load"]
    out["frozen_tensors"] = len(after)
    out["frozen_sha256_after_load"] = hashlib.sha256(json.dumps(out["frozen_sha256_after_load"], sort_keys=True).encode()).hexdigest()
    return out, layer_out


def save_layers(d, layer_out, topk):
    """Each decoder layer's output as ``layerNNN.pt`` and the routers' top-k as ``nf4.pt``; returns the layers' sha256."""
    sha = []
    for i, t in enumerate(layer_out):
        if t is None:
            continue
        torch.save(t, os.path.join(d, f"layer{i:03d}.pt"))
        sha.append(_sha(t))
    torch.save({"router_topk": topk}, os.path.join(d, "nf4.pt"))
    return sha


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
    ap.add_argument("--plants", action="store_true", help="with --repeat: Amendment 3's in-process null and planted errors")
    a = ap.parse_args()
    if a.device == "cpu" and a.enabler == "fast":
        ap.error("the fused training kernels need CUDA; use --enabler reference on the CPU")
    if a.plants and (not a.repeat or a.plant):
        ap.error("--plants rides the fresh-process repeat (--repeat), and replaces --plant")
    out, layer_out = run(a)
    os.makedirs(a.out, exist_ok=True)
    out["layer_sha256"] = save_layers(a.out, layer_out, out.pop("_topk"))
    for kind, rec in out.pop("_plants", {}).items():               # Amendment 3: each in-process plant in its own directory
        d = os.path.join(a.out, f"plant_{kind}")
        os.makedirs(d, exist_ok=True)
        save_layers(d, rec["_layers"], rec["_topk"])
        json.dump({"loss": rec["loss"]}, open(os.path.join(d, "nf4.json"), "w"))
        json.dump({"plant": {"kind": kind, "layer": rec["layer"]}, "in_process": True, "layer_sha256": rec["layer_sha256"],
                   "pre_plant_max_rel_diff": rec["pre_plant_max_rel_diff"], "pre_plant_rel_diff": rec["pre_plant_rel_diff"],
                   "pre_plant_equal_null": rec["pre_plant_equal_null"], "planted_layer_differs": rec["planted_layer_differs"],
                   "repeat": {"loss": rec["loss"], "forward_s": rec["forward_s"]}}, open(os.path.join(d, "box.json"), "w"), indent=1)
    loss = out["repeat"]["loss"] if a.repeat else out["in_process"]["loss_0"]
    json.dump({"loss": loss}, open(os.path.join(a.out, "nf4.json"), "w"))
    json.dump(out, open(os.path.join(a.out, "box.json"), "w"), indent=1, default=str)
    brief = {k: out[k] for k in ("plant", "in_process", "rules", "frozen_sha256_after_both_steps_equal") if k in out}
    for k in ("step", "step1"):
        if k in out:
            brief[k] = {f: out[k][f] for f in ("loss", "elapsed_s", "h2d_bytes_offload", "max_memory_allocated", "host_peak_rss")}
    if "repeat" in out:
        brief["repeat"] = out["repeat"]
    if "plants" in out:
        brief["plants"] = {k: {f: v[f] for f in ("loss", "pre_plant_max_rel_diff", "pre_plant_equal_null", "planted_layer_differs")}
                           for k, v in out["plants"].items()}
    print(json.dumps(brief, default=str))
    bad = strict_failures(out, repeat=a.repeat)
    if bad:
        print(f"STRICT FAIL: {', '.join(bad)}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
