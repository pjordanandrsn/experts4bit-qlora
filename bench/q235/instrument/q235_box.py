# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's step: one full-depth QLoRA optimizer step with the records bench/q235/PREREG-q235.md registers.

The same script runs the registered box, the gate-1 rehearsal (a depth-truncated checkpoint, ``--device cpu`` or the RTX A2000) and
the rehearsal's planted errors (``--plant routing`` / ``--plant epilogue``, at layer floor(n/2), Amendment 1). ``--repeat`` is the
fresh-process repeat of step 0: it loads, runs the step-0 forward and records its loss, and stops. ``--repeat --plants`` (Amendment 3)
then runs each planted error in the same process after that clean forward (the null): plant, forward, unplant, each checked against
the null and structurally unplanted; their records go to ``plant_routing/`` and ``plant_epilogue/`` under ``--out``. The fp32 copies
behind the gradient and delta checks live on the host (Amendment 4), so the device peak is the training step's own.

Shape B (Amendment 6, ``--micro-batch M``): each optimizer step accumulates the rows as ``rows / M`` micro-batches of ``M`` rows, in
order, each trimmed to its own longest row and weighted by its share of the valid labels (the window's token-weighted mean, as the
field recipe's trainer normalizes). The same rows as one micro-batch (shape A) give the in-process gradients ``g0``/``g1``; each
step's accumulated gradient is compared with shape A's at the same parameters (``ga_check``), and two planted accumulation bugs (a
mean of micro-batch means; each micro-batch's gradient overwriting the last) are compared the same way, before the steps.

Records (one JSON): host facts; versions; the load (timed); frozen-base sha256 after load and after the step (every frozen parameter and
buffer, host homes included); the in-process spread (two forward-backward passes, no optimizer step); the step's loss, per-adapter
gradient checks (finite, nonzero count, the global norm) and optimizer delta (every trained tensor changed); device memory (allocated
and reserved peaks); host memory (peak RSS); bytes moved host-to-device by the expert offload (``E4B_OFFLOAD_STATS=1``: its per-copy
counters, on for the step) and NVMe bytes (``/proc/diskstats``) around the load and around the step; the step's synchronized elapsed
times; and each decoder layer's output on the step-0 forward, for the fidelity read against the layer-streamed bf16 reference.

Amendment 8 (the ladder's L2, after the R2 gate-1 ruling): each step records ``dx``, every decoder layer's output gradient (dL/dh_i,
i = 0..n-1) as its fp32 norm, per forward-backward of the step's own pass (``DxProbe``), and ``rules`` gains its STRICT
``step0_dx_finite_nonzero`` / ``step1_dx_finite_nonzero``. With ``--heldout TOKENS.pt`` (the registered held-out rows,
``q235_tokens.py --first``), ``heldout`` records the loss on those rows before the update, after step 0 and after step 1, each one
no-gradient forward at shape A, with ``heldout_finite`` STRICT; the step's own rows are read the same way at the same points
(``heldout.train``, reported only). Rule 10, the verdict's, is one-sided: no post-update held-out read above ``before x 1.10``.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
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


#: Amendment 8's STRICT rules: every decoder layer's output gradient recorded, finite and nonzero at both steps; and, with
#: ``--heldout``, the held-out loss finite at every read. Required on a record that ran under Amendment 8 (``amendment8``).
STRICT_RULES_A8 = ("step0_dx_finite_nonzero", "step1_dx_finite_nonzero")


def strict_failures(out, repeat=False):
    """The STRICT checks a box record fails, by name: Amendment 2's rules and the frozen base across both steps, and on an
    Amendment 8 record its dX rules (and the held-out read's, when it ran one). Empty = none.
    Fails CLOSED: outside --repeat (step 0's forward only, which records neither), a rule or the frozen-base equality that is
    absent or not exactly True is a failure, so a record that stopped early never reads as clean."""
    if repeat:
        return []
    rules = out.get("rules") or {}
    a8 = out.get("amendment8") or {}
    need = STRICT_RULES + (STRICT_RULES_A8 if a8 else ()) + (("heldout_finite",) if a8.get("heldout") else ())
    bad = [k for k in need if rules.get(k) is not True]
    bad += [k for k, v in rules.items() if k not in need and v is not True]
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


def micro_batches(tokens, m):
    """Shape B (Amendment 6): the rows as ``len(rows) / m`` micro-batches of ``m`` rows, in order, each trimmed to its longest real row
    (the rows are right-padded), with each micro-batch's count of valid shifted labels -- its weight in the window's token mean."""
    ids, mask, labels = tokens["input_ids"], tokens["attention_mask"], tokens["labels"]
    rows = ids.shape[0]
    if m < 1 or rows % m:
        raise ValueError(f"--micro-batch {m} does not divide the {rows} rows")
    out = []
    for i in range(0, rows, m):
        w = int(mask[i:i + m].sum(1).max())
        lab = labels[i:i + m, :w]
        out.append({"input_ids": ids[i:i + m, :w], "attention_mask": mask[i:i + m, :w], "labels": lab,
                    "valid": int((lab[:, 1:] != -100).sum())})
    return out


def compare_grads(ref, got):
    """Amendment 6's views of a gradient against shape A's at the same parameters: the per-tensor relative difference (the STRICT
    quantity; over tensors whose reference is nonzero), tensors exactly zero in the reference but not here, the global cosine and the
    norm ratio of the whole adapter gradient, the largest relative difference per decoder layer, and the per-tensor differences in
    full with their p50 and p99 (nearest rank). A tensor with no gradient here (``p.grad`` None: no token reached it) counts as a zero
    gradient."""
    rel, zero_mismatch, per_layer = {}, [], {}
    dot = nr = ng = 0.0
    for n, r in ref.items():
        r64 = r.double()
        g = got.get(n)
        g64 = torch.zeros_like(r64) if g is None else g.double()
        rn = float(r64.norm())
        dot += float((r64 * g64).sum())
        nr += rn ** 2
        ng += float(g64.norm()) ** 2
        if rn == 0.0:
            if bool((g64 != 0).any()):
                zero_mismatch.append(n)
            continue
        rel[n] = float((g64 - r64).norm()) / rn
        parts = n.split(".")
        if "layers" in parts:
            i = int(parts[parts.index("layers") + 1])
            per_layer[i] = max(per_layer.get(i, 0.0), rel[n])
    for n, g in got.items():                                    # a gradient where shape A has none
        if n not in ref and bool((g != 0).any()):
            zero_mismatch.append(n)
            ng += float(g.double().norm()) ** 2
    worst = max(rel, key=rel.get) if rel else None
    srt = sorted(rel.values())

    def rank(q):
        return srt[min(len(srt) - 1, max(0, -(-int(q * 100) * len(srt) // 100) - 1))] if srt else None

    return {"tensors": len(ref), "compared": len(rel), "max_rel": rel[worst] if worst else None, "max_rel_tensor": worst,
            "rel_p50": rank(0.50), "rel_p99": rank(0.99), "rel": rel,
            "zero_mismatch": sorted(zero_mismatch), "cosine": dot / (nr ** 0.5 * ng ** 0.5) if nr and ng else None,
            "norm_ratio": (ng / nr) ** 0.5 if nr else None, "per_layer_max_rel": {str(k): per_layer[k] for k in sorted(per_layer)}}


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


class MoETimer:
    """``--time-moe N``: when the MoE experts module of N evenly spaced layers is staged and when it computes, for ONE extra
    forward and backward after both registered steps. REPORTED only: ``q235_verdict.py`` does not read it, and the steps run
    before any of these hooks exist, so their records carry none of its overhead.

    Per sampled layer, from CUDA events in stream order (``perf_counter`` on the CPU rehearsal, which has no offload), each mark
    also numbered in the order the host enqueued it:
    - ``stage_ms``: from just before e4b's offload pre-hook (``_stage_pre_hook``, which calls ``_ExpertOffload.stage``) to just
      after it. That is the layer's synchronous host-to-device copy, enqueued on the compute stream.
    - ``compute_ms``: from the end of staging to the experts module's return. That is gate_up, the activation and down
      together; e4b's fused path runs them inside one module call. ``None`` for a call that never returned: a checkpoint
      recompute that stopped early at the region's last saved tensor.
    - Each of those in the ``forward`` and, under gradient checkpointing, the ``recompute``: the same hooks firing inside the
      backward (``engines.offload._in_backward``).
    - ``backward``: the experts module's own backward window, from its full backward pre-hook to its full backward hook.
      Under a non-reentrant checkpoint the layer's recompute runs INSIDE that window (it starts at the first saved tensor
      read, which is in this module); under the reentrant one (the fast path's default) it ran before it.
      ``recompute_inside`` says which, and ``own_ms`` is the window after the recompute's compute end, the dgrad and the LoRA
      gradients (an upper bound by the layer's ops after this module), or ``None`` if the recompute ran inside the window
      but never returned.

    The hooks never change an argument, output or gradient (each returns None), and they are removed before the record is
    read."""

    PHASES = ("forward", "recompute")

    def __init__(self, model, n, cuda):
        import itertools
        from experts4bit_qlora.engines.offload import _in_backward
        self._in_backward, self.cuda, self._seq = _in_backward, cuda, itertools.count()
        L = len(model.model.layers)
        n = max(1, min(int(n), L))
        self.layers = sorted({round(k * (L - 1) / (n - 1)) for k in range(n)}) if n > 1 else [0]
        self.marks = {i: {} for i in self.layers}
        self.staged, self.handles = {}, []
        for i in self.layers:
            mod = model.model.layers[i].mlp.experts
            self.staged[i] = any(getattr(h, "__name__", "") == "_stage_pre_hook" for h in mod._forward_pre_hooks.values())
            self.handles += [
                mod.register_forward_pre_hook(self._forward(i, "stage_start"), prepend=True),   # before e4b's staging
                mod.register_forward_pre_hook(self._forward(i, "stage_end")),                    # after it
                mod.register_forward_hook(self._forward(i, "compute_end"), prepend=True),       # before e4b's evict
                mod.register_full_backward_pre_hook(self._mark(i, "backward:start")),
                mod.register_full_backward_hook(self._mark(i, "backward:end")),
            ]

    def _stamp(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def _put(self, i, key):
        self.marks[i].setdefault(key, []).append((next(self._seq), self._stamp()))

    def _mark(self, i, key):
        def hook(*_):
            self._put(i, key)
        return hook

    def _forward(self, i, what):
        def hook(*_):
            self._put(i, f"{'recompute' if self._in_backward() else 'forward'}:{what}")
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _ms(self, a, b):
        a, b = a[1], b[1]
        return round(a.elapsed_time(b) if self.cuda else (b - a) * 1e3, 4)

    def record(self):
        if self.cuda:
            torch.cuda.synchronize()
        out = {}
        for i in self.layers:
            m, row = self.marks[i], {"staged_by_offload": self.staged[i]}
            for ph in self.PHASES:
                s0, s1, c1 = (m.get(f"{ph}:{w}", []) for w in ("stage_start", "stage_end", "compute_end"))
                n = min(len(s0), len(s1))
                # a call's compute end is the first one after its stage end and before the next call's stage start
                ends = [next((e for e in c1 if s1[k][0] < e[0] < (s0[k + 1][0] if k + 1 < n else float("inf"))), None)
                        for k in range(n)]
                row[ph] = {"calls": n, "stage_ms": [self._ms(s0[k], s1[k]) for k in range(n)],
                           "compute_ms": [self._ms(s1[k], e) if e else None for k, e in enumerate(ends)]}
            b0, b1 = m.get("backward:start", []), m.get("backward:end", [])
            rs, rc = m.get("recompute:stage_start", []), m.get("recompute:compute_end", [])
            bw = {"calls": min(len(b0), len(b1)), "ms": [], "recompute_inside": [], "own_ms": []}
            for k in range(bw["calls"]):
                lo, hi = b0[k][0], b1[k][0]
                inside = [e for e in rs if lo < e[0] < hi]
                end = [e for e in rc if lo < e[0] < hi]
                bw["ms"].append(self._ms(b0[k], b1[k]))
                bw["recompute_inside"].append(bool(inside))
                bw["own_ms"].append(self._ms(end[-1], b1[k]) if end else None if inside else self._ms(b0[k], b1[k]))
            row["backward"] = bw
            out[str(i)] = row
        return out


def time_moe(model, fwd_bwd, n, cuda, sync, bound=False, micro_batches=None):
    """The extra timed pass (see MoETimer): ``fwd_bwd`` is the registered step's own forward-backward, shape B's accumulation under
    ``--micro-batch`` (Amendment 6), so each sampled layer gets one stage/compute pair per micro-batch, call ``k`` being micro-batch
    ``k`` (``micro_batches``: their rows, width, positions and valid labels). Every pass runs with the checkpoint's early stop off,
    so a non-reentrant recompute reaches the experts module's return and its compute is observed (the reentrant checkpoint always
    runs the region whole). Each pass's time includes ``fwd_bwd``'s own device-to-host copy of the adapter gradients, after the
    hooks' region; it is recorded and left out of every comparison. Never fails the box: an error is recorded and the hooks are
    removed.

    ``bound`` (Amendment 7's overhead read, as the maintainer registered it): one discarded warm-up pass, then four in ABBA order,
    untimed, timed, timed, untimed. ``overhead_s`` is the mean timed pass minus the mean untimed one; ``half_s`` is the order and
    drift noise of that read, half the difference between the two timed-minus-untimed pairs ``(on1 - off1)`` and ``(on2 - off2)``.
    The read is decision-grade only when ``half_s <= overhead_s / 2``; otherwise it is reporting-only. Both timed passes carry the
    hooks, so the layers' record then holds two calls per micro-batch, call ``k`` being micro-batch ``k mod M`` of timed pass
    ``k // M``. Every key is always written; a quantity with no base is None."""
    from torch.utils.checkpoint import set_checkpoint_early_stop
    rec = {"reported_only": True, "read_by_verdict": False, "pass": "the registered shape's forward-backward after both registered steps, "
           "no optimizer step, checkpoint early stop off", "timer": "cuda events" if cuda else "perf_counter (CPU rehearsal, no offload)",
           "micro_batches": [{**m, "positions": m["rows"] * m["width"]} for m in micro_batches or []]}

    def one_pass():
        """The pass's seconds without its grads copy, and the copy's seconds; unrounded."""
        sync()
        t = time.perf_counter()
        r = fwd_bwd()
        sync()
        copy = float(r[4]) if isinstance(r, tuple) and len(r) >= 5 else 0.0
        return time.perf_counter() - t - copy, copy
    timer = None
    try:
        with set_checkpoint_early_stop(False):
            warm = one_pass() if bound else None
            off1 = one_pass() if bound else None
            timer = MoETimer(model, n, cuda)
            rec["sampled_layers"] = timer.layers
            ons = [one_pass() for _ in range(2 if bound else 1)]
            timer.remove()
            off2 = one_pass() if bound else None
        rec["layers"] = timer.record()
        rec["timed_passes"] = len(ons)
        rec["pass_on_s"] = [round(x, 4) for x, _ in ons]
        rec["grads_copy_s"] = {"on": [round(c, 4) for _, c in ons]}
        if bound:
            on1, on2 = ons[0][0], ons[1][0]
            d1, d2 = on1 - off1[0], on2 - off2[0]
            over, half = (on1 + on2) / 2 - (off1[0] + off2[0]) / 2, abs(d1 - d2) / 2
            rec.update({"order": "warm-up, off, on, on, off", "pass_warmup_s": round(warm[0], 4),
                        "pass_off_s": [round(off1[0], 4), round(off2[0], 4)], "pairs_s": [round(d1, 4), round(d2, 4)],
                        "overhead_s": round(over, 4), "half_s": round(half, 4),
                        "overhead_ratio": round(over / ((off1[0] + off2[0]) / 2), 4) if off1[0] + off2[0] > 0 else None,
                        "decision_grade": half <= over / 2})
            rec["grads_copy_s"].update(warmup=round(warm[1], 4), off=[round(off1[1], 4), round(off2[1], 4)])
    except Exception as e:  # noqa: BLE001 - reported only: a failure here is a reading, never a box failure
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    finally:
        if timer is not None:
            timer.remove()
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


class DxProbe:
    """Amendment 8: each decoder layer's output gradient, dL/dh_i for i = 0..n-1 -- the gradient the layer's own adapters and the
    layers below it receive -- as its fp32 norm, once per forward-backward of the registered step (one per micro-batch under
    shape B).

    Where it is taken: layer i's output as the INPUT of the next module, through a forward pre-hook on layer i+1 (and on the final
    norm for the last layer). There it is an outer graph tensor under either checkpoint mode. Under a reentrant checkpoint (e4b's
    fast path, ``ckpt_offload``) the checkpointed forward runs without gradients, so the output a layer's own forward hook sees
    never carries one; the tensor handed to the next module is the checkpoint's own output, which does.

    The pre-hooks register a tensor hook only while ``armed`` (the step's own forward), so the checkpoint recompute inside the
    backward, which calls the modules again, registers nothing. A layer whose output carries no gradient, or whose hook never
    fires, stays ``None`` and fails the rule: the record fails closed."""

    def __init__(self, model):
        layers = list(model.model.layers)
        self.n = len(layers)
        self.armed, self.cur, self.passes = False, None, []
        nxt = layers[1:] + [model.model.norm]                   # the module that receives layer i's output
        self.handles = [m.register_forward_pre_hook(self._hook(i), with_kwargs=True) for i, m in enumerate(nxt)]

    def _hook(self, i):
        def h(mod, args, kwargs):
            if not self.armed or self.cur is None or self.cur[i] is not None:
                return None
            t = args[0] if args else kwargs.get("hidden_states")
            if not (torch.is_tensor(t) and t.requires_grad):
                return None
            cur = self.cur
            cur[i] = "pending"

            def g(grad):
                cur[i] = float(grad.detach().float().norm())
            t.register_hook(g)
            return None
        return h

    def arm(self):
        """Before the step's forward (each micro-batch's)."""
        self.cur = [None] * self.n
        self.passes.append(self.cur)
        self.armed = True

    def disarm(self):
        """After the forward returns, before the backward: the recompute registers nothing."""
        self.armed = False

    def remove(self):
        for h in self.handles:
            h.remove()

    def record(self):
        bad = [[p, i] for p, norms in enumerate(self.passes) for i, v in enumerate(norms)
               if not isinstance(v, float) or not math.isfinite(v) or v <= 0]
        vals = [v for norms in self.passes for v in norms if isinstance(v, float)]
        return {"layers": self.n, "passes": len(self.passes), "norms": self.passes,
                "finite_nonzero": bool(self.passes) and not bad, "failing": bad[:20], "failing_count": len(bad),
                "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                "what": "dL/dh_i at each decoder layer's output (fp32 norm), taken as the next module's input, per forward-backward "
                        "of the step (Amendment 8)"}


def heldout_loss(model, held, sync):
    """Amendment 8: the loss on the registered held-out rows at the current parameters, no gradient (one forward, shape A)."""
    with torch.no_grad():
        loss = model(input_ids=held["input_ids"], attention_mask=held["attention_mask"], labels=held["labels"]).loss
    sync()
    return float(loss)


@contextlib.contextmanager
def deterministic():
    """Amendment 6: ``torch.use_deterministic_algorithms(True, warn_only=True)`` for rule 8's check passes only. The previous setting
    (enabled, warn_only) is restored on exit, however the block ends, so the registered steps run on the shipped default path."""
    prev = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])


def algorithms():
    """Which mode a pass runs in: ``deterministic`` or ``default``."""
    return "deterministic" if torch.are_deterministic_algorithms_enabled() else "default"


def route_flips(topk_a, routes_b, mbs, mask):
    """Amendment 6, reported: per decoder layer, the real tokens whose routed expert SET differs between shape A's forward (``topk_a``,
    one ``[rows x width, k]`` tensor per layer) and shape B's micro-batch forwards at the same parameters (``routes_b``: per
    micro-batch, ``{layer: [m x w, k]}``). The rows are right-padded, so micro-batch ``j``'s row ``r`` and position ``t`` are shape A's
    row ``j * m + r`` at ``t``."""
    rows, width = mask.shape
    per_layer = []
    for i, a in enumerate(topk_a):
        if a is None:
            continue
        k = a.shape[-1]
        a = a.reshape(rows, width, k).sort(-1).values
        flipped = real = 0
        r0 = 0
        for j, mb in enumerate(mbs):
            m, w = mb["input_ids"].shape
            b = routes_b[j][i].reshape(m, w, k).sort(-1).values
            sel = mask[r0:r0 + m, :w].bool().cpu()
            diff = (a[r0:r0 + m, :w] != b).any(-1)
            flipped += int((diff & sel).sum())
            real += int(sel.sum())
            r0 += m
        per_layer.append({"layer": i, "flipped": flipped, "real": real})
    tot_f, tot_n = sum(x["flipped"] for x in per_layer), sum(x["real"] for x in per_layer)
    return {"per_layer": per_layer, "flipped": tot_f, "token_layers": tot_n, "flipped_fraction": tot_f / tot_n if tot_n else None,
            "layers_with_flips": sum(1 for x in per_layer if x["flipped"])}


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
    argv = {k: v for k, v in vars(a).items() if v or not k.startswith("time_moe")}   # off: the record carries no trace of it
    out = {"argv": argv, "host": host_facts(), "versions": versions(), "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dev = a.device
    cuda = dev == "cuda"
    tokens = torch.load(a.tokens, weights_only=True)
    out["tokens_sha256"] = {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in tokens.items()}
    held = None
    if getattr(a, "heldout", None):                             # Amendment 8: the registered held-out rows
        held = torch.load(a.heldout, weights_only=True)
        out["heldout"] = {"tokens_sha256": {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in held.items()},
                          "rows": int(held["input_ids"].shape[0]), "real_tokens": int(held["attention_mask"].sum())}
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

    def fwd_bwd(probe=None):
        model.zero_grad(set_to_none=True)
        sync()
        t = time.time()
        if probe is not None:
            probe.arm()
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        if probe is not None:
            probe.disarm()
        sync()
        tf = time.time()
        loss.backward()
        sync()
        tb = time.time()
        grads = {n: host_copy(p.grad) for n, p in params.items() if p.grad is not None}
        return float(loss), grads, tf - t, tb - tf, time.time() - tb

    mb = getattr(a, "micro_batch", None)
    mbs = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in m.items()} for m in micro_batches(tokens, mb)] if mb else None
    n_valid = sum(m["valid"] for m in mbs) if mbs else None

    def fwd_bwd_b(mode="token", routes=None, probe=None):
        """Shape B's accumulation over the micro-batches (Amendment 6). ``token``: each micro-batch's mean loss weighted by its share of
        the window's valid labels, the gradients accumulating in ``p.grad``. Planted bugs: ``mean`` weights each micro-batch 1 / k (a
        mean of micro-batch means); ``overwrite`` clears ``p.grad`` before each backward, so only the last micro-batch survives.
        ``separate``: the same-shapes reference -- each micro-batch's weighted gradient taken on its own (``p.grad`` cleared before
        each backward) and summed on the host in fp32, so it differs from ``token`` only in how the four backwards accumulate.
        ``routes`` (a list) collects each micro-batch's router top-k from its forward (not the checkpoint recompute)."""
        cur, rhooks = {}, []
        if routes is not None:
            def route(i):
                def h(mod, args, o):
                    if i not in cur:
                        cur[i] = o[2].detach().to("cpu")
                return h
            rhooks = [lay.mlp.gate.register_forward_hook(route(i)) for i, lay in enumerate(model.model.layers)]
        model.zero_grad(set_to_none=True)
        sync()
        t, tf_sum, total, acc = time.time(), 0.0, 0.0, None
        for m in mbs:
            cur.clear()
            if mode in ("overwrite", "separate"):
                model.zero_grad(set_to_none=True)
            t_m = time.time()
            if probe is not None:
                probe.arm()
            loss = model(input_ids=m["input_ids"], attention_mask=m["attention_mask"], labels=m["labels"]).loss
            if probe is not None:
                probe.disarm()
            sync()
            tf_sum += time.time() - t_m
            w = (1.0 / len(mbs)) if mode == "mean" else m["valid"] / n_valid
            total += w * float(loss)
            (loss * w).backward()
            sync()
            if routes is not None:
                routes.append(dict(cur))
            if mode == "separate":
                acc = {} if acc is None else acc
                for n, p in params.items():
                    if p.grad is not None:
                        g = host_copy(p.grad)
                        if n in acc:
                            acc[n].add_(g)
                        else:
                            acc[n] = g
        for h in rhooks:
            h.remove()
        tb = time.time()
        grads = acc if mode == "separate" else {n: host_copy(p.grad) for n, p in params.items() if p.grad is not None}
        return total, grads, tf_sum, tb - t - tf_sum, time.time() - tb

    def rule8_check(routes=None):
        """Rule 8's check at the current parameters, every pass under deterministic algorithms (Amendment 6): the host-sum reference
        (the micro-batches taken apart), the device accumulation (four backwards into ``p.grad``; a pass of its own, not the
        registered step) and a second accumulation for the floor. Returns the record and the reference, which the step's own
        default-mode accumulation is also read against (reported). At most four host copies, with ``g0`` or the step-1 Shape A
        gradient alive."""
        with deterministic():
            mode = algorithms()
            _, ref, *_ = fwd_bwd_b("separate", routes=routes)
            _, acc, *_ = fwd_bwd_b("token")
            _, acc2, *_ = fwd_bwd_b("token")
            rec = {"algorithms": mode, "check": compare_grads(ref, acc), "floor": compare_grads(acc, acc2)}
            del acc, acc2
        rec["restored"] = algorithms()
        return rec, ref

    l0, g0, *_ = fwd_bwd()                                     # step-0 forward: the layer outputs are recorded here
    for h in hooks:
        h.remove()
    l1, g1, *_ = fwd_bwd()
    spread = max(((g1[n] - g0[n]).abs().max() / g0[n].abs().max().clamp_min(1e-30)).item() for n in g0)
    out["in_process"] = {"loss_0": l0, "loss_1": l1, "loss_equal": l0 == l1, "grad_maxrel": spread}
    out["amendment8"] = {"dx": True, "heldout": held is not None}
    train_rows = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    if held is not None:
        held = {k: v.to(dev) for k, v in held.items()}
        out["heldout"]["loss_before"] = heldout_loss(model, held, sync)        # at load: B is zero, so the adapters add nothing
        out["heldout"]["train"] = {"loss_before": heldout_loss(model, train_rows, sync)}   # the step's rows, the same read (reported)
    if mbs:                                                     # Amendment 6: at most four host copies alive at once, as Amendment 5
        del g1
        out["shape_b"] = {"micro_batch": mb, "accumulation": len(mbs), "valid_per_micro_batch": [m["valid"] for m in mbs],
                          "valid_total": n_valid, "widths": [int(m["input_ids"].shape[1]) for m in mbs]}
        out["ga_plants"] = {}
        routes_b = []
        check0, ref0 = rule8_check(routes=routes_b)            # rule 8's check at step 0's parameters (deterministic)
        out["route_flips"] = {**route_flips(topk, routes_b, mbs, tokens["attention_mask"]),
                              "modes": {"shape_a": "default", "shape_b": "deterministic"}}
        del routes_b
        with deterministic():                                   # the plants are check passes too
            for kind in ("mean", "overwrite"):
                lp, gp, *_ = fwd_bwd_b(kind)
                out["ga_plants"][kind] = {"loss": lp, "algorithms": algorithms(), **compare_grads(g0, gp),
                                          "vs_micro_batches": compare_grads(ref0, gp)}
                del gp

    if cuda:
        import bitsandbytes as bnb
        opt, opt_name = bnb.optim.AdamW8bit(list(params.values()), lr=a.lr, weight_decay=a.weight_decay), "bitsandbytes AdamW8bit"
    else:                                                       # the CPU rehearsal: mechanics only, not the registered optimizer
        opt, opt_name = torch.optim.AdamW(list(params.values()), lr=a.lr, weight_decay=a.weight_decay), "torch AdamW (CPU rehearsal)"
    from experts4bit_qlora.engines import offload as off
    is_a = {n: n.endswith("lora_A") for n in params}
    a_names = [n for n in params if is_a[n]]
    b_names = [n for n in params if not is_a[n]]

    def one_step(ref=None, mb_ref=None):
        """One forward-backward and optimizer step at the registered shape, instrumented: its record, gradients and deltas. Under shape
        B the forward-backward is the accumulation, and ``ref`` (shape A's gradient at the same parameters) is compared with it."""
        before = {n: host_copy(p) for n, p in params.items()}
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        os.environ["E4B_OFFLOAD_STATS"] = "1"
        off.reset_offload_stats()
        d1 = diskstats()
        t_step = time.time()
        mode = algorithms()
        probe = DxProbe(model)                                      # Amendment 8: this step's own pass only
        try:
            loss, grads, t_fwd, t_bwd, t_copy = fwd_bwd_b(probe=probe) if mbs else fwd_bwd(probe=probe)
        finally:
            probe.remove()
        sync()
        t_o = time.time()
        opt.step()
        sync()
        t_opt = time.time() - t_o
        rec = {
            "algorithms": mode, "loss": loss, "forward_s": round(t_fwd, 3), "backward_s": round(t_bwd, 3), "optimizer_s": round(t_opt, 3),
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
        if ref is not None:
            rec["ga_check"] = compare_grads(ref, grads)
        if mb_ref is not None:
            rec["ga_mb_check"] = compare_grads(mb_ref, grads)
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
        rec["dx"] = probe.record()
        return rec

    # Amendment 2. Step 0: B starts at zero, so every A's gradient is exactly zero (it flows through B) -- a nonzero one means B was
    # not zero-initialized (VOID, a harness error); B's gradients are nonzero and every B changes. Step 1, after step 0 has moved B:
    # every A's gradient is nonzero and every A changes (a broken A path cannot pass), and B's rules repeat. The frozen base's hashes
    # hold across both steps.
    if mbs:                                                     # Amendment 6: rule 8's check, then the registered step (default mode)
        s0 = one_step(ref=g0, mb_ref=ref0)
        s0["rule8"] = check0
        del g0, ref0
        if held is not None:
            out["heldout"]["loss_after_step0"] = heldout_loss(model, held, sync)
            out["heldout"]["train"]["loss_after_step0"] = heldout_loss(model, train_rows, sync)
        _, ga1, *_ = fwd_bwd()                                  # shape A at step 1's parameters, before its accumulation
        check1, ref1 = rule8_check()
        s1 = one_step(ref=ga1, mb_ref=ref1)
        s1["rule8"] = check1
        del ga1, ref1
    else:
        s0 = one_step()
        if held is not None:
            out["heldout"]["loss_after_step0"] = heldout_loss(model, held, sync)
            out["heldout"]["train"]["loss_after_step0"] = heldout_loss(model, train_rows, sync)
        s1 = one_step()
    if held is not None:
        h = out["heldout"]
        h["loss_after"] = heldout_loss(model, held, sync)                      # after both updates
        h["train"]["loss_after"] = heldout_loss(model, train_rows, sync)
        for rec in (h, h["train"]):                                             # each post-update read against the read before
            rec["rel_change_step0"] = (rec["loss_after_step0"] - rec["loss_before"]) / rec["loss_before"]
            rec["rel_change"] = (rec["loss_after"] - rec["loss_before"]) / rec["loss_before"]
        h["all_finite"] = all(math.isfinite(h[k]) for k in ("loss_before", "loss_after_step0", "loss_after"))
    out["step"], out["step1"] = s0, s1
    out["rules"] = {
        "step0_all_finite": s0["gradients"]["all_finite"], "step0_every_B_nonzero": s0["gradients"]["every_B_nonzero"],
        "step0_every_A_zero": s0["gradients"]["every_A_zero"], "step0_every_B_changed": s0["optimizer_delta"]["every_B_changed"],
        "step1_all_finite": s1["gradients"]["all_finite"], "step1_every_A_nonzero": s1["gradients"]["every_A_nonzero"],
        "step1_every_B_nonzero": s1["gradients"]["every_B_nonzero"], "step1_every_A_changed": s1["optimizer_delta"]["every_A_changed"],
        "step1_every_B_changed": s1["optimizer_delta"]["every_B_changed"],
        # Amendment 8: every layer's output gradient finite and nonzero at both steps, and the held-out reads finite
        "step0_dx_finite_nonzero": s0["dx"]["finite_nonzero"], "step1_dx_finite_nonzero": s1["dx"]["finite_nonzero"],
        **({"heldout_finite": out["heldout"]["all_finite"]} if held is not None else {})}
    after = frozen_hashes(model)
    out["frozen_sha256_after_both_steps_equal"] = after == out["frozen_sha256_after_load"]
    out["frozen_tensors"] = len(after)
    out["frozen_sha256_after_load"] = hashlib.sha256(json.dumps(out["frozen_sha256_after_load"], sort_keys=True).encode()).hexdigest()
    if a.time_moe:                      # REPORTED only, after every registered record has been taken (MoETimer)
        def shape(m_ids, m_labels):
            return {"rows": int(m_ids.shape[0]), "width": int(m_ids.shape[1]), "valid": int((m_labels[:, 1:] != -100).sum())}
        if mbs:                         # the registered shape: B's accumulation, one pair per micro-batch (Amendment 6)
            pass_fn, shapes = (lambda: fwd_bwd_b("token")), [shape(m["input_ids"], m["labels"]) for m in mbs]
        else:
            pass_fn, shapes = fwd_bwd, [shape(ids, labels)]
        out["moe_timing"] = time_moe(model, pass_fn, a.time_moe, cuda, sync, bound=a.time_moe_bound, micro_batches=shapes)
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
    ap.add_argument("--micro-batch", type=int, default=None, help="shape B (Amendment 6): rows per micro-batch, accumulated per step")
    ap.add_argument("--time-moe", type=int, default=0, metavar="N",
                    help="after both steps, one extra forward-backward timing the MoE experts' staging and compute on N evenly "
                         "spaced layers (REPORTED only, under moe_timing; off by default)")
    ap.add_argument("--time-moe-bound", action="store_true",
                    help="with --time-moe: Amendment 7's overhead read (a warm-up, then untimed, timed, timed, untimed)")
    ap.add_argument("--heldout", default=None, metavar="TOKENS.pt",
                    help="Amendment 8: the registered held-out rows (q235_tokens.py --first), read before the update, after step 0 "
                         "and after step 1")
    a = ap.parse_args()
    if a.device == "cpu" and a.enabler == "fast":
        ap.error("the fused training kernels need CUDA; use --enabler reference on the CPU")
    if a.plants and (not a.repeat or a.plant):
        ap.error("--plants rides the fresh-process repeat (--repeat), and replaces --plant")
    if a.micro_batch and (a.repeat or a.plant):
        ap.error("--micro-batch shapes the box's steps; the repeat and the plants run shape A's forward")
    if a.time_moe < 0 or (a.time_moe and a.repeat):
        ap.error("--time-moe N (N >= 1) rides the stepped run, after both registered steps; not --repeat")
    if a.time_moe_bound and not a.time_moe:
        ap.error("--time-moe-bound needs --time-moe")
    if a.heldout and (a.repeat or a.plant):
        ap.error("--heldout rides the stepped run; the repeat and the plants run step 0's forward only")
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
