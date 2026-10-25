#!/usr/bin/env python3
"""p238_op.py -- P238 level 1: one framework's layer-0 experts module, on the model AS TRAINING LOADS IT, timed per arm.

One process per framework per draw (PREREG "Level 1"). The model is loaded through tc1_arm's own path (``build_parser``,
``apply_unsloth_knobs``, ``load_e4b`` / ``load_unsloth``, the fp32 adapter cast, ``apply_matched_init``), so the timed module is the
one the step level trains: the same patches, wrappers, adapters and bytes. The process then:

1. writes its premise table (``p238_bytes``) and the arm-independent facts;
2. for each registered cell (rows per expert x routing), builds one seeded routed input and checks every arm's forward output, dX
   and LoRA gradients against ONE fp32 reference (bitsandbytes' ``dequantize_4bit`` of the held bytes, the LoRA term in fp32);
3. times the arms in a palindrome per rep (20 reps, 50 calls a visit between two CUDA events), switching the arm's knob between
   visits (both knobs are read per call), recording each visit's counter deltas;
4. measures energy per arm over >= 10 s windows (NVML's total-energy counter; ``power.draw.instant`` as the cross-check).

The capture mode is EAGER for every arm: e4b's fused expert forward refuses CUDA-graph capture (``_refuse_under_capture``,
experts4bit_qlora/engines/fast.py), and the PREREG requires one mode across all arms. The probe that shows it is recorded.

``--dry`` runs every step on CPU with a stand-in experts module (no library, no model) so the harness, the generator, the reference
and the record are testable at $0; a dry record can never read (the reducer refuses it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

E, TOPK, HIDDEN, INTER = 256, 8, 2048, 512          # Qwen3.6-35B-A3B (config at revision 995ad96e)
ROWS = (32, 64, 128, 256, 512)                      # rows per expert, the registered sweep
ROUTINGS = ("uniform", "skew")
ZIPF_S, SKEW_SEED, ZERO_EXPERT = 1.1, 238, E - 1    # the skew: Zipf over experts, seeded, expert 255 forced to 0 rows
REPS, CALLS, WARM = 20, 50, 10
TOL = 2e-2                                          # max|err| <= TOL x max|ref|
ENERGY_S = 10.0
ARMS = {"e4b": ("E_F", "E_A"), "unsloth": ("U0", "U1", "UA")}
KNOB = {"E_F": ("GNF4_TRAIN_GEMM", "fused"), "E_A": ("GNF4_TRAIN_GEMM", "auto"),
        "U0": ("UNSLOTH_MOE_NF4_FUSED", "0"), "U1": ("UNSLOTH_MOE_NF4_FUSED", "1"), "UA": ("UNSLOTH_MOE_NF4_FUSED", "auto")}


# -- the routed input (pure, CPU-testable) -----------------------------------------------------------------------------------------
def tokens_for(rows_per_expert):
    """Tokens whose top-8 routing gives ``rows_per_expert`` rows per expert on average: tokens x 8 = rows x 256."""
    t, r = divmod(rows_per_expert * E, TOPK)
    assert r == 0, rows_per_expert
    return t


def routing(rows_per_expert, kind, seed=SKEW_SEED):
    """(top_k_index [T, 8] int64, top_k_weights [T, 8] fp32). ``uniform``: token t takes experts (8t + j) mod 256, so every expert
    gets exactly ``rows_per_expert`` rows and a token's 8 experts are distinct. ``skew``: each token draws 8 distinct experts
    without replacement from Zipf(s) weights over a seeded expert permutation, with expert ``ZERO_EXPERT`` excluded (0 rows).
    Weights: a seeded softmax over 8 logits per token, as a router's top-8 renormalised."""
    import torch
    T = tokens_for(rows_per_expert)
    g = torch.Generator().manual_seed(seed * 1000 + rows_per_expert)
    if kind == "uniform":
        idx = (torch.arange(T).unsqueeze(1) * TOPK + torch.arange(TOPK).unsqueeze(0)) % E
    elif kind == "skew":
        perm = torch.randperm(E, generator=g)
        w = torch.zeros(E, dtype=torch.float64)
        w[perm] = 1.0 / torch.arange(1, E + 1, dtype=torch.float64) ** ZIPF_S
        w[ZERO_EXPERT] = 0.0
        idx = torch.multinomial(w.expand(T, E), TOPK, replacement=False, generator=g)
    else:
        raise ValueError(kind)
    weights = torch.softmax(torch.randn(T, TOPK, generator=g), dim=-1)
    return idx.to(torch.int64), weights.to(torch.float32)


def counts(idx):
    """Rows per expert of a routing."""
    import torch
    return torch.bincount(idx.reshape(-1), minlength=E)


# -- the fp32 reference (pure, CPU-testable) ---------------------------------------------------------------------------------------
def reference(x, idx, w, W_gu, W_d, lora, scaling):
    """fp32 forward of a Qwen MoE experts block with per-expert LoRA on both projections, before the activation (e4b ExpertsLoRA and
    Unsloth's separated LoRA both add the delta to the projection's output): for each (token, slot) routed to expert e,
    h = x W_gu[e]^T + s (x A_gu[e]^T) B_gu[e]^T; gate, up = h.chunk(2); a = silu(gate) * up; y = a W_d[e]^T + s (a A_d[e]^T) B_d[e]^T;
    out[token] += w * y. ``lora`` = (A_gu [E,r,H], B_gu [E,2I,r], A_d [E,r,I], B_d [E,H,r]) or None. Returns out [T, H] fp32."""
    import torch
    import torch.nn.functional as F
    x = x.float()
    out = torch.zeros(x.shape[0], W_d.shape[1], dtype=torch.float32, device=x.device)
    for e in torch.unique(idx).tolist():
        tok, slot = (idx == e).nonzero(as_tuple=True)
        xe = x[tok]
        h = xe @ W_gu[e].float().t()
        if lora is not None:
            h = h + scaling * ((xe @ lora[0][e].float().t()) @ lora[1][e].float().t())
        gate, up = h.chunk(2, dim=-1)
        a = F.silu(gate) * up
        y = a @ W_d[e].float().t()
        if lora is not None:
            y = y + scaling * ((a @ lora[2][e].float().t()) @ lora[3][e].float().t())
        out.index_add_(0, tok, y * w[tok, slot].unsqueeze(1))
    return out


def rel_err(got, ref):
    """max|got - ref| / max|ref| (fp32). A zero reference reads inf unless got is zero too."""
    d = (got.float() - ref.float()).abs().max().item()
    m = ref.float().abs().max().item()
    return 0.0 if d == 0 else (d / m if m > 0 else math.inf)


# -- the visit schedule (pure, CPU-testable) ---------------------------------------------------------------------------------------
def palindrome(arms, rep):
    """A rep's visit order: the arms, then reversed (E_F E_A E_A E_F); odd reps start from the other end."""
    seq = list(arms) + list(reversed(arms))
    return seq if rep % 2 == 0 else list(reversed(seq))


# -- bytes -> fp32 stacks (bitsandbytes' own dequant of the held bytes) ------------------------------------------------------------
def quant_state_e4b(base, which):
    """A bitsandbytes QuantState over e4b's nested buffers (absmax_dq.py:316-319), fp32 output, for ``dequantize_4bit``."""
    import torch
    from bitsandbytes.functional import QuantState
    n, k = getattr(base, f"_{which}_shape")
    s2 = QuantState(absmax=getattr(base, f"{which}_absmax_s"), code=getattr(base, f"{which}_absmax_code"), blocksize=256,
                    dtype=torch.float32)
    return QuantState(absmax=getattr(base, f"{which}_absmax_q").reshape(-1), shape=torch.Size((base.num_experts, n, k)), code=base.code,
                      blocksize=base.blocksize, quant_type="nf4", dtype=torch.float32, offset=getattr(base, f"{which}_absmax_off"),
                      state2=s2)


def dequant_fp32(packed, qs):
    """bitsandbytes' dequantize_4bit of the held bytes, in fp32 (a copy of ``qs`` with dtype fp32 when it is not)."""
    import copy

    import torch
    from bitsandbytes.functional import dequantize_4bit
    if qs.dtype != torch.float32:
        qs = copy.copy(qs)
        qs.dtype = torch.float32
    return dequantize_4bit(packed.reshape(-1, 1) if packed.dim() != 2 else packed, qs).reshape(qs.shape)


# -- the loaded model (tc1_arm's own path) -----------------------------------------------------------------------------------------
TC1_FLAGS = {   # the step arms' registered flags, minus the point's shape and the arm's knob (the PREREG's tables are the authority)
    "common": ["--fam", "qwen3_5", "--model", "Qwen/Qwen3.6-35B-A3B", "--revision", "995ad96eacd98c81ed38be0c5b274b04031597b0",
               "--r", "8", "--alpha", "16", "--lora-init", "matched:3407", "--adapter-dtype", "fp32", "--seq", "2048",
               "--grad-ckpt", "unsloth", "--optim", "adamw_8bit"],
    "e4b": ["--framework", "e4b", "--arm", "fused", "--attn-4bit", "1", "--absmax-dq", "1"],
    "unsloth": ["--framework", "unsloth", "--arm", "unsloth", "--unsloth-loader", "FastModel", "--unsloth-double-quant", "default",
                "--unsloth-targets", "q_proj,k_proj,v_proj,o_proj",
                "--unsloth-target-parameters", "mlp.experts.gate_up_proj,mlp.experts.down_proj"],
}


def load_model(framework, tc1_path):
    """Load exactly as a step arm does, up to its first step. Returns (model, x, a)."""
    import importlib.util

    import torch
    spec = importlib.util.spec_from_file_location("tc1_arm", tc1_path)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    a = t.build_parser().parse_args(TC1_FLAGS["common"] + TC1_FLAGS[framework])
    t.apply_unsloth_knobs(a)
    torch.manual_seed(a.seed)
    model, x = (t.load_e4b if framework == "e4b" else t.load_unsloth)(a)
    tr = t.trainable_census(model)[0]
    for _, p in tr:
        if p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
    matched = t.apply_matched_init(model, a.lora_init, x.get("n_layers"), getattr(model, "config", None))
    return model, x, a, matched


def layer0_experts(model, framework):
    """(name, called, inner): the module at ``layers.0.mlp.experts`` as its parent holds it (the object the training forward CALLS,
    which under PEFT ``target_parameters`` may be a ParamWrapper around the experts module), and the experts module inside it
    (e4b: the ``ExpertsLoRA``'s ``ExpertsNbit`` base; Unsloth: the ``*MoeExperts`` reached through ``base_layer``)."""
    import re
    for name, mod in model.named_modules():
        if not re.search(r"(?:^|\.)layers\.0\.mlp\.experts$", name):
            continue
        inner = mod
        if framework == "e4b":
            if type(mod).__name__ != "ExpertsLoRA":
                raise SystemExit(f"p238_op: {name} is {type(mod).__name__}, not ExpertsLoRA")
            return name, mod, mod.base
        while hasattr(inner, "base_layer") and not type(inner).__name__.endswith("MoeExperts"):
            inner = inner.base_layer
        if not type(inner).__name__.endswith("MoeExperts"):
            raise SystemExit(f"p238_op: no *MoeExperts under {name} ({type(mod).__name__})")
        return name, mod, inner
    raise SystemExit(f"p238_op: no layer-0 experts module for {framework}")


def peft_wrappers(called):
    """{parameter_name: ParamWrapper} for the expert stacks under the called module (PEFT's target_parameters wrappers)."""
    out = {}
    for m in [called] + [m for m in called.modules() if m is not called]:
        pn = getattr(m, "parameter_name", None)
        if pn in ("gate_up_proj", "down_proj") and hasattr(m, "lora_A"):
            out.setdefault(pn, m)
    return out


# -- one cell: parity with a non-zero B, then the timed palindrome ------------------------------------------------------------------
def diff(c0, c1):
    """c1 - c0 over nested dicts of ints; non-numeric leaves are kept from c1 only when they changed."""
    out = {}
    for k, v in (c1 or {}).items():
        b = (c0 or {}).get(k)
        if isinstance(v, dict):
            d = diff(b if isinstance(b, dict) else {}, v)
            if d:
                out[k] = d
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            if v != (b or 0):
                out[k] = v - (b or 0)
        elif v != b:
            out[k] = v
    return out


def run_cell(ctx, rows_per_expert, kind):
    """One (rows, routing) cell: the routed input, the parity check against the fp32 reference with a seeded NON-ZERO B (the
    applied-delta layout check), B restored to exactly zero, the warm-up, then REPS palindrome reps of CALLS calls a visit."""
    x, idx, w, gup = ctx.inputs(rows_per_expert, kind)
    n = counts(idx)
    cell = {"rows": rows_per_expert, "routing": kind, "tokens": int(idx.shape[0]), "rows_min": int(n.min()), "rows_max": int(n.max()),
            "experts_empty": int((n == 0).sum()), "parity": {}, "visits": []}
    ctx.set_lora_b(nonzero=True)
    ref = ctx.reference(x, idx, w, gup)
    for arm in ctx.arms:
        ctx.set_arm(arm)
        c0 = ctx.counters()
        got = ctx.call(x, idx, w, gup)
        errs = {k: rel_err(got[k], ref[k]) for k in ref}
        cell["parity"][arm] = {"errs": errs, "ok": all(v <= TOL for v in errs.values()), "counters": diff(c0, ctx.counters()),
                               "bits_equal_self": bool(all(ctx.call(x, idx, w, gup)[k].equal(got[k]) for k in got))}
    cell["b_restored_zero"] = ctx.set_lora_b(nonzero=False)
    for arm in ctx.arms:
        ctx.set_arm(arm)
        for _ in range(WARM):
            ctx.call(x, idx, w, gup)
    for rep in range(REPS):
        for pos, arm in enumerate(palindrome(ctx.arms, rep)):
            ctx.set_arm(arm)
            c0 = ctx.counters()
            dev_ms, wall_ms = ctx.timed(lambda: ctx.call(x, idx, w, gup), CALLS)
            cell["visits"].append({"rep": rep, "pos": pos, "arm": arm, "device_ms_per_call": dev_ms, "wall_ms_per_call": wall_ms,
                                   "counters": diff(c0, ctx.counters())})
    return cell


class DryCtx:
    """CPU stand-in: the reference math as the 'module' (bf16 storage, fp32 compute), small dims, perf_counter timing. Exercises
    every step of the harness and the record; never a reading."""

    def __init__(self):
        import torch
        global E, HIDDEN, INTER, ZERO_EXPERT
        E, HIDDEN, INTER, ZERO_EXPERT = 16, 64, 32, 15
        g = torch.Generator().manual_seed(0)
        self.arms, self.r, self.scaling = ["D_A", "D_B"], 8, 2.0
        self.W_gu = (torch.randn(E, 2 * INTER, HIDDEN, generator=g) * 0.02).to(torch.bfloat16)
        self.W_d = (torch.randn(E, HIDDEN, INTER, generator=g) * 0.02).to(torch.bfloat16)
        self.lora = [torch.randn(E, self.r, HIDDEN, generator=g) * 0.02, torch.zeros(E, 2 * INTER, self.r),
                     torch.randn(E, self.r, INTER, generator=g) * 0.02, torch.zeros(E, HIDDEN, self.r)]
        for t in self.lora:
            t.requires_grad_(True)
        self.calls = 0

    def facts(self):
        return {"dry": True, "dims": {"E": E, "H": HIDDEN, "I": INTER, "r": self.r}}

    def inputs(self, rows_per_expert, kind):
        import torch
        idx, w = routing(rows_per_expert, kind)
        g = torch.Generator().manual_seed(rows_per_expert)
        x = torch.randn(idx.shape[0], HIDDEN, generator=g).to(torch.bfloat16)
        gup = torch.randn(idx.shape[0], HIDDEN, generator=g)
        return x, idx, w, gup

    def set_arm(self, arm):
        self.arm = arm

    def counters(self):
        return {"calls": self.calls}

    def set_lora_b(self, nonzero):
        import torch
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for i in (1, 3):
                self.lora[i].copy_(torch.randn(self.lora[i].shape, generator=g) * 0.02 if nonzero else torch.zeros_like(self.lora[i]))
        return all(bool((self.lora[i] == 0).all()) for i in (1, 3))

    def _run(self, x, idx, w, gup, fp32):
        import torch
        xl = (x.float() if fp32 else x).detach().requires_grad_(True)
        lora = [t.detach().clone().requires_grad_(True) for t in self.lora]
        out = reference(xl, idx, w, self.W_gu, self.W_d, lora, self.scaling)
        if not fp32:
            out = out.to(torch.bfloat16)
        out.backward(gup.to(out.dtype))
        return {"out": out.detach(), "dx": xl.grad, "dA_gu": lora[0].grad, "dB_gu": lora[1].grad, "dA_d": lora[2].grad,
                "dB_d": lora[3].grad}

    def reference(self, x, idx, w, gup):
        return self._run(x, idx, w, gup, fp32=True)

    def call(self, x, idx, w, gup):
        self.calls += 1
        return self._run(x, idx, w, gup, fp32=False)

    def timed(self, fn, n):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        ms = (time.perf_counter() - t0) * 1e3 / n
        return ms, ms

    def energy(self, rows_per_expert):
        return {}


class GpuCtx:
    """The box: the model loaded through tc1_arm's own path, its layer-0 experts module, the libraries' own counters."""

    def __init__(self, framework, tc1_path):
        import torch
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "..", "common"))
        import p238_bytes
        self.fw, self.arms, self.dev = framework, list(ARMS[framework]), torch.device("cuda")
        self.counter = None
        if framework == "unsloth":
            from unsloth_moe_counter import UnslothMoeCounter
            self.counter = UnslothMoeCounter().install()            # before the load, as the step arms (post_import)
        self.model, self.x, self.a, self.matched = load_model(framework, tc1_path)
        self.name, self.mod, self.inner = layer0_experts(self.model, framework)
        if self.counter is not None:
            self.counter.patch_loaded([m for _, m in self.model.named_modules() if type(m).__name__.endswith("MoeExperts")])
        self.table = (p238_bytes.e4b_table if framework == "e4b" else p238_bytes.unsloth_table)(self.model)
        self.W_gu, self.W_d = self._base_fp32("gate_up"), self._base_fp32("down")
        self.scaling = float(self._lora_handles()[1])

    # the bytes and the adapters, each in its own library's layout
    def _base_fp32(self, which):
        if self.fw == "e4b":
            base = self.inner
            return dequant_fp32(getattr(base, f"{which}_proj"), quant_state_e4b(base, which))
        from p238_bytes import _base_param
        p = _base_param(getattr(self.inner, f"{which}_proj"))
        return dequant_fp32(p.data, p.quant_state)

    def _lora_handles(self):
        """((A_gu, B_gu, A_d, B_d) as the LIBRARY's parameters, scaling)."""
        if self.fw == "e4b":
            m = self.mod
            return (m.gate_up_lora_A, m.gate_up_lora_B, m.down_lora_A, m.down_lora_B), m.scaling
        pw = peft_wrappers(self.mod)
        if set(pw) != {"gate_up_proj", "down_proj"}:
            raise SystemExit(f"p238_op: PEFT wrappers under {self.name}: {sorted(pw)} (want gate_up_proj and down_proj)")
        wr = [pw["gate_up_proj"], pw["down_proj"]]
        k = [list(x.lora_A.keys())[0] for x in wr]
        return ((wr[0].lora_A[k[0]].weight, wr[0].lora_B[k[0]].weight, wr[1].lora_A[k[1]].weight, wr[1].lora_B[k[1]].weight),
                wr[0].scaling[k[0]])

    def _canon(self, i, t):
        """A library tensor (param or grad) -> the per-expert canonical layout [E, r, in] (A) / [E, out, r] (B). PEFT: A rows are
        expert-major (E, r, in); B columns are rank-major, read by get_delta_weight as [out, r, E] (peft layer.py:2552-2554)."""
        if self.fw == "e4b":
            return t
        if i in (0, 2):
            return t.reshape(E, -1, t.shape[-1])
        return t.reshape(t.shape[0], -1, E).permute(2, 0, 1)

    def set_lora_b(self, nonzero):
        import torch
        (_, B_gu, _, B_d), _ = self._lora_handles()
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for i, B in ((1, B_gu), (3, B_d)):
                if not nonzero:
                    B.zero_()
                    continue
                canon = self._canon(i, B)
                vals = (torch.randn(canon.shape, generator=g) * 0.02).to(B.dtype).to(B.device)
                canon.copy_(vals)                                    # a view in the library's own storage: written in ITS layout
        return all(bool((B == 0).all()) for B in (B_gu, B_d))

    def facts(self):
        import torch
        f = {"dry": False, "experts_module": self.name, "called_type": type(self.mod).__name__,
             "inner_type": type(self.inner).__name__, "matched_init": self.matched,
             "device": torch.cuda.get_device_name(0), "cap": list(torch.cuda.get_device_capability()),
             "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "premise_table": self.table,
             "loader_x": {k: self.x.get(k) for k in ("loader_used", "unsloth_double_quant", "absmax_dq", "n_patched", "reason",
                                                      "moe_backend_selected", "unsloth_target_parameters")}}
        if self.fw == "e4b":
            from experts4bit_qlora.engines.fast import FAST_TRAIN_STATS
            f["fast_train"] = json.loads(json.dumps(FAST_TRAIN_STATS, default=str))
        else:
            f["counter_served"] = self.counter.snapshot().get("served")
            f["counter_expert_forwards"] = self.counter.snapshot().get("expert_forwards")
        f["capture_probe"] = self._capture_probe()
        return f

    def _capture_probe(self):
        import torch
        x, idx, w, gup = self.inputs(ROWS[0], "uniform")
        s = torch.cuda.Stream()
        try:
            with torch.cuda.stream(s):
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    self.mod(x, idx, w)
            return {"capturable": True}
        except Exception as e:  # noqa: BLE001 -- recorded: one arm that cannot capture puts every arm in eager
            torch.cuda.synchronize()
            return {"capturable": False, "why": f"{type(e).__name__}: {str(e)[:200]}"}

    def inputs(self, rows_per_expert, kind):
        import torch
        idx, w = routing(rows_per_expert, kind)
        g = torch.Generator().manual_seed(rows_per_expert)
        x = torch.randn(idx.shape[0], HIDDEN, generator=g).to(torch.bfloat16)
        gup = torch.randn(idx.shape[0], HIDDEN, generator=g).to(torch.bfloat16)
        return x.to(self.dev), idx.to(self.dev), w.to(torch.bfloat16).to(self.dev), gup.to(self.dev)

    def set_arm(self, arm):
        k, v = KNOB[arm]
        os.environ[k] = v

    def counters(self):
        if self.fw == "e4b":
            import nf4_qlora
            import nf4_route
            return {"dgrad": json.loads(json.dumps(nf4_qlora.DGRAD_STATS)), "route": dict(nf4_route.ROUTE_STATS)}
        s = self.counter.snapshot()
        return {"counts": s["counts"], "nf4": s["library_delta"].get("nf4"), "generic": s["library_delta"].get("generic"),
                "dispatch": s["dispatch"]}

    def call(self, x, idx, w, gup):
        (A_gu, B_gu, A_d, B_d), _ = self._lora_handles()
        for p in (A_gu, B_gu, A_d, B_d):
            p.grad = None
        xl = x.detach().requires_grad_(True)
        out = self.mod(xl, idx, w)
        out.backward(gup)
        return {"out": out.detach(), "dx": xl.grad, "dA_gu": self._canon(0, A_gu.grad), "dB_gu": self._canon(1, B_gu.grad),
                "dA_d": self._canon(2, A_d.grad), "dB_d": self._canon(3, B_d.grad)}

    def reference(self, x, idx, w, gup):
        import torch
        (A_gu, B_gu, A_d, B_d), _ = self._lora_handles()
        lora = [self._canon(i, t).detach().float().clone().requires_grad_(True) for i, t in enumerate((A_gu, B_gu, A_d, B_d))]
        xl = x.detach().float().requires_grad_(True)
        out = reference(xl, idx, w.float(), self.W_gu, self.W_d, lora, self.scaling)
        out.backward(gup.float())
        torch.cuda.synchronize()
        return {"out": out.detach(), "dx": xl.grad, "dA_gu": lora[0].grad, "dB_gu": lora[1].grad, "dA_d": lora[2].grad,
                "dB_d": lora[3].grad}

    def timed(self, fn, n):
        import torch
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        e0.record()
        for _ in range(n):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n, (time.perf_counter() - t0) * 1e3 / n

    def energy(self, rows_per_expert):
        """Per arm at the uniform cell: NVML's total-energy counter over >= ENERGY_S of back-to-back calls (J/call), and
        power.draw.instant sampled every 0.25 s by nvidia-smi over the same window (the cross-check, read with >= 40 samples)."""
        import subprocess

        import pynvml
        import torch
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        x, idx, w, gup = self.inputs(rows_per_expert, "uniform")
        out = {"rows": rows_per_expert, "arms": {}}
        for arm in self.arms:
            self.set_arm(arm)
            try:
                mj0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            except pynvml.NVMLError as e:
                mj0, out["counter_unsupported"] = None, str(e)
            smi = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw.instant", "--format=csv,noheader,nounits", "-lms", "250"],
                                   stdout=subprocess.PIPE, text=True)
            torch.cuda.synchronize()
            t0, n = time.perf_counter(), 0
            while time.perf_counter() - t0 < ENERGY_S or n < CALLS:
                self.call(x, idx, w, gup)
                n += 1
                if n % 10 == 0:
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            mj1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) if mj0 is not None else None
            smi.terminate()
            samples = [float(v) for v in smi.communicate()[0].split() if v.replace(".", "", 1).isdigit()]
            out["arms"][arm] = {"calls": n, "seconds": secs, "j_per_call": (mj1 - mj0) / 1e3 / n if mj0 is not None else None,
                                "sampled_w": samples, "sampled_n": len(samples)}
        return out


# -- the record ----------------------------------------------------------------------------------------------------------------------
def write_json(path, obj):
    data = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()
    with open(path, "wb") as f:                     # LF bytes on every platform
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=sorted(ARMS), required=True)
    ap.add_argument("--draw", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tc1", default=os.path.join(os.path.dirname(__file__), "..", "tc1", "tc1_arm.py"))
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    ap.add_argument("--dry", action="store_true", help="CPU, stand-in module, no library: a record that never reads")
    a = ap.parse_args(argv)
    rows = [int(r) for r in a.rows.split(",")]
    if any(r not in ROWS for r in rows):
        raise SystemExit(f"p238_op: --rows {a.rows} is not inside the registered {ROWS}")
    rec = {"p238_op": 1, "framework": a.framework, "draw": a.draw, "dry": bool(a.dry), "arms": list(ARMS[a.framework]),
           "rows": rows, "routings": list(ROUTINGS), "reps": REPS, "calls": CALLS, "tol": TOL, "capture": "eager",
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": []}
    ctx = DryCtx() if a.dry else GpuCtx(a.framework, a.tc1)
    rec["facts"] = ctx.facts()
    for r in rows:
        for kind in ROUTINGS:
            rec["cells"].append(run_cell(ctx, r, kind))
    rec["energy"] = [ctx.energy(r) for r in rows] if not a.dry else []
    rec["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("P238_OP " + write_json(a.out, rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
