#!/usr/bin/env python3
"""p238_hook.py -- P238's TC1_ARM_HOOK module (``TC1_ARM_HOOK=p238_hook:make`` with ``TC1_ARM_HOOK_SHA256`` its staged sha256;
tc1_arm.py's load_arm_hook calls the factory with no arguments and refuses any other bytes).

OBSERVE ONLY. Every wrapper it installs returns the wrapped function's own output, unchanged (the tests show it per wrapper), and
nothing here sets a knob, a dispatch or a route. The phases tc1_arm calls:
- ``post_import(args)``: Unsloth arms install ``UnslothMoeCounter`` (bench/common/unsloth_moe_counter.py); e4b arms wrap
  ``nf4_route.train_gemm_route`` (read by grouped-nf4-gemm at call time, nf4_qlora.py:247-248) to log (n_groups, route) per call;
- ``post_load(args, model)``: after the load and the framework's patching, before the first forward. Unsloth arms call
  ``patch_loaded`` with the loaded ``*MoeExperts`` modules (the zoo's compile-cache copy of moe_utils can appear during the load).
  Every arm records the premise table (``p238_bytes``), the adapter census per (layer, projection), the absmax facts, and puts a
  forward PRE-hook (it returns None, so the call is unchanged) on each called experts module, counting forwards and those under grad;
- ``step_edge(i)``: a snapshot of every counter, and NVML's total-energy counter (mJ; the PREREG's primary step energy, over
  edges 10..60), before optimizer step i and once after the last step;
- ``record()``: all of it, for the receipt.

The non-zero-B layout check is NOT here: it changes B and runs extra forwards, so it runs in the op processes (p238_op.py), which
load the model the same way.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERTS_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")


def _sha_t(t):
    import torch
    t = t.detach().float().cpu().contiguous()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


class P238Hook:
    def __init__(self):
        self.fw, self.micro_batches = None, 1                       # read from tc1_arm's args at post_import
        self.counter = None
        self._nvml = None
        self.route_log = {}                  # "n_groups:route" -> calls
        self.fwd = {"calls": 0, "under_grad": 0}
        self.edges = []
        self.facts = {}
        self._saved = []

    # -- post_import ----------------------------------------------------------------------------------------------------------
    def post_import(self, args):
        self.fw = getattr(args, "framework", None)
        self.micro_batches = int(getattr(args, "accum", 1) or 1)   # tc1_arm: one optimizer step = accum micro-batches
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:  # noqa: BLE001 -- recorded: the energy counter is then unread, and the sampled cross-check stands alone
            self.facts["nvml"] = f"unavailable: {type(e).__name__}"
        if self.fw == "unsloth":
            sys.path.insert(0, os.path.join(HERE, "..", "common"))
            from unsloth_moe_counter import UnslothMoeCounter
            self.counter = UnslothMoeCounter().install()
        elif self.fw == "e4b":
            import nf4_route
            fn, log = nf4_route.train_gemm_route, self.route_log

            def train_gemm_route(dev=None, n_groups=None):
                out = fn(dev, n_groups)
                key = f"{n_groups}:{out}"
                log[key] = log.get(key, 0) + 1
                return out

            train_gemm_route.__wrapped__ = fn
            self._saved.append((nf4_route, "train_gemm_route", fn))
            nf4_route.train_gemm_route = train_gemm_route

    # -- post_load ------------------------------------------------------------------------------------------------------------
    def post_load(self, args, model):
        sys.path.insert(0, HERE)
        import p238_bytes
        called = [(n, m) for n, m in model.named_modules() if EXPERTS_RE.search(n)]
        if self.fw == "unsloth":
            inner = [m for _, m in model.named_modules() if type(m).__name__.endswith("MoeExperts")]
            self.counter.patch_loaded(inner)
            self.facts["premise_table"] = p238_bytes.unsloth_table(model)
        else:
            self.facts["premise_table"] = p238_bytes.e4b_table(model)
            from experts4bit_qlora.absmax_dq import is_absmax_compressed
            from experts4bit_qlora.engines.fast import FAST_TRAIN_STATS
            stacks = [m.base for _, m in called if hasattr(m, "base")] or [m for _, m in called]
            self.facts["absmax_compressed"] = sum(bool(is_absmax_compressed(m)) for m in stacks)
            self.facts["fast_train_absmax_dq"] = json.loads(json.dumps(FAST_TRAIN_STATS.get("absmax_dq"), default=str))
            self.facts["fast_train_patched"] = FAST_TRAIN_STATS.get("patched")
            self.facts["fast_train_skipped"] = dict(FAST_TRAIN_STATS.get("skipped") or {})
        self.facts["called_experts"] = len(called)
        self.facts["adapters"] = adapter_census(model, self.fw)
        fwd = self.fwd

        def pre_hook(module, inputs):
            import torch
            fwd["calls"] += 1
            if torch.is_grad_enabled():
                fwd["under_grad"] += 1
            return None                      # observe: the inputs pass unchanged

        self._handles = [m.register_forward_pre_hook(pre_hook) for _, m in called]

    # -- step_edge / record ---------------------------------------------------------------------------------------------------
    def _energy_mj(self):
        if self._nvml is None:
            return None
        try:
            import pynvml
            return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self._nvml))
        except Exception:  # noqa: BLE001 -- an unsupported counter reads None; the reducer then uses the sampled cross-check, stated
            return None

    def snapshot(self):
        import time
        s = {"fwd": dict(self.fwd), "t": time.time(), "energy_mj": self._energy_mj()}
        if self.counter is not None:
            s["unsloth"] = self.counter.snapshot()
        if self.fw == "e4b":
            import nf4_qlora
            import nf4_route
            s["dgrad"] = json.loads(json.dumps(nf4_qlora.DGRAD_STATS))
            s["route"] = dict(nf4_route.ROUTE_STATS)
            s["route_log"] = dict(self.route_log)
        return s

    def step_edge(self, i):
        if self.counter is not None:
            self.counter.mark_step(i)        # passes after this edge belong to step i (observation: a bookkeeping index)
        self.edges.append({"i": int(i), **self.snapshot()})

    def record(self):
        rec = {"p238_hook": 1, "framework": self.fw, "facts": self.facts, "edges": self.edges, "micro_batches": self.micro_batches}
        if self.counter is not None:         # the backward BY STRUCTURE over the marked steps (edges 0..N close N steps)
            rec["structural"] = self.counter.structural(self.micro_batches, max(len(self.edges) - 1, 0))
        return rec


def adapter_census(model, fw):
    """Per (layer, projection): the trainable count, the per-expert A/B shapes, the scaling, and sha256 of A in the canonical
    per-expert layout [E, r, in] (fp32). e4b: ExpertsLoRA's own tensors (lora.py:514-517). Unsloth: PEFT's ParamWrapper, A rows
    expert-major (peft layer.py:2552). Attention adapters are counted per layer too."""
    out = {}
    for name, m in model.named_modules():
        mm = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if mm is None:
            continue
        L = int(mm.group(1))
        if fw == "e4b" and type(m).__name__ == "ExpertsLoRA":
            E = m.gate_up_lora_A.shape[0]
            for which in ("gate_up", "down"):
                A, B = getattr(m, f"{which}_lora_A"), getattr(m, f"{which}_lora_B")
                out[f"{L}:{which}_proj"] = {"trainable": int(A.numel() + B.numel()), "A": list(A.shape), "B": list(B.shape),
                                            "experts": E, "scaling": float(m.scaling), "A_sha": _sha_t(A)}
        pn = getattr(m, "parameter_name", None)
        if fw == "unsloth" and pn in ("gate_up_proj", "down_proj") and hasattr(m, "lora_A"):
            k = list(m.lora_A.keys())[0]
            A, B, E = m.lora_A[k].weight, m.lora_B[k].weight, int(getattr(m, "num_experts", 1))
            Ac = A.reshape(E, -1, A.shape[-1])
            out[f"{L}:{pn}"] = {"trainable": int(A.numel() + B.numel()), "A": list(Ac.shape),
                                "B": [E, B.shape[0], B.shape[1] // E], "experts": E, "scaling": float(m.scaling[k]), "A_sha": _sha_t(Ac)}
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if name.endswith(f".{proj}"):
                n = sum(p.numel() for pn_, p in m.named_parameters() if "lora_" in pn_ and p.requires_grad)
                if n:
                    out[f"{L}:{proj}"] = {"trainable": int(n)}
    return out


def make():
    """The factory tc1_arm calls, with no arguments (TC1_ARM_HOOK=p238_hook:make)."""
    return P238Hook()
