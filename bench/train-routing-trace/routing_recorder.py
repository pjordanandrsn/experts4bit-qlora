"""Record a MoE model's router decisions during TRAINING: which experts each layer picks, per micro-batch, per pass.

A training step routes every micro-batch twice under gradient checkpointing: the forward, and the recompute inside
backward. The dgrad of a layer's experts runs on the recompute's routing, so the recompute set is also the dgrad set
(by construction, not separately hooked). This module records the forward and the recompute, and checks that they agree.

Counts and expert ids only. No timings are recorded: the trace answers "which rows would a residency tier have to
hold", not "how fast".

Usage (see run_capture.py for the training-harness wrapper)::

    rec = RoutingRecorder()
    rec.attach(model)            # hooks every `*.layers.<i>.mlp.gate` router
    with rec.backward_phase():   # around loss.backward(): router calls inside are the recompute
        loss.backward()
    rec.end_step()               # after optimizer.step()
    rec.save("trace.npz", meta)
"""
from __future__ import annotations

import contextlib
import json
import re

import numpy as np
import torch

#: a router module's qualified name: `...layers.<i>.mlp.gate` (Qwen3-MoE, OLMoE, Mixtral's `block_sparse_moe.gate`)
ROUTER_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:mlp|block_sparse_moe)\.gate$")

FWD, RECOMPUTE = "fwd", "recompute"


def router_indices(output):
    """The top-k expert ids a router returned. transformers 5 Qwen3-MoE's `Qwen3MoeTopKRouter` returns
    `(router_logits, router_scores, router_indices)`; anything else is refused rather than guessed."""
    if isinstance(output, tuple) and len(output) == 3 and torch.is_tensor(output[2]) and not output[2].is_floating_point():
        return output[2]
    raise TypeError("router output is not (logits, scores, int indices): refusing to guess the expert ids "
                    f"(got {type(output).__name__} of length {len(output) if isinstance(output, tuple) else '-'})")


class RoutingRecorder:
    """Collects per-(step, micro-batch, pass, layer) top-k expert ids from forward hooks on the routers."""

    def __init__(self, n_experts: int | None = None):
        self.n_experts = n_experts
        self.step = 0                 # optimizer steps completed
        self.mb = -1                  # micro-batch index within the current step (set by the layer-0 forward)
        self.phase = FWD
        self.records: dict[tuple[int, int, str, int], np.ndarray] = {}
        self.layers: list[int] = []
        self._handles = []

    # -- wiring -------------------------------------------------------------------------------------------------
    def attach(self, model) -> int:
        """Hook every router by qualified name. Returns the number hooked; refuses a model with none."""
        for name, mod in model.named_modules():
            m = ROUTER_NAME.search(name)
            if m:
                layer = int(m.group(1))
                self.layers.append(layer)
                self._handles.append(mod.register_forward_hook(self._hook(layer)))
        if not self._handles:
            raise RuntimeError("no router modules matched `layers.<i>.mlp.gate`: nothing to record")
        self.layers.sort()
        return len(self._handles)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @contextlib.contextmanager
    def backward_phase(self):
        """Router calls inside this block are the checkpoint recompute of the current micro-batch."""
        prev, self.phase = self.phase, RECOMPUTE
        try:
            yield
        finally:
            self.phase = prev

    def end_step(self):
        """Call after the optimizer step: the next forward starts micro-batch 0 of the next step."""
        self.step += 1
        self.mb = -1

    def _hook(self, layer: int):
        def hook(_module, _inputs, output):
            if self.phase == FWD and not torch.is_grad_enabled():
                return                                # an eval forward (no_grad): not part of a training step
            if self.phase == FWD and layer == self.layers[0]:
                self.mb += 1                          # the first router of a training forward opens a micro-batch
            ids = router_indices(output).detach()
            idx = ids.reshape(-1, ids.shape[-1])
            key = (self.step, self.mb, self.phase, layer)
            arr = idx.to("cpu", torch.int64).numpy().astype(np.uint8 if (self.n_experts or 256) <= 256 else np.int16)
            # a layer recomputed in chunks (or a checkpoint segment) appends rather than overwrites
            self.records[key] = arr if key not in self.records else np.concatenate([self.records[key], arr])
        return hook

    # -- checks -------------------------------------------------------------------------------------------------
    def check_fwd_equals_recompute(self) -> dict[str, list[tuple[int, int, int]]]:
        """Compare each (step, micro-batch, layer)'s recompute ids with its forward ids.

        `differ`: the recompute exists and disagrees -- a VOID result for the census (or itself the finding).
        `no_recompute`: a forward with no backward after it (a probe, a forward-only pass) -- excluded from replay.
        Both empty means every training micro-batch routed identically in its forward and its recompute."""
        differ, missing = [], []
        for (step, mb, phase, layer), fwd in self.records.items():
            if phase != FWD:
                continue
            rec = self.records.get((step, mb, RECOMPUTE, layer))
            if rec is None:
                missing.append((step, mb, layer))
            elif rec.shape != fwd.shape or not np.array_equal(rec, fwd):
                differ.append((step, mb, layer))
        return {"differ": sorted(differ), "no_recompute": sorted(missing)}

    def check_every_layer_fired(self) -> list[tuple[int, int, str]]:
        """Every (step, micro-batch, pass) that is missing a layer: the hook did not see the whole model."""
        seen: dict[tuple[int, int, str], set[int]] = {}
        for (step, mb, phase, layer) in self.records:
            seen.setdefault((step, mb, phase), set()).add(layer)
        want = set(self.layers)
        return sorted(k for k, v in seen.items() if v != want)

    # -- output -------------------------------------------------------------------------------------------------
    def save(self, path: str, meta: dict) -> dict:
        """Write the trace as one .npz (one array per key) plus the meta, and return the summary written."""
        arrays = {f"s{s}_mb{m}_{p}_L{layer}": a for (s, m, p, layer), a in sorted(self.records.items())}
        summary = {
            "steps": self.step, "layers": self.layers, "records": len(self.records),
            "fwd_recompute_mismatches": self.check_fwd_equals_recompute(),
            "missing_layers": self.check_every_layer_fired(),
        }
        np.savez_compressed(path, **arrays)
        with open(path + ".meta.json", "w") as f:
            json.dump({"meta": meta, "summary": summary}, f, indent=1, default=list)
        return summary
