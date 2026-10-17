"""Assumption A1 (README.md, option (b)): does autograd's backward run layer-major across micro-batches when every
(layer, micro-batch) is its own non-reentrant checkpoint? CPU only; a toy model with e4b's single-slot staging rule.

The staging rule mirrors experts4bit_qlora/engines/offload.py:
- a forward pre-hook stages the layer's experts: idempotent if that layer is already resident, otherwise it evicts the resident
  layer and counts one copy (``stage()``);
- a forward post-hook evicts the layer, except inside a backward (``_evict_post_hook``: ``if _in_backward(): return``), where
  ``_in_backward()`` is ``torch._C._current_graph_task_id() != -1``.

Two schedules, the same micro-batches, the same loss weights:
- as-is (Shape B as registered): per micro-batch, a forward through every layer, then ``(loss * w).backward()``;
- layer-major: for each layer, every micro-batch through it; the loss per micro-batch; one backward of the weighted sum.

Reported: the copies per pass (prediction: as-is M(2n-1), layer-major 2n-1), the order of the recompute calls in the backward,
and the gradients' agreement (same math, so equal up to summation order).

    python a1_layer_major.py [--layers 94 --micro-batches 4]
"""
import argparse
import json

import torch
from torch.utils.checkpoint import checkpoint


class Slot:
    """e4b's single resident slot: at most one layer's experts staged at a time."""

    def __init__(self):
        self.resident, self.copies, self.log, self.hold = None, 0, [], False

    def stage(self, layer):
        self.log.append(("stage", layer, torch._C._current_graph_task_id() != -1))
        if self.resident == layer:
            return                      # idempotent: already staged
        self.resident = layer           # evicts the previous layer
        self.copies += 1

    def evict_after_forward(self, layer):
        if self.hold or torch._C._current_graph_task_id() != -1:
            return                      # inside a backward (or a held layer-major forward): the next stage() evicts instead
        if self.resident == layer:
            self.resident = None


class Layer(torch.nn.Module):
    def __init__(self, i, h, slot):
        super().__init__()
        self.i, self.slot = i, slot
        self.frozen = torch.nn.Linear(h, h, bias=False).requires_grad_(False)     # the offloaded experts
        self.lora = torch.nn.Linear(h, h, bias=False)                              # a trained adapter
        self.register_forward_pre_hook(lambda m, a: self.slot.stage(self.i))
        self.register_forward_hook(lambda m, a, o: self.slot.evict_after_forward(self.i))

    def forward(self, x):
        return x + torch.tanh(self.frozen(x) + self.lora(x))


def build(n, h, seed=0):
    torch.manual_seed(seed)
    slot = Slot()
    layers = torch.nn.ModuleList(Layer(i, h, slot) for i in range(n))
    head = torch.nn.Linear(h, 1, bias=False)
    return slot, layers, head


def micro_batches(m, h, seed=1):
    g = torch.Generator().manual_seed(seed)
    lens = [3, 5, 9, 4][:m] + [6] * max(0, m - 4)       # unequal, as Shape B's are
    return [torch.randn(t, h, generator=g) for t in lens]


def as_is(n, h, mbs, ws):
    slot, layers, head = build(n, h)
    for x, w in zip(mbs, ws):
        for lay in layers:
            x = checkpoint(lay, x, use_reentrant=False)
        (head(x).pow(2).mean() * w).backward()
    return slot, [p.grad.clone() for lay in layers for p in lay.lora.parameters()]


def layer_major(n, h, mbs, ws):
    slot, layers, head = build(n, h)
    xs = list(mbs)
    for lay in layers:
        xs = [checkpoint(lay, x, use_reentrant=False) for x in xs]
    sum(head(x).pow(2).mean() * w for x, w in zip(xs, ws)).backward()
    return slot, [p.grad.clone() for lay in layers for p in lay.lora.parameters()]


def layer_major_held(n, h, mbs, ws):
    """Layer-major with the driver holding each layer resident across its micro-batches in the forward (no evict between them):
    the change (b) needs beyond A1, because e4b's forward post-hook evicts after every call outside a backward."""
    slot, layers, head = build(n, h)
    xs = list(mbs)
    slot.hold = True
    for lay in layers:
        xs = [checkpoint(lay, x, use_reentrant=False) for x in xs]
    slot.hold = False
    sum(head(x).pow(2).mean() * w for x, w in zip(xs, ws)).backward()
    return slot, [p.grad.clone() for lay in layers for p in lay.lora.parameters()]


def summarize(slot, n, m):
    back = [i for kind, i, in_bwd in slot.log if kind == "stage" and in_bwd]
    runs = []                               # the backward's stage() calls, compressed into runs of one layer
    for i in back:
        if runs and runs[-1][0] == i:
            runs[-1][1] += 1
        else:
            runs.append([i, 1])
    layer_major_order = [r[0] for r in runs] == list(range(n - 1, -1, -1)) and all(r[1] == m for r in runs)
    return {"copies": slot.copies, "backward_stage_calls": len(back), "backward_runs_first": runs[:6],
            "backward_is_layer_major": layer_major_order}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=94)
    ap.add_argument("--micro-batches", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=16)
    a = ap.parse_args()
    n, m, h = a.layers, a.micro_batches, a.hidden
    mbs = micro_batches(m, h)
    ws = [x.shape[0] / sum(t.shape[0] for t in mbs) for x in mbs]
    s_a, g_a = as_is(n, h, mbs, ws)
    s_b, g_b = layer_major(n, h, mbs, ws)
    s_c, g_c = layer_major_held(n, h, mbs, ws)

    def rel(ga, gb):
        return max(((x - y).abs().max() / x.abs().max().clamp_min(1e-30)).item() for x, y in zip(ga, gb))
    print(json.dumps({
        "torch": torch.__version__, "layers": n, "micro_batches": m,
        "prediction": {"as_is": m * (2 * n - 1), "layer_major": 2 * n - 1},
        "note": "a fresh toy starts with nothing staged, so each schedule's first forward also stages layer 0: +1 copy vs the box",
        "as_is": summarize(s_a, n, m), "layer_major_e4b_hooks": summarize(s_b, n, m),
        "layer_major_held_forward": summarize(s_c, n, m),
        "grad_max_rel_diff_vs_as_is": {"layer_major": rel(g_a, g_b), "layer_major_held": rel(g_a, g_c)}}, indent=1))


if __name__ == "__main__":
    main()
