#!/usr/bin/env python3
"""The distinct-expert count per layer per decode step -- the one soft number in #564's expert-tier roofline.

P54 registered this as an untimed ``--amort on --series-out`` arm and the harness refused it: ``step_decomp``'s
B>1 stage asserts ``--amort off`` ("amort-armed runs keep the baseline dispatch path (not capturable)"), so
the count could not come from the census harness at B=16 at all. This is the replacement (P54 amendment 1):
a forward hook on every MoE block's router that records, per layer, how many DISTINCT experts each decode
step's batch touched. It is model-generic (any block whose router lives at ``.gate`` and whose config carries
``num_experts_per_tok``), it never touches the serving kernels, and its step time is not quoted.

**Capture-safe (P57 amendment 3).** The first version kept a Python list of ``int(torch.unique(ids).numel())``
per call. Under the harness's B>1 stage every decode step after warm-up runs inside a captured CUDA graph, and
``torch.unique`` needs the result size on the host -- a device->host synchronisation that is illegal during
capture (``p57b-5090-2`` / ``p57c-5090-2``: ``operation not permitted when stream is capturing``, 3 warm-up
steps counted, then the harness died). This version does **no host work in the hook**: for a decode call it
scatters the routed ids into a preallocated per-layer ``[E]`` buffer, sums it (the distinct count as a device
scalar) and accumulates -- in place, static shapes, no allocation, no sync -- into per-layer device counters:
the sum of distinct counts, the min and max, the number of decode calls, and how often each expert was touched.
Recorded during capture, those ops replay with the graph, so every replayed step is counted. ``report()`` moves
the counters to the host once, after the run. Prefill chunks (more rows than the batch) are counted separately
and never enter the statistics
the check is on ``ids.shape[0]``, a Python int, so it does not sync either.

    python distinct_experts.py --self-test        # CPU: a tiny random Qwen3MoE, the hook end to end
    python distinct_experts.py --capture-test     # CUDA: a fake router under torch.cuda.graph capture + replay,
                                                  # the counter checked against a CPU reference step by step

Output: ``{"batch", "top_k", "num_experts", "layers", "steps", "method", "per_layer": [{"steps", "mean_distinct",
"min", "max", "raw_calls", "prefill_calls", "touched_frac": [E floats]}], "mean_distinct_over_layers",
"min_layer_mean", "max_layer_mean", "uniform_random_expectation"}``. ``series`` is gone (it needed the host per
step)
``touched_frac[e]`` = the fraction of decode steps in which expert ``e`` was touched, per layer. The
expectation for a uniform router is ``E * (1 - (1 - k/E)**B)`` for B rows drawing k of E experts each -- 82.4
for E=128, k=8, B=16 -- and is printed beside the measurement so the skew is visible.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch
from torch import nn


class DistinctExpertCounter:
    """Hooks every ``gate`` module under a module whose class name ends in ``SparseMoeBlock`` (transformers'
    Qwen3-MoE naming; ``block_suffix`` overrides) and accumulates distinct-expert statistics ON DEVICE."""

    def __init__(self, model: nn.Module, top_k: int, num_experts: int, batch: int,
                 device: torch.device | str | None = None, block_suffix: str = "SparseMoeBlock"):
        self.top_k, self.num_experts, self.batch = int(top_k), int(num_experts), int(batch)
        blocks = [(n, m) for n, m in model.named_modules() if type(m).__name__.endswith(block_suffix)]
        if not blocks:
            raise RuntimeError(f"no module whose class name ends in {block_suffix!r}: cannot place the router hook")
        if device is None:
            device = next(model.parameters()).device
        L, E = len(blocks), self.num_experts
        self.layers = L
        self._touched = torch.zeros(L, E, dtype=torch.float32, device=device)     # scratch, zeroed per decode call
        self.touched_sum = torch.zeros(L, E, dtype=torch.float32, device=device)  # steps in which expert e was touched
        self.cnt_sum = torch.zeros(L, dtype=torch.float32, device=device)         # sum over steps of distinct counts
        self.cnt_min = torch.full((L,), float("inf"), dtype=torch.float32, device=device)
        self.cnt_max = torch.full((L,), float("-inf"), dtype=torch.float32, device=device)
        self.n_dec = torch.zeros(L, dtype=torch.float32, device=device)           # decode calls (rows == batch)
        self.n_other = torch.zeros(L, dtype=torch.float32, device=device)         # prefill / other calls
        self.handles = []
        for li, (name, blk) in enumerate(blocks):
            gate = getattr(blk, "gate", None)
            if not isinstance(gate, nn.Module):
                raise RuntimeError(f"{name}: expected the router module at .gate, found {type(gate).__name__}")
            self.handles.append(gate.register_forward_hook(self._make_hook(li)))

    @staticmethod
    def _routed_ids(out, top_k: int) -> torch.Tensor:
        # transformers >= 5.x: the router is its own module returning (logits, scores, indices) -- take the
        # indices it actually routed on. Older trees: a bare nn.Linear `gate` returning logits -- the
        # selection is argmax-invariant to the softmax, so top-k on the logits IS that router's top-k.
        if isinstance(out, tuple) and len(out) == 3 and not torch.is_floating_point(out[2]):
            return out[2]
        logits = out[0] if isinstance(out, tuple) else out
        return torch.topk(logits.reshape(-1, logits.shape[-1]), top_k, dim=-1).indices

    def _make_hook(self, li: int):
        def hook(_mod, _inp, out):
            ids = self._routed_ids(out, self.top_k)
            if ids.shape[0] != self.batch:                 # a Python int: prefill chunk / other -- no sync
                self.n_other[li] += 1
                return
            t = self._touched[li]
            t.zero_()
            t.scatter_(0, ids.reshape(-1).to(torch.long), 1.0)   # 0/1 per expert, in place, static shape
            cnt = t.sum()                                          # the distinct count, a device scalar
            self.touched_sum[li] += t
            self.cnt_sum[li] += cnt
            self.cnt_min[li].copy_(torch.minimum(self.cnt_min[li], cnt))
            self.cnt_max[li].copy_(torch.maximum(self.cnt_max[li], cnt))
            self.n_dec[li] += 1
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def report(self) -> dict:
        """Host side, once, after the run (never inside the hook)."""
        n = self.n_dec.cpu()
        cs = self.cnt_sum.cpu()
        mn = self.cnt_min.cpu()
        mx = self.cnt_max.cpu()
        no = self.n_other.cpu()
        ts = self.touched_sum.cpu()
        per_layer = []
        for li in range(self.layers):
            steps = int(n[li].item())
            mean = float(cs[li].item() / steps) if steps else None
            per_layer.append({"steps": steps, "mean_distinct": mean,
                              "min": int(mn[li].item()) if steps else None, "max": int(mx[li].item()) if steps else None,
                              "raw_calls": steps + int(no[li].item()), "prefill_calls": int(no[li].item()),
                              "touched_frac": [round(float(x) / steps, 4) for x in ts[li]] if steps else None})
        means = [x["mean_distinct"] for x in per_layer if x["mean_distinct"] is not None]
        e, k, b = self.num_experts, self.top_k, self.batch
        return {"batch": b, "top_k": k, "num_experts": e, "layers": self.layers,
                "steps": min((x["steps"] for x in per_layer), default=0),
                "method": "on-device accumulation in the router hook (capture-safe, P57 amendment 3): every eager and every graph-replayed decode call is counted; no per-step series",
                "mean_distinct_over_layers": statistics.mean(means) if means else None,
                "min_layer_mean": min(means) if means else None, "max_layer_mean": max(means) if means else None,
                "uniform_random_expectation": e * (1 - (1 - k / e) ** b),
                "per_layer": per_layer}


class _FakeBlock(nn.Module):
    """A 'SparseMoeBlock' whose router returns routed ids from a table, advanced by a DEVICE step counter --
    every op static-shaped and on device, so it captures and replays like the real router."""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.gate = _FakeRouter(table)

    def forward(self, x):
        return self.gate(x)


class _FakeRouter(nn.Module):
    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.register_buffer("table", table)                                      # [S, rows, k] int64
        self.register_buffer("step", torch.zeros(1, dtype=torch.long, device=table.device))

    def forward(self, x):
        ids = torch.index_select(self.table, 0, self.step)[0]
        self.step += 1
        logits = torch.zeros(ids.shape[0], self.table.shape[-1], device=ids.device)
        return logits, logits, ids


class _FakeSparseMoeBlock(_FakeBlock):
    pass


def _capture_test() -> int:
    """CUDA: eager steps, then one step captured into a CUDA graph and replayed; the counter must equal the CPU
    reference over exactly the executed steps (eager + replays; the capture pass itself executes nothing)."""
    if not torch.cuda.is_available():
        print("capture-test needs CUDA")
        return 2
    dev = torch.device("cuda")
    torch.manual_seed(0)
    E, k, B, L, S, eager, replays = 128, 8, 16, 3, 64, 3, 20
    tables = [torch.stack([torch.stack([torch.randperm(E)[:k] for _ in range(B)]) for _ in range(S)]).to(dev) for _ in range(L)]
    model = nn.Sequential(*[_FakeSparseMoeBlock(t) for t in tables]).to(dev)
    counter = DistinctExpertCounter(model, top_k=k, num_experts=E, batch=B, device=dev)
    x = torch.zeros(B, 8, device=dev)

    def step():
        for blk in model:
            blk(x)
    for _ in range(eager):
        step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()                                                     # warm on the side stream (executes: counts as a step)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        step()                                                     # captured, NOT executed
    for _ in range(replays):
        g.replay()
    torch.cuda.synchronize()
    rep = counter.report()
    executed = eager + 1 + replays
    # CPU reference over the steps that executed: eager (0..2), the side-stream warm (3), replays (4..23):
    # the captured step reads table[step] with the step counter's value AT CAPTURE, and replays re-read the
    # same captured index but the counter increments each replay... so recompute exactly what the device did.
    ok = True
    for li, t in enumerate(tables):
        # what executed: steps 0..eager-1 and eager (warm) eagerly; then the graph replays: each replay re-runs
        # index_select(table, step) with the CURRENT step buffer value (incremented by the previous replay).
        seq = list(range(eager + 1)) + [eager + 1 + i for i in range(replays)]
        ref = [len(set(t[i].flatten().tolist())) for i in seq]
        pl = rep["per_layer"][li]
        got_mean, got_min, got_max, got_steps = pl["mean_distinct"], pl["min"], pl["max"], pl["steps"]
        exp_mean = statistics.mean(ref)
        line = f"layer {li}: steps {got_steps} (expected {executed}), mean {got_mean:.4f} (ref {exp_mean:.4f}), min {got_min}/{min(ref)}, max {got_max}/{max(ref)}"
        if got_steps != executed or abs(got_mean - exp_mean) > 1e-4 or got_min != min(ref) or got_max != max(ref):
            ok = False
            line += "  <-- MISMATCH"
        print(line)
    frac_ok = all(abs(sum(pl["touched_frac"]) - pl["mean_distinct"]) < 1e-3 for pl in rep["per_layer"])
    print("touched_frac sums equal the mean distinct per layer:", frac_ok)
    print("capture-test", "OK" if (ok and frac_ok) else "FAILED", f"-- {executed} decode steps counted under capture+replay, uniform expectation {rep['uniform_random_expectation']:.2f}")
    return 0 if (ok and frac_ok) else 1


def _self_test() -> int:
    """CPU: a tiny random Qwen3MoE
    the hook must see every layer on every step and count sensibly."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_experts=16,
                         num_experts_per_tok=4, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, vocab_size=512, max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).eval()
    B, T, steps = 5, 7, 4
    counter = DistinctExpertCounter(model, top_k=cfg.num_experts_per_tok, num_experts=cfg.num_experts, batch=B)
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        for _ in range(steps):
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
    counter.remove()
    rep = counter.report()                                   # prefill (B*T rows) is dropped by row count
    assert rep["layers"] == 3 and rep["steps"] == steps, rep
    assert all(pl["prefill_calls"] == 1 and pl["raw_calls"] == steps + 1 for pl in rep["per_layer"]), rep["per_layer"][0]
    for pl in rep["per_layer"]:
        assert 1 <= pl["min"] <= pl["max"] <= min(cfg.num_experts, B * cfg.num_experts_per_tok), pl
        assert abs(sum(pl["touched_frac"]) - pl["mean_distinct"]) < 1e-6, pl
    print(json.dumps({k: v for k, v in rep.items() if k != "per_layer"}, indent=1))
    print("self-test OK: 3 layers x", steps, "decode steps hooked; uniform expectation",
          round(rep["uniform_random_expectation"], 2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--capture-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if a.capture_test:
        return _capture_test()
    raise SystemExit("box mode is wired by the lane's runner (p57_run.sh) around the harness's own load path; "
                     "run --self-test (CPU) or --capture-test (CUDA) here")


if __name__ == "__main__":
    sys.exit(main())
