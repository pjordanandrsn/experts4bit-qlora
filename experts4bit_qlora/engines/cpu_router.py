# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""CPU router — Phase 1 of the hybrid CPU/GPU tier (gate G1).

Moves router logits + top-k off the GPU and onto the host, killing the
per-layer blocking D2H sync class found in the #105/#108 campaign (~5 host
syncs per layer-step in paths whose design law is "nothing synchronizes").
The CPU becomes the dispatcher: it reads the post-attention hidden state via
an async D2H into pinned per-layer staging, computes routing against a
host-resident FP32 copy of the router weights, and pushes the index/weight
vectors H2D on a side stream. The compute stream never host-syncs — it
waits device-side on the tiny H2D event.

Design contract (one round trip per layer, decode shapes):

    compute stream:  ... attention ... post_attn_norm ─┐        ┌─ experts
                                                       │        │   (waits
    copy stream:                 wait ── D2H(hidden) ──┤        │   ev_idx
                                                       ▼        │   device-
    host:                                   ev_d2h.sync()       │   side)
                                            fp32 logits+topk    │
                                            pinned idx/wts ─ H2D┴ ev_idx

The FP32 router copy (~1M params/layer) is the documented, deliberate
exemption to the one-artifact invariant — it is a *router* copy, never an
expert weight.

Determinism: top-k uses a stable descending sort, so ties break to the
LOWER expert index, identically on every backend. Router math is fp32 on
host (deterministic per box); the resulting weights are cast to the module
dtype. Enabling this engine changes routing arithmetic from device
bf16-linear to host fp32-linear — selection can legitimately differ from
the unpatched model only where the reference's own logit gap is within
bf16 noise; ``assert_every=N`` cross-checks exactly that (and raising on a
real disagreement is a directive stop condition).

Falls back to the saved reference forward under grad/training, on non-CUDA
inputs, for prefill (rows > ``max_rows``), and inside CUDA-graph capture
(the reference router is pure device ops and captures fine; a host
callback cannot).

Supported router classes (explicit table — unknown routers are skipped
loudly, never guessed): OlmoeTopKRouter, Qwen3MoeTopKRouter (softmax→topk,
``norm_topk_prob``), GptOssTopKRouter (bias, topk→softmax). DeepSeek-V4/K3
grouped-topk routers are out of scope for Phase 1 and reported as skipped.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..util import log

_MARKER = "_e4b_cpu_router_ref"
_STATE = "_cpu_router"

# class-name -> epilogue kind
_SOFTMAX_THEN_TOPK = "softmax_then_topk"   # olmoe / qwen3_moe
_TOPK_THEN_SOFTMAX = "topk_then_softmax"   # gpt_oss
_ROUTER_KINDS = {
    "OlmoeTopKRouter": _SOFTMAX_THEN_TOPK,
    "Qwen3MoeTopKRouter": _SOFTMAX_THEN_TOPK,
    "GptOssTopKRouter": _TOPK_THEN_SOFTMAX,
}

_COPY_STREAMS: dict[int, torch.cuda.Stream] = {}


def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    s = _COPY_STREAMS.get(device.index)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _COPY_STREAMS[device.index] = s
    return s


def deterministic_topk(logits32: torch.Tensor, k: int):
    """Top-k with the backend-fixed rule: stable descending sort, ties to
    the lower expert index. Returns (values [T,k] fp32, indices [T,k] i64)."""
    vals, idx = torch.sort(logits32, dim=-1, descending=True, stable=True)
    return vals[:, :k], idx[:, :k]


def route_on_host(logits32, k, kind, norm_topk_prob):
    """The routing epilogue, fp32, mirroring the reference math of each
    supported arch. Returns (weights [T,k] fp32, indices [T,k] i64)."""
    top_logits, top_idx = deterministic_topk(logits32, k)
    if kind == _SOFTMAX_THEN_TOPK:
        # reference: softmax over ALL logits (fp32), then select
        probs = torch.softmax(logits32, dim=-1)
        w = probs.gather(-1, top_idx)
        if norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
    else:  # _TOPK_THEN_SOFTMAX
        w = torch.softmax(top_logits, dim=-1)
    return w, top_idx


class _CpuRouter:
    """Per-router-module state: FP32 host weights, pinned staging (2-slot
    ring), device landing tensors, reusable events."""

    def __init__(self, mod, kind: str, device: torch.device, max_rows: int,
                 assert_every: int):
        self.kind = kind
        self.k = int(mod.top_k)
        self.n_experts = int(mod.num_experts)
        self.max_rows = max_rows
        self.assert_every = assert_every
        self.calls = 0
        self.device = device
        self.model_dtype = mod.weight.dtype
        # host FP32 router copy — [H, E] contiguous so logits = h32 @ w_t
        self.w_t = mod.weight.detach().to("cpu", torch.float32).t().contiguous()
        b = getattr(mod, "bias", None)
        self.bias32 = None if b is None else b.detach().to("cpu", torch.float32)
        hid = int(mod.hidden_dim)
        # pinned staging: hidden in model dtype (D2H is a plain memcpy),
        # idx/weights outbound. Two hidden slots (ring) per the double-
        # buffer contract; idx/wts are consumed synchronously by the H2D
        # enqueue on the same host thread, so one slot each suffices.
        # staging stays in MODEL dtype: a cross-dtype D2H is a cast kernel
        # plus a copy — two API submissions, ~14 µs measured — where the
        # same-dtype copy is one. The bf16->fp32 cast happens host-side in
        # numpy as an exact uint16 zero-extend (<<16), ~2 µs.
        self.pin_hidden = [
            torch.empty(max_rows, hid, dtype=self.model_dtype, pin_memory=True)
            for _ in range(2)
        ]
        # fused landing row: [k × int64 idx][k × model-dtype wts][pad to 8 B]
        # — ONE pinned buffer, ONE H2D enqueue for both tensors. Every CUDA
        # API call is ~3-8 µs of the 35 µs budget; two copies were two.
        wb = self.k * self.model_dtype.itemsize
        self.row_bytes = self.k * 8 + ((wb + 7) & ~7)
        self.pin_land = torch.empty(max_rows * self.row_bytes,
                                    dtype=torch.uint8, pin_memory=True)
        self.dev_land = torch.empty(max_rows * self.row_bytes,
                                    dtype=torch.uint8, device=device)
        rw8 = self.row_bytes // 8
        self.pin_idx = self.pin_land.view(torch.int64).as_strided(
            (max_rows, self.k), (rw8, 1))
        self.dev_idx = self.dev_land.view(torch.int64).as_strided(
            (max_rows, self.k), (rw8, 1))
        rwi = self.row_bytes // self.model_dtype.itemsize
        woff = (self.k * 8) // self.model_dtype.itemsize
        self.pin_wts = self.pin_land.view(self.model_dtype).as_strided(
            (max_rows, self.k), (rwi, 1), woff)
        self.dev_wts = self.dev_land.view(self.model_dtype).as_strided(
            (max_rows, self.k), (rwi, 1), woff)
        # host scratch, preallocated. The epilogue runs in NUMPY over
        # zero-copy views of these buffers: a torch CPU op costs ~2-5 µs of
        # dispatch, numpy ~1 µs, and the 35 µs round-trip budget is spent
        # almost entirely on dispatch, not arithmetic. One torch op remains
        # (the bf16->fp32 cast of the staged hidden; numpy has no bf16).
        self.np_w_t = self.w_t.numpy()                     # [H, E] fp32 view
        # one float per cache line: summing this strided view streams the
        # whole weight matrix into LLC during the wake-spin (the gemv is
        # DRAM-bound cold — ~10 µs of the budget — and the host is
        # otherwise idle while the GPU finishes attention + the D2H)
        self.np_w_lines = self.np_w_t.reshape(-1)[::16]
        self.np_bias = None if self.bias32 is None else self.bias32.numpy()
        if self.model_dtype == torch.bfloat16:
            self.np_hidden_u16 = [p.view(torch.uint16).numpy()
                                  for p in self.pin_hidden]
        else:                                   # fp16 staging: torch cast path
            self.np_hidden_u16 = None
            self.np_hidden = [p.numpy() for p in self.pin_hidden]
        self.np_h32u = np.empty((max_rows, hid), dtype=np.uint32)
        self.np_h32f = self.np_h32u.view(np.float32)
        self.np_logits = np.empty((max_rows, self.n_experts), dtype=np.float32)
        self.np_exp = np.empty_like(self.np_logits)
        self.np_w32 = np.empty((max_rows, self.k), dtype=np.float32)
        self.t_w32 = torch.from_numpy(self.np_w32)         # wrapped ONCE
        self.np_idx = np.lib.stride_tricks.as_strided(     # pinned idx view
            self.pin_land.numpy().view(np.int64),
            shape=(max_rows, self.k), strides=(self.row_bytes, 8))
        self.spin_wait = 400          # ev queries before falling back to sync
        self.next_w_lines = None      # layer L+1's weight lines (chain warm)
        self.slot = 0
        self.ev_start = torch.cuda.Event(enable_timing=True)
        self.ev_d2h = torch.cuda.Event(enable_timing=True)
        self.ev_idx = torch.cuda.Event(enable_timing=True)
        # round-trip stats (ms), appended per served call when timing is on
        self.timing = False
        self.trip_ms: list[float] = []
        self.host_ms: list[float] = []
        self.seg_ms: list[tuple[float, float, float]] = []  # wake/math/push

    def will_serve(self, mod, hidden: torch.Tensor) -> bool:
        if mod.training:
            # reentrant gradient checkpointing runs its first forward under
            # no_grad while still in train mode; serving it would disagree
            # with the grad-enabled recompute (same footgun fast.py guards)
            return False
        if torch.is_grad_enabled() and hidden.requires_grad:
            return False
        if not hidden.is_cuda:
            return False
        if hidden.shape[0] > self.max_rows:
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        return True

    def route(self, mod, hidden: torch.Tensor):
        """The served path. ``hidden`` is [T, H] on device, T <= max_rows."""
        t = hidden.shape[0]
        cur = torch.cuda.current_stream(self.device)
        side = _copy_stream(self.device)
        pin_h = self.pin_hidden[self.slot]
        self.slot ^= 1

        side.wait_stream(cur)
        with torch.cuda.stream(side):
            if self.timing:
                self.ev_start.record(side)
            pin_h[:t].copy_(hidden, non_blocking=True)
            self.ev_d2h.record(side)
        hidden.record_stream(side)

        # host wait: event-scoped, on the copy stream only — the compute
        # stream keeps draining whatever is queued behind it. Spin-poll
        # first: cudaEventSynchronize's blocking wake costs ~10-20 µs, a
        # query loop wakes in ~2-5; the D2H is µs-scale by construction, so
        # the spin is short. Fall back to the blocking wait if the GPU is
        # genuinely far behind (deep queue) rather than burning a core.
        if self.timing:
            t0 = time.perf_counter()
        if not self.ev_d2h.query():
            # GPU still on attention/D2H: use the dead time to pull the
            # router weights into cache so the gemv below runs warm
            float(self.np_w_lines.sum())
            for _ in range(self.spin_wait):
                if self.ev_d2h.query():
                    break
            else:
                self.ev_d2h.synchronize()
        if self.timing:
            t_wake = time.perf_counter()
        # trimmed epilogue in numpy over preallocated buffers: identical
        # math to route_on_host() (pinned by
        # test_route_inline_matches_route_on_host). Stable argsort of the
        # NEGATED logits = descending with ties to the lower index — the
        # backend-fixed rule. softmax(x)[idx] is computed as
        # exp(x-m)[idx]/Z, same values, fewer dispatches.
        if self.np_hidden_u16 is not None:         # exact bf16->fp32: <<16
            h32u = self.np_h32u[:t]
            h32u[:] = self.np_hidden_u16[self.slot ^ 1][:t]
            np.left_shift(h32u, 16, out=h32u)
            nh = self.np_h32f[:t]
        else:
            nh = self.np_hidden[self.slot ^ 1][:t].astype(np.float32)
        logits = self.np_logits[:t]
        np.dot(nh, self.np_w_t, out=logits)
        if self.np_bias is not None:
            logits += self.np_bias
        order = np.argsort(-logits, axis=-1, kind="stable")
        idx = order[:, :self.k]
        w32 = self.np_w32[:t]
        if self.kind == _SOFTMAX_THEN_TOPK:
            e = self.np_exp[:t]
            np.subtract(logits, logits.max(axis=-1, keepdims=True), out=e)
            np.exp(e, out=e)
            w32[:] = np.take_along_axis(e, idx, axis=-1)
            w32 /= e.sum(axis=-1, keepdims=True)
            if bool(getattr(mod, "norm_topk_prob", False)):
                w32 /= w32.sum(axis=-1, keepdims=True)
        else:
            top = np.take_along_axis(logits, idx, axis=-1)
            np.subtract(top, top.max(axis=-1, keepdims=True), out=top)
            np.exp(top, out=top)
            np.divide(top, top.sum(axis=-1, keepdims=True), out=w32[:, :])
        if self.timing:
            t_math = time.perf_counter()
        self.np_idx[:t] = idx                      # writes land in pinned
        self.pin_wts[:t].copy_(self.t_w32[:t])     # fp32->model dtype, one op
        if self.timing:
            now = time.perf_counter()
            self.host_ms.append((now - t0) * 1e3)
            self.seg_ms.append(((t_wake - t0) * 1e3, (t_math - t_wake) * 1e3,
                                (now - t_math) * 1e3))

        with torch.cuda.stream(side):
            nb = t * self.row_bytes                # ONE fused H2D for both
            self.dev_land[:nb].copy_(self.pin_land[:nb], non_blocking=True)
            self.ev_idx.record(side)
        cur.wait_event(self.ev_idx)
        if self.timing:
            self.ev_idx.synchronize()
            self.trip_ms.append(self.ev_start.elapsed_time(self.ev_idx))

        # chain warm: pull the NEXT router's weights toward cache now, off
        # the critical path (the GPU is busy with the next layer's
        # attention; a cold gemv costs ~10 µs ON the path, this costs the
        # same ~10 µs beside it). In a deep-queued real decode the
        # spin-warm above never fires — this is what keeps the gemv warm
        # there.
        if self.next_w_lines is not None:
            float(self.next_w_lines.sum())

        self.calls += 1
        if self.assert_every and self.calls % self.assert_every == 0:
            self._cross_check(mod, hidden,
                              torch.from_numpy(np.ascontiguousarray(idx)))
        # router_logits is returned as None on the served path: all three
        # supported blocks discard it in inference, and pushing an [T,E]
        # tensor is 3 dispatches the 35 µs budget cannot spare. Aux-loss /
        # training callers run the reference path (grad fallback) and get
        # real logits.
        return (None, self.dev_wts[:t], self.dev_idx[:t])

    def _cross_check(self, mod, hidden, idx):
        """Debug-mode GPU-reference comparison (synchronizes; never on the
        hot path unless requested). A disagreement the reference's own
        logit gap cannot explain is a stop condition — raise."""
        ref_logits = torch.nn.functional.linear(
            hidden, mod.weight, getattr(mod, "bias", None)
        ).float().cpu()
        _, ref_idx = deterministic_topk(ref_logits, self.k)
        if torch.equal(ref_idx, idx):
            return
        # allowed only where the swap crosses a near-tie in the reference
        eps = 1e-2 * ref_logits.abs().max().clamp(min=1.0)
        for row in range(idx.shape[0]):
            ours, ref = set(idx[row].tolist()), set(ref_idx[row].tolist())
            if ours == ref:
                continue
            gap = min(
                abs(ref_logits[row, a] - ref_logits[row, b])
                for a in ours - ref for b in ref - ours
            )
            if gap > eps:
                raise RuntimeError(
                    f"cpu_router disagrees with GPU reference beyond "
                    f"tie-break: row {row} ours={sorted(ours)} "
                    f"ref={sorted(ref)} gap={gap:.4f} eps={eps:.4f} — "
                    f"stop condition, do not work around"
                )


def cpu_router_available() -> bool:
    """True when the served path can run at all (CUDA + pinned alloc)."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.empty(8, pin_memory=True)
        return True
    except RuntimeError:
        return False


def enable_cpu_router(model, *, max_rows: int = 8, assert_every: int = 0,
                      timing: bool = False, verbose: bool = False) -> int:
    """Patch every supported router module; returns the patch count (0 is
    'not engaged' — record it, per the config-matrix lesson)."""
    n = 0
    skipped: list[str] = []
    for name, mod in model.named_modules():
        kind = _ROUTER_KINDS.get(type(mod).__name__)
        if kind is None:
            if type(mod).__name__.endswith("TopKRouter"):
                skipped.append(f"{name} ({type(mod).__name__})")
            continue
        if hasattr(mod, _MARKER):
            raise RuntimeError(f"{name} already has a cpu_router patch")
        device = mod.weight.device
        if device.type != "cuda":
            skipped.append(f"{name} (weights on {device})")
            continue
        state = _CpuRouter(mod, kind, device, max_rows, assert_every)
        state.timing = timing
        ref = mod.forward

        def patched(hidden_states, _mod=mod, _state=state, _ref=ref):
            h = hidden_states.reshape(-1, _state.w_t.shape[0])
            if not _state.will_serve(_mod, h):
                return _ref(hidden_states)
            return _state.route(_mod, h)

        setattr(mod, _MARKER, ref)
        setattr(mod, _STATE, state)
        mod.forward = patched
        n += 1
        if verbose:
            log(f"cpu_router: patched {name} ({kind}, E={state.n_experts}, "
                f"k={state.k})")
    # chain the states in discovery (= layer) order so each warms its
    # successor's weights after pushing its own indices
    states = [getattr(m, _STATE) for _, m in model.named_modules()
              if hasattr(m, _STATE)]
    for i, st in enumerate(states):
        if len(states) > 1:
            st.next_w_lines = states[(i + 1) % len(states)].np_w_lines
    for s in skipped:
        log(f"cpu_router: SKIPPED unsupported router {s}")
    if verbose or n == 0:
        log(f"cpu_router: patched {n} router modules")
    return n


def disable_cpu_router(model) -> int:
    """Restore every patched router and drop all stamped state."""
    n = 0
    for _, mod in model.named_modules():
        ref = getattr(mod, _MARKER, None)
        if ref is None:
            continue
        mod.forward = ref
        delattr(mod, _MARKER)
        if hasattr(mod, _STATE):
            delattr(mod, _STATE)
        n += 1
    return n


def router_trip_stats(model) -> dict:
    """Aggregate per-layer round-trip timings (requires ``timing=True``).
    trip = D2H enqueue -> indices landed on device (CUDA events);
    host = event wait + fp32 routing + pinned writes (perf_counter)."""
    trips: list[float] = []
    hosts: list[float] = []
    for _, mod in model.named_modules():
        st = getattr(mod, _STATE, None)
        if st is not None:
            trips.extend(st.trip_ms)
            hosts.extend(st.host_ms)
    if not trips:
        return {"served_calls": 0}
    trips.sort()
    hosts.sort()

    def pct(v, p):
        return v[min(len(v) - 1, int(p * len(v)))]

    segs: list[tuple[float, float, float]] = []
    for _, mod in model.named_modules():
        st = getattr(mod, _STATE, None)
        if st is not None:
            segs.extend(st.seg_ms)
    out = {
        "served_calls": len(trips),
        "trip_us_p50": pct(trips, 0.50) * 1e3,
        "trip_us_p99": pct(trips, 0.99) * 1e3,
        "host_us_p50": pct(hosts, 0.50) * 1e3,
        "host_us_p99": pct(hosts, 0.99) * 1e3,
    }
    if segs:
        for i, name in enumerate(("wake", "math", "push")):
            col = sorted(s[i] for s in segs)
            out[f"{name}_us_p50"] = pct(col, 0.50) * 1e3
    return out
