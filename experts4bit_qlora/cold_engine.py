"""Cold engine — host-CPU compute of the cold expert tail.

:func:`enable_hot_residency` pins each layer's hottest experts in VRAM and
*streams* the routed cold tail's NF4 to the device per token — the right call
where the host CPU is weak and the PCIe link is not the sole bottleneck. On a
strong-CPU server the opposite wins: keep the cold NF4 in host RAM and compute
the cold experts *on the host* (a GGUF runtime's ``--n-cpu-moe`` regime), so
the only per-token traffic is activation-sized (``[rows, hidden]`` each way)
instead of weight-sized. This module is that other instrument.

The partition, id bookkeeping, and the hot (device) side are shared with
:mod:`.hot_residency` — the cold engine only replaces the cold branch. The math
stays identical to the reference ``ExpertsNbit`` forward: both paths decode the
same NF4 values (the host dequant is bit-exact against
``bitsandbytes.functional.dequantize_4bit``; correctness-gated in the suite),
round through the module's ``compute_dtype``, and accumulate in fp32.

Host dequant backend (``dequant=``):

* ``"bnb"``   — ``bitsandbytes``' CPU ``dequantize_4bit``. Carries an AVX-512
  kernel; **on AVX2-only hosts it silently falls back below naive torch**
  (grouped-nf4-gemm ``bench/cold-engine/receipts-floor_bnb-qnap.json``:
  0.041 GB/s vs the 0.067 GB/s naive floor on the Comet Lake box), so it is
  only auto-picked when ``avx512f`` is present.
* ``"torch"`` — pure-torch nibble-unpack + codebook gather, bit-exact with the
  bnb decode. The AVX2-cliff-safe default — a correctness reference, not a
  bandwidth path (Phase-0 rooflines put it at ~0.6% of memcpy ceiling on
  gpt-oss expert geometry); the ggml-style AVX2 decode kernel slots in here
  as a third backend when the cold-engine lane's Phase 2 lands.
* ``"auto"``  — ``"bnb"`` iff the CPU reports ``avx512f`` *and* a probe decode
  succeeds; otherwise ``"torch"``.

An all-cold configuration (every ``hot_sets`` entry empty, ``device="cpu"``)
needs neither a CUDA device nor the fused ``nf4_grouped`` kernel — the whole
MoE runs on the host. Non-empty hot sets require ``[fast]`` + CUDA exactly as
hot residency does.

Usage::

    from experts4bit_qlora import enable_cold_engine
    # hot_sets[i] = 1-D LongTensor of hot expert ids for the i-th MoE layer
    # ([] = that layer runs fully on the host).
    n = enable_cold_engine(model, hot_sets, device="cuda", dequant="auto")
"""
from __future__ import annotations

import functools
from typing import Sequence

import torch

from .hot_residency import _HotResidency, _eligible


# --------------------------------------------------------------------------- #
# host dequant backends
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=1)
def _cpu_has_avx512() -> bool:
    """Linux-only probe for the avx512f ISA flag; False anywhere it can't be read."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags") and " avx512f" in line:
                    return True
        return False
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def _bnb_cpu_dequant_ok() -> bool:
    """Whether bitsandbytes' CPU ``dequantize_4bit`` actually runs on this host."""
    try:
        from bitsandbytes import functional as F
        code = F.get_4bit_type("nf4", device="cpu")
        st = F.QuantState(absmax=torch.ones(2), shape=torch.Size((1, 128)), code=code,
                          blocksize=64, quant_type="nf4", dtype=torch.float32)
        F.dequantize_4bit(torch.zeros(64, 1, dtype=torch.uint8), quant_state=st)
        return True
    except Exception:
        return False


def _resolve_dequant(spec: str) -> str:
    if spec == "torch":
        return "torch"
    if spec == "bnb":
        if not _bnb_cpu_dequant_ok():
            raise RuntimeError(
                "dequant='bnb' requested but bitsandbytes' CPU dequantize_4bit is not "
                "functional on this host — use dequant='torch' (bit-exact) instead")
        return "bnb"
    if spec == "auto":
        # bnb's CPU 4-bit dequant is AVX-512-only-fast; on AVX2-only hosts it lands
        # below the naive torch decode (own receipts), so avx512f gates the pick.
        return "bnb" if (_cpu_has_avx512() and _bnb_cpu_dequant_ok()) else "torch"
    raise ValueError(f"dequant must be 'auto', 'bnb', or 'torch'; got {spec!r}")


def _dequant_torch(packed: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor,
                   n: int, k: int) -> torch.Tensor:
    """Pure-torch NF4 decode of one expert: ``packed [n, k//2] uint8`` +
    ``absmax [n, k//64] fp32`` -> fp32 ``[n, k]``. First element rides the HIGH
    nibble (bitsandbytes' quantize_4bit packing); bit-exact vs the bnb decode
    (gated by ``test_cold_engine.py::test_torch_dequant_bit_exact_vs_bnb``)."""
    pk = packed.reshape(-1)
    idx = torch.empty(pk.numel() * 2, dtype=torch.long)
    idx[0::2] = (pk >> 4).long()
    idx[1::2] = (pk & 0xF).long()
    vals = code.index_select(0, idx)                                  # fp32 [n*k]
    return (vals.reshape(-1, 64) * absmax.reshape(-1, 1)).reshape(n, k)


def _dequant_bnb(packed: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor,
                 n: int, k: int) -> torch.Tensor:
    """bitsandbytes CPU decode of one expert, mirroring the reference
    ``_dequantize_expert`` call shape-for-shape (``[packed, 1]`` layout)."""
    from bitsandbytes import functional as F
    st = F.QuantState(absmax=absmax.reshape(-1), shape=torch.Size((n, k)), code=code,
                      blocksize=64, quant_type="nf4", dtype=torch.float32)
    return F.dequantize_4bit(packed.reshape(-1, 1), quant_state=st)


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #

class _ColdEngine(_HotResidency):
    """Hot side identical to :class:`_HotResidency`; cold side computed on the
    host from the CPU-resident NF4 stacks (weights never cross the bus)."""

    _PIN_COLD = False  # cold weights are never H2D-streamed — keep them pageable

    def __init__(self, mod, hot_ids, device, dequant: str = "auto"):
        super().__init__(mod, hot_ids, device)
        self.dequant_backend = _resolve_dequant(dequant)
        self._dequant = _dequant_bnb if self.dequant_backend == "bnb" else _dequant_torch
        # the shared NF4 codebook, host-resident fp32 (identical for every expert)
        self.code_cpu = mod.code.detach().to("cpu", torch.float32)
        # gpt-oss per-expert biases, host-resident copies aligned to cold local ids
        # (the parent keeps device-resident copies for its streaming path).
        if self.gptoss:
            ci = self.cold_ids
            self.cc_gu_b = mod.gate_up_bias.detach().index_select(0, ci).to("cpu", torch.float32).contiguous()
            self.cc_dn_b = mod.down_bias.detach().index_select(0, ci).to("cpu", torch.float32).contiguous()

    def _cold_contrib(self, x, flat, row_token, row_slot, cr, top_k_weights, out, dev):
        n1, k1, n2, k2 = self.shapes
        cold_glob = flat.index_select(0, cr).cpu()
        local = self.g2c_cpu.index_select(0, cold_glob)               # [Rc] local cold id per row
        # one activation-sized D2H copy; decoded values round through compute_dtype
        # exactly as the fused kernel sees them, then fp32 math on the host.
        cd = self.mod.compute_dtype if self.mod.compute_dtype is not None else x.dtype
        xr = x.index_select(0, row_token.index_select(0, cr)).to("cpu", torch.float32)

        order = torch.argsort(local)
        sorted_local = local.index_select(0, order)
        xs = xr.index_select(0, order)
        uniq, counts = torch.unique_consecutive(sorted_local, return_counts=True)
        dn_sorted = torch.empty(xs.shape[0], n2, dtype=torch.float32)

        start = 0
        for e_local, cnt in zip(uniq.tolist(), counts.tolist()):
            rows = xs.narrow(0, start, cnt)
            w_gu = self._dequant(self.c_gu_p[e_local], self.c_gu_a[e_local], self.code_cpu, n1, k1)
            w_dn = self._dequant(self.c_dn_p[e_local], self.c_dn_a[e_local], self.code_cpu, n2, k2)
            w_gu = w_gu.to(cd).to(torch.float32)
            w_dn = w_dn.to(cd).to(torch.float32)
            gu = rows @ w_gu.T
            if self.gptoss:
                gu = gu + self.cc_gu_b[e_local]
                gate, up = gu.chunk(2, dim=-1)                        # de-interleaved at load
                gate = gate.clamp(max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                h = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
                dn = h @ w_dn.T + self.cc_dn_b[e_local]
            elif self.has_gate:
                gate, up = gu.chunk(2, dim=-1)
                h = self.act_fn(gate) * up
                dn = h @ w_dn.T
            else:
                dn = self.act_fn(gu) @ w_dn.T
            dn_sorted.narrow(0, start, cnt).copy_(dn)
            start += cnt

        dn_all = torch.empty_like(dn_sorted)
        dn_all.index_copy_(0, order, dn_sorted)                       # back to row order
        w = top_k_weights[row_token.index_select(0, cr), row_slot.index_select(0, cr)].to(torch.float32)
        out.index_add_(0, row_token.index_select(0, cr), dn_all.to(dev) * w[:, None])


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def cold_engine_available(with_hot: bool = False) -> bool:
    """All-cold (``hot_sets`` of empties, ``device="cpu"``) runs anywhere torch
    does. ``with_hot=True`` asks about a non-empty hot side, which needs the
    fused kernel + CUDA exactly like hot residency."""
    if not with_hot:
        return True
    from .hot_residency import hot_residency_available
    return hot_residency_available()


def enable_cold_engine(model, hot_sets: Sequence, device: str = "cuda",
                       dequant: str = "auto", verbose: bool = False) -> int:
    """Partition every eligible ``ExpertsNbit`` under ``model`` into a resident
    device hot-stack + a host-computed cold-stack, in MoE-layer order.

    Semantics mirror :func:`enable_hot_residency` (one ``hot_sets`` entry per
    module in order, skipped layers still consume their entry, re-enable
    rebuilds from current weights, gpt-oss epilogue supported, inference only)
    with one relaxation: the fused ``nf4_grouped`` kernel is required **only if
    any hot set is non-empty** — an all-cold configuration is a pure-host MoE
    and imports nothing beyond torch (+ bitsandbytes when ``dequant="bnb"``).

    ``dequant`` picks the host decode: ``"auto"`` (avx512f-gated bnb, else
    torch), ``"bnb"``, or ``"torch"``. The resolved choice is exposed as
    ``module._cold_engine.dequant_backend``."""
    any_hot = any(len(torch.as_tensor(h).reshape(-1)) for h in hot_sets)
    if any_hot:
        try:
            from nf4_grouped import gemm_4bit_grouped  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "enable_cold_engine with non-empty hot sets runs the hot side on the "
                "fused grouped-GEMM kernel (module nf4_grouped); install it with: "
                "pip install 'experts4bit-qlora[fast]' — or pass all-empty hot_sets "
                "for a pure-host MoE") from e
    _resolve_dequant(dequant)  # fail fast on a bad/broken backend request
    from experts4bit_qlora import Experts4bit, ExpertsNbit

    stock_forwards = {ExpertsNbit.forward, Experts4bit.forward}
    try:
        from experts4bit_qlora.gptoss import GptOssExperts4bit, GptOssExpertsNbit
        stock_forwards |= {GptOssExperts4bit.forward, GptOssExpertsNbit.forward}
    except ImportError:
        pass
    if hasattr(model, "modules"):
        try:
            from experts4bit_qlora.lora import ExpertsLoRA
            wrapped = {id(m.base) for m in model.modules()
                       if isinstance(m, ExpertsLoRA) and hasattr(m, "base")}
        except ImportError:
            wrapped = set()
        all_nbit = [m for m in model.modules() if isinstance(m, ExpertsNbit)]
        mods = [m for m in all_nbit if id(m) not in wrapped]
        if wrapped and not mods:
            raise NotImplementedError(
                "every ExpertsNbit here is an ExpertsLoRA.base (the streaming-loader / "
                "offload path) — see enable_hot_residency; the cold engine has the same "
                "standalone-module support boundary.")
    else:
        mods = [model]
    if len(hot_sets) != len(mods):
        raise ValueError(
            f"hot_sets has {len(hot_sets)} entries but the model has {len(mods)} "
            f"ExpertsNbit modules — exactly one entry per MoE layer in module "
            f"order is required (skipped layers still consume their entry, so "
            f"alignment never silently shifts and trailing entries are never "
            f"dropped)")
    patched = 0
    for i, mod in enumerate(mods):  # hot_sets[i] belongs to mods[i], patched or not
        if type(mod).forward not in stock_forwards and not hasattr(mod, "_e4b_cold_ref"):
            if verbose:
                print(f"[cold_engine] skip {type(mod).__name__}: custom forward")
            continue
        reason = _eligible(mod)
        if reason is not None:
            if verbose:
                print(f"[cold_engine] skip {type(mod).__name__}: {reason}")
            continue
        for ref, name in (("_e4b_fast_ref", "[fast]"), ("_e4b_hot_ref", "hot residency"),
                          ("_e4b_pipe_ref", "pipelined residency")):
            if hasattr(mod, ref):
                if verbose:
                    print(f"[cold_engine] skip {type(mod).__name__}: {name} enabled — disable it first")
                break
        else:
            if hasattr(mod, "_cold_engine"):
                # rebuild every time — a caller may have reloaded a checkpoint; a
                # cached partition must never go stale.
                mod._cold_engine = _ColdEngine(mod, hot_sets[i], device, dequant)
                patched += 1
                continue
            state = _ColdEngine(mod, hot_sets[i], device, dequant)
            mod._e4b_cold_ref = mod.forward
            mod._cold_engine = state

            def _fwd(hidden, top_k_index, top_k_weights, _m=mod):
                st = _m._cold_engine
                cd = _m.compute_dtype if _m.compute_dtype is not None else hidden.dtype
                # fp32 is host-math-only: the fused hot kernel takes bf16/fp16 inputs,
                # so an fp32 compute_dtype rides the engine only on all-cold partitions.
                ok = (torch.bfloat16, torch.float16) if st.hot_ids.numel() else (
                    torch.bfloat16, torch.float16, torch.float32)
                if cd not in ok:
                    return _m._e4b_cold_ref(hidden, top_k_index, top_k_weights)
                if torch.is_grad_enabled() and (
                    hidden.requires_grad or any(p.requires_grad for p in _m.parameters())
                ):
                    return _m._e4b_cold_ref(hidden, top_k_index, top_k_weights)
                return st.forward(hidden, top_k_index, top_k_weights)

            mod.forward = _fwd
            patched += 1
    return patched


def disable_cold_engine(model) -> int:
    """Undo :func:`enable_cold_engine`; returns the number of modules restored."""
    mods = model.modules() if hasattr(model, "modules") else [model]
    restored = 0
    for mod in mods:
        if hasattr(mod, "_e4b_cold_ref") and hasattr(mod, "_cold_engine"):
            mod.forward = mod._e4b_cold_ref
            del mod._e4b_cold_ref, mod._cold_engine
            restored += 1
    return restored
