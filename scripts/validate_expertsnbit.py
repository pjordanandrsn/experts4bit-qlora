"""One-command ExpertsNbit validation report.

Runs the per-scheme correctness contract on synthetic fused-expert stacks — no model downloads,
no pytest — and prints one grep-able line per check:

    build            quantize a [E, 2*inter, hidden] / [E, hidden, inter] stack
    forward_parity   forward relerr vs a float SwiGLU-MoE reference, under the calibrated ceiling
    state_roundtrip  state_dict save/load (strict=True) reproduces the forward, metadata validated
    lora_step        LoRA-over-frozen-base backward + optimizer step: adapters move, packed frozen
    decode_sanity    single-token no_grad decode loop (synthetic single layer — NOT a model tok/s)
    offload_identity offloaded forward matches resident forward; placeholders after evict

plus one global check:

    metadata_guard   an nf4 checkpoint refuses to load into an fp4-built module (same shapes)

SKIP always carries the reason; the summary line and a nonzero exit code report any FAIL.
Big-model validation stays where it was: the manual bench/ scripts and `python -m
experts4bit_qlora.infer` (see README "Benchmarks" / PROVENANCE.md).

Usage: python scripts/validate_expertsnbit.py
"""

import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from experts4bit_qlora import Experts4bit, ExpertsLoRA, ExpertsNbit, enable_expert_offload

MODES = ("nf4", "fp4", "int8", "fp8", "bf16", "fp16")
PER_MODE_CHECKS = ("forward_parity", "state_roundtrip", "lora_step", "decode_sanity", "offload_identity")
# bnb signals a missing/broken quantize backend in several ways (same tuple as the test suite).
_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
E, HID, INTER, TOP_K, N_TOK = 6, 128, 256, 2, 40
DECODE_STEPS = 64
# Forward-parity ceilings — keep in sync with tests/test_reference_parity.py (the source of
# truth; calibrated against CPU + A2000 kernels, bnb 0.49.2, 2026-07-04).
TOL_FWD = {"nf4": 0.25, "fp4": 0.30, "int8": 0.03, "fp8": 0.08, "bf16": 8e-3, "fp16": 1e-3}

_counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}


def _line(status: str, mode: str, check: str, detail: str = "") -> None:
    _counts[status] += 1
    print(f"[{status}] {mode:<5} {check:<16} {detail}")


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).float().norm() / b.float().norm().clamp_min(1e-9)).item()


def _ref_swiglu_moe(hs, idx, wts, gate_up, down):
    """Float SwiGLU-MoE reference in experts4bit's convention (lifted from the parity tests)."""
    out = torch.zeros_like(hs)
    for e in range(gate_up.shape[0]):
        tok, pos = (idx == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        g, u = F.linear(hs[tok], gate_up[e]).chunk(2, dim=-1)
        out.index_add_(0, tok, F.linear(F.silu(g) * u, down[e]) * wts[tok, pos, None])
    return out


def _weights(seed: int):
    torch.manual_seed(seed)
    gate_up = (torch.randn(E, 2 * INTER, HID) * 0.1).to(DEVICE)
    down = (torch.randn(E, HID, INTER) * 0.1).to(DEVICE)
    return gate_up, down


def _cls(mode: str):
    return Experts4bit if mode in ("nf4", "fp4") else ExpertsNbit  # mirror the loader's dispatch


def _header() -> None:
    try:
        from importlib.metadata import version

        ver = version("experts4bit-qlora")
    except Exception:
        import experts4bit_qlora

        ver = getattr(experts4bit_qlora, "__version__", "?")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        r = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
        commit = r.stdout.strip() or "-"
    except Exception:
        commit = "-"
    try:
        import bitsandbytes

        bnb_ver = bitsandbytes.__version__
    except Exception:
        bnb_ver = "-"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"
    cuda = "yes" if torch.cuda.is_available() else "no"
    print(
        f"experts4bit-qlora validate | v{ver} | commit {commit} | torch {torch.__version__} | "
        f"bnb {bnb_ver} | cuda {cuda} | {gpu}"
    )


def _check_mode(mode: str) -> None:
    gate_up, down = _weights(seed=0)
    torch.manual_seed(1)
    hs = torch.randn(N_TOK, HID, device=DEVICE)
    idx = torch.randint(0, E, (N_TOK, TOP_K), device=DEVICE)
    wts = torch.rand(N_TOK, TOP_K, device=DEVICE)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    try:
        base = _cls(mode).from_float(gate_up, down, quant_type=mode, compute_dtype=torch.float32)
    except _QUANTIZE_UNAVAILABLE as e:
        _line("SKIP", mode, "build", f"bitsandbytes {mode} quantize unavailable on {DEVICE}: {type(e).__name__}: {e}")
        for check in PER_MODE_CHECKS:
            _line("SKIP", mode, check, "build skipped")
        return
    _line("PASS", mode, "build", f"{time.perf_counter() - t0:.2f}s")

    # forward_parity — under the calibrated per-scheme ceiling.
    with torch.no_grad():
        got = base(hs, idx, wts)
    err = _relerr(got, _ref_swiglu_moe(hs, idx, wts, gate_up, down))
    status = "PASS" if err < TOL_FWD[mode] else "FAIL"
    _line(status, mode, "forward_parity", f"relerr={err:.4f} (tol {TOL_FWD[mode]})")

    # state_roundtrip — a strict load into a differently-seeded same-config module reproduces the
    # forward bit-for-bit (same packed bytes => same decode), with metadata validated on the way.
    try:
        dst = _cls(mode).from_float(*_weights(seed=7), quant_type=mode, compute_dtype=torch.float32)
        dst.load_state_dict(base.state_dict(), strict=True)
        with torch.no_grad():
            got2 = dst(hs, idx, wts)
        if torch.equal(got2, got):
            _line("PASS", mode, "state_roundtrip", "strict=True, extra_state validated")
        else:
            _line("FAIL", mode, "state_roundtrip", f"forward mismatch after load (relerr={_relerr(got2, got):.2e})")
    except Exception as e:
        _line("FAIL", mode, "state_roundtrip", f"{type(e).__name__}: {e}")

    # lora_step — adapters get gradients and move; the frozen packed storage does not.
    try:
        lora = ExpertsLoRA(base, r=4, alpha=8, dtype=torch.float32).to(DEVICE)
        lora(hs.clone().requires_grad_(True), idx, wts).sum().backward()
        packed_before = base.gate_up_proj.detach().clone()
        b_before = lora.gate_up_lora_B.detach().clone()
        torch.optim.SGD([p for p in lora.parameters() if p.requires_grad], lr=1e-2).step()
        moved = not torch.equal(lora.gate_up_lora_B.detach(), b_before)
        frozen = torch.equal(base.gate_up_proj.detach(), packed_before)
        if moved and frozen:
            _line("PASS", mode, "lora_step", "adapters moved, packed storage frozen")
        else:
            _line("FAIL", mode, "lora_step", f"adapters_moved={moved} packed_frozen={frozen}")
    except Exception as e:
        _line("FAIL", mode, "lora_step", f"{type(e).__name__}: {e}")
        lora = None

    # decode_sanity — synthetic single-layer single-token loop. A smoke number for THIS module
    # only (labels itself as such); model-level tok/s comes from `python -m experts4bit_qlora.infer`.
    try:
        hs1 = torch.randn(1, HID, device=DEVICE)
        idx1 = torch.randint(0, E, (1, TOP_K), device=DEVICE)
        wts1 = torch.rand(1, TOP_K, device=DEVICE)
        module = lora.eval() if lora is not None else base
        with torch.no_grad():
            module(hs1, idx1, wts1)  # warm the path
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(DECODE_STEPS):
                out1 = module(hs1, idx1, wts1)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = f" peak={torch.cuda.max_memory_allocated() / 1e9:.2f}GB" if DEVICE == "cuda" else ""
        if torch.isfinite(out1).all():
            _line("PASS", mode, "decode_sanity", f"{DECODE_STEPS / dt:.0f} tok/s (synthetic single layer){peak}")
        else:
            _line("FAIL", mode, "decode_sanity", "non-finite decode output")
    except Exception as e:
        _line("FAIL", mode, "decode_sanity", f"{type(e).__name__}: {e}")

    # offload_identity — offload changes tensor location, not math; evict leaves placeholders.
    try:
        if lora is None:
            raise RuntimeError("lora_step failed; no module to offload")
        with torch.no_grad():
            ref_out = lora(hs, idx, wts)
        enable_expert_offload(lora, DEVICE, pin=True)
        with torch.no_grad():
            off_out = lora(hs, idx, wts)
        identical = torch.allclose(ref_out, off_out, atol=1e-6, rtol=1e-6)
        evicted = base.gate_up_proj.numel() == 0
        if identical and evicted:
            _line("PASS", mode, "offload_identity", f"max|diff|={(ref_out - off_out).abs().max().item():.1e}")
        else:
            _line("FAIL", mode, "offload_identity", f"identical={identical} evicted_after_forward={evicted}")
    except Exception as e:
        _line("FAIL", mode, "offload_identity", f"{type(e).__name__}: {e}")


def _check_metadata_guard() -> None:
    """nf4 and fp4 pack to byte-identical shapes; the state_dict metadata must catch the swap."""
    try:
        src = _cls("nf4").from_float(*_weights(seed=0), quant_type="nf4", compute_dtype=torch.float32)
        dst = _cls("fp4").from_float(*_weights(seed=7), quant_type="fp4", compute_dtype=torch.float32)
    except _QUANTIZE_UNAVAILABLE as e:
        _line("SKIP", "-", "metadata_guard", f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {type(e).__name__}")
        return
    try:
        dst.load_state_dict(src.state_dict(), strict=True)
        _line("FAIL", "-", "metadata_guard", "nf4 checkpoint loaded into fp4 module without error")
    except ValueError as e:
        _line("PASS", "-", "metadata_guard", f"raised ValueError: {str(e)[:60]}...")
    except Exception as e:
        _line("FAIL", "-", "metadata_guard", f"wrong error type {type(e).__name__}: {e}")


def main() -> int:
    _header()
    for mode in MODES:
        _check_mode(mode)
    _check_metadata_guard()
    code = 1 if _counts["FAIL"] else 0
    print(f"SUMMARY pass={_counts['PASS']} fail={_counts['FAIL']} skip={_counts['SKIP']} -> exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
