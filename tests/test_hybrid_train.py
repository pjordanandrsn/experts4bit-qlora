# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Hybrid backward (Phase 5): gradients through the three-tier engine match
a full-precision autograd reference over the SAME dequantized weights, the
Function is checkpoint-by-construction, and inference calls never enter the
training seam."""

import json
import struct

import pytest
import torch

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series")
pytest.importorskip("nvme_residency")
cpu_grouped = pytest.importorskip("cpu_grouped")

from nvme_arena import bake_expert_tensors  # noqa: E402

from experts4bit_qlora import Experts4bit  # noqa: E402
from experts4bit_qlora.engines import hybrid as hy  # noqa: E402
from experts4bit_qlora.engines import hybrid_train as ht  # noqa: E402
from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS  # noqa: E402

E, INTER, H, K = 8, 64, 128, 2

needs_stack = pytest.mark.skipif(
    not hy.hybrid_available(), reason="needs CUDA + gnf4_native CPU kernels"
)


def _st_bytes(tensors):
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _module(seed):
    g = torch.Generator().manual_seed(seed)
    gate_up = torch.randn(E, 2 * INTER, H, generator=g) * 0.05
    down = torch.randn(E, H, INTER, generator=g) * 0.05
    return Experts4bit.from_float(gate_up.to(torch.bfloat16),
                                  down.to(torch.bfloat16), has_gate=True,
                                  activation=torch.nn.functional.silu,
                                  quant_type="nf4",
                                  compute_dtype=torch.bfloat16)


@pytest.fixture()
def one_layer(tmp_path):
    mod = _module(23)
    dt = {torch.uint8: "U8", torch.float32: "F32"}
    tensors = {}
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    payload = {
        "nf4.gate_up_blocks": mod.gate_up_proj.view(E, n1, k1 // 2),
        "nf4.gate_up_absmax": mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
        "nf4.down_blocks": mod.down_proj.view(E, n2, k2 // 2),
        "nf4.down_absmax": mod.down_absmax.view(E, n2, k2 // 64).float(),
    }
    for kind, stack in payload.items():
        for e in range(E):
            t = stack[e].contiguous().cpu()
            tensors[f"model.layers.0.mlp.experts.{e}.{kind}"] = (
                tuple(t.shape), dt[t.dtype], t.numpy().tobytes())
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    arena_path = str(tmp_path / "m.arena")
    bake_expert_tensors(
        str(snap), arena_path,
        name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
        kinds=tuple(NF4_SEGMENTS.values()), align=4096, log=lambda *a: None)
    return torch.nn.ModuleList([mod.to("cuda")]), arena_path


def _manifest(spec):
    tiers = {"vram": [], "dram": [], "nvme": []}
    for tier, ids in spec.items():
        tiers[tier] += [[0, e] for e in ids]
    return {"schema": "e4b-placement/1", "tiers": tiers,
            "masses": {"vram_frac": 0, "dram_frac": 0, "nvme_frac": 0}}


def _dequant_stack(packed, absmax, n_out, k_in):
    """Whole-stack NF4 dequant to fp32 via the oracle (bitsandbytes) — the
    values every bus computes on, bit-identically."""
    import bitsandbytes.functional as F
    from bitsandbytes.functional import QuantState

    g = packed.shape[0]
    state = QuantState(
        absmax=absmax.reshape(-1).float(), shape=torch.Size((g * n_out, k_in)),
        dtype=torch.float32, blocksize=64, quant_type="nf4",
        code=F.get_4bit_type("nf4", device=packed.device))
    return F.dequantize_4bit(packed.reshape(-1, 1), quant_state=state).view(
        g, n_out, k_in)


def _reference_grads(mod, hidden, idx, wts, grad_out, lora=None):
    """Full-precision autograd over the SAME dequantized weights: dequant
    each routed expert to fp32 (bit-identical values to what every branch
    computes on) and run the expert MLP (+ LoRA deltas, when given) and
    weighted combine in plain torch."""
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    wgu = _dequant_stack(mod.gate_up_proj.view(E, n1, k1 // 2),
                         mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
                         n1, k1)
    wdn = _dequant_stack(mod.down_proj.view(E, n2, k2 // 2),
                         mod.down_absmax.view(E, n2, k2 // 64).float(),
                         n2, k2)
    h = hidden.detach().to(torch.float32).requires_grad_()
    w = wts.detach().to(torch.float32).requires_grad_()
    params = [h, w]
    la = lb = lda = ldb = None
    if lora is not None:
        la = lora.gate_up_lora_A.detach().float().requires_grad_()
        lb = lora.gate_up_lora_B.detach().float().requires_grad_()
        lda = lora.down_lora_A.detach().float().requires_grad_()
        ldb = lora.down_lora_B.detach().float().requires_grad_()
        params += [la, lb, lda, ldb]
    out = torch.zeros(h.shape[0], H, dtype=torch.float32, device="cuda")
    for t in range(h.shape[0]):
        for s in range(idx.shape[1]):
            e = int(idx[t, s])
            gu = wgu[e] @ h[t]
            if lora is not None:
                gu = gu + lora.scaling * (lb[e] @ (la[e] @ h[t]))
            gate, up = gu.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            dn = wdn[e] @ act
            if lora is not None:
                dn = dn + lora.scaling * (ldb[e] @ (lda[e] @ act))
            out = out.index_put((torch.tensor([t], device="cuda"),),
                                dn * w[t, s], accumulate=True)
    grads = torch.autograd.grad(out, params,
                                grad_outputs=grad_out.to(torch.float32))
    return grads


@needs_stack
@pytest.mark.parametrize("placement", [
    {"vram": [0, 1, 2], "dram": [3, 4, 5], "nvme": [6, 7]},
    {"vram": [], "dram": list(range(E)), "nvme": []},
    {"vram": list(range(E)), "dram": [], "nvme": []},
])
def test_hybrid_grads_match_full_precision_reference(one_layer, placement):
    model, arena = one_layer
    mod = model[0]
    n = ht.enable_hybrid_train(model, arena, _manifest(placement),
                               hot_rows=E, threads=2)
    assert n == 1
    try:
        torch.manual_seed(5)
        t = 6
        hidden = (torch.randn(t, H, device="cuda", dtype=torch.bfloat16)
                  .requires_grad_())
        idx = torch.randint(0, E, (t, K), device="cuda")
        wts = (torch.rand(t, K, device="cuda", dtype=torch.bfloat16)
               .requires_grad_())
        grad_out = torch.randn(t, H, device="cuda", dtype=torch.bfloat16)

        out = mod(hidden, idx, wts)
        assert out.requires_grad, "training seam did not engage"
        out.backward(grad_out)

        ref_gh, ref_gw = _reference_grads(mod, hidden, idx, wts, grad_out)
        # documented tolerance: the forward computes in bf16 on the GPU
        # buses and fp32 on the DRAM bus; the reference is fp32 end to end.
        # bf16 has ~3 decimal digits — grads agree to that, not better.
        torch.testing.assert_close(hidden.grad.float(), ref_gh,
                                   rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(wts.grad.float(), ref_gw,
                                   rtol=3e-2, atol=3e-2)
    finally:
        ht.disable_hybrid_train(model)


@needs_stack
def test_qlora_adapter_grads_match_reference_across_tiers(one_layer):
    """The directive's parity requirement: hybrid QLoRA grads vs the
    full-GPU full-precision reference, adapters included. B is seeded
    nonzero so the gate_up-A path carries signal (B=0 kills dL/dA)."""
    from experts4bit_qlora.lora import ExpertsLoRA
    model, arena = one_layer
    mod = model[0]
    lora = ExpertsLoRA(mod, r=4, alpha=8, dtype=torch.float32).to("cuda")
    torch.manual_seed(11)
    with torch.no_grad():
        lora.gate_up_lora_B.normal_(std=0.05)
        lora.down_lora_B.normal_(std=0.05)
    wrapped = torch.nn.ModuleList([lora])
    n = ht.enable_hybrid_train(
        wrapped, arena,
        _manifest({"vram": [0, 1, 2], "dram": [3, 4, 5], "nvme": [6, 7]}),
        hot_rows=E, threads=2)
    assert n == 1
    try:
        t = 5
        hidden = (torch.randn(t, H, device="cuda", dtype=torch.bfloat16)
                  .requires_grad_())
        idx = torch.randint(0, E, (t, K), device="cuda")
        wts = (torch.rand(t, K, device="cuda", dtype=torch.bfloat16)
               .requires_grad_())
        grad_out = torch.randn(t, H, device="cuda", dtype=torch.bfloat16)

        out = lora(hidden, idx, wts)
        assert out.requires_grad
        out.backward(grad_out)

        ref = _reference_grads(mod, hidden, idx, wts, grad_out, lora=lora)
        ref_gh, ref_gw, ref_la, ref_lb, ref_lda, ref_ldb = ref
        tol = dict(rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(hidden.grad.float(), ref_gh, **tol)
        torch.testing.assert_close(wts.grad.float(), ref_gw, **tol)
        torch.testing.assert_close(lora.gate_up_lora_A.grad.float(),
                                   ref_la, **tol)
        torch.testing.assert_close(lora.gate_up_lora_B.grad.float(),
                                   ref_lb, **tol)
        torch.testing.assert_close(lora.down_lora_A.grad.float(),
                                   ref_lda, **tol)
        torch.testing.assert_close(lora.down_lora_B.grad.float(),
                                   ref_ldb, **tol)
    finally:
        ht.disable_hybrid_train(wrapped)


@needs_stack
def test_inference_calls_bypass_the_training_seam(one_layer):
    model, arena = one_layer
    ht.enable_hybrid_train(
        model, arena, _manifest({"vram": [0, 1, 2, 3], "dram": [4, 5, 6, 7],
                                 "nvme": []}), hot_rows=E, threads=2)
    try:
        hidden = torch.randn(2, H, device="cuda", dtype=torch.bfloat16)
        idx = torch.randint(0, E, (2, K), device="cuda")
        wts = torch.rand(2, K, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out = model[0](hidden, idx, wts)
        assert out.grad_fn is None
        # the training path recomputes per-projection in fp32 (the serve
        # path is fused) — same math, different rounding: close, not equal
        h2 = hidden.clone().requires_grad_()
        out2 = model[0](h2, idx, wts)
        assert out2.grad_fn is not None
        torch.testing.assert_close(out.float(), out2.detach().float(),
                                   rtol=3e-2, atol=3e-2)
    finally:
        ht.disable_hybrid_train(model)


@needs_stack
def test_train_mode_no_grad_never_enters_the_lora_inline_path(one_layer):
    """The Bugbot HIGH: ExpertsLoRA's delegation predicate refuses under
    model.training, so a no-grad forward routed through it falls into the
    inline path that reads base storage the streaming loader never
    materializes (reentrant checkpointing's outer pass, train()+no_grad
    validation). Routing must key on the ADAPTER's state instead: zero
    adapter -> the fused tier serve bitwise; trained adapter -> the train
    forward's math grad-free."""
    from experts4bit_qlora.lora import ExpertsLoRA
    from experts4bit_qlora.engines.hybrid_train import _train_forward
    model, arena = one_layer
    mod = model[0]
    lora = ExpertsLoRA(mod, r=4, alpha=8, dtype=torch.float32).to("cuda")
    wrapped = torch.nn.ModuleList([lora])
    ht.enable_hybrid_train(
        wrapped, arena,
        _manifest({"vram": [0, 1, 2], "dram": [3, 4, 5], "nvme": [6, 7]}),
        hot_rows=E, threads=2)
    try:
        st = mod._hot_residency
        lora.train()                     # the flag that broke delegation
        hidden = torch.randn(3, H, device="cuda", dtype=torch.bfloat16)
        idx = torch.randint(0, E, (3, K), device="cuda")
        wts = torch.rand(3, K, device="cuda", dtype=torch.bfloat16)

        # zero adapter: must be the fused tier serve path, bitwise
        with torch.no_grad():
            got = lora(hidden, idx, wts)
            want = mod.forward(hidden, idx, wts)
        assert torch.equal(got, want)

        # trained adapter: deltas must still apply — the grad-free train
        # forward, bitwise (and nothing read from base torch storage)
        with torch.no_grad():
            lora.gate_up_lora_B.normal_(std=0.05)
            lora.down_lora_B.normal_(std=0.05)
        lora._delegate_ok = None         # raw .data mutation: manual reset
        with torch.no_grad():
            got2 = lora(hidden, idx, wts)
            want2 = _train_forward(st, lora, hidden, idx, wts)
        assert torch.equal(got2, want2)
        assert not torch.equal(got2, want), "deltas did not apply"
    finally:
        ht.disable_hybrid_train(wrapped)
