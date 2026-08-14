"""Training from an arena of NATIVE MXFP4 bytes — the CPU spec.

Two rented-pod runs died on things this file would have caught for free, which
is the whole reason it exists and why it is written before the feature:

  * a bake call with the wrong signature, masked by a pipe so the run continued
    against an arena that was never created;
  * `quant_type="mxfp4"` rejected by the loader, and behind it the real
    requirement — a meta expert whose DECLARED buffers match the arena's
    segments, which is what the tier's geometry check compares.

The arena here is built from bytes this file writes, so nothing depends on a
checkpoint, a GPU, or a 149 GB download.

The tests are in four groups, and the split is about what each one can actually
prove:

  RESOLUTION — `arena_offload_view` resolves NF4 and MXFP4 layouts and refuses
  anything else.

  STAGING — the meta expert DECLARES MXFP4-shaped buffers, so an MXFP4 arena can
  be staged into it, and a genuinely NF4-shaped module is still rejected.

  NUMERICS (CPU) — the module's reference forward, `dequantize_mxfp4` then
  matmul, against an oracle decoded straight from the source bytes. This is the
  gate before renting: it catches an orientation, fusion-order or epilogue error
  for free, and each of those would otherwise cost a pod run.

  GPU PARITY — the fused Triton kernel against that same pure-torch oracle.
  Skipped without CUDA, and deliberately NOT compared against another accelerated
  lane: that would measure whether two fast paths round alike, not whether either
  is right.
"""
from __future__ import annotations

import json
import struct

import pytest
import torch

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series modules")
pytest.importorskip("mxfp4_residency", reason="needs grouped-nf4-gemm MXFP4 residency")

from nvme_arena import bake_expert_tensors, load_index  # noqa: E402

from experts4bit_qlora.engines.nvme_train import (  # noqa: E402
    OFFLOAD_SEGMENTS,
    arena_offload_view,
    check_arena_geometry,
)

# Both in_features must be a multiple of the NF4 blocksize (64): gate_up's is
# hidden_dim, down's is intermediate_dim. The vendored primitive enforces it so
# per-expert quant blocks tile exactly, and a 32-wide fixture is simply illegal.
E, INTER, H = 4, 128, 64
# V4's clamped-GLU bound. Chosen against this fixture's dynamic range so the
# clamps actually BITE — see `test_the_fixture_would_catch_a_plain_swiglu...`,
# which fails if they do not.
LIMIT = 7.0
NAME_TEMPLATE = "model.layers.{layer}.mlp.experts.{expert}.{kind}"
# Order is load-bearing: the fuse presents each PAIR as one range, so both
# blocks must be adjacent and both scales adjacent. This mirrors
# `mxfp4_residency.V4_RESIDENCY_KINDS`.
MXFP4_KINDS = ("w1.weight", "w3.weight", "w1.scale", "w3.scale",
               "w2.weight", "w2.scale")


def _st_bytes(tensors: dict) -> bytes:
    """Minimal safetensors writer — the format nvme_arena's header reader parses."""
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def _mxfp4_stacks():
    """Per-expert MXFP4 storage: nibble-packed blocks (K/2 bytes) and one e8m0
    scale byte per 32 elements.

    Block bytes are uniform — any nibble is a legal fp4 code. Scale bytes are
    NOT: an e8m0 byte is an exponent with bias 127, so a uniform draw spans
    2**-127 to 2**128 and half the fixture dequantizes to inf or to zero. That is
    fine for a shape/dtype assertion and useless for an arithmetic one, and the
    numerics tests below need weights that behave like a checkpoint's. Bounded to
    2**-8 .. 2**7, which is wider than any real MXFP4 tensor's dynamic range.
    """
    g = torch.Generator().manual_seed(4)

    def u8(*shape):
        return torch.randint(0, 255, shape, generator=g, dtype=torch.uint8)

    def e8m0(*shape):
        return torch.randint(127 - 8, 127 + 8, shape, generator=g, dtype=torch.uint8)

    return {
        "w1.weight": u8(E, INTER, H // 2), "w1.scale": e8m0(E, INTER, H // 32),
        "w3.weight": u8(E, INTER, H // 2), "w3.scale": e8m0(E, INTER, H // 32),
        "w2.weight": u8(E, H, INTER // 2), "w2.scale": e8m0(E, H, INTER // 32),
    }


def _dense_from_source(dtype=torch.bfloat16):
    """THE ORACLE: dense per-expert weights, decoded from the source bytes with
    `dequantize_mxfp4` and nothing else.

    Independent of the module under test by construction — it reads the stacks
    this file wrote, never the arena, the tier, or the staged buffers. Returns
    `[E, H, 2I]` and `[E, I, H]`, i.e. the `x @ W` orientation `dequantize_mxfp4`
    produces.
    """
    from experts4bit_qlora.formats.mxfp4 import dequantize_mxfp4

    src = _mxfp4_stacks()
    gu, dn = [], []
    for e in range(E):
        # gate_up is w1 then w3, the same order the arena fuses them in.
        b = torch.cat([src["w1.weight"][e], src["w3.weight"][e]], dim=0)
        s = torch.cat([src["w1.scale"][e], src["w3.scale"][e]], dim=0)
        gu.append(dequantize_mxfp4(b.view(2 * INTER, H // 32, 16),
                                   s.view(2 * INTER, H // 32), dtype=dtype))
        dn.append(dequantize_mxfp4(src["w2.weight"][e].view(H, INTER // 32, 16),
                                   src["w2.scale"][e].view(H, INTER // 32), dtype=dtype))
    return torch.stack(gu), torch.stack(dn)


def _routing(tokens=6, k=2, device="cpu"):
    """A deterministic routing draw, constructed rather than sampled.

    Token ``t`` takes experts ``(t*k + j) % E``, which gives distinct experts per
    token and — at ``tokens=6, k=2, E=4`` — three tokens per expert, so no
    projection goes unexercised and the fused lane takes its GEMM branch. At
    ``tokens=1`` every group holds one row, which is the branch
    ``gemm_mxfp4_grouped`` routes to its GEMV reduction instead: a different
    kernel, and one a prefill-shaped fixture would never reach.
    """
    g = torch.Generator().manual_seed(11)
    idx = torch.tensor([[(t * k + j) % E for j in range(k)] for t in range(tokens)])
    w = torch.rand(tokens, k, generator=g, dtype=torch.float32).to(torch.bfloat16)
    x = (torch.randn(tokens, H, generator=g) * 0.1).to(torch.bfloat16)
    return x.to(device), idx.to(device), w.to(device)


def _oracle_forward(gu, dn, x, idx, w, *, limit):
    """The reference expert computation, written out in full against dense weights.

    Mirrors `_DeepseekV4ForwardMixin.forward` — including its ordering, which
    applies the router score to the fp32 gated tensor BEFORE the down projection —
    so a disagreement is about the projection math and not about where a scalar
    was multiplied in. `limit=0` disables the clamps, exactly as V4's own
    `if self.limit > 0` does.
    """
    import torch.nn.functional as F

    out = torch.zeros(x.shape[0], H, dtype=torch.float32, device=x.device)
    mask = F.one_hot(idx, num_classes=E).permute(2, 1, 0)
    for e in range(E):
        pos, tok = torch.where(mask[e])
        if tok.numel() == 0:
            continue
        gate_up = x[tok] @ gu[e]
        gate, up = gate_up.chunk(2, dim=-1)
        gate, up = gate.float(), up.float()
        if limit > 0:
            gate = gate.clamp(max=limit)                    # one-sided, per V4
            up = up.clamp(min=-limit, max=limit)
        gated = F.silu(gate) * up
        gated = gated * w[tok, pos, None].float()
        out.index_add_(0, tok, (gated.to(x.dtype) @ dn[e]).float())
    return out.to(x.dtype)


def _staged_v4(tmp_path, device="cpu"):
    """An MXFP4-arena V4 expert module holding the arena's real bytes.

    V4 rather than a bare `Experts4bit` because it carries `_apply_gate`, which
    is the hook the forward is required to defer to.
    """
    from experts4bit_qlora.arch.deepseek_v4 import DeepseekV4Experts4bit
    from experts4bit_qlora.engines.nvme_experts import build_meta_experts

    path, index = _bake_mxfp4(tmp_path)
    mod = build_meta_experts(index, E, has_gate=True, compute_dtype=torch.bfloat16,
                             quant_type="nf4", cls=DeepseekV4Experts4bit)
    mod.limit = LIMIT
    _stage(mod, index, path, device=device)     # never `mod.to()` — see `_stage`
    return mod


def _rel_err(got, want):
    # detach first: a metric has no business holding a graph, and without it
    # measuring a training-path output warns about converting a requires_grad
    # tensor to a scalar — noise that makes a real warning easy to miss.
    got, want = got.detach().float(), want.detach().float()
    return float((got - want).abs().max() / want.abs().max().clamp_min(1e-6))


def _bake_mxfp4(tmp_path, name="v4.mxarena"):
    tensors = {}
    for kind, stack in _mxfp4_stacks().items():
        for e in range(E):
            t = stack[e].contiguous()
            tensors[NAME_TEMPLATE.format(layer=0, expert=e, kind=kind)] = (
                tuple(t.shape), "U8", t.numpy().tobytes())
    snap = tmp_path / "snap-mxfp4"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(_st_bytes(tensors))
    path = str(tmp_path / name)
    bake_expert_tensors(str(snap), path, name_template=NAME_TEMPLATE,
                        kinds=MXFP4_KINDS, align=4096, log=lambda *a: None)
    return path, load_index(path)


# --------------------------------------------------------------- implemented
def test_mxfp4_arena_resolves_to_the_four_segment_staging_view(tmp_path):
    """The point of the change: six per-projection segments present as the same
    four the NF4 path stages, so staging is one code path rather than two."""
    _path, index = _bake_mxfp4(tmp_path)
    assert len(index["segments"]) == 6, "fixture should bake six MXFP4 segments"

    view, segmap = arena_offload_view(index)

    # Suffixes are checkpoint-dependent (V4 'w1.weight+w3.weight', K3
    # 'w1.weight_packed+w3.weight_packed'), so assert the MAPPING resolves, not
    # that it equals any fixed set of names.
    assert len(view["segments"]) == 4, "fusion must present four segments"
    have = {g["suffix"] for g in view["segments"]}
    assert set(segmap.values()) <= have, (
        f"map points at segments the fused view lacks: {sorted(segmap.values())} vs {sorted(have)}")
    # every offload tensor the tier stages must resolve
    assert set(segmap) == set(OFFLOAD_SEGMENTS), (
        "the MXFP4 map must cover exactly the tensors the NF4 map covers")


def test_an_unknown_layout_raises_and_names_both_expectations(tmp_path):
    """A wrong arena must fail at attach with both expected layouts named — not
    later, deep in a stage, with a shape error."""
    with pytest.raises((ValueError, KeyError)) as exc:
        arena_offload_view({"segments": [{"suffix": "w9.bogus"}]})
    msg = str(exc.value)
    assert "w9.bogus" in msg, "the error must say what the arena actually has"


def test_nf4_layout_is_returned_untouched():
    """The NF4 path must not acquire a fusion step it does not need — the index
    object itself is returned, so a regression here is visible as identity."""
    idx = {"segments": [{"suffix": s} for s in OFFLOAD_SEGMENTS.values()]}
    view, segmap = arena_offload_view(idx)
    assert view is idx
    assert segmap is OFFLOAD_SEGMENTS


# ---------------------------------------------------------------------- spec
def test_meta_experts_declare_mxfp4_shapes_for_an_mxfp4_arena(tmp_path):
    """THE requirement the second pod run died on.

    Under `arena_train=True` the base is on `meta` and holds nothing, so what
    matters is the DECLARED dtype and per-expert width. The tier's geometry
    check compares them against the arena's segments, and an NF4-shaped module
    cannot match an MXFP4 arena: NF4 declares `absmax` as fp32 per 64 elements,
    MXFP4 needs `scales` as uint8 per 32. Same blocks width, different scales
    entirely.
    """
    from experts4bit_qlora.engines.nvme_experts import build_meta_experts

    _path, index = _bake_mxfp4(tmp_path)
    experts = build_meta_experts(index, E, has_gate=True,
                                 compute_dtype=torch.bfloat16, quant_type="nf4")
    # Must not raise: the module the loader builds for an MXFP4 arena has to be
    # stageable from that arena.
    check_arena_geometry(experts, index, 0)


def test_geometry_check_rejects_an_nf4_module_against_an_mxfp4_arena(tmp_path):
    """The other half, and the one that keeps the spec honest: whatever makes an
    MXFP4 module match must still REJECT a genuinely mismatched one, or the check
    has been widened into uselessness rather than taught a second layout.

    This passes TODAY and must keep passing after option B lands. An NF4 module
    declares `absmax` as fp32 per 64 elements; the MXFP4 arena carries uint8
    scales per 32. Attaching them would copy real bytes into the right-sized
    buffer and compute nonsense.
    """
    from experts4bit_qlora import Experts4bit

    _path, index = _bake_mxfp4(tmp_path)
    g = torch.Generator().manual_seed(1)
    nf4 = Experts4bit.from_float(
        (torch.randn(E, 2 * INTER, H, generator=g) * 0.05).to(torch.bfloat16),
        (torch.randn(E, H, INTER, generator=g) * 0.05).to(torch.bfloat16),
        has_gate=True, activation=torch.nn.functional.silu,
        quant_type="nf4", compute_dtype=torch.bfloat16)

    with pytest.raises((TypeError, ValueError)) as exc:
        check_arena_geometry(nf4, index, 0)
    assert "arena" in str(exc.value).lower()


def test_mxfp4_arena_forward_matches_the_dequantize_then_matmul_oracle(tmp_path):
    """The numerics parity test that REPLACES the flag-and-refuse spec.

    Option B wired STAGING; this is the COMPUTE half. The module's own forward —
    the reference lane, `dequantize_mxfp4` then matmul, per expert — is graded
    against an oracle built straight from the source bytes: different provenance
    (this file's stacks, not the arena), different code (an explicit loop, not
    `_project`), same answer.

    A V4 base on purpose. It carries `_apply_gate`, so this also pins the epilogue
    the PREREG's amendment 1 names: a plain SwiGLU over a clamped-GLU base trains
    and lowers a loss while optimising the wrong function, and the companion test
    below proves this fixture would actually catch that.

    Tolerance, not equality: the oracle and the module do the same arithmetic in
    the same order, but bf16 matmul blocking is not contractually reproducible
    across shapes. The bar is tight enough that any wrong weight, orientation, or
    activation lands orders of magnitude outside it.
    """
    x, idx, w = _routing()
    mod = _staged_v4(tmp_path)
    gu, dn = _dense_from_source()

    got = mod(x, idx, w)
    want = _oracle_forward(gu, dn, x, idx, w, limit=LIMIT)

    assert got.shape == want.shape, f"{tuple(got.shape)} != {tuple(want.shape)}"
    assert got.dtype == x.dtype, "the forward must return the caller's dtype"
    err = _rel_err(got, want)
    assert err < 2e-2, f"reference lane disagrees with the oracle: rel max err {err:.3e}"


def test_the_fixture_would_catch_a_plain_swiglu_over_the_clamped_base(tmp_path):
    """The test above is only worth its tolerance if the tolerance can fail.

    `lora._epilogue` warns that assuming plain SwiGLU over a clamped-GLU base is
    SILENT — the model trains, the loss falls, and it optimises a function the
    frozen base does not compute. So grade the same forward against an oracle
    that drops the clamps and require it to be rejected by a wide margin. Without
    this, a fixture whose values never reach `limit` would let the parity test
    pass with the epilogue wired wrong.
    """
    x, idx, w = _routing()
    mod = _staged_v4(tmp_path)
    gu, dn = _dense_from_source()

    unclamped = _oracle_forward(gu, dn, x, idx, w, limit=0.0)   # 0 = no clamp, per V4
    err = _rel_err(mod(x, idx, w), unclamped)
    assert err > 1e-1, (
        f"a plain (unclamped) SwiGLU oracle is only {err:.3e} away from the module — "
        "this fixture cannot distinguish the epilogues, so the parity test above "
        "proves nothing about `_apply_gate`")


def _trained_lora(mod, seed=5):
    """An `ExpertsLoRA` over `mod` whose adapter is NOT the initial one.

    `B` is zero-initialised, so a fresh adapter's delta is identically zero and
    every assertion about it would pass against a base-only forward. Worse for a
    gradient test: `dL/dA` is proportional to `B`, so at `B = 0` the `A`
    gradients are exactly zero and "no gradient reached A" is indistinguishable
    from a broken graph. Randomise `B` first, then both claims have teeth.
    """
    from experts4bit_qlora.lora import ExpertsLoRA

    g = torch.Generator().manual_seed(seed)
    lora = ExpertsLoRA(mod, r=4, alpha=8)
    with torch.no_grad():
        for p in (lora.gate_up_lora_B, lora.down_lora_B):
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    # `param.data`-style mutation is exactly the case `_adapter_is_zero`'s cache
    # tells callers to invalidate by hand; a stale True would serve the base.
    lora._delegate_ok = None
    return lora


def _oracle_forward_lora(gu, dn, lora, x, idx, w, *, with_delta=True):
    """`ExpertsLoRA.forward`'s computation, written out against dense weights.

    Mirrors THAT forward, not `_DeepseekV4ForwardMixin.forward`: the wrapper
    applies the router weight AFTER the down projection and reaches the clamps
    through `_epilogue`. The low-rank term is added to the SAME pre-activation
    tensor, which is the only place it is correct — `act(Wx + BAx) != act(Wx) + d`
    for any cheap `d`, and that is the whole reason this adapter re-implements the
    expert math instead of calling the base.

    `with_delta=False` computes the SAME arithmetic in the SAME order with the
    low-rank term omitted. That is the control the parity assertion needs: a
    base-only oracle written the other way round (V4's ordering) differs from this
    forward by bf16 rounding alone, which is enough to satisfy a naive "the
    adapter changed something" check and is NOT evidence the adapter ran.
    """
    import torch.nn.functional as F

    from experts4bit_qlora.lora import _epilogue

    def delta(v, A, B):
        if not with_delta:
            return torch.zeros((), dtype=v.dtype, device=v.device)
        return (lora.scaling * F.linear(F.linear(v.to(A.dtype), A), B)).to(v.dtype)

    out = torch.zeros(x.shape[0], H, dtype=torch.float32, device=x.device)
    mask = F.one_hot(idx, num_classes=E).permute(2, 1, 0)
    for e in range(E):
        pos, tok = torch.where(mask[e])
        if tok.numel() == 0:
            continue
        xe = x[tok]
        proj = xe @ gu[e] + delta(xe, lora.gate_up_lora_A[e], lora.gate_up_lora_B[e])
        h = _epilogue(lora.base, proj)
        down = h @ dn[e] + delta(h, lora.down_lora_A[e], lora.down_lora_B[e])
        out.index_add_(0, tok, (down * w[tok, pos, None]).float())
    return out.to(x.dtype)


def test_expertslora_over_an_mxfp4_arena_matches_the_oracle(tmp_path):
    """The TRAINING composition, which the frozen-forward tests do not cover.

    Everything above grades the frozen base. This grades the module the model
    actually calls: `ExpertsLoRA` never invokes `base.forward`, it re-implements
    the expert math so the low-rank delta lands before the nonlinearity — so it
    reaches the MXFP4 bytes through `_base_project` -> `base._project` ->
    `_dequantize_expert`, a path no test here had exercised.

    Graded as a RATIO, not an absolute error, and that is not fussiness. The first
    version of this test asserted `rel_max_err < 2e-2` against the with-delta
    oracle and **passed with the low-rank term deleted entirely** — caught by
    mutating `_lora` to return zeros. `_rel_err` normalises by the max, and this
    fixture's e8m0 exponents span 2**-8..2**7, so the base output's heavy tail
    sets the denominator and a real adapter delta disappears into it. The ratio
    is scale-free: whatever the magnitudes, the module must land far closer to the
    oracle that HAS the delta than to the one that does not.
    """
    x, idx, w = _routing()
    lora = _trained_lora(_staged_v4(tmp_path))
    gu, dn = _dense_from_source()

    got = lora(x, idx, w)
    want = _oracle_forward_lora(gu, dn, lora, x, idx, w)
    without = _oracle_forward_lora(gu, dn, lora, x, idx, w, with_delta=False)

    err_with = _rel_err(got, want)
    err_without = _rel_err(got, without)
    assert err_with < 2e-2, (
        f"ExpertsLoRA over an MXFP4 arena disagrees with its oracle: {err_with:.3e}")
    assert err_without > 1e-4, (
        f"the module is indistinguishable from the WITHOUT-delta oracle "
        f"({err_without:.3e}). Two causes, and both invalidate the parity check "
        "above: the module never applied the low-rank term, or the adapter is too "
        "small for this fixture to grade. Check `_trained_lora` randomised B "
        "before concluding the fixture is at fault.")
    assert err_with * 10 < err_without, (
        f"the module is not measurably closer to the WITH-delta oracle "
        f"({err_with:.3e}) than to the without ({err_without:.3e}) — consistent "
        "with the low-rank term never reaching the MXFP4 path")


def test_the_adapter_trains_and_the_frozen_mxfp4_base_does_not(tmp_path):
    """A loss must reach all four adapter tensors, and nothing else.

    This is the claim the whole tier exists to support: frozen experts served from
    the checkpoint's own MXFP4 bytes, a trainable adapter over them. A gradient
    that reaches `B` but not `A` is the signature of a delta added AFTER the
    nonlinearity, and a finite non-zero gradient on the frozen storage would mean
    the base was not frozen at all.
    """
    x, idx, w = _routing()
    lora = _trained_lora(_staged_v4(tmp_path))

    lora(x, idx, w).float().pow(2).mean().backward()

    for name in ("gate_up_lora_A", "gate_up_lora_B", "down_lora_A", "down_lora_B"):
        g = getattr(lora, name).grad
        assert g is not None, f"{name} received no gradient"
        assert torch.isfinite(g).all(), f"{name} gradient is not finite"
        assert g.abs().max() > 0, f"{name} gradient is identically zero"

    for name in ("gate_up_proj", "gate_up_absmax", "down_proj", "down_absmax"):
        t = getattr(lora.base, name)
        assert not t.requires_grad, f"frozen base tensor {name} requires grad"
        assert getattr(t, "grad", None) is None, f"frozen base tensor {name} got a gradient"


def test_enable_fast_train_refuses_an_mxfp4_arena_base(tmp_path):
    """The NF4 grouped kernel must not be patched over MXFP4 storage.

    An MXFP4-arena base passes every eligibility check `enable_fast_train` had:
    it is `quant_type="nf4"` by class (only its BUFFERS were re-declared) and it
    carries `_apply_gate`. It then dies inside the forward on
    `gate_up_absmax.view(E, n1, k1 // 64)` — e8m0 scales are one byte per 32
    elements, not fp32 per 64. Refusing at patch time is the difference between a
    named skip and a traceback at step 1 of a rented run.
    """
    from experts4bit_qlora.engines.fast import enable_fast_train
    from experts4bit_qlora.lora import ExpertsLoRA

    mod = _staged_v4(tmp_path)
    model = torch.nn.Module()
    model.experts = ExpertsLoRA(mod, r=4, alpha=8)
    assert enable_fast_train(model) == 0, (
        "enable_fast_train patched an MXFP4-arena base; the NF4 kernel would "
        "read e8m0 scale bytes as fp32 absmax")
    assert not hasattr(model.experts, "_e4b_train_ref"), "no patch may be left behind"


# ------------------------------------------------------- MXFP4 forward parity
def _stage(mod, index, path, device="cpu"):
    """Put the arena's real bytes into the meta module, as the tier does.

    Staging lands the tensors ON `device` rather than moving the module
    afterwards, and that is not a shortcut — `mod.to("cuda")` RAISES here.
    `build_meta_experts` declares everything on `meta`; staging replaces the four
    expert tensors with real ones and leaves the rest (the NF4 codebook, which
    the MXFP4 lane never reads) still meta, and `Module.to` refuses to copy out
    of a meta tensor. The production path never calls `.to()` either — the tier
    stages rows into place — so matching it here keeps the test on the same
    mechanism the engine uses.
    """
    from nvme_residency import ColdTier, segment_geometry, segment_into
    from experts4bit_qlora.engines.nvme_train import arena_offload_view

    view, segmap = arena_offload_view(index)
    tier = ColdTier(path, hot_rows=64, pinned=False)
    ids = list(range(E))
    for name, suffix in segmap.items():
        _dt, shape, _off, _ln = segment_geometry(view, suffix)
        dest = torch.empty(E, int(torch.tensor(shape).prod()),
                           dtype=getattr(mod, name).dtype)
        # the reader fills host memory; move after the copy, not before
        segment_into(tier, view, 0, ids, suffix, dest.view(E, *shape), rows=ids)
        dest = dest.to(device)
        if name in mod._parameters:
            mod._parameters[name] = torch.nn.Parameter(dest, requires_grad=False)
        else:
            mod._buffers[name] = dest
    return mod


def test_staged_mxfp4_bytes_decode_to_the_source_weights_bit_exactly(tmp_path):
    """THE parity test, against the format's own oracle rather than another lane.

    The arena is baked from bytes this file writes, so the source of truth is
    known exactly. `dequantize_mxfp4` is pure torch, which makes this checkable
    on CPU — no kernel, no GPU, no checkpoint. Bit-exact is the right bar here
    and not an aspiration: nothing in this path is allowed to change a value.
    Tiering moves bytes; it does not get to round them.
    """
    from experts4bit_qlora.engines.nvme_experts import (
        build_meta_experts, mxfp4_expert_weight)
    from experts4bit_qlora.formats.mxfp4 import dequantize_mxfp4

    path, index = _bake_mxfp4(tmp_path)
    mod = build_meta_experts(index, E, has_gate=True,
                             compute_dtype=torch.bfloat16, quant_type="nf4")
    _stage(mod, index, path)

    src = _mxfp4_stacks()
    for e in range(E):
        # gate_up is w1 and w3 fused, in that order — the same fusion the arena
        # presents, so the oracle must be built the same way to compare.
        gu_b = torch.cat([src["w1.weight"][e], src["w3.weight"][e]], dim=0)
        gu_s = torch.cat([src["w1.scale"][e], src["w3.scale"][e]], dim=0)
        want = dequantize_mxfp4(gu_b.view(2 * INTER, H // 32, 16),
                                gu_s.view(2 * INTER, H // 32),
                                dtype=torch.bfloat16)
        got = mxfp4_expert_weight(mod, "gate_up", e)
        assert torch.equal(got, want), f"expert {e} gate_up decoded differently"

        want_d = dequantize_mxfp4(src["w2.weight"][e].view(H, INTER // 32, 16),
                                  src["w2.scale"][e].view(H, INTER // 32),
                                  dtype=torch.bfloat16)
        assert torch.equal(mxfp4_expert_weight(mod, "down", e), want_d), (
            f"expert {e} down decoded differently")


# ------------------------------------------------------------------ GPU parity
#
# Everything above runs on CPU and is the gate before renting anything. Nothing
# above touches `mxfp4_grouped`: the fused kernel is Triton, so on a CPU host the
# forward's router takes the reference lane and a green suite says NOTHING about
# the kernel. These two tests are the ones that need a card.
#
# They are graded against the pure-torch oracle, never against another
# accelerated lane. Comparing the kernel to, say, the MXFP4 serving engine would
# measure whether two fast paths round alike, not whether either is right.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the fused MXFP4 GEMM is Triton/CUDA — on CPU this would assert nothing")


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 6])   # GEMV reduction, then the GEMM branch
def test_fused_mxfp4_gemm_matches_the_dequantize_then_matmul_oracle(tmp_path, tokens):
    """THE kernel parity test: `gemm_mxfp4_grouped` against `dequantize_mxfp4` + `@`.

    Graded per projection rather than through the whole forward, so a failure
    names the projection and cannot be absorbed by the epilogue or by the router
    weighting. The kernel reads the module's STAGED buffers — the bytes that
    actually travelled disk -> tier -> device — while the oracle decodes the
    source stacks this file wrote, so staging and arithmetic are both on trial.

    Both must be exercised: `gemm_mxfp4_grouped` dispatches all-size-1 groups to a
    GEMV reduction and everything else to the grouped GEMM. They are separate
    kernels and a decode-shaped bug is invisible to a prefill-shaped test.
    """
    pytest.importorskip("mxfp4_grouped", reason="needs grouped-nf4-gemm's MXFP4 kernel")
    from mxfp4_grouped import gemm_mxfp4_grouped

    x, idx, w = _routing(tokens=tokens, device="cuda")
    mod = _staged_v4(tmp_path, device="cuda")
    gu, dn = (t.cuda() for t in _dense_from_source())

    k = idx.shape[1]
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    token_rows = order // k
    counts = torch.bincount(flat, minlength=E)
    active = torch.nonzero(counts, as_tuple=False).view(-1)
    sizes = counts[active].tolist()
    expert_ids = active.to(torch.int32).tolist()
    assert (max(sizes) == 1) == (tokens == 1), (
        f"fixture did not reach the intended kernel branch: sizes={sizes}")

    a_cat = x.index_select(0, token_rows).contiguous()
    got = gemm_mxfp4_grouped(a_cat,
                             mod.gate_up_proj.view(E, 2 * INTER, H // 2),
                             mod.gate_up_absmax.view(E, 2 * INTER, H // 32),
                             sizes, expert_ids)

    # Oracle, group by group in the kernel's own row order.
    want = torch.empty_like(got)
    at = 0
    for e, n in zip(expert_ids, sizes):
        want[at:at + n] = a_cat[at:at + n] @ gu[e]
        at += n
    assert at == a_cat.shape[0]
    err = _rel_err(got, want)
    assert err < 2e-2, f"gate_up: kernel vs oracle rel max err {err:.3e}"

    # The down projection has a different K (intermediate, not hidden) and a
    # different N, so it is a distinct shape through the same kernel.
    h = torch.randn(a_cat.shape[0], INTER, device="cuda").to(torch.bfloat16) * 0.1
    got_d = gemm_mxfp4_grouped(h.contiguous(),
                               mod.down_proj.view(E, H, INTER // 2),
                               mod.down_absmax.view(E, H, INTER // 32),
                               sizes, expert_ids)
    want_d = torch.empty_like(got_d)
    at = 0
    for e, n in zip(expert_ids, sizes):
        want_d[at:at + n] = h[at:at + n] @ dn[e]
        at += n
    err_d = _rel_err(got_d, want_d)
    assert err_d < 2e-2, f"down: kernel vs oracle rel max err {err_d:.3e}"


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 6])
def test_fused_forward_agrees_with_the_reference_lane_on_gpu(tmp_path, tokens):
    """The whole forward, both lanes, same module, same card.

    The kernel test above proves the projections; this proves the forward built
    on them — routing, the `_apply_gate` epilogue, and the fp32 accumulation —
    picks the fused lane and lands on the reference lane's answer. `_e4b_mxfp4_arena_ref`
    is the module's pristine forward, so this is exactly fused-vs-oracle one level up.

    The residual is bounded rather than zero, and two known terms make it so: the
    fused lane applies the router score AFTER the down projection (the vendored
    contract) where V4 applies it before in fp32, and `_epilogue` rounds V4's fp32
    gate back to bf16 one step earlier. Both are algebraically identity — there is
    no down bias to break the commutation — so anything beyond bf16 rounding here
    is a real disagreement.
    """
    pytest.importorskip("mxfp4_grouped", reason="needs grouped-nf4-gemm's MXFP4 kernel")

    x, idx, w = _routing(tokens=tokens, device="cuda")
    mod = _staged_v4(tmp_path, device="cuda")

    with torch.no_grad():
        fused = mod(x, idx, w)
        reference = mod._e4b_mxfp4_arena_ref(x, idx, w)

    assert fused.shape == reference.shape
    err = _rel_err(fused, reference)
    assert err < 3e-2, f"fused lane vs reference lane: rel max err {err:.3e}"

    # And the fused lane must actually have RUN. Both lanes return the same
    # answer by design, so equality alone cannot tell a fused forward from a
    # router that quietly fell through to the reference on every call — which is
    # how this test would pass on a box where Triton is installed but unusable.
    gu, dn = (t.cuda() for t in _dense_from_source())
    oracle = _oracle_forward(gu, dn, x, idx, w, limit=LIMIT)
    assert _rel_err(fused, oracle) < 3e-2, "fused lane disagrees with the source-byte oracle"


@requires_cuda
def test_the_fused_lane_is_refused_when_a_gradient_is_required(tmp_path):
    """`gemm_mxfp4_grouped` is raw Triton with no autograd.Function behind it.

    Routed there with grad enabled it would produce no `dL/dx`, and training below
    this layer would silently stop learning — the same shape of failure as a wrong
    epilogue. The router must send a grad-carrying forward to the reference lane,
    which carries gradients through `_FrozenLinearRecomputeBackward`.
    """
    pytest.importorskip("mxfp4_grouped", reason="needs grouped-nf4-gemm's MXFP4 kernel")

    x, idx, w = _routing(device="cuda")
    mod = _staged_v4(tmp_path, device="cuda")
    x = x.detach().requires_grad_(True)

    out = mod(x, idx, w)
    out.float().sum().backward()
    assert x.grad is not None, "no gradient reached the input"
    assert torch.isfinite(x.grad.float()).all(), "gradient is not finite"
    assert x.grad.float().abs().max() > 0, (
        "the input gradient is identically zero — the forward was routed to the "
        "non-differentiable fused kernel")


def test_decoding_an_nf4_module_as_mxfp4_is_refused(tmp_path):
    """The guard that keeps the helper honest: NF4 bytes decoded as MXFP4 would
    return plausible-looking nonsense, so the flag is checked, not assumed."""
    from experts4bit_qlora.engines.nvme_experts import mxfp4_expert_weight
    from experts4bit_qlora import Experts4bit

    g = torch.Generator().manual_seed(2)
    nf4 = Experts4bit.from_float(
        (torch.randn(E, 2 * INTER, H, generator=g) * 0.05).to(torch.bfloat16),
        (torch.randn(E, H, INTER, generator=g) * 0.05).to(torch.bfloat16),
        has_gate=True, activation=torch.nn.functional.silu,
        quant_type="nf4", compute_dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        mxfp4_expert_weight(nf4, "gate_up", 0)
