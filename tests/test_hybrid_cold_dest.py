# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Stage 3 / workstream 1: cold experts have two execution destinations.

A cold expert's bytes must be read from NVMe either way; the open question
is which engine turns them into a contribution first. These tests pin the
mechanism — the destination rule, its refusals, and the equivalence that
makes the choice safe to make at all — not the policy, which is a threshold
here and stays one until the prereg's gate 2 says otherwise.

The rule and the refusals run anywhere. The end-to-end equivalence needs
CUDA plus the native kernels and is gated accordingly.
"""
import importlib.util
import json
import struct

import pytest
import torch

pytest.importorskip("nvme_arena", reason="needs grouped-nf4-gemm N-series")
pytest.importorskip("nvme_residency")
cpu_grouped = pytest.importorskip("cpu_grouped")

from nvme_arena import bake_expert_tensors, load_index  # noqa: E402

from experts4bit_qlora import Experts4bit  # noqa: E402
from experts4bit_qlora.engines import hybrid as hy  # noqa: E402
from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS  # noqa: E402

E, INTER, H, K = 8, 64, 128, 2

needs_stack = pytest.mark.skipif(
    not hy.hybrid_available(), reason="needs CUDA + gnf4_native CPU kernels"
)
# Conditional on the INSTALLED gnf4, the way this suite already gates other
# capability-dependent tests: an older gnf4 satisfies the dependency pin and
# simply skips these rather than failing.
needs_view = pytest.mark.skipif(
    importlib.util.find_spec("cold_cpu_view") is None,
    reason="needs a gnf4 carrying ColdCpuView (the cold CPU destination)",
)


# ------------------------------------------------------ the rule, anywhere --

class _Stub:
    """Only what `_cold_to_cpu` reads. The rule is a pure decision over the
    step's routing and the configured destination; giving it a real module
    would test the harness, not the rule."""

    def __init__(self, dest, view=object()):
        self._cold_dest = hy._parse_cold_dest(dest)
        self.view = view


def _decide(dest, rows_per_expert=1, uniq=2, view=object()):
    flat = torch.arange(uniq).repeat_interleave(rows_per_expert)
    nr = torch.arange(flat.numel())
    return hy._HybridTier._cold_to_cpu(_Stub(dest, view), nr, flat)


def test_default_destination_is_the_pre_stage3_path():
    assert _decide("gpu") is False
    assert _decide("gpu", rows_per_expert=64) is False


def test_explicit_cpu_destination_takes_the_cpu():
    assert _decide("cpu") is True


def test_cpu_destination_without_a_view_falls_back_rather_than_crashing():
    """A gnf4 without ColdCpuView refuses at ENABLE. If one ever reaches the
    hot path with view=None anyway, serving degrades to the GPU path instead
    of raising mid-forward."""
    assert _decide("cpu", view=None) is False


def test_threshold_flips_on_rows_per_unique_expert():
    assert _decide(4.0, rows_per_expert=2) is False     # 2 rows/expert < 4
    assert _decide(4.0, rows_per_expert=4) is True      # at the threshold
    assert _decide(4.0, rows_per_expert=8) is True


def test_threshold_is_rows_per_expert_not_row_count():
    """The statistic is per UNIQUE expert — the same shape the DRAM tier's
    offload_rows uses. Eight rows spread over eight experts is one row each
    and must not trip a threshold of 4."""
    assert _decide(4.0, rows_per_expert=1, uniq=8) is False


@pytest.mark.parametrize("bad", ["dram", "", None, -1, 0, float("nan"), object()])
def test_bad_cold_dest_is_a_named_error(bad):
    with pytest.raises(ValueError, match="cold_dest"):
        hy._parse_cold_dest(bad)


def test_good_cold_dest_round_trips():
    assert hy._parse_cold_dest("gpu") == "gpu"
    assert hy._parse_cold_dest("cpu") == "cpu"
    assert hy._parse_cold_dest(8) == 8.0
    assert hy._parse_cold_dest("2.5") == 2.5


def test_cold_stats_on_an_unpatched_model_is_zeros_not_a_crash():
    st = hy.cold_stats(torch.nn.Linear(4, 4))
    assert st["cold_rows"] == 0 and st["cold_cpu_frac"] == 0.0


# ------------------------------------------------ the equivalence, on a box --

class _Router(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.top_k, self.num_experts, self.hidden_dim = K, E, H
        self.norm_topk_prob = False
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(E, H, generator=g) * 0.3)


class _Block(torch.nn.Module):
    def __init__(self, experts, seed):
        super().__init__()
        self.router = _Router(seed)
        self.experts = experts


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
    mod = _module(11)
    dt = {torch.uint8: "U8", torch.float32: "F32"}
    n1, k1 = mod._gate_up_shape
    n2, k2 = mod._down_shape
    payload = {
        "nf4.gate_up_blocks": mod.gate_up_proj.view(E, n1, k1 // 2),
        "nf4.gate_up_absmax": mod.gate_up_absmax.view(E, n1, k1 // 64).float(),
        "nf4.down_blocks": mod.down_proj.view(E, n2, k2 // 2),
        "nf4.down_absmax": mod.down_absmax.view(E, n2, k2 // 64).float(),
    }
    tensors = {}
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
    model = torch.nn.ModuleList([_Block(mod.to("cuda"), seed=21)])
    return model, arena_path, load_index(arena_path)


def _manifest(spec):
    tiers = {"vram": [], "dram": [], "nvme": []}
    for tier, ids in spec.items():
        tiers[tier] += [[0, e] for e in ids]
    return {"schema": "e4b-placement/1", "tiers": tiers,
            "masses": {"vram_frac": 0, "dram_frac": 0, "nvme_frac": 0}}


def _run(model, idx, wts, hidden):
    with torch.no_grad():
        return model[0].experts(hidden, idx, wts)


@needs_stack
@needs_view
def test_cold_on_cpu_is_bit_identical_to_the_same_expert_placed_in_dram(one_layer):
    """The equivalence that makes the destination safe to choose at runtime.

    A cold expert executed on the CPU and a DRAM-placed expert executed on
    the CPU run the SAME kernel over the SAME packed bytes with the same
    locked summation tree — only the bytes' provenance differs (tier row vs
    setup-time stack). So this is BITWISE, not a tolerance. If it ever needs
    a tolerance, the cold path has stopped reading the bytes it claims to.
    """
    model, path, _ = one_layer
    torch.manual_seed(3)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    wts = torch.rand(1, K, device="cuda", dtype=torch.bfloat16)
    idx = torch.tensor([[2, 3]], device="cuda")

    hy.enable_hybrid_tier(model, path, _manifest(
        {"vram": [0], "dram": [2, 3], "nvme": [1, 4, 5, 6, 7]}), hot_rows=E)
    warm = _run(model, idx, wts, hidden)
    hy.disable_hybrid_tier(model)

    hy.enable_hybrid_tier(model, path, _manifest(
        {"vram": [0], "dram": [], "nvme": [1, 2, 3, 4, 5, 6, 7]}), hot_rows=E,
        cold_dest="cpu")
    cold = _run(model, idx, wts, hidden)
    st = hy.cold_stats(model)
    hy.disable_hybrid_tier(model)

    assert st["cold_rows_cpu"] == 2 and st["cold_rows_gpu"] == 0
    assert torch.equal(cold, warm), (
        "a cold expert computed on the CPU must equal the same expert "
        "computed on the CPU from DRAM — same kernel, same bytes")


@needs_stack
def test_cold_on_gpu_is_bit_identical_to_the_same_expert_placed_in_vram(one_layer):
    """The GPU destination's half of the equivalence, and the one the suite was
    missing when #171 was filed.

    A cold expert executed on the GPU and a VRAM-placed expert executed on the
    GPU run the SAME `_fused_over_stack` over the SAME arena bytes on the same
    device — only the bytes' provenance differs (tier row vs setup-time stack).
    So this is BITWISE, exactly like the CPU destination's test above.

    It is the test that separates the two things #171 could have been. A cold
    gather that mis-indexed an expert, or a router weight applied to the wrong
    row, lands here as an O(1) difference. Rounding does not: a matched
    destination cannot round differently from itself. What #171 measured was a
    cold-GPU arm compared against a control in which those experts sat in the
    DRAM tier and therefore executed on the CPU in fp32 — a cross-placement
    rounding change, which is why the same manifest at `cold_dest="cpu"`
    matched the control and this one did not. `bench/hybrid-g9/issue171/`
    measures that pair at 4.622e-03 relative RMS, and measures DRAM against
    VRAM — no cold path in it at all — at the same 4.622e-03.
    """
    model, path, _ = one_layer
    torch.manual_seed(3)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    wts = torch.rand(1, K, device="cuda", dtype=torch.bfloat16)
    idx = torch.tensor([[2, 3]], device="cuda")

    hy.enable_hybrid_tier(model, path, _manifest(
        {"vram": [0, 2, 3], "dram": [], "nvme": [1, 4, 5, 6, 7]}), hot_rows=E)
    try:
        resident = _run(model, idx, wts, hidden)
    finally:
        hy.disable_hybrid_tier(model)

    hy.enable_hybrid_tier(model, path, _manifest(
        {"vram": [0], "dram": [], "nvme": [1, 2, 3, 4, 5, 6, 7]}), hot_rows=E)
    try:
        cold = _run(model, idx, wts, hidden)
        st = hy.cold_stats(model)
    finally:
        hy.disable_hybrid_tier(model)

    assert st["cold_rows_gpu"] == 2 and st["cold_rows_cpu"] == 0
    assert torch.equal(cold, resident), (
        "a cold expert computed on the GPU must equal the same expert placed "
        "in VRAM — same kernel, same bytes, same device (max abs diff "
        f"{(cold.float() - resident.float()).abs().max().item():.3e})")


@needs_stack
@needs_view
def test_default_still_routes_cold_rows_to_the_gpu(one_layer):
    """Invariant 9: the feature is inert unless asked for."""
    model, path, _ = one_layer
    torch.manual_seed(3)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    wts = torch.rand(1, K, device="cuda", dtype=torch.bfloat16)
    idx = torch.tensor([[2, 3]], device="cuda")
    man = _manifest({"vram": [0], "dram": [], "nvme": [1, 2, 3, 4, 5, 6, 7]})

    hy.enable_hybrid_tier(model, path, man, hot_rows=E)
    base = _run(model, idx, wts, hidden)
    st = hy.cold_stats(model)
    assert st["cold_rows_gpu"] == 2 and st["cold_rows_cpu"] == 0
    assert model[0].experts._hot_residency.view is None, \
        "cold_dest='gpu' must not build a view it will never read"
    hy.disable_hybrid_tier(model)
    assert base is not None


@needs_stack
@needs_view
def test_the_two_destinations_are_not_interchangeable_and_the_gap_is_bf16_scale(
        one_layer):
    """The law the two tests above bracket, stated as a measurement.

    Each destination is bitwise against its OWN matched reference — CPU against
    DRAM, GPU against VRAM — and the two destinations are NOT bitwise against
    each other: the CPU kernels run a locked fp32 tree while the GPU kernel
    lands each grouped GEMM in the compute dtype and runs the SwiGLU epilogue
    there. Choosing `cold_dest` therefore chooses a rounding path, which is why
    it belongs in run identity next to the manifest.

    Asserted from both ends. A floor, because a zero here would mean one of the
    destinations stopped being the engine it claims to be. A ceiling at bf16
    mantissa scale, because anything larger is a real defect and not rounding —
    that is the discrimination #171 needed and did not have.
    """
    model, path, _ = one_layer
    torch.manual_seed(3)
    hidden = torch.randn(8, H, dtype=torch.bfloat16, device="cuda")
    wts = torch.rand(8, K, device="cuda", dtype=torch.bfloat16)
    # every token routes to two DISTINCT cold experts (offset 3 is coprime
    # with 7), so no token's k slots collapse onto one expert
    idx = torch.stack([torch.arange(8) % 7 + 1,
                       (torch.arange(8) + 3) % 7 + 1], dim=1).cuda()
    man = _manifest({"vram": [0], "dram": [], "nvme": [1, 2, 3, 4, 5, 6, 7]})

    hy.enable_hybrid_tier(model, path, man, hot_rows=E, cold_dest="gpu")
    try:
        on_gpu = _run(model, idx, wts, hidden).float()
    finally:
        hy.disable_hybrid_tier(model)

    hy.enable_hybrid_tier(model, path, man, hot_rows=E, cold_dest="cpu")
    try:
        on_cpu = _run(model, idx, wts, hidden).float()
    finally:
        hy.disable_hybrid_tier(model)

    rel = ((on_gpu - on_cpu).pow(2).mean().sqrt()
           / on_cpu.pow(2).mean().sqrt()).item()
    assert rel > 0, ("the destinations came out bit-identical — one of them is "
                     "no longer running the engine it claims to")
    assert rel < 2 ** -5, (
        f"cold_dest gap is {rel:.3e} relative RMS, past bf16 mantissa scale: "
        "that is a defect in a destination, not its rounding path")


@needs_stack
@needs_view
def test_a_repeated_cold_step_reads_the_disk_once(one_layer):
    """The cold CPU path must not re-read bytes the tier already holds — the
    'never duplicate a cold read' clause, measured rather than asserted."""
    model, path, _ = one_layer
    torch.manual_seed(3)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    wts = torch.rand(1, K, device="cuda", dtype=torch.bfloat16)
    idx = torch.tensor([[2, 3]], device="cuda")

    hy.enable_hybrid_tier(model, path, _manifest(
        {"vram": [0], "dram": [], "nvme": [1, 2, 3, 4, 5, 6, 7]}), hot_rows=E,
        cold_dest="cpu")
    try:
        _run(model, idx, wts, hidden)
        after_first = hy.cold_stats(model)["disk_reads"]
        _run(model, idx, wts, hidden)
        assert hy.cold_stats(model)["disk_reads"] == after_first
    finally:
        hy.disable_hybrid_tier(model)
