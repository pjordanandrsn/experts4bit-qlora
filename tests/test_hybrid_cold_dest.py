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


class _StubTier:
    """Only the surface cold_stats reads: a stats() dict."""

    def __init__(self, **kw):
        self._s = kw

    def stats(self):
        return dict(self._s)


def test_cold_stats_forwards_the_reuse_ratio_s_own_denominator():
    """`reuse_before_overwrite` is resurrections / (resurrections +
    reclaimable_overwritten). Forwarding the ratio without its inputs makes
    a receipt that cannot be audited AND actively misleads: a reader
    defaulting the absent key to 0 computes 1.0 and silently disagrees with
    the reported value. logical_evictions cannot substitute -- rows still
    sitting reclaimable have resolved neither way, so it is strictly larger
    than the denominator."""
    m = torch.nn.Linear(4, 4)
    m._e4b_cold_tier = _StubTier(
        resurrections=219, spec_resurrections=0,
        reclaimable_overwritten=3271, logical_evictions=3522,
        reuse_before_overwrite=219 / (219 + 3271),
        protected_rows=96, reclaimable_rows=32,
        evictions=3271, disk_reads=3409, disk_bytes=0)
    st = hy.cold_stats(m)

    for k in ("resurrections", "spec_resurrections",
              "reclaimable_overwritten", "logical_evictions",
              "protected_rows", "reclaimable_rows"):
        assert k in st, f"cold_stats dropped {k}"

    num = st["resurrections"] + st["spec_resurrections"]
    den = num + st["reclaimable_overwritten"]
    assert abs(st["reuse_before_overwrite"] - num / den) < 1e-12
    assert st["logical_evictions"] > den, (
        "unresolved reclaimable rows belong to neither term")


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


# ------------------------------------------- the cold landing selector --

class _FakeTier:
    """Only what build_cold_view touches. A real ColdTier needs an arena on
    disk; the selector's job is choosing a path from GEOMETRY, which is
    exactly what should be testable without one."""

    def __init__(self):
        self.hot_rows = 8
        self.attached = None

    def attach_landing(self, cb):
        self.attached = cb


_ITEMSIZE = {"U8": 1, "BF16": 2, "F16": 2, "F32": 4}


def _index(absmax_dtype="F32", aligned=True):
    """A minimal arena index. shape_per_expert must AGREE with dtype and
    length — a mismatch is a broken fixture, not a geometry the code should
    have to survive, and getting it wrong once already sent this test
    through the fallback path while asserting the fast one."""
    ln = 4096 if aligned else 4097
    segs, off = [], 0
    for k, dt in ((NF4_SEGMENTS["c_gu_p"], "U8"),
                  (NF4_SEGMENTS["c_gu_a"], absmax_dtype),
                  (NF4_SEGMENTS["c_dn_p"], "U8"),
                  (NF4_SEGMENTS["c_dn_a"], absmax_dtype)):
        segs.append({"suffix": k, "dtype": dt, "seg_off": off, "length": ln,
                     "shape_per_expert": [ln // _ITEMSIZE[dt]]})
        off += ln
    return {"align": 4096, "row_stride": off, "row_bytes": off,
            "segments": segs, "n_layers": 1, "n_experts_per_layer": 8}


def test_narrow_absmax_falls_back_and_says_why():
    """A DMA has nowhere to widen bf16 to fp32, so direct is impossible and
    the reason must be recorded, not swallowed."""
    v = hy.build_cold_view(_FakeTier(), _index(absmax_dtype="BF16"),
                           direct=True)
    assert v.e4b_path == "copy"
    assert "widening cast" in v.e4b_fallback_reason


def test_direct_is_recorded_when_it_is_taken():
    t = _FakeTier()
    v = hy.build_cold_view(t, _index(), direct=True)
    assert getattr(v, "e4b_fallback_reason", None) is None, (
        "direct was refused: %s" % getattr(v, "e4b_fallback_reason", None))
    assert v.e4b_path == "direct-scatter"
    assert t.attached is not None, "the tier must receive the landing"


def test_direct_false_forces_the_copy_path_for_an_ab():
    v = hy.build_cold_view(_FakeTier(), _index(), direct=False)
    assert v.e4b_path == "copy"


def test_the_view_is_shared_not_per_module():
    """One tier, one view: a per-module view sized tier.hot_rows allocated
    that much host RAM PER LAYER while only using its own layer's slots."""
    t = _FakeTier()
    a = hy.build_cold_view(t, _index())
    assert t._e4b_cold_view is a
    b = hy.build_cold_view(t, _index())
    assert t._e4b_cold_view is b


# ------------------------------------------------ the deadline rule (WS4) --

def _costs():
    cd = pytest.importorskip("cold_deadline")
    return cd.Costs(cpu_us_fixed=55.0, cpu_us_per_row=2.0, b_dram_gbs=380.1,
                    b_vram_gbs=1574.2, b_link_gbs=28.47,
                    bytes_per_expert=3538944)


class _DeadlineStub:
    """Only what `_cold_to_cpu_deadline` reads. A real _HybridTier needs a
    model and a GPU; the rule itself is arithmetic over row counts and two
    residency masks, which is exactly what should be testable without one."""

    offload_rows = None

    def __init__(self, E, vram, dram, costs):
        self.is_vram = torch.zeros(E, dtype=torch.bool)
        self.is_dram = torch.zeros(E, dtype=torch.bool)
        for e in vram:
            self.is_vram[e] = True
        for e in dram:
            self.is_dram[e] = True
        self.costs = costs
        self._gpu_only = False
        self.dram_thin = False
        self.deadline_log = []
        self.deadline_flips = 0

    _group = hy._HybridTier._group
    _cold_to_cpu_deadline = hy._HybridTier._cold_to_cpu_deadline


def test_deadline_is_a_valid_destination():
    assert hy._parse_cold_dest("deadline") == "deadline"


def test_deadline_without_measured_costs_refuses_by_name():
    """Guessing the constants would put a spec-sheet number into a
    scheduling decision."""
    st = _DeadlineStub(8, [], [], costs=None)
    flat = torch.tensor([0, 1, 2, 3])
    with pytest.raises(RuntimeError, match="measured cost constants"):
        st._cold_to_cpu_deadline(torch.arange(4), flat)


def test_a_busy_gpu_pushes_cold_work_to_the_cpu():
    """Gate 2's shape, through the executor's own accounting: heavy VRAM
    routing this step means the GPU is committed, so the cold group reaches
    the join sooner on the CPU."""
    c = _costs()
    st = _DeadlineStub(8, vram=[0, 1], dram=[], costs=c)
    # 256 rows on the two VRAM experts, 4 cold rows on expert 7
    flat = torch.cat([torch.zeros(128, dtype=torch.long),
                      torch.ones(128, dtype=torch.long),
                      torch.full((4,), 7, dtype=torch.long)])
    nr = torch.arange(256, 260)
    assert st._cold_to_cpu_deadline(nr, flat) is True
    assert st.deadline_log and st.deadline_log[-1]["dest"] == "cpu"


def test_an_idle_gpu_keeps_cold_work_on_the_gpu_for_a_fat_group():
    """The complement: nothing committed on either side, many rows per
    expert, so the GPU's flat-in-rows cost wins and nothing flips."""
    c = _costs()
    st = _DeadlineStub(8, vram=[], dram=[], costs=c)
    flat = torch.full((256,), 7, dtype=torch.long)
    assert st._cold_to_cpu_deadline(torch.arange(256), flat) is False
    assert st.deadline_log[-1]["flipped_by_backlog"] is False


def test_dram_work_counts_as_cpu_backlog_unless_it_is_offloaded():
    """DRAM rows are the CPU's committed work — except when they are being
    routed to the GPU, in which case they are not the CPU's at all."""
    c = _costs()
    flat = torch.cat([torch.zeros(256, dtype=torch.long),
                      torch.full((4,), 7, dtype=torch.long)])
    nr = torch.arange(256, 260)
    busy = _DeadlineStub(8, vram=[], dram=[0], costs=c)
    idle = _DeadlineStub(8, vram=[], dram=[0], costs=c)
    idle.dram_thin = True                    # those rows go to the GPU
    busy._cold_to_cpu_deadline(nr, flat)
    idle._cold_to_cpu_deadline(nr, flat)
    assert busy.deadline_log[-1]["cpu_committed_us"] > 0
    assert idle.deadline_log[-1]["cpu_committed_us"] == 0


def test_every_decision_records_its_counterfactual():
    """A scheduler that logs only its choice cannot be scored."""
    c = _costs()
    st = _DeadlineStub(8, vram=[0], dram=[], costs=c)
    flat = torch.cat([torch.zeros(32, dtype=torch.long),
                      torch.full((4,), 7, dtype=torch.long)])
    st._cold_to_cpu_deadline(torch.arange(32, 36), flat)
    r = st.deadline_log[-1]
    for k in ("dest", "cpu_join_us", "gpu_join_us", "margin_us",
              "flipped_by_backlog", "rows", "uniq", "cpu_committed_us",
              "gpu_committed_us"):
        assert k in r, k


def test_an_empty_cold_group_decides_nothing():
    c = _costs()
    st = _DeadlineStub(8, vram=[], dram=[], costs=c)
    assert st._cold_to_cpu_deadline(torch.empty(0, dtype=torch.long),
                                    torch.empty(0, dtype=torch.long)) is False
    assert st.deadline_log == []


def test_direct_landing_is_refused_when_a_cold_row_could_take_the_gpu():
    """The direct landing and the GPU cold path cannot share a tier: the GPU
    path reads raw rows via tier.row(), which an external-landing tier
    refuses. Only cold_dest="cpu" guarantees no cold row goes to the GPU, so
    every other destination downgrades to the copy path -- recorded, because
    a silent downgrade is a silent regression."""
    for dest, want_direct in (("cpu", True), ("deadline", False),
                              (4.0, False)):
        assert (hy._parse_cold_dest(dest) == "cpu") is want_direct


def test_dram_routed_to_the_gpu_is_charged_to_the_gpu_side():
    """e4b#179. offload_rows can send DRAM to the GPU in the same step. If
    that cost is charged to neither side the estimator sees a busy CPU beside
    an idle GPU that is in fact doing the DRAM work, and can send cold rows
    to the engine already carrying it."""
    c = _costs()
    flat = torch.cat([torch.zeros(256, dtype=torch.long),      # DRAM, fat
                      torch.full((4,), 7, dtype=torch.long)])  # cold
    nr = torch.arange(256, 260)
    st = _DeadlineStub(8, vram=[], dram=[0], costs=c)
    st.offload_rows = 4.0                     # 256 rows / 1 expert >= 4 -> GPU
    st._cold_to_cpu_deadline(nr, flat)
    r = st.deadline_log[-1]
    assert r["cpu_committed_us"] == 0, "DRAM went to the GPU; not CPU work"
    assert r["gpu_committed_us"] > 0, "so it must be charged to the GPU side"


def test_dram_below_the_offload_threshold_stays_cpu_backlog():
    c = _costs()
    flat = torch.cat([torch.zeros(2, dtype=torch.long),
                      torch.full((4,), 7, dtype=torch.long)])
    st = _DeadlineStub(8, vram=[], dram=[0], costs=c)
    st.offload_rows = 4.0                     # 2 rows / 1 expert < 4 -> CPU
    st._cold_to_cpu_deadline(torch.arange(2, 6), flat)
    r = st.deadline_log[-1]
    assert r["cpu_committed_us"] > 0 and r["gpu_committed_us"] == 0
