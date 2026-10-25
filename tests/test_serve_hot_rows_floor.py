# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_PAGED_HOT_ROWS`` below its hard floor at a tiered placement is refused at startup, by name, never mid-request.

The cold tier fetches every unique NVMe-tier expert one forward routes in ONE ``ColdTier.ensure`` request and protects
each slot from eviction, so a tier with fewer pinned rows raises inside a forward (lane P233's shakedown: solver
placement at the server's defaults asked for 95 unique rows of ``hot_rows`` 64, in its first long prefill). One
forward of this server carries at most ``F = max(chunk_tokens, max_seqs)`` tokens, so a layer asks for at most
``min(F * top_k, its NVMe experts)`` rows, and :func:`serve_paged.hot_rows_floor` is the largest of those.

The floor is tied to the REAL failure, not argued: the hybrid tier is built on CPU through ``enable_hybrid_tier``
(the tier's landing buffer unpinned, which needs no driver, and the fused NF4 kernel stubbed, since what is measured
is the request the tier sees, not the math), and a forward of F tokens that routes every NVMe expert of the binding
layer passes at the floor and raises nvme_residency's own error one row below it. VRAM-hot and DRAM experts never
reach the tier, so they do not count.
"""
import inspect

import pytest
import torch  # noqa: F401

pytest.importorskip("fastapi")
from experts4bit_qlora import serve_paged as sp  # noqa: E402
from experts4bit_qlora.engines import placement as pl  # noqa: E402


def _man(n_layers, per_layer):
    """A manifest from ``{layer: (vram ids, dram ids, nvme ids)}`` on the solver's 0..L-1 layers."""
    tiers = {"vram": [], "dram": [], "nvme": []}
    for lay in range(n_layers):
        v, d, n = per_layer.get(lay, ((), (), ()))
        for t, ids in (("vram", v), ("dram", d), ("nvme", n)):
            tiers[t] += [[lay, e] for e in ids]
    return {"tiers": tiers}


# ------------------------------------------------------------------------------------------------- the floor --
def test_all_vram_has_no_floor():
    man = _man(3, {lay: (range(8), (), ()) for lay in range(3)})
    rec = sp.hot_rows_floor(man, 3, 8, 2, 512, 16)
    assert rec["floor"] == 0 and sp.hot_rows_refusal(0, rec) is None and sp.hot_rows_refusal(64, rec) is None


def test_the_largest_layer_nvme_count_binds_when_f_times_k_is_larger():
    man = _man(3, {0: ([0], [1], [2, 3]), 1: ([0], [], [1, 2, 3, 4, 5, 6, 7]), 2: ([0, 1, 2], [3, 4], [5])})
    rec = sp.hot_rows_floor(man, 3, 8, 2, 512, 16)
    assert rec["floor"] == 7 and rec["worst_layer"] == 1 and rec["worst_layer_nvme_experts"] == 7
    assert rec["tokens_per_forward"] == 512


def test_f_times_k_binds_when_smaller_and_f_is_the_larger_of_chunk_and_max_seqs():
    man = _man(1, {0: ([0], [], list(range(1, 128)))})
    assert sp.hot_rows_floor(man, 1, 128, 8, 4, 2)["floor"] == 32         # F = 4 (the chunk), 4 x 8
    assert sp.hot_rows_floor(man, 1, 128, 8, 4, 9)["floor"] == 72         # F = 9 (decode width), 9 x 8
    assert sp.hot_rows_floor(man, 1, 128, 8, 512, 16)["floor"] == 127     # the layer's 127 NVMe experts


def test_vram_and_dram_experts_do_not_count():
    man = _man(1, {0: (range(4), range(4, 120), range(120, 128))})
    assert sp.hot_rows_floor(man, 1, 128, 8, 512, 16)["floor"] == 8


def test_p233s_numbers_refuse_64_and_accept_128():
    """Qwen3-30B-A3B at solver: E 128, top-k 8, the default chunk 512 and 16 slots; 95 of a layer's 128 off VRAM, all
    of them on NVMe (shake-1's request). The default 64 refuses, naming the knob and every input; 128 (P233 Am. 3) and
    the floor itself pass."""
    man = _man(48, {lay: (range(33), (), range(33, 128)) for lay in range(48)})
    rec = sp.hot_rows_floor(man, 48, 128, 8, 512, 16)
    assert rec["floor"] == 95
    why = sp.hot_rows_refusal(64, rec)
    for want in ("E4B_PAGED_HOT_ROWS=64", "below its floor 95", "E4B_PAGED_CHUNK_TOKENS 512", "max_seqs 16",
                 "top_k 8", "4096 routed rows", "MoE layer 0", "95 of its 128 experts on the NVMe tier",
                 "E4B_PAGED_HOT_ROWS >= 95"):
        assert want in why, (want, why)
    assert sp.hot_rows_refusal(95, rec) is None and sp.hot_rows_refusal(128, rec) is None


CALIB = {"cpu_bench": {"scatter_best": {"gbs": 200.0}}, "gpu_bench": {"devices": [{"b_vram_triad_gbs": 1600.0}]}}


def test_the_floor_on_the_real_solvers_manifest():
    """solve_placement's own output: a budget that leaves experts on NVMe gives the largest per-layer NVMe count (at
    a wide F); the server's all-vram override of the same manifest leaves no floor."""
    L, E, bpe = 4, 16, 10 << 20
    man = pl.solve_placement(n_layers=L, n_experts=E, bytes_per_expert=bpe, vram_budget_bytes=8 * bpe,
                             dram_budget_bytes=24 * bpe, calibration=CALIB, profile_path=None, batch=1, top_k=2)
    per = [sum(1 for lay, _ in man["tiers"]["nvme"] if lay == i) for i in range(L)]
    assert sum(per) == L * E - 32 and max(per) > 0
    assert sp.hot_rows_floor(man, L, E, 2, 512, 16)["floor"] == max(per)
    assert sp.hot_rows_floor(man, L, E, 2, 1, 1)["floor"] == min(2, max(per))
    pairs = sorted(tuple(p) for t in ("vram", "dram", "nvme") for p in man["tiers"][t])
    override = {"tiers": {"vram": [list(p) for p in pairs], "dram": [], "nvme": []}}
    assert sp.hot_rows_floor(override, L, E, 2, 512, 16)["floor"] == 0


def test_build_engine_checks_the_floor_after_the_placement_and_before_the_tier_pins_rows():
    """The order, read from build_engine itself: the solver, the all-vram override, then the floor and its refusal,
    before the arena-id renumbering and before enable_hybrid_tier pins any row; /health records it."""
    src = inspect.getsource(sp.build_engine)
    solve = src.index("man = solve_placement(")
    override = src.index('if cfg.placement == "all-vram":', solve)
    floor = src.index("hot_floor = hot_rows_floor(man, L, E, k, cfg.chunk_tokens, cfg.max_seqs)")
    refuse = src.index("why = hot_rows_refusal(cfg.hot_rows, hot_floor)")
    renumber = src.index("ids = arena_layer_ids(cfg.arena, L)")
    tier = src.index("n = enable_hybrid_tier(")
    assert solve < override < floor < refuse < renumber < tier
    assert 'info["hot_rows_floor"] = {**hot_floor, "hot_rows": cfg.hot_rows}' in src
    # the floor counts one layer's rows: the tier is built without prefetch, which would share the slots with the next
    # layer's speculative rows (hot_rows_floor's docstring)
    call = src[tier:src.index(")", src.index("merged_arena=merge", tier)) + 1]
    assert "prefetch" not in call or "prefetch=False" in call, call
