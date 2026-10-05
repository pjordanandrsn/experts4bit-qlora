"""The paged server's estimate (serve_recipe): the KV pool arithmetic IS what Fp8PagedKV allocates (constructed on CPU),
the topology carries the server's KV geometry, and placements that are not priced are refused in words."""
import pytest

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers")
pytest.importorskip("accelerate")
pytest.importorskip("row_pool", reason="needs grouped-nf4-gemm")
pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm")

from experts4bit_qlora.arch.topology import describe_moe  # noqa: E402
from experts4bit_qlora.engines import fp8_paged_kv  # noqa: E402
from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV  # noqa: E402
from experts4bit_qlora.recipe import QLoRASetup, _module_bytes, _stack_modules  # noqa: E402
from experts4bit_qlora.serve_recipe import (ARENA_ALIGN, BLOCK_TOKENS, ServeSetup,  # noqa: E402
                                             bytes_per_expert, estimate_serve_footprint, paged_kv_pool_bytes,
                                             solver_tiers)


def _allocated(kv) -> int:
    ts = [kv.kp.dev, kv.vp.dev, kv._bt_all, kv.seq_lens]
    return sum(t.numel() * t.element_size() for t in ts)


@pytest.mark.parametrize("L,H,D,batch,T,scratch,kg", [
    (3, 2, 128, 4, 100, 0, None), (2, 4, 64, 1, 16, 16, None), (5, 8, 128, 16, 4096, 16, 4), (1, 1, 256, 2, 17, 0, None),
])
def test_pool_bytes_are_what_fp8_paged_kv_allocates(L, H, D, batch, T, scratch, kg):
    kv = Fp8PagedKV(L, H, D, batch=batch, max_tokens_per_seq=T, k_groups=kg, device="cpu", scratch_slots=scratch)
    assert paged_kv_pool_bytes(L, H, D, batch=batch, max_tokens_per_seq=T, k_groups=kg, scratch_slots=scratch) == _allocated(kv)


def test_per_layer_geometry_matches_too():
    Hs, Ds = [2, 4, 2], [128, 64, 128]
    kv = Fp8PagedKV(3, Hs, Ds, batch=2, max_tokens_per_seq=64, device="cpu")
    assert paged_kv_pool_bytes(3, Hs, Ds, batch=2, max_tokens_per_seq=64) == _allocated(kv)
    with pytest.raises(ValueError):
        paged_kv_pool_bytes(2, Hs, Ds, batch=2, max_tokens_per_seq=64)


def test_block_tokens_is_the_servers():
    assert BLOCK_TOKENS == fp8_paged_kv.BLOCK_TOKENS


def _qwen3():
    return tr.Qwen3MoeConfig(hidden_size=128, intermediate_size=256, moe_intermediate_size=64, num_experts=8,
                             num_experts_per_tok=2, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=32, vocab_size=192, max_position_embeddings=64, decoder_sparse_step=1)


def test_topology_carries_the_servers_kv_geometry():
    topo = describe_moe(_qwen3())
    assert (topo.kv_layers, topo.kv_heads, topo.kv_head_dims) == (3, 2, 32)


def test_estimate_items_and_scaling():
    topo = describe_moe(_qwen3())
    small = estimate_serve_footprint(topo, ServeSetup(max_seqs=1, max_tokens_per_seq=256, graphs=False))
    big = estimate_serve_footprint(topo, ServeSetup(max_seqs=8, max_tokens_per_seq=4096, graphs=True))
    kv = lambda f: next(i for i in f.items if i.name == "FP8 paged KV pool")  # noqa: E731
    assert kv(small).basis == "derived" and kv(big).bytes > 8 * 16 * kv(small).bytes // 2
    assert kv(small).bytes == paged_kv_pool_bytes(3, 2, 32, batch=1, max_tokens_per_seq=256)
    assert {i.name for i in big.items} >= {"frozen expert stacks (all VRAM)", "dense weights (bf16)", "FP8 paged KV pool",
                                            "prefill/decode working set"}
    assert any("CUDA graph" in u for u in big.unmodelled) and not any("CUDA graph" in u for u in small.unmodelled)
    slab = next(i for i in big.items if i.name.startswith("frozen expert stacks"))
    assert slab.bytes == sum(_module_bytes(_stack_modules(st, QLoRASetup())[0]) for st in topo.expert_stacks)  # cache


def test_unknown_placements_and_batched_graphs_under_the_solver_are_refused_in_words():
    topo = describe_moe(_qwen3())
    assert any("placement must be" in r for r in estimate_serve_footprint(topo, ServeSetup(placement="tiers")).refusals)
    f = estimate_serve_footprint(topo, ServeSetup(placement="solver", max_seqs=4, graphs=True))
    assert f.items == () and any("all-vram placement" in r for r in f.refusals)


def test_solver_tiers_fill_vram_then_dram_then_nvme():
    mib = 2**20
    assert solver_tiers(2, 10, mib, vram_gb=5 / 1024, dram_gb=7 / 1024) == {"vram": 5, "dram": 7, "nvme": 8}
    assert solver_tiers(2, 10, mib, vram_gb=1.0, dram_gb=1.0) == {"vram": 20, "dram": 0, "nvme": 0}


def test_without_a_profile_the_bandwidths_decide_nothing():
    from experts4bit_qlora.engines.placement import solve_placement
    for b_vram, b_dram in ((500.0, 20.0), (20.0, 500.0), (1.0, 1.0)):
        man = solve_placement(n_layers=3, n_experts=8, bytes_per_expert=2**20, vram_budget_bytes=5 * 2**20,
                              dram_budget_bytes=9 * 2**20, calibration={}, profile_path=None,
                              b_vram_override=b_vram, b_dram_override=b_dram, batch=1)
        assert {t: len(v) for t, v in man["tiers"].items()} == solver_tiers(3, 8, 2**20, 5 / 1024, 9 / 1024)


def test_solver_estimate_splits_the_slab_across_tiers():
    topo = describe_moe(_qwen3())
    allv = estimate_serve_footprint(topo, ServeSetup(max_seqs=1, graphs=False))
    slab = next(i.bytes for i in allv.items if i.name.startswith("frozen expert stacks"))
    rows, bpe = len(topo.expert_stacks) * topo.expert_stacks[0].n_experts, bytes_per_expert(topo.expert_stacks[0])
    f = estimate_serve_footprint(topo, ServeSetup(placement="solver", max_seqs=1, graphs=False,
                                                  vram_gb=5 * bpe / 2**30, dram_gb=7 * bpe / 2**30, hot_rows=4))
    assert not f.refusals
    by = {i.name: i for i in f.items}
    vram, dram, nvme = (by["expert stacks, VRAM tier"], by["expert stacks, DRAM tier (computed on the CPU)"],
                        by["expert rows on NVMe (read through the cold tier)"])
    assert (vram.where, dram.where, nvme.where) == ("device", "host", "nvme")
    assert (vram.bytes, dram.bytes) == (5 * bpe, 7 * bpe) and vram.bytes + dram.bytes + nvme.bytes == rows * bpe
    assert 0 <= slab - rows * bpe < 1024 * len(topo.expert_stacks)   # only the stacks' per-stack constants remain
    assert by["cold view (rows land here)"].bytes == 4 * bpe
    assert any("CPU tier" in u for u in f.unmodelled)


def test_the_hybrid_tier_host_buffers_are_priced_at_either_placement():
    from nvme_residency import pinned_request_cost
    topo = describe_moe(_qwen3())
    f = estimate_serve_footprint(topo, ServeSetup(max_seqs=1, graphs=False, hot_rows=64))
    by = {i.name: i for i in f.items}
    bpe = bytes_per_expert(topo.expert_stacks[0])
    stride = -(-bpe // ARENA_ALIGN) * ARENA_ALIGN
    assert by["cold tier landing (pinned)"].bytes == pinned_request_cost(64 * stride)
    assert by["setup tier (while the stacks are built)"].bytes == 64 * stride
    assert "cold view (rows land here)" not in by                  # nothing lives on NVMe at all-VRAM


def test_to_env_is_what_the_server_reads_back(monkeypatch):
    from experts4bit_qlora.serve_paged import PagedServeConfig

    monkeypatch.setenv("E4B_PAGED_DEVICE", "cpu")      # host-independent: no GPU facts enter from_env
    for st in (ServeSetup(), ServeSetup(max_seqs=3, max_tokens_per_seq=777, chunk_tokens=128, graphs=False,
                                        buckets=(1, 2), kv_groups=4, prefill_graph="0", vram_gb=2.5, dram_gb=0.75,
                                        hot_rows=128, placement="solver")):
        for k, v in st.to_env().items():
            monkeypatch.setenv(k, v)
        cfg = PagedServeConfig.from_env()
        assert {f: getattr(cfg, f) for f in ("placement", "max_seqs", "max_tokens_per_seq", "chunk_tokens", "graphs")} \
            == {f: getattr(st, f) for f in ("placement", "max_seqs", "max_tokens_per_seq", "chunk_tokens", "graphs")}
        assert tuple(cfg.buckets) == tuple(st.buckets) and str(cfg.kv_groups) == str(st.kv_groups)
        assert cfg.prefill_graph == st.prefill_graph
        assert (cfg.vram_gb, cfg.dram_gb, cfg.hot_rows) == (st.vram_gb, st.dram_gb, st.hot_rows)


def test_the_prefill_graph_pool_is_named_where_the_graph_can_engage():
    topo = describe_moe(_qwen3())
    named = lambda f: any("prefill graph" in u for u in f.unmodelled)  # noqa: E731
    assert named(estimate_serve_footprint(topo, ServeSetup(max_seqs=4)))                      # the server's default
    assert not named(estimate_serve_footprint(topo, ServeSetup(max_seqs=4, prefill_graph="0")))
    assert not named(estimate_serve_footprint(topo, ServeSetup(max_seqs=4, graphs=False)))   # needs device grouping
    assert not named(estimate_serve_footprint(topo, ServeSetup(max_seqs=1)))
    assert estimate_serve_footprint(topo, ServeSetup(prefill_graph="on")).refusals


def test_bytes_per_expert_is_the_arena_row_not_a_share_of_the_stack():
    """The bake writes per expert: gate_up and down packed 4-bit plus one fp32 absmax per 64 weights. The stack also
    holds per-stack constants, which a slab / n_experts share would smear across every row (OLMoE: 3538945 vs 3538944)."""
    topo = describe_moe(_qwen3())
    st = topo.expert_stacks[0]
    gate_up, down = 2 * st.intermediate * st.hidden, st.hidden * st.intermediate
    row = gate_up // 2 + gate_up // 64 * 4 + down // 2 + down // 64 * 4
    assert bytes_per_expert(st) == row
    f = estimate_serve_footprint(topo, ServeSetup(placement="solver", max_seqs=1, graphs=False,
                                                  vram_gb=3 * row / 2**30, dram_gb=0.0))
    assert next(i.bytes for i in f.items if i.name == "expert stacks, VRAM tier") == 3 * row
