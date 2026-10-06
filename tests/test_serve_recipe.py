"""The paged server's estimate (serve_recipe): the KV pool arithmetic IS what Fp8PagedKV allocates (constructed on CPU),
the topology carries the server's KV geometry, and placements that are not priced are refused in words."""
from dataclasses import replace

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
                                             bytes_per_expert, estimate_serve_footprint, min_hot_rows,
                                             paged_kv_pool_bytes, prefill_staging_tokens, solver_tiers,
                                             staging_bytes_per_token)


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
                                                  vram_gb=5 * bpe / 2**30, dram_gb=7 * bpe / 2**30, hot_rows=8))
    assert not f.refusals
    by = {i.name: i for i in f.items}
    vram, dram, nvme = (by["expert stacks, VRAM tier"], by["expert stacks, DRAM tier (computed on the CPU)"],
                        by["expert rows on NVMe (read through the cold tier)"])
    assert (vram.where, dram.where, nvme.where) == ("device", "host", "nvme")
    assert (vram.bytes, dram.bytes) == (5 * bpe, 7 * bpe) and vram.bytes + dram.bytes + nvme.bytes == rows * bpe
    assert 0 <= slab - rows * bpe < 1024 * len(topo.expert_stacks)   # only the stacks' per-stack constants remain
    assert by["cold view (rows land here)"].bytes == 8 * bpe
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


def test_a_cold_layer_needs_as_many_rows_as_it_can_route_in_one_step():
    topo = describe_moe(_qwen3())                       # 3 layers x 8 experts, top-2
    row = bytes_per_expert(topo.expert_stacks[0])
    gib = lambda n: n * row / 2**30  # noqa: E731
    nvme_one_layer = dict(placement="solver", max_seqs=1, graphs=False, vram_gb=gib(8), dram_gb=gib(8))
    assert min_hot_rows(topo, ServeSetup(**nvme_one_layer, chunk_tokens=512)) == 8          # all 8 experts of the layer
    assert min_hot_rows(topo, ServeSetup(**nvme_one_layer, chunk_tokens=2)) == 4            # top-2 x 2 tokens
    assert min_hot_rows(topo, ServeSetup(placement="solver", max_seqs=1, graphs=False, vram_gb=gib(20),
                                         dram_gb=gib(2), chunk_tokens=512)) == 2           # only 2 rows are cold
    assert min_hot_rows(topo, ServeSetup(placement="solver", max_seqs=1, graphs=False, vram_gb=1.0)) == 0
    f = estimate_serve_footprint(topo, ServeSetup(**nvme_one_layer, hot_rows=4))
    assert f.items == () and any("hot_rows 4 is below the 8" in r for r in f.refusals)
    assert not estimate_serve_footprint(topo, ServeSetup(**nvme_one_layer, hot_rows=8)).refusals


def test_the_cold_rows_device_stack_is_priced_where_rows_are_cold():
    """Measured on an RTX A2000 (OLMoE, solver 1.2 / 1.5 GiB): the whole gap between this estimate and the allocator
    peak was hot_residency._cold_contrib staging one layer's routed NVMe experts on the GPU (54 rows x 3.375 MiB)."""
    topo = describe_moe(_qwen3())
    row = bytes_per_expert(topo.expert_stacks[0])
    gib = lambda n: n * row / 2**30  # noqa: E731
    cold = ServeSetup(placement="solver", max_seqs=1, graphs=False, vram_gb=gib(8), dram_gb=gib(8), hot_rows=8)
    by = {i.name: i for i in estimate_serve_footprint(topo, cold).items}
    assert by["cold rows' device stack (one layer call)"].bytes == min_hot_rows(topo, cold) * row
    warm = ServeSetup(placement="solver", max_seqs=1, graphs=False, vram_gb=gib(24), dram_gb=0.0)
    assert not any(i.name.startswith("cold rows") for i in estimate_serve_footprint(topo, warm).items)


def test_prefill_staging_is_the_longest_prompt_plus_the_next_chunk():
    """Measured on an RTX A2000 (OLMoE, all-VRAM, four 1024-token prompts): 128 MiB of bf16 K/V staged for every
    attention layer at the generation peak, one prompt's worth -- 1024 x 16 layers x 2 x 16 heads x 128 x 2 B."""
    topo = describe_moe(_qwen3())                      # 3 attention layers, 2 KV heads, head_dim 32
    assert staging_bytes_per_token(topo) == 3 * 2 * 2 * 32 * 2
    assert prefill_staging_tokens(ServeSetup(max_seqs=1, max_tokens_per_seq=4096)) == 4096
    assert prefill_staging_tokens(ServeSetup(max_seqs=4, max_tokens_per_seq=4096, chunk_tokens=512)) == 4096 + 512
    assert prefill_staging_tokens(ServeSetup(max_seqs=2, max_tokens_per_seq=100, chunk_tokens=512)) == 200
    f = estimate_serve_footprint(topo, ServeSetup(max_seqs=4, max_tokens_per_seq=4096, graphs=False))
    item = next(i for i in f.items if i.name.startswith("prefill staging"))
    assert item.where == "device" and item.bytes == (4096 + 512) * staging_bytes_per_token(topo)


def test_int4_store_bytes_are_what_pack_int4_b32_returns():
    pack = pytest.importorskip("int4_pack_ref", reason="needs grouped-nf4-gemm").pack_int4_b32
    from experts4bit_qlora.serve_recipe import int4_store_bytes
    for n, k in ((64, 128), (96, 32), (256, 2048)):
        packed, scales = pack(torch.randn(n, k))
        assert int4_store_bytes(n, k) == packed.numel() * packed.element_size() + scales.numel() * scales.element_size()
        assert int4_store_bytes(n, k, experts=8) == 8 * int4_store_bytes(n, k)


def test_the_int4_attention_rule_is_the_swaps_and_the_topology_carries_it():
    from experts4bit_qlora.engines.int4_attn import attention_linears
    cfg = _qwen3()
    topo = describe_moe(cfg)
    real = [(lin.out_features, lin.in_features) for _m, _n, lin in attention_linears(tr.Qwen3MoeForCausalLM(cfg))]
    assert list(topo.int4_attention_linears) == real and len(real) == 3 * 4      # q/k/v/o on each of 3 layers
    assert sum(n * k for n, k in real) == topo.attention.numel


def test_exp_int4_replaces_the_nf4_stacks_and_prices_the_repack():
    from experts4bit_qlora.serve_recipe import _int4_split_k, int4_store_bytes
    topo = describe_moe(_qwen3())
    st = ServeSetup(max_seqs=4, max_tokens_per_seq=1024, graphs=False)
    nf4 = estimate_serve_footprint(topo, st)
    f = estimate_serve_footprint(topo, replace(st, exp_int4=True))
    assert not f.refusals
    by = {i.name: i for i in f.items}
    assert "frozen expert stacks (all VRAM)" not in by
    want = 0
    for s in topo.expert_stacks:                       # gate/up [E, 2I, H], down [E, H, I], top-2 partials
        want += int4_store_bytes(2 * 64, 128, 8) + int4_store_bytes(128, 64, 8)
        want += 4 * 2 * (_int4_split_k(2 * 64, 128)[0] * 2 * 64 + _int4_split_k(128, 64)[0] * 128)
    assert by["int4 expert stores (all VRAM; the NF4 stacks freed)"].bytes == want
    from experts4bit_qlora.serve_recipe import INT4_REPACK_HOST_BYTES_PER_PARAM
    repack = by["int4 repack: one layer's experts in fp32 (load)"]
    assert repack.where == "host" and repack.bytes == INT4_REPACK_HOST_BYTES_PER_PARAM * topo.expert_stacks[0].numel
    assert any("source checkpoint" in u for u in f.unmodelled)
    # the load-time overlap, where it exists, lifts the device total to exactly the repack's peak
    slab = next(i.bytes for i in nf4.items if i.name.startswith("frozen expert stacks"))
    at_load = slab + 2 * topo.dense_numel + int4_store_bytes(2 * 64, 128, 8) + int4_store_bytes(128, 64, 8)
    serving = sum(i.bytes for i in f.items if i.where == "device" and not i.name.startswith("int4 repack at load"))
    assert f.device_bytes == max(at_load, serving)
    env = replace(st, exp_int4=True).to_env()
    assert (env["E4B_SERVE_EXP_INT4"], env["E4B_SERVE_ATTN_INT4"], env["E4B_INT4_KEEP_NF4"]) == ("1", "0", "0")
    assert ServeSetup().to_env()["E4B_SERVE_EXP_INT4"] == "0" and "E4B_INT4_KEEP_NF4" not in ServeSetup().to_env()
    from experts4bit_qlora.serve_paged import LEVER_ENV
    assert {"E4B_SERVE_EXP_INT4", "E4B_SERVE_ATTN_INT4", "E4B_INT4_KEEP_NF4"} <= set(LEVER_ENV)   # the names it reads


def test_attn_int4_prices_the_grid_and_the_bf16_copy_it_keeps():
    from experts4bit_qlora.serve_recipe import int4_store_bytes
    topo = describe_moe(_qwen3())
    st = ServeSetup(max_seqs=1, max_tokens_per_seq=256, graphs=False)
    bf16 = {i.name: i for i in estimate_serve_footprint(topo, st).items}
    f = estimate_serve_footprint(topo, replace(st, attn_int4=True))
    by = {i.name: i for i in f.items}
    numel = topo.attention.numel
    assert by["dense weights (bf16)"].bytes == bf16["dense weights (bf16)"].bytes - 2 * numel
    assert by["attention projections on the int4-b32 grid"].bytes == sum(int4_store_bytes(n, k)
                                                                         for n, k in topo.int4_attention_linears)
    assert by["attention projections' bf16 copy (kept from the first prefill)"].bytes == 2 * numel
    assert by["int4 attention workspaces"].bytes > 0
    # a speed lever, not a memory one: the attention weights cost more than bf16 once a prompt is served
    assert f.device_bytes > estimate_serve_footprint(topo, st).device_bytes


def test_int4_levers_are_refused_where_the_server_refuses_them():
    topo = describe_moe(_qwen3())
    f = estimate_serve_footprint(topo, ServeSetup(placement="solver", max_seqs=1, graphs=False, exp_int4=True))
    assert f.items == () and any("all-VRAM collapsed" in r for r in f.refusals)
    # o_proj reads 3 heads x 80 = 240 features: not a whole number of int4-b32 scale blocks (the NF4 experts are fine)
    odd = describe_moe(tr.Qwen3MoeConfig(hidden_size=128, intermediate_size=256, moe_intermediate_size=64, num_experts=8,
                                         num_experts_per_tok=2, num_hidden_layers=2, num_attention_heads=3,
                                         num_key_value_heads=1, head_dim=80, vocab_size=192, max_position_embeddings=64))
    assert any("multiple of 32" in r for r in estimate_serve_footprint(odd, ServeSetup(attn_int4=True)).refusals)
    assert not estimate_serve_footprint(odd, ServeSetup(exp_int4=True)).refusals


def _stateful_hybrids():
    """Tiny configs, one per verdict: LFM2's conv layers (a type the runner keeps no state for), granite-4.0-h's Mamba
    layers (labelled linear_attention, which the per-slot pool cannot drive), Qwen3.5's Gated DeltaNet (kept)."""
    lfm2 = tr.Lfm2MoeConfig(vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, num_dense_layers=1,
                            num_experts=4, num_experts_per_tok=2, max_position_embeddings=256,
                            layer_types=["conv", "full_attention", "conv", "full_attention"])
    granite = tr.GraniteMoeHybridConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                        num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
                                        num_experts_per_tok=2, shared_intermediate_size=64, mamba_n_heads=4,
                                        mamba_n_groups=1, mamba_d_state=16, mamba_d_head=32, mamba_d_conv=4,
                                        max_position_embeddings=256,
                                        layer_types=["linear_attention", "attention", "linear_attention", "attention"])
    qwen35 = tr.Qwen3_5MoeConfig(text_config=dict(
        vocab_size=128, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        moe_intermediate_size=64, shared_expert_intermediate_size=64, num_experts=4, num_experts_per_tok=2,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=24,
        linear_conv_kernel_dim=4, max_position_embeddings=256, layer_types=["linear_attention", "full_attention"] * 2))
    return {"lfm2": lfm2, "granite": granite, "qwen35": qwen35}


def _built(cfg):
    """The tiny model itself (a composite config's text model), for the tests that run the runner's own code."""
    return tr.AutoModelForCausalLM.from_config(getattr(cfg, "text_config", None) or cfg).eval()


def test_the_estimate_refuses_the_state_the_paged_runner_does_not_keep():
    """LFM2-8B-A1B and granite-4.0-h-tiny planned as feasible serves; serve_paged refuses both when it builds the
    runner. The topology now carries the runner's own verdict, and the estimate refuses with it."""
    cfgs = _stateful_hybrids()
    verdict = {name: describe_moe(cfg).paged_state_refusal for name, cfg in cfgs.items()}
    assert "['conv']" in verdict["lfm2"]
    assert "Gated DeltaNet only" in verdict["granite"]
    assert verdict["qwen35"] is None and describe_moe(_qwen3()).paged_state_refusal is None
    for name in ("lfm2", "granite"):
        f = estimate_serve_footprint(describe_moe(cfgs[name]), ServeSetup(max_seqs=1, graphs=False))
        assert f.items == () and any("the paged server refuses this model" in r for r in f.refusals)
    assert not estimate_serve_footprint(describe_moe(cfgs["qwen35"]), ServeSetup(max_seqs=1, graphs=False)).refusals


def test_the_refusal_is_the_runner_s_own_verdict():
    """Built for real (tiny, CPU), each model is refused by the runner's layer plan exactly when the verdict says so."""
    pytest.importorskip("transformers.cache_utils", reason="needs transformers' cache_utils")
    from transformers.cache_utils import LinearAttentionLayer  # noqa: F401  (the pool's state carrier, >= 5.13)

    from experts4bit_qlora.engines.paged_runner import layer_plan, paged_state_refusal
    for name, cfg in _stateful_hybrids().items():
        model = _built(cfg)
        refusal = paged_state_refusal(model)
        try:
            layer_plan(model, model.config.num_hidden_layers, 2)
            refused = False
        except NotImplementedError:
            refused = True
        assert refused == (refusal is not None), (name, refusal)


def test_the_linear_state_pool_bytes_are_what_the_pool_allocates():
    """A hybrid's per-slot linear-attention state, priced from the topology, IS what LinearStatePool holds after each
    layer's first forward: a bf16 conv window and an fp32 recurrent state per slot (transformers keeps it in fp32)."""
    pytest.importorskip("transformers.cache_utils", reason="needs transformers' cache_utils")
    from transformers.cache_utils import LinearAttentionLayer

    from experts4bit_qlora.engines.linear_state import LinearStatePool, _linear_classes, _OneLayerCache
    from experts4bit_qlora.serve_recipe import linear_state_pool_bytes

    cfg = _stateful_hybrids()["qwen35"]
    topo = describe_moe(cfg)
    assert [g[0] for g in topo.linear_state_layers] == [0, 2]
    model = _built(cfg).to(torch.bfloat16)
    pool = LinearStatePool(5)
    for mod in (m for m in model.modules() if isinstance(m, _linear_classes())):
        lal = LinearAttentionLayer()
        try:
            with torch.no_grad():
                mod(torch.randn(2, 3, model.config.hidden_size, dtype=torch.bfloat16),
                    cache_params=_OneLayerCache(mod.layer_idx, lal))
        except RuntimeError as e:          # a CUDA-only causal_conv1d build on a CPU box
            pytest.skip(f"the Gated DeltaNet forward cannot run here: {e}")
        pool.store(mod.layer_idx, [0, 1], lal)
    assert pool.nbytes() == linear_state_pool_bytes(topo.linear_state_layers, 5)


def test_a_hybrid_serve_prices_its_state_pool_per_slot():
    from experts4bit_qlora.serve_recipe import linear_state_pool_bytes
    topo = describe_moe(_stateful_hybrids()["qwen35"])
    f = estimate_serve_footprint(topo, ServeSetup(max_seqs=4, max_tokens_per_seq=256, graphs=True, buckets=(1, 2, 4)))
    item = next(i for i in f.items if i.name == "linear-attention state pool")
    assert item.where == "device" and item.basis == "derived"
    assert item.bytes == linear_state_pool_bytes(topo.linear_state_layers, 4 + 4)    # seqs + graph scratch slots
    assert not any("recurrent state" in u for u in f.unmodelled)                    # every linear layer is priced
    assert describe_moe(_qwen3()).linear_state_layers == ()


def test_a_model_with_leading_dense_layers_keeps_kv_for_every_layer():
    """ERNIE-4.5's layer 0 is dense but has attention: the paged KV pool (and so the estimate) sizes for every decoder
    layer, not for the MoE layers. Sized by its 27 MoE layers, ERNIE's pool indexed past its end at layer 27."""
    cfg = tr.Qwen3MoeConfig(hidden_size=128, intermediate_size=256, moe_intermediate_size=64, num_experts=8,
                            num_experts_per_tok=2, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=32, vocab_size=192, max_position_embeddings=64, mlp_only_layers=[0])
    topo = describe_moe(cfg)
    assert len(topo.expert_stacks) == 2 and topo.kv_layers == 3
    from experts4bit_qlora.engines.paged_runner import decoder_layers, kv_layers
    model = tr.AutoModelForCausalLM.from_config(cfg)
    assert kv_layers(model, decoder_layers(model.config)) == 3
    f = estimate_serve_footprint(topo, ServeSetup(max_seqs=1, max_tokens_per_seq=256, graphs=False))
    kv = next(i for i in f.items if i.name == "FP8 paged KV pool")
    assert kv.bytes == paged_kv_pool_bytes(3, 2, 32, batch=1, max_tokens_per_seq=256)


def test_the_dram_tier_s_prefill_on_the_gpu_is_priced_by_its_excess_over_the_cold_stack():
    """hybrid._dram_on_gpu computes a prefill chunk's routed DRAM experts on the GPU: their NF4 bytes with absmax cast
    to bf16, and the chunk's routed rows (bf16 in, two fp32 outs). Measured on ERNIE-4.5-21B-A3B (RTX A2000, no NVMe
    rows): 457.5 MiB live at the generation peak, 64 experts x 5.98 MiB + 75 MiB, exactly this arithmetic."""
    from experts4bit_qlora.serve_recipe import dram_rows_per_layer
    topo = describe_moe(_qwen3())                      # 3 layers x 8 experts, top-2, H=128
    st0, bpe = topo.expert_stacks[0], bytes_per_expert(topo.expert_stacks[0])
    setup = ServeSetup(placement="solver", max_seqs=1, graphs=False, vram_gb=5 * bpe / 2**30, dram_gb=7 * bpe / 2**30,
                       hot_rows=8)
    dram_max = dram_rows_per_layer(3, 8, bpe, setup.vram_gb, setup.dram_gb)
    assert 0 < dram_max <= 8
    per_expert = st0.numel // 8
    rows = 2 * setup.chunk_tokens
    dram_gpu = min(dram_max, rows) * (per_expert // 2 + 2 * (per_expert // 64)) + rows * 128 * 10
    f = estimate_serve_footprint(topo, setup)
    by = {i.name: i for i in f.items}
    cold = by["cold rows' device stack (one layer call)"].bytes
    item = by["DRAM experts run on the GPU at prefill (one layer call)"]
    assert item.where == "device" and item.basis == "derived" and item.bytes == dram_gpu - cold
    # every expert in VRAM, or nothing in DRAM: no item
    assert dram_rows_per_layer(3, 8, bpe, 1.0, 0.0) == 0
    allv = estimate_serve_footprint(topo, ServeSetup(placement="solver", max_seqs=1, graphs=False, vram_gb=1.0,
                                                     dram_gb=0.0, hot_rows=8))
    assert not any(i.name.startswith("DRAM experts run on the GPU") for i in allv.items)


def test_the_server_captures_only_the_buckets_its_sequences_can_use(monkeypatch):
    """A bucket above max_seqs never runs; on Qwen3.6 served for one sequence, buckets 2-16 failed to capture (lane
    SV3). The server keeps the buckets it can use, and the estimate sizes the scratch slots by the same rule."""
    from experts4bit_qlora.serve_paged import PagedServeConfig
    from experts4bit_qlora.serve_recipe import usable_buckets
    assert usable_buckets(1, (1, 2, 4, 8, 16)) == (1,)
    assert usable_buckets(12, (1, 2, 4, 8, 16)) == (1, 2, 4, 8, 12)
    assert usable_buckets(32, (1, 2, 4, 8, 16)) == (1, 2, 4, 8, 16)
    monkeypatch.setenv("E4B_PAGED_DEVICE", "cpu")
    for k, v in ServeSetup(max_seqs=1).to_env().items():
        monkeypatch.setenv(k, v)
    assert tuple(PagedServeConfig.from_env().buckets) == (1,)
    topo = describe_moe(_qwen3())
    f = estimate_serve_footprint(topo, ServeSetup(max_seqs=1, max_tokens_per_seq=256, graphs=True))
    kv = next(i for i in f.items if i.name == "FP8 paged KV pool")
    assert kv.bytes == paged_kv_pool_bytes(3, 2, 32, batch=1, max_tokens_per_seq=256, scratch_slots=1)


def test_multi_head_latent_attention_is_refused_before_the_pool_meets_it():
    """DeepSeek-V2-Lite planned a feasible serve; the server built the pool head_dim (64) wide and its first prompt's
    append refused keys of 192 and values of 128. MLA is refused by its config, in the estimate and before any weight
    is read."""
    from experts4bit_qlora.engines.paged_runner import kv_layout_refusal
    cfg = tr.DeepseekV2Config(vocab_size=128, hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, n_routed_experts=8,
                              num_experts_per_tok=2, n_shared_experts=1, kv_lora_rank=32, q_lora_rank=None,
                              qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32, first_k_dense_replace=1,
                              max_position_embeddings=64)
    why = kv_layout_refusal(cfg)
    assert why and "keys 48 and values 32" in why
    assert kv_layout_refusal(_qwen3()) is None
    topo = describe_moe(cfg)
    assert topo.paged_state_refusal == why or topo.loader_refusal
    if not topo.loader_refusal:
        f = estimate_serve_footprint(topo, ServeSetup(max_seqs=1, graphs=False))
        assert f.items == () and any("multi-head latent attention" in r for r in f.refusals)


@pytest.mark.parametrize("L,H,D,T,kg", [(3, 2, 128, 4096, None), (5, 8, 128, 512, 4), (48, 4, 128, 8192, None), (2, 16, 64, 100, None)])
def test_the_bulk_flush_bound_is_the_pool_s_own(L, H, D, T, kg):
    """fp8_paged_kv.append_prompt_peak_bytes (pure) equals Fp8PagedKV.append_prompt_peak_bytes on a constructed pool."""
    from experts4bit_qlora.engines.fp8_paged_kv import append_prompt_peak_bytes
    kv = Fp8PagedKV(L, H, D, batch=1, max_tokens_per_seq=T, k_groups=kg, device="cpu")
    assert append_prompt_peak_bytes([(H, D, kv.kgs[i]) for i in range(L)], T) == kv.append_prompt_peak_bytes(T)


def test_the_bulk_kv_flush_is_priced_at_the_slot_s_capacity():
    """Lane SV5: Qwen3-30B-A3B all-VRAM 8 x 8192 on an RTX 4090 ran out of memory at 8,000-token prompts with the flush
    unpriced. The estimate now carries it at the slot's capacity, as the server's own prefill-graph headroom check does."""
    from experts4bit_qlora.engines.fp8_paged_kv import append_prompt_peak_bytes
    topo = describe_moe(_qwen3())                      # 3 pool layers, 2 KV heads, head_dim 32
    on = estimate_serve_footprint(topo, ServeSetup(max_seqs=2, max_tokens_per_seq=250, graphs=False))
    off = estimate_serve_footprint(topo, ServeSetup(max_seqs=2, max_tokens_per_seq=250, graphs=False, bulk_kv=False))
    item = next(i for i in on.items if i.name.startswith("bulk KV flush"))
    kv = Fp8PagedKV(3, 2, 32, batch=2, max_tokens_per_seq=250, device="cpu")
    assert item.basis == "derived" and item.bytes == kv.append_prompt_peak_bytes(kv.blocks_per_seq * kv.bt)
    assert on.device_bytes - off.device_bytes == item.bytes
    assert ServeSetup(bulk_kv=False).to_env()["E4B_PAGED_BULK_KV"] == "0"
    assert append_prompt_peak_bytes([], 100) == 0
