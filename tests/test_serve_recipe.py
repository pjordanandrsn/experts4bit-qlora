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
from experts4bit_qlora.serve_recipe import (BLOCK_TOKENS, ServeSetup, estimate_serve_footprint,  # noqa: E402
                                             paged_kv_pool_bytes)


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


def test_unpriced_placements_are_refused_in_words():
    topo = describe_moe(_qwen3())
    f = estimate_serve_footprint(topo, ServeSetup(placement="solver"))
    assert f.items == () and any("not priced yet" in r for r in f.refusals)


def test_to_env_is_what_the_server_reads_back(monkeypatch):
    from experts4bit_qlora.serve_paged import PagedServeConfig

    monkeypatch.setenv("E4B_PAGED_DEVICE", "cpu")      # host-independent: no GPU facts enter from_env
    for st in (ServeSetup(), ServeSetup(max_seqs=3, max_tokens_per_seq=777, chunk_tokens=128, graphs=False,
                                        buckets=(1, 2), kv_groups=4, prefill_graph="0")):
        for k, v in st.to_env().items():
            monkeypatch.setenv(k, v)
        cfg = PagedServeConfig.from_env()
        assert {f: getattr(cfg, f) for f in ("placement", "max_seqs", "max_tokens_per_seq", "chunk_tokens", "graphs")} \
            == {f: getattr(st, f) for f in ("placement", "max_seqs", "max_tokens_per_seq", "chunk_tokens", "graphs")}
        assert tuple(cfg.buckets) == tuple(st.buckets) and str(cfg.kv_groups) == str(st.kv_groups)
        assert cfg.prefill_graph == st.prefill_graph


def test_the_prefill_graph_pool_is_named_where_the_graph_can_engage():
    topo = describe_moe(_qwen3())
    named = lambda f: any("prefill graph" in u for u in f.unmodelled)  # noqa: E731
    assert named(estimate_serve_footprint(topo, ServeSetup(max_seqs=4)))                      # the server's default
    assert not named(estimate_serve_footprint(topo, ServeSetup(max_seqs=4, prefill_graph="0")))
    assert not named(estimate_serve_footprint(topo, ServeSetup(max_seqs=4, graphs=False)))   # needs device grouping
    assert not named(estimate_serve_footprint(topo, ServeSetup(max_seqs=1)))
    assert estimate_serve_footprint(topo, ServeSetup(prefill_graph="on")).refusals
