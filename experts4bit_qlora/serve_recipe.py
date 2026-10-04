"""The paged server's memory, before loading: :class:`ServeSetup` and :func:`estimate_serve_footprint`.

The training counterpart is :mod:`experts4bit_qlora.recipe`. This prices what :func:`experts4bit_qlora.serve_paged.build_engine`
allocates for a model, from its :class:`~experts4bit_qlora.arch.topology.MoETopology` and the server's own knobs:

* the frozen expert stacks under ``placement="all-vram"`` -- every (layer, expert) row resident, sized by constructing
  this package's ``Experts4bit`` on ``meta`` exactly as the training estimate does (derived);
* the dense weights in bf16, with no adapters (``build_engine`` loads with an arena, so the expert LoRA is never built);
* the FP8 paged KV pool, by :func:`paged_kv_pool_bytes` -- the same arithmetic ``Fp8PagedKV`` allocates with, asserted
  equal to a constructed pool in the tests (derived);
* a prefill/decode working set (heuristic, stated).

Only the all-VRAM placement is priced; the solver's VRAM/DRAM/NVMe tiers are refused here until they are measured.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .recipe import Footprint, FootprintItem, QLoRASetup, _module_bytes, _stack_modules

BLOCK_TOKENS = 16          # experts4bit_qlora.engines.fp8_paged_kv.BLOCK_TOKENS (asserted equal in the tests)
DEFAULT_BUCKETS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class ServeSetup:
    """The paged server's memory-relevant knobs (``PagedServeConfig`` names; ``serve_paged`` reads them from env)."""

    placement: str = "all-vram"
    #: E4B_PAGED_MAX_SEQS: batch width == KV slots
    max_seqs: int = 16
    #: E4B_PAGED_MAX_TOKENS_PER_SEQ: prompt + output per sequence
    max_tokens_per_seq: int = 4096
    #: E4B_PAGED_CHUNK_TOKENS: the prefill chunk
    chunk_tokens: int = 512
    #: E4B_PAGED_GRAPHS: bucketed decode graphs add ``max(buckets)`` scratch KV slots
    graphs: bool = True
    buckets: tuple = DEFAULT_BUCKETS
    #: E4B_PAGED_KV_GROUPS: "auto" or an int
    kv_groups: object = "auto"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_env(self) -> dict:
        """The ``serve_paged`` environment that builds this setup (``PagedServeConfig.from_env`` reads it back)."""
        return {"E4B_PAGED_PLACEMENT": self.placement, "E4B_PAGED_MAX_SEQS": str(self.max_seqs),
                "E4B_PAGED_MAX_TOKENS_PER_SEQ": str(self.max_tokens_per_seq),
                "E4B_PAGED_CHUNK_TOKENS": str(self.chunk_tokens), "E4B_PAGED_GRAPHS": "1" if self.graphs else "0",
                "E4B_PAGED_BUCKETS": ",".join(str(int(b)) for b in self.buckets),
                "E4B_PAGED_KV_GROUPS": str(self.kv_groups)}


def paged_kv_pool_bytes(n_layers: int, n_kv_heads, head_dim, *, batch: int, max_tokens_per_seq: int,
                        k_groups=None, scratch_slots: int = 0) -> int:
    """Device bytes ``Fp8PagedKV(n_layers, n_kv_heads, head_dim, batch=..., ...)`` allocates for its K and V row pools
    and its block tables: the same normalization and row arithmetic, so a planner can size the pool before building it."""
    from fp8_kv import kv_block_bytes

    from .engines.fp8_paged_kv import _auto_k_groups

    hs = [int(x) for x in (n_kv_heads if hasattr(n_kv_heads, "__len__") else [n_kv_heads] * n_layers)]
    ds = [int(x) for x in (head_dim if hasattr(head_dim, "__len__") else [head_dim] * n_layers)]
    if len(hs) != n_layers or len(ds) != n_layers:
        raise ValueError(f"per-layer geometry needs {n_layers} entries, got {len(hs)} heads / {len(ds)} dims")
    kgs = [_auto_k_groups(d) for d in ds] if k_groups is None else [int(k_groups)] * n_layers
    k_row = max(kv_block_bytes(BLOCK_TOKENS, h, d) + BLOCK_TOKENS * h * 4 * (kg - 1) for h, d, kg in zip(hs, ds, kgs))
    v_row = max(kv_block_bytes(BLOCK_TOKENS, h, d) for h, d in zip(hs, ds))
    blocks_per_seq = -(-max_tokens_per_seq // BLOCK_TOKENS)
    rows = batch * blocks_per_seq + scratch_slots
    tables = n_layers * (batch + scratch_slots) * blocks_per_seq * 4 + n_layers * (batch + scratch_slots) * 4
    return n_layers * rows * (k_row + v_row) + tables


def serve_setup_refusals(topology, setup: ServeSetup) -> tuple:
    out = []
    if topology.loader_refusal:
        return (f"the loader refuses this model: {topology.loader_refusal}",)
    if not topology.expert_stacks:
        out.append("no fused expert stack at the loader's expert path")
    if setup.placement != "all-vram":
        out.append(f"placement {setup.placement!r} is not priced yet (only all-vram); the solver's tiers need measured "
                   "bandwidths and a routing profile")
    if topology.kv_layers is None:
        out.append(f"the paged KV geometry is not describable: {topology.provenance.get('kv')}")
    if setup.max_seqs < 1 or setup.max_tokens_per_seq < 2:
        out.append("max_seqs >= 1 and max_tokens_per_seq >= 2 are required")
    try:
        import fp8_kv  # noqa: F401
    except ImportError:
        out.append("grouped-nf4-gemm's fp8_kv is not importable: the paged server cannot build its KV pool")
    return tuple(out)


def estimate_serve_footprint(topology, setup: ServeSetup) -> Footprint:
    """Device bytes ``serve_paged.build_engine`` holds for ``setup`` on ``topology``, item by item."""
    refusals = serve_setup_refusals(topology, setup)
    if refusals:
        return Footprint(items=(), refusals=refusals)
    items, unmodelled = [], []
    qs = QLoRASetup()                      # nf4, blocksize 64: the arena's default bake
    slab = sum(_module_bytes(_stack_modules(st, qs)[0]) for st in topology.expert_stacks)
    items.append(FootprintItem("frozen expert stacks (all VRAM)", "device", slab, "derived",
                               f"{len(topology.expert_stacks)} layers x all experts, nf4 blocksize 64 (packed + absmax)"))
    items.append(FootprintItem("dense weights (bf16)", "device", 2 * topology.dense_numel, "derived",
                               "embeddings, attention, norms, routers, dense/shared MLPs; no adapters under an arena load"))
    scratch = max(setup.buckets) if setup.graphs else 0
    kv = paged_kv_pool_bytes(topology.kv_layers, topology.kv_heads, topology.kv_head_dims, batch=setup.max_seqs,
                             max_tokens_per_seq=setup.max_tokens_per_seq,
                             k_groups=None if setup.kv_groups == "auto" else int(setup.kv_groups), scratch_slots=scratch)
    items.append(FootprintItem("FP8 paged KV pool", "device", kv, "derived",
                               f"{topology.kv_layers} layers x ({setup.max_seqs} seqs x {setup.max_tokens_per_seq} tokens"
                               f" + {scratch} graph scratch slots), fp8 payload + fp32 scales (Fp8PagedKV's arithmetic)"))
    st0 = topology.expert_stacks[0]
    first_out = 1
    for d in st0.first_shape:
        first_out *= d
    first_out //= st0.n_experts * st0.hidden
    c = setup.chunk_tokens
    work = c * topology.hidden_size * 2 * 8 + c * (topology.top_k or 0) * (first_out + st0.intermediate + st0.hidden) * 2 \
        + setup.max_seqs * topology.vocab_size * 4
    items.append(FootprintItem("prefill/decode working set", "device", work, "heuristic",
                               f"a {c}-token prefill chunk's hidden states and routed expert activations + fp32 logits "
                               f"for {setup.max_seqs} sequences"))
    if setup.graphs:
        unmodelled.append("CUDA graph memory pools for the decode buckets")
    if topology.attention is not None and topology.attention.layers < topology.n_layers:
        unmodelled.append(f"recurrent state of the {topology.n_layers - topology.attention.layers} non-attention layers")
    unmodelled.append("CUDA context, cuBLAS/Triton workspaces and allocator fragmentation (the caller's to add)")
    return Footprint(items=tuple(items), unmodelled=tuple(unmodelled))
