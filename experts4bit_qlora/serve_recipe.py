"""The paged server's memory, before loading: :class:`ServeSetup` and :func:`estimate_serve_footprint`.

The training counterpart is :mod:`experts4bit_qlora.recipe`. This prices what :func:`experts4bit_qlora.serve_paged.build_engine`
allocates for a model, from its :class:`~experts4bit_qlora.arch.topology.MoETopology` and the server's own knobs:

* the frozen expert stacks, sized by constructing this package's ``Experts4bit`` on ``meta`` exactly as the training
  estimate does (derived). Under ``placement="all-vram"`` every (layer, expert) row is in VRAM. Under
  ``placement="solver"`` the rows split into VRAM, DRAM and NVMe tiers by
  :func:`~experts4bit_qlora.engines.placement.solve_placement` itself, with the same budgets the server passes;
* the hybrid tier's host buffers, which ``enable_hybrid_tier`` builds at either placement: the cold tier's pinned
  landing (``hot_rows`` rows, priced by grouped-nf4-gemm's ``pinned_request_cost``), the setup tier used while the
  stacks are built, and, when rows live on NVMe, the cold view those rows land in (derived);
* the dense weights in bf16, with no adapters (``build_engine`` loads with an arena, so the expert LoRA is never built);
* the FP8 paged KV pool, by :func:`paged_kv_pool_bytes` -- the same arithmetic ``Fp8PagedKV`` allocates with, asserted
  equal to a constructed pool in the tests (derived);
* a prefill/decode working set (heuristic, stated).

Arena geometry is the bake's default (NF4 blocksize 64, fp32 absmax, rows aligned to 4096 bytes).
"""
from __future__ import annotations

import functools
from dataclasses import asdict, dataclass, replace

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
    #: E4B_PAGED_PREFILL_GRAPH: "auto" (the server's default), "1" or "0". Its graph's private pool is not priced
    prefill_graph: str = "auto"
    #: E4B_PAGED_VRAM_GB / E4B_PAGED_DRAM_GB: the solver's tier budgets in GiB (placement="solver" only)
    vram_gb: float = 1.2
    dram_gb: float = 6.0
    #: E4B_PAGED_HOT_ROWS: the cold tier's row capacity (pinned landing; the cold view), at either placement
    hot_rows: int = 64

    def to_dict(self) -> dict:
        return asdict(self)

    def to_env(self) -> dict:
        """The ``serve_paged`` environment that builds this setup (``PagedServeConfig.from_env`` reads it back)."""
        return {"E4B_PAGED_PLACEMENT": self.placement, "E4B_PAGED_MAX_SEQS": str(self.max_seqs),
                "E4B_PAGED_MAX_TOKENS_PER_SEQ": str(self.max_tokens_per_seq),
                "E4B_PAGED_CHUNK_TOKENS": str(self.chunk_tokens), "E4B_PAGED_GRAPHS": "1" if self.graphs else "0",
                "E4B_PAGED_BUCKETS": ",".join(str(int(b)) for b in self.buckets),
                "E4B_PAGED_KV_GROUPS": str(self.kv_groups), "E4B_PAGED_PREFILL_GRAPH": str(self.prefill_graph),
                "E4B_PAGED_VRAM_GB": repr(float(self.vram_gb)), "E4B_PAGED_DRAM_GB": repr(float(self.dram_gb)),
                "E4B_PAGED_HOT_ROWS": str(int(self.hot_rows))}


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


#: grouped-nf4-gemm's arena bakes align every row to this many bytes (``nvme_arena`` / ``nvme_bake_nf4`` default)
ARENA_ALIGN = 4096


def solver_tiers(n_layers: int, n_experts: int, bytes_per_expert: int, vram_gb: float, dram_gb: float) -> dict:
    """Expert rows per tier under ``placement="solver"``, from :func:`~.engines.placement.solve_placement` itself.

    ``build_engine`` passes no routing profile, so every expert weighs the same and the solver's greedy fills VRAM, then
    DRAM, then NVMe; the bandwidths it is given decide nothing here, so unit overrides stand in for a calibration."""
    from .engines.placement import solve_placement

    man = solve_placement(n_layers=n_layers, n_experts=n_experts, bytes_per_expert=bytes_per_expert,
                          vram_budget_bytes=int(vram_gb * 2**30), dram_budget_bytes=int(dram_gb * 2**30),
                          calibration={}, profile_path=None, b_vram_override=1.0, b_dram_override=1.0, batch=1)
    return {t: len(v) for t, v in man["tiers"].items()}


def bytes_per_expert(stack, qsetup=None) -> int:
    """One expert's frozen bytes (packed + absmax), the arena's row before alignment: the growth of the stack from one
    expert to ``n_experts``, so per-stack constants (the NF4 code table) do not leak into every row."""
    qsetup = qsetup or QLoRASetup()
    one = _frozen_stack_bytes(replace(stack, layer=0, n_experts=1), qsetup)
    if stack.n_experts == 1:
        return one
    return (_frozen_stack_bytes(replace(stack, layer=0), qsetup) - one) // (stack.n_experts - 1)


def min_hot_rows(topology, setup) -> int:
    """The fewest cold-tier rows ``setup`` can serve with: grouped-nf4-gemm's ColdTier refuses a demand window larger than
    its slots ("Size hot_rows >= max routed experts per layer"). A step routes at most ``top_k`` experts per token over
    ``max(chunk_tokens, max_seqs)`` tokens, a layer has ``n_experts``, and only the layer's NVMe rows are cold. Without a
    routing profile the solver fills layer by layer, so NVMe holds whole trailing layers. 0 when nothing is on NVMe."""
    st = topology.expert_stacks[0]
    if setup.placement != "solver":
        return 0
    n_nvme = solver_tiers(len(topology.expert_stacks), st.n_experts, bytes_per_expert(st), setup.vram_gb,
                          setup.dram_gb)["nvme"]
    if not n_nvme:
        return 0
    routed = (topology.top_k or st.n_experts) * max(setup.chunk_tokens, setup.max_seqs)
    return int(min(st.n_experts, routed, n_nvme))


def _hybrid_host_items(hot_rows: int, bpe: int, stride: int, n_nvme: int) -> list:
    """The host buffers ``enable_hybrid_tier`` builds at either placement (the server always builds the hybrid tier)."""
    try:
        from nvme_residency import pinned_request_cost
    except ImportError:                      # pragma: no cover - the KV refusal already names grouped-nf4-gemm
        def pinned_request_cost(n):
            return n
    setup_rows = max(8, min(hot_rows, 64))
    out = [FootprintItem("cold tier landing (pinned)", "host", pinned_request_cost(hot_rows * stride), "derived",
                         f"ColdTier(hot_rows={hot_rows}) pins hot_rows x {stride} B; PyTorch's pinned allocator rounds "
                         "the request up to a power of two (grouped-nf4-gemm pinned_request_cost)"),
           FootprintItem("setup tier (while the stacks are built)", "host", setup_rows * stride, "derived",
                         f"a {setup_rows}-row pageable tier for the one-shot reads that build the resident stacks; "
                         "closed after construction, counted in full: measured shared memory on OLMoE never fell "
                         "below the total that includes it")]
    if n_nvme:
        out.append(FootprintItem("cold view (rows land here)", "host", hot_rows * bpe, "derived",
                                 f"hot_rows x {bpe} B of page-aligned shared memory the NVMe rows are read into; "
                                 "filled as rows land, so this is its ceiling"))
    return out


@functools.lru_cache(maxsize=64)
def _frozen_stack_bytes(stack, qsetup) -> int:
    """One stack's frozen bytes, built on meta. Keyed by shape (callers pass ``layer=0``): a planner prices the same
    stacks once per context/concurrency it tries, and a 48-layer model has one stack shape."""
    return _module_bytes(_stack_modules(stack, qsetup)[0])


def serve_setup_refusals(topology, setup: ServeSetup) -> tuple:
    out = []
    if topology.loader_refusal:
        return (f"the loader refuses this model: {topology.loader_refusal}",)
    if not topology.expert_stacks:
        out.append("no fused expert stack at the loader's expert path")
    if setup.placement not in ("all-vram", "solver"):
        out.append(f"placement must be 'all-vram' or 'solver', got {setup.placement!r}")
    if setup.placement == "solver":
        if setup.graphs and setup.max_seqs > 1 and max(setup.buckets) > 1:
            out.append("batched decode graphs bind to the all-vram placement (serve_paged refuses them with the solver); "
                       "plan graphs=False")
        if topology.expert_bias_tensors:
            out.append("the hybrid tier cannot serve per-expert biases from the arena (gpt-oss)")
        if setup.vram_gb < 0 or setup.dram_gb < 0:
            out.append("vram_gb and dram_gb must be >= 0")
        if len({st.n_experts for st in topology.expert_stacks}) > 1:
            out.append("the solver assumes one expert count for every MoE layer")
    if setup.hot_rows < 1:
        out.append("hot_rows must be >= 1")
    if topology.kv_layers is None:
        out.append(f"the paged KV geometry is not describable: {topology.provenance.get('kv')}")
    if setup.max_seqs < 1 or setup.max_tokens_per_seq < 2:
        out.append("max_seqs >= 1 and max_tokens_per_seq >= 2 are required")
    if str(setup.prefill_graph) not in ("auto", "0", "1"):
        out.append(f"prefill_graph must be 'auto', '0' or '1', got {setup.prefill_graph!r}")
    try:
        import fp8_kv  # noqa: F401
    except ImportError:
        out.append("grouped-nf4-gemm's fp8_kv is not importable: the paged server cannot build its KV pool")
    return tuple(out)


def estimate_serve_footprint(topology, setup: ServeSetup) -> Footprint:
    """What ``serve_paged.build_engine`` holds for ``setup`` on ``topology``, item by item: device, host and (under the
    solver) NVMe bytes."""
    refusals = serve_setup_refusals(topology, setup)
    if refusals:
        return Footprint(items=(), refusals=refusals)
    items, unmodelled = [], []
    qs = QLoRASetup()                      # nf4, blocksize 64: the arena's default bake
    slab = sum(_frozen_stack_bytes(replace(st, layer=0), qs) for st in topology.expert_stacks)
    n_layers, n_experts = len(topology.expert_stacks), topology.expert_stacks[0].n_experts
    bpe = bytes_per_expert(topology.expert_stacks[0], qs)
    stride = -(-bpe // ARENA_ALIGN) * ARENA_ALIGN
    if setup.placement == "all-vram":
        items.append(FootprintItem("frozen expert stacks (all VRAM)", "device", slab, "derived",
                                   f"{n_layers} layers x all experts, nf4 blocksize 64 (packed + absmax)"))
        n_nvme = 0
    else:
        tiers = solver_tiers(n_layers, n_experts, bpe, setup.vram_gb, setup.dram_gb)
        n_nvme = tiers["nvme"]
        how = (f"solve_placement with the server's budgets ({setup.vram_gb:g} / {setup.dram_gb:g} GiB) and no routing "
               "profile, as build_engine calls it: VRAM fills first, then DRAM, then NVMe")
        items.append(FootprintItem("expert stacks, VRAM tier", "device", tiers["vram"] * bpe, "derived",
                                   f"{tiers['vram']} of {n_layers * n_experts} expert rows x {bpe} B; {how}"))
        items.append(FootprintItem("expert stacks, DRAM tier (computed on the CPU)", "host", tiers["dram"] * bpe, "derived",
                                   f"{tiers['dram']} rows, pageable host memory, fp32 absmax"))
        items.append(FootprintItem("expert rows on NVMe (read through the cold tier)", "nvme", n_nvme * bpe, "derived",
                                   f"{n_nvme} rows streamed from the arena on demand"))
        need = min_hot_rows(topology, setup)
        if need:
            items.append(FootprintItem("cold rows' device stack (one layer call)", "device", need * bpe, "derived",
                                       f"the routed NVMe experts of one layer call, streamed to the GPU and run there "
                                       f"(hot_residency._cold_contrib): at most {need} rows x {bpe} B, the same bound as "
                                       "min_hot_rows; transient, so this is its ceiling"))
        if setup.hot_rows < need:
            return Footprint(items=(), refusals=(
                f"hot_rows {setup.hot_rows} is below the {need} a cold layer can route in one step (top_k x "
                f"max(chunk_tokens, max_seqs), at most n_experts): grouped-nf4-gemm's ColdTier would refuse the demand "
                "window mid-request; raise hot_rows",))
    items += _hybrid_host_items(setup.hot_rows, bpe, stride, n_nvme)
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
    hybrid = topology.attention is not None and topology.attention.layers < topology.n_layers
    if str(setup.prefill_graph) != "0" and setup.graphs and setup.max_seqs > 1 and max(setup.buckets) > 1 and not hybrid:
        unmodelled.append("the first-chunk prefill graph's private pool, which the server keeps for its life when the "
                          "graph engages (E4B_PAGED_PREFILL_GRAPH=auto engages it only if that much is still free after "
                          "capture; lane SC2b measured +3.3 GiB on Qwen3-30B-A3B int4); prefill_graph='0' bounds memory "
                          "by this estimate")
    if topology.attention is not None and topology.attention.layers < topology.n_layers:
        unmodelled.append(f"recurrent state of the {topology.n_layers - topology.attention.layers} non-attention layers")
    if setup.placement == "solver":
        unmodelled.append("the CPU tier's compute buffers")
    unmodelled.append("CUDA context, cuBLAS/Triton workspaces and allocator fragmentation (the caller's to add)")
    return Footprint(items=tuple(items), unmodelled=tuple(unmodelled))
