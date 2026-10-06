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
* a prefill/decode working set (heuristic, stated);
* the int4 serving levers when set (``E4B_SERVE_EXP_INT4``, ``E4B_SERVE_ATTN_INT4``, round-to-nearest): the int4-b32
  expert stores that replace the NF4 stacks, with the repack's load-time overlap and its host read; the int4-b32
  attention projections, with the bf16 copy each keeps once a call has more than 16 rows (derived).

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
    #: E4B_SERVE_EXP_INT4: repack every expert stack onto the int4-b32 grid from the source checkpoint at load
    #: (round-to-nearest, all-VRAM only) and free the NF4 stacks (E4B_INT4_KEEP_NF4=0)
    exp_int4: bool = False
    #: E4B_SERVE_ATTN_INT4: store the attention projections on the int4-b32 grid (round-to-nearest)
    attn_int4: bool = False

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
                "E4B_PAGED_HOT_ROWS": str(int(self.hot_rows)),
                "E4B_SERVE_EXP_INT4": "1" if self.exp_int4 else "0", "E4B_SERVE_ATTN_INT4": "1" if self.attn_int4 else "0",
                **({"E4B_INT4_KEEP_NF4": "0"} if self.exp_int4 else {})}


def usable_buckets(max_seqs: int, buckets) -> tuple:
    """The decode-graph buckets a server of ``max_seqs`` sequences can use: those below ``max_seqs``, then ``max_seqs``
    itself, capped at the largest given (a wider step runs in chunks of it). A decode step never carries more rows than
    sequences and the runner pads a step to the next bucket, so a bucket above ``max_seqs`` never runs. It still costs a
    graph and, as the largest, the scratch slots: a full slot of a hybrid model's linear-attention state each. Lane SV3
    measured worse: on Qwen3.6-35B-A3B served for one sequence, buckets 2-16 failed to capture."""
    keep = {int(b) for b in buckets if int(b) < max_seqs}
    return tuple(sorted(keep | {min(int(max_seqs), max(int(b) for b in buckets))}))


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


def linear_state_pool_bytes(layers, n_slots: int, conv_bytes: int = 2) -> int:
    """Device bytes ``engines.linear_state.LinearStatePool`` holds for ``n_slots`` slots once every linear-attention layer
    has run: per layer a conv window ``[n_slots, conv_dim, conv_kernel]`` in the model's dtype (``conv_bytes``; bf16 when
    serving) and a recurrent state ``[n_slots, v_heads, head_k_dim, head_v_dim]`` in fp32. ``layers`` is
    ``MoETopology.linear_state_layers``."""
    return sum(int(n_slots) * (cd * k * conv_bytes + vh * hk * hv * 4) for _layer, cd, k, vh, hk, hv in layers)


#: grouped-nf4-gemm's arena bakes align every row to this many bytes (``nvme_arena`` / ``nvme_bake_nf4`` default)
ARENA_ALIGN = 4096


@functools.lru_cache(maxsize=64)
def _solver_manifest(n_layers: int, n_experts: int, bytes_per_expert: int, vram_gb: float, dram_gb: float) -> dict:
    """:func:`~.engines.placement.solve_placement` as ``build_engine`` calls it, with no routing profile: every expert
    weighs the same and the solver's greedy fills VRAM, then DRAM, then NVMe; the bandwidths it is given decide nothing
    here, so unit overrides stand in for a calibration."""
    from .engines.placement import solve_placement

    return solve_placement(n_layers=n_layers, n_experts=n_experts, bytes_per_expert=bytes_per_expert,
                           vram_budget_bytes=int(vram_gb * 2**30), dram_budget_bytes=int(dram_gb * 2**30),
                           calibration={}, profile_path=None, b_vram_override=1.0, b_dram_override=1.0, batch=1)


def solver_tiers(n_layers: int, n_experts: int, bytes_per_expert: int, vram_gb: float, dram_gb: float) -> dict:
    """Expert rows per tier under ``placement="solver"``, from :func:`~.engines.placement.solve_placement` itself."""
    man = _solver_manifest(n_layers, n_experts, bytes_per_expert, float(vram_gb), float(dram_gb))
    return {t: len(v) for t, v in man["tiers"].items()}


def dram_rows_per_layer(n_layers: int, n_experts: int, bytes_per_expert: int, vram_gb: float, dram_gb: float) -> int:
    """The most DRAM-tier experts any one layer holds under the solver's placement. Without a routing profile the
    solver fills whole layers in order, so this is usually every expert of a layer."""
    from collections import Counter

    man = _solver_manifest(n_layers, n_experts, bytes_per_expert, float(vram_gb), float(dram_gb))
    return max(Counter(int(layer) for layer, _e in man["tiers"]["dram"]).values(), default=0)


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


def staging_bytes_per_token(topology) -> int:
    """bf16 K + V bytes one prompt token stages across the paged pool's attention layers (per-layer geometry honoured)."""
    n = topology.kv_layers
    hs = topology.kv_heads if hasattr(topology.kv_heads, "__len__") else [topology.kv_heads] * n
    ds = topology.kv_head_dims if hasattr(topology.kv_head_dims, "__len__") else [topology.kv_head_dims] * n
    return sum(2 * int(h) * int(d) * 2 for h, d in zip(list(hs)[:n], list(ds)[:n]))


def prefill_staging_tokens(setup) -> int:
    """The most prompt tokens the server can hold in prefill staging at once. The scheduler spends each step's prefill
    budget (``chunk_tokens``) on admitted prompts in order, so one prompt can be finishing while the next starts its
    first chunk; a prompt's staging is freed when it completes."""
    one = setup.max_tokens_per_seq
    return min(one + (setup.chunk_tokens if setup.max_seqs > 1 else 0), setup.max_seqs * one)


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


#: grouped-nf4-gemm's int4-b32 grid (``int4_pack_ref.BLOCK``): one fp16 scale per this many weights along K
INT4_BLOCK = 32
#: host bytes per expert parameter of one MoE layer that ``enable_serve_experts_int4``'s repack holds at its peak
#: (heuristic; measured 10.4-11.2 on OLMoE-1B-7B, the anonymous host peak over the NF4 build's after-load, for a
#: 403M-parameter layer)
INT4_REPACK_HOST_BYTES_PER_PARAM = 12


def int4_store_bytes(n: int, k: int, experts: int = 1) -> int:
    """What ``pack_int4_b32`` returns for ``experts`` weights of ``[n, k]``: two nibbles a byte, one fp16 scale per
    :data:`INT4_BLOCK` weights."""
    return experts * n * (k // 2 + 2 * (k // INT4_BLOCK))


def _int4_split_k(n: int, k: int):
    """``(sk, exact)``: the split count ``int4_b32._plan(n, k)`` gives the single-row GEMV, which is what the int4 stores
    size their partials buffers with; its ceiling 16 where the kernel package (it needs triton) does not import."""
    try:
        from int4_b32 import _plan
    except ImportError:
        return 16, False
    return int(_plan(n, k)[2]), True


def _smallm_split_k(n: int, k: int):
    """``(sk, exact)``: the K16 small-M GEMM's split count for ``[n, k]`` (``int4_smallm.plan_smallm``, as
    ``Int4Linear`` sizes its workspace), or its ceiling 4 where the kernel package does not import."""
    try:
        from int4_smallm import plan_smallm
    except ImportError:
        return 4, False
    return int(plan_smallm(n, k)[2]), True


def _first_out(stack) -> int:
    """The first expert projection's output width (gate+up fused, or up alone), whatever the stack's layout."""
    n = 1
    for d in stack.first_shape:
        n *= d
    return n // (stack.n_experts * stack.hidden)


def _int4_expert_stores(topology):
    """``(device bytes, one layer's stores, exact)`` of the int4-b32 expert stores ``enable_serve_experts_int4`` installs:
    gate/up ``[E, first_out, hidden]`` and down ``[E, hidden, intermediate]``, each with an fp32 partials buffer of
    ``sk x top_k x N``."""
    total, layer_max, exact = 0, 0, True
    for st in topology.expert_stacks:
        n_gu, n_dn = _first_out(st), st.hidden
        sk_gu, ok_gu = _int4_split_k(n_gu, st.hidden)
        sk_dn, ok_dn = _int4_split_k(n_dn, st.intermediate)
        stores = int4_store_bytes(n_gu, st.hidden, st.n_experts) + int4_store_bytes(n_dn, st.intermediate, st.n_experts)
        total += stores + 4 * topology.top_k * (sk_gu * n_gu + sk_dn * n_dn)
        layer_max, exact = max(layer_max, stores), exact and ok_gu and ok_dn
    return total, layer_max, exact


def _int4_attention_workspaces(linears):
    """``(bytes, exact)`` each ``Int4Linear`` preallocates beside its weight: the single-row GEMV's ``sk x N`` fp32
    partials and expert-id scalar, and the K16 small-M route's ``sk x 16 x N`` fp32 workspace and ``cdiv(N, 64)``
    counters (priced as if the route is on, its default wherever the kernel package carries it)."""
    total, exact = 0, True
    for n, k in linears:
        sk, ok = _int4_split_k(n, k)
        sk_sm, ok_sm = _smallm_split_k(n, k)
        total += 4 * sk * n + 4 + 4 * sk_sm * 16 * n + 4 * (-(-n // 64))
        exact = exact and ok and ok_sm
    return total, exact


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
    if topology.paged_state_refusal:
        out.append(f"the paged server refuses this model: {topology.paged_state_refusal}")
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
    if setup.exp_int4:
        if setup.placement != "all-vram":
            out.append("exp_int4 repacks the all-VRAM collapsed stacks only (enable_serve_experts_int4 refuses tiered "
                       "layers); plan placement='all-vram'")
        if topology.expert_bias_tensors:
            out.append("under exp_int4 gpt-oss's experts are served natively as MXFP4, a store this estimate does not "
                       "price")
        if not topology.top_k:
            out.append("exp_int4 sizes its partials by the routed experts per token, which this config does not state")
    if setup.attn_int4:
        if not topology.int4_attention_linears:
            out.append("attn_int4 matches no attention projection on this model (enable_serve_attn_int4 refuses)")
        elif any(k % INT4_BLOCK for _n, k in topology.int4_attention_linears):
            out.append(f"attn_int4 needs every attention projection's input width to be a multiple of {INT4_BLOCK}")
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
    int4_layer = 0
    if setup.placement == "all-vram" and setup.exp_int4:
        stores, int4_layer, exact = _int4_expert_stores(topology)
        items.append(FootprintItem("int4 expert stores (all VRAM; the NF4 stacks freed)", "device", stores,
                                   "derived" if exact else "heuristic",
                                   f"{n_layers} layers x all experts on the int4-b32 grid (packed nibbles + one fp16 "
                                   f"scale per {INT4_BLOCK}), repacked from the source checkpoint, plus each projection's "
                                   "fp32 split-K partials for top_k rows" + ("" if exact else
                                                                             " (split count at its ceiling 16: "
                                                                             "grouped-nf4-gemm's int4_b32 not importable)")))
        n_nvme = 0
    elif setup.placement == "all-vram":
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
        dram_max = dram_rows_per_layer(n_layers, n_experts, bpe, setup.vram_gb, setup.dram_gb)
        if dram_max:
            per_expert = topology.expert_stacks[0].numel // n_experts
            upload = per_expert // 2 + 2 * (per_expert // 64)            # NF4 nibbles + absmax cast to bf16
            rows = (topology.top_k or n_experts) * setup.chunk_tokens
            routed = min(dram_max, rows)
            dram_gpu = routed * upload + rows * topology.hidden_size * (2 + 4 + 4)
            cold = need * bpe if need else 0
            if dram_gpu > cold:
                items.append(FootprintItem(
                    "DRAM experts run on the GPU at prefill (one layer call)", "device", dram_gpu - cold, "derived",
                    f"a prefill chunk computes the DRAM tier's routed experts on the GPU "
                    f"(hybrid._dram_on_gpu): up to {routed} experts x {upload} B (NF4 + bf16 absmax) and the chunk's "
                    f"{rows} routed rows (bf16 input, two fp32 outputs); transient, priced by its excess over the cold "
                    f"rows' stack, since a layer call streams one or the other"))
        if setup.hot_rows < need:
            return Footprint(items=(), refusals=(
                f"hot_rows {setup.hot_rows} is below the {need} a cold layer can route in one step (top_k x "
                f"max(chunk_tokens, max_seqs), at most n_experts): grouped-nf4-gemm's ColdTier would refuse the demand "
                "window mid-request; raise hot_rows",))
    items += _hybrid_host_items(setup.hot_rows, bpe, stride, n_nvme)
    attn = topology.int4_attention_linears if setup.attn_int4 else ()
    attn_numel = sum(n * k for n, k in attn)
    items.append(FootprintItem("dense weights (bf16)", "device", 2 * (topology.dense_numel - attn_numel), "derived",
                               "embeddings, attention, norms, routers, dense/shared MLPs; no adapters under an arena load"
                               + ("; the int4 attention projections' weights are priced below" if attn else "")))
    if attn:
        items.append(FootprintItem("attention projections on the int4-b32 grid", "device",
                                   sum(int4_store_bytes(n, k) for n, k in attn), "derived",
                                   f"{len(attn)} projections (engines.int4_attn.attention_linears), packed nibbles + "
                                   f"one fp16 scale per {INT4_BLOCK}; a projection bias stays bf16 in the dense weights"))
        items.append(FootprintItem("attention projections' bf16 copy (kept from the first prefill)", "device",
                                   2 * attn_numel, "derived",
                                   "Int4Linear serves a call of more than 16 rows (any prefill chunk; more than 1 row "
                                   "without the K16 route) with cuBLAS on a dequantised bf16 weight it builds once and "
                                   "keeps, so the attention weights cost more than bf16 alone once a prompt is served"))
        ws, exact = _int4_attention_workspaces(attn)
        items.append(FootprintItem("int4 attention workspaces", "device", ws, "derived" if exact else "heuristic",
                                   "each projection's single-row split-K partials and its K16 small-M workspace, "
                                   "preallocated at the swap" + ("" if exact else
                                                                 " (split counts at their ceilings: grouped-nf4-gemm's "
                                                                 "int4 kernels not importable)")))
    scratch = max(usable_buckets(setup.max_seqs, setup.buckets)) if setup.graphs else 0      # what the server captures
    kv = paged_kv_pool_bytes(topology.kv_layers, topology.kv_heads, topology.kv_head_dims, batch=setup.max_seqs,
                             max_tokens_per_seq=setup.max_tokens_per_seq,
                             k_groups=None if setup.kv_groups == "auto" else int(setup.kv_groups), scratch_slots=scratch)
    items.append(FootprintItem("FP8 paged KV pool", "device", kv, "derived",
                               f"{topology.kv_layers} layers x ({setup.max_seqs} seqs x {setup.max_tokens_per_seq} tokens"
                               f" + {scratch} graph scratch slots), fp8 payload + fp32 scales (Fp8PagedKV's arithmetic)"))
    if topology.linear_state_layers:
        slots = setup.max_seqs + scratch
        items.append(FootprintItem("linear-attention state pool", "device",
                                   linear_state_pool_bytes(topology.linear_state_layers, slots), "derived",
                                   f"{len(topology.linear_state_layers)} linear-attention layers x {slots} slots "
                                   f"({setup.max_seqs} seqs + {scratch} graph scratch slots): a bf16 conv window and an "
                                   "fp32 recurrent state per slot (engines.linear_state.LinearStatePool, allocated at "
                                   "each layer's first prompt chunk)"))
    staged = prefill_staging_tokens(setup)
    items.append(FootprintItem("prefill staging (bf16 K/V of prompts mid-prefill)", "device",
                               staged * staging_bytes_per_token(topology), "derived",
                               f"a prompt's K/V for every attention layer stays bf16 until the prompt completes "
                               f"(paged_attention's staging buffer); at most {staged} tokens staged at once (one prompt "
                               f"of up to {setup.max_tokens_per_seq} tokens finishing while the next starts a "
                               f"{setup.chunk_tokens}-token chunk); its ceiling, since prompt lengths are the caller's"))
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
    if int4_layer:
        # The repack builds layer by layer before the KV pool exists: each layer's int4 store is allocated while that
        # layer's NF4 stack is still held, and the attention projections are still bf16. Its peak only matters where it
        # exceeds everything the server holds once it serves.
        at_load = slab + 2 * topology.dense_numel + int4_layer
        serving = sum(i.bytes for i in items if i.where == "device")
        if at_load > serving:
            items.append(FootprintItem("int4 repack at load, above the serving total", "device", at_load - serving,
                                       "derived",
                                       f"the NF4 stacks ({slab} B) + bf16 dense weights + one layer's int4 store "
                                       f"({int4_layer} B), held together before the KV pool is built"))
        items.append(FootprintItem("int4 repack: one layer's experts in fp32 (load)", "host",
                                   max(INT4_REPACK_HOST_BYTES_PER_PARAM * st.numel for st in topology.expert_stacks),
                                   "heuristic",
                                   f"{INT4_REPACK_HOST_BYTES_PER_PARAM} B per parameter of the largest layer: the "
                                   "per-expert fp32 reads (4), their fused copy (4), the packed lists and their stack "
                                   "(~1.1) and the bf16 source reads in flight; measured 10.4-11.2 on OLMoE-1B-7B (RTX "
                                   "A2000 host). Gone after load: each layer's heap is handed back (engines.host_heap)"))
    if setup.graphs:
        unmodelled.append("CUDA graph memory pools for the decode buckets (lane SV1: +60 MiB allocated on OLMoE-1B-7B, "
                          "16 seqs, RTX 5090, NF4" + ("; the int4 store's batched decode allocates its split-K partials "
                                                      "through these pools, unmeasured)" if setup.exp_int4 else ")"))
    hybrid = topology.attention is not None and topology.attention.layers < topology.n_layers
    if str(setup.prefill_graph) != "0" and setup.graphs and setup.max_seqs > 1 and max(setup.buckets) > 1 and not hybrid:
        unmodelled.append("the first-chunk prefill graph's private pool, which the server keeps for its life when the "
                          "graph engages (E4B_PAGED_PREFILL_GRAPH=auto engages it only if that much is still free after "
                          "capture; lane SV1 measured +0.24 GiB on OLMoE-1B-7B and +0.57 GiB on Qwen3-30B-A3B at NF4, "
                          "SC2b +3.3 GiB on Qwen3-30B-A3B int4); prefill_graph='0' bounds memory by this estimate")
    if topology.attention is not None and topology.attention.layers < topology.n_layers \
            and len(topology.linear_state_layers) < topology.n_layers - topology.attention.layers:
        unmodelled.append(f"recurrent state of the {topology.n_layers - topology.attention.layers} non-attention layers "
                          f"({len(topology.linear_state_layers)} priced as the linear-attention state pool)")
    if setup.placement == "solver":
        unmodelled.append("the CPU tier's compute buffers")
    if setup.exp_int4:
        unmodelled.append("the source checkpoint on local disk: the int4 repack reads its safetensors (snapshot_download), "
                          "never the arena")
    unmodelled.append("the bulk KV flush transient (E4B_PAGED_BULK_KV, on by default since SC2c/SC2d): one prompt's K/V "
                      "staged for the single bulk write; lane SC2d recorded 168 MiB on gpt-oss-20b")
    unmodelled.append("CUDA context, cuBLAS/Triton workspaces and allocator fragmentation (the caller's to add)")
    return Footprint(items=tuple(items), unmodelled=tuple(unmodelled))
