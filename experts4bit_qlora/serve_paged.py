# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""OpenAI-compatible HTTP server over the continuous-batching engine (opt-in, v1).

``experts4bit_qlora.serve`` admits one generation at a time through HF ``generate``. This module
serves the OTHER engine -- :class:`~.engines.scheduler.ContinuousScheduler` driving
:class:`~.engines.paged_runner.PagedModelRunner` over :class:`~.engines.fp8_paged_kv.Fp8PagedKV`
-- behind ``/v1/completions`` so that request-level serving benchmarks (TTFT, inter-token latency
and throughput under Poisson arrivals, as ``vllm bench serve`` and ``sglang.bench_serving`` drive
them) run against e4b exactly as they run against vLLM, SGLang and llama.cpp. It is the server the
serving campaign benchmarks against; it is not a general-purpose deployment.

::

    E4B_PAGED_MODEL=Qwen/Qwen3-30B-A3B E4B_PAGED_ARENA=/arenas/qwen3.nf4 E4B_PAGED_CALIB=/calib.json \\
      python -m experts4bit_qlora.serve_paged                      # 127.0.0.1:8778

**The served stack is the harness's stack.** :func:`build_engine` reproduces the construction
``bench/p39/step_decomp.py`` and ``bench/p44/serve_stack.build_served_model`` use, in the same
order: ``load_moe_4bit_streaming`` through the NF4 arena -> the placement solver, then every
expert moved into the VRAM tier (``E4B_PAGED_PLACEMENT=all-vram``, the point every certified
serving number was measured at) -> ``enable_hybrid_tier`` -> the int4 levers the lane hook
(``bench/p42/hook/usercustomize.py``) applies right after the tier, read from the SAME environment
names (``E4B_SERVE_EXP_INT4``, ``E4B_SERVE_EXP_INT4_CALIB``, ``E4B_SERVE_ATTN_INT4``,
``E4B_SERVE_ATTN_INT4_CALIB``, ``E4B_SERVE_LMHEAD_INT4_CALIB``, ``E4B_SERVE_DENSE_INT4_CALIB``,
``E4B_CALIB_NSEQ``, ``E4B_CALIB_SOURCE``, ``E4B_INT4_ARTIFACT_DIR``, ``E4B_INT4_EXPECTED_FINGERPRINT``,
``E4B_INT4_DUMP_ARTIFACT_DIR``, ``E4B_CALIB_LAYERS_PER_PASS``; ``E4B_INT4_ASSIGNMENT`` is read by the
library itself) -> amortisation counters off (``--amort off``, the production shape) -> the paged
attention registered -> the fusions, at ONE assembly point as the harness has them: with
``E4B_PAGED_FUSE_QKV=1``, ``fuse_qkv`` (lane P54), which applies the three env-gated folds
(``E4B_FUSE_T1_GLUE``, ``E4B_FUSE_T1_GLUE_R2``, ``E4B_FUSE_ROUTER_EPI``) ITSELF after fusing -- the
registered B=1 fused stack (P54 / P58 / P88) is ``--fuse-qkv`` WITH those flags set; without it, the
three folds called directly. The census reports what each fold returned either way (in the fused
branch the fold functions are wrapped on their modules for the duration of the ``fuse_qkv`` call,
which imports them inside its body) -> ``Fp8PagedKV(batch=max_seqs, scratch_slots=max(buckets))`` ->
``PagedModelRunner`` -> ``enable_decode_graphs`` when graphs are on (``E4B_PAGED_GRAPHS``: ``auto``, the default since lane
P109, captures them on a CUDA device at the all-resident placement; with more than one sequence, after the batched
lane's sync-free device grouping is switched on, as ``_bv3_stage`` does) -> the scheduler. A
lever that is set and patches nothing RAISES at startup (the lanes' ``_lever_check`` rule); the
census -- how many modules each lever patched -- is reported at ``GET /health`` so a reader can
tell which stack answered. The four fusion knobs (``E4B_PAGED_FUSE_QKV`` and the three folds) each
also take ``auto`` (:func:`_fusion_env`): apply where the module structure and the installed kernels
license it, patch nothing -- without raising -- where they do not; ``1`` keeps the refusal. Unset, each resolves
per family at the build (:func:`resolve_fusion_modes`): ``auto`` on a family with a registered passing read
(:data:`FUSION_DEFAULT_FAMILIES`, lane P115) and ``0`` everywhere else. Explicit ``auto`` stays structural and, off
that list, logs one warning naming the read the family lacks or failed. ``/health`` reports each knob's resolution,
its source and what each fold skipped.

**Semantics (stated, not silently approximated).** Greedy only: ``temperature`` must be 0 or
absent (the runner argmaxes; a nonzero temperature is a 400, never ignored). ``max_tokens`` is
honoured exactly; ``ignore_eos`` runs the request to ``max_tokens``; ``min_tokens`` suppresses the
EOS stop until that many tokens are out. ``logprobs``, ``echo``, ``n > 1``, ``best_of > 1``,
``suffix``, stop STRINGS and the penalties are 400s (``stop_token_ids`` is honoured; ``top_p`` /
``top_k`` / ``seed`` are accepted because they cannot change a greedy result). A prompt is a string
(tokenised with the model's tokenizer, ``add_special_tokens`` per request, default on as vLLM's
completions endpoint has it) or a list of token ids. Streaming emits one SSE chunk per generated
token with the incremental text delta (the detokenizer keeps a window over the last tokens and
holds a delta back while it ends in U+FFFD, so a code point split across byte-fallback tokens is
never emitted in pieces), ``finish_reason`` on the last token's chunk, a ``usage``-only chunk when
``stream_options.include_usage`` is set, then ``data: [DONE]``.

**Engine facts a benchmark reader needs (read from the engine code, 2026-10-01):**

* **EOS.** Before this module the engine had no stop set: ``PagedModelRunner`` stores ``eos_id``
  and never consults it, and ``ContinuousScheduler._emit`` ended a sequence only at
  ``max_new_tokens``. Every registered serving measurement ran that way (fixed output length). The
  fix is additive in the scheduler -- ``add_request(..., stop_ids=, min_tokens=)`` -- so an
  EOS-finished request frees its KV slot in the step that produced the EOS rather than decoding
  wasted tokens to ``max_tokens``; the default (``stop_ids=None``) is the old contract and the
  harness is unchanged. The stop token is kept in ``out`` and counted in ``completion_tokens``
  (vLLM's convention); its text is dropped with ``skip_special_tokens``.
* **Length.** A sequence owns ``max_tokens_per_seq`` tokens of KV. The KV's ``append`` raises
  ``ValueError("... overflows its ... blocks")`` INSIDE an engine step, which is an engine fault
  for every resident request, so admission refuses (400) any request with
  ``prompt_len + max_tokens > E4B_PAGED_MAX_TOKENS_PER_SEQ`` before it reaches the scheduler.
  A prefilling prompt's bf16 K/V is staged on the GPU until the prompt completes (then quantised
  into the pool once); that staging is the engine's VRAM cost per concurrently prefilling prompt.
* **Prefill.** A prompt is ingested ``chunk_tokens`` per step (``E4B_PAGED_CHUNK_TOKENS``), and a
  step's total prefill budget is ``max_prefill_tokens_per_step`` (default = chunk, the harness's
  ``--chunk``). The scheduler prefers prefill over decode within a step and each chunk is its own
  batch-1 forward, so resident decoders see one prefill forward of extra latency per step while a
  prompt is ingesting -- the TTFT/ITL trade the chunk size sets.
* **Capacity.** Admission is bounded by ``max_seqs`` AND free KV slots (``kv_slots = max_seqs``);
  a request past capacity WAITS in the scheduler's FIFO queue and its TTFT includes that wait
  (``Request.arrival`` is stamped at HTTP arrival and handed to ``add_request(now=...)``). There
  is no eviction or preemption anywhere in the engine and this server adds none.
  ``E4B_PAGED_MAX_QUEUE`` (default 0 = unbounded) can cap in-flight requests with a 503.
* **Graphs.** ``enable_decode_graphs`` captures one CUDA graph per bucket on scratch slots and pads
  a decode step to the next bucket; an active set larger than the largest bucket is split into
  consecutive chunks of at most that size (two replays per step at ``max_seqs=32``, buckets up to
  16). A bucket whose capture failed runs the same padded step eagerly and says so in
  ``/health`` (``engine.graph_status``).
* **Clocks.** ``GET /stats`` returns ``scheduler.stats()`` (TTFT p50/p99 FROM ARRIVAL, queue wait,
  per-stream rate) plus the runner's ``graph_stats``; ``E4B_PAGED_TRACE=<path>`` appends one JSON
  line per finished request (arrival / admitted_at / first_token_at / finished_at on the engine's
  monotonic clock, ``arrival_epoch`` on the wall clock, prompt_len, out_len, finish_reason) so
  server-side TTFT and ITL can be read beside the client's. ``E4B_PAGED_STEP_TRACE=<path>`` appends one JSON
  line per engine STEP (:mod:`.engines.step_trace`): what it carried, its host time by segment, and when the GPU
  finished each part, read after the step's own syncs.
* **KV bookkeeping.** Per request the engine resets the slot at admission and at finish, flushes the prompt's
  K/V into the FP8 pool, and (with decode graphs) claims every block the slot can reach at its first decode.
  Per layer and per block that is ~13.5k host-issued launches a request on Qwen3-30B-A3B at 2048 tokens a slot,
  serialized ahead of every resident decode (``bench/stall-census-2026-10-05``). ``E4B_PAGED_BULK_KV=1`` does
  the same work in a launch count independent of layers and blocks, leaving the same pool, tables and lengths. It is
  the default since lanes SC2c (#1166: DEFAULT_LICENSED) and SC2d (#1192: engaged and output-identical on a hybrid
  and on gpt-oss); ``0`` keeps the per-layer path.
* **Decode lookahead.** A decode step reads its tokens back before the step ends, so the GPU idles while the host
  emits them, retires finished requests, plans the next step and copies its inputs in. ``E4B_PAGED_DECODE_LOOKAHEAD=1``
  (opt-in until lane P118 reads it; needs decode graphs) issues the next decode step before reading the previous one
  back, its input ids taken on the device (:meth:`~.engines.paged_runner.PagedModelRunner.issue_decode`). Tokens and
  finish reasons are the synchronous path's; a request that ends on a stop id has had one more step computed and
  discarded, and frees its slot one step later.

Not in v1: sampling, logprobs, stop strings, adapters, prefix caching, per-request timeouts.
Everything above the engine seam is testable on CPU with a fake runner (``tests/test_serve_paged.py``);
:func:`build_engine` is the one function that needs a GPU, and it was written from the harness
rather than measured here.
"""
import asyncio
import collections
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .engines.scheduler import ContinuousScheduler, Request

DEFAULT_BUCKETS = (1, 2, 4, 8, 16)

# lever flags the lane hook reads; snapshotted into /health so the served stack is legible
LEVER_ENV = ("E4B_SERVE_EXP_INT4", "E4B_SERVE_EXP_INT4_CALIB", "E4B_SERVE_ATTN_INT4", "E4B_SERVE_ATTN_INT4_CALIB",
             "E4B_SERVE_LMHEAD_INT4_CALIB", "E4B_SERVE_DENSE_INT4_CALIB", "E4B_CALIB_NSEQ", "E4B_CALIB_SOURCE",
             "E4B_INT4_ARTIFACT_DIR", "E4B_INT4_EXPECTED_FINGERPRINT", "E4B_INT4_DUMP_ARTIFACT_DIR",
             "E4B_INT4_ASSIGNMENT", "E4B_CALIB_LAYERS_PER_PASS", "E4B_INT4_KEEP_NF4",
             "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI", "E4B_FUSED_KV_APPEND")
FUSION_ENV = ("E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
FUSE_QKV_ENV = "E4B_PAGED_FUSE_QKV"
FUSION_KNOBS = (FUSE_QKV_ENV,) + FUSION_ENV
# Lane P115's family-scoped default (PREREG-p115.md Amendment 3, mechanism (B)): an UNSET fusion knob resolves to
# ``auto`` on a model_type with a registered passing read and to ``0`` everywhere else. Explicit ``auto`` stays
# structural, as Phase C measured it.
FUSION_DEFAULT_FAMILIES = {
    "qwen3_moe": "P115 Phases A and B: speed and quality (#1328)",
    "qwen3_5_moe": "P115 Phase C: engagement and SANE (#1342)",
    "granitemoe": "P115 Phase C: Phase B's quality read, GRANITE_LICENSED (#1342)",
}
# families with a registered read that FAILED, named in the warning an explicit ``auto`` logs on them
FUSION_FAILED_READS = {
    "gpt_oss": "P115 Phase C: SANE argmax agreement 0.924 < 0.95 (#1342)",
}
FUSION_UNSET = "default"          # from_env's value for an unset knob, resolved per family by resolve_fusion_modes
_FUSION_WARNED: set = set()


def _capability(device: str):
    from .engines.fp8_paged_kv import cuda_capability

    return cuda_capability(device)


def _graphs_env(value: str, device: str, placement: str, capability=None) -> bool:
    """``E4B_PAGED_GRAPHS``: ``auto`` (the default since lane P109, also when unset or empty) captures bucketed decode
    graphs on a CUDA device at the ``all-vram`` placement and decodes eagerly anywhere else; ``1`` forces them (and is
    refused where batched graphs are refused); ``0`` keeps eager decode. Anything else is refused rather than read as one
    of these. P109 (e4b#770, ``bench/p109/RESULTS-p109.md``) read the default server both ways on an RTX 5090.

    ``capability`` ((major, minor), None if unknown): bucketed graphs need the fused KV append, which needs sm_89+
    (``fp8_paged_kv.fused_append_unsupported``). Below that ``auto`` decodes eagerly and ``1`` is refused."""
    from .engines.fp8_paged_kv import fused_append_unsupported

    v = (value or "auto").strip().lower() or "auto"
    why = fused_append_unsupported(capability)
    if v == "auto":
        return str(device).startswith("cuda") and placement == "all-vram" and why is None
    if v == "1" and why:
        raise ValueError(f"E4B_PAGED_GRAPHS=1 but bucketed decode graphs need the fused KV append, and {why}")
    if v in ("0", "1"):
        return v == "1"
    raise ValueError(f"E4B_PAGED_GRAPHS={value!r}: expected 'auto', '0' or '1'")


def _last_logits_env(value: str) -> bool:
    """``E4B_PAGED_LAST_LOGITS``: opt-in final-position prefill logits (0/1).

    Off until a served-prefill quality read licenses the LM-head shape change.
    Forced on requires an explicit supported model.forward keyword at startup.
    """
    v = (value or "0").strip() or "0"
    if v not in ("0", "1"):
        raise ValueError(f"E4B_PAGED_LAST_LOGITS={value!r}: expected '0' or '1'")
    return v == "1"


def _prefill_graph_env(value: str) -> str:
    """``E4B_PAGED_PREFILL_GRAPH``, one of three settings. Every first chunk of exactly ``E4B_PAGED_CHUNK_TOKENS``
    tokens is served from one CUDA graph (:meth:`~.engines.paged_runner.PagedModelRunner.enable_prefill_graph`) when it
    engages:

    * ``auto`` (the default since lane SC2b, also when unset or empty) engages wherever its startup check passes and
      the device keeps as much memory free as the graph's pool takes. Otherwise prefill stays eager, the server
      still starts, and ``/health`` reports ``refused`` with the reason. SC2b (#846, ``bench/sc2/``, one RTX 5090,
      Qwen3-30B-A3B int4, 16 sequences) read serial TTFT 1.30-1.65x faster with byte-identical text in both draws,
      the capacity ceiling unchanged, and +3.3 GiB of VRAM for the graph's pool.
    * ``1`` engages, or refuses at startup with the reason.
    * ``0`` prefills eagerly.

    Anything else is refused rather than read as one of these."""
    v = (value or "auto").strip().lower() or "auto"
    if v in ("auto", "0", "1"):
        return v
    raise ValueError(f"E4B_PAGED_PREFILL_GRAPH={value!r}: expected 'auto', '0' or '1'")


def _max_seqs_env(value: str) -> int:
    """``E4B_PAGED_MAX_SEQS``: ``auto`` (the default since lane SC2e, also when unset or empty) or a positive int.
    ``auto`` is resolved when the engine builds (:func:`resolve_max_seqs`): the widest width lane SC2e read that the
    serve estimate fits in the device's free memory (64, 32 or 16 with the default ``E4B_PAGED_BUCKETS=auto``; 64 or 16
    with an explicit list such as ``1,2,4,8,16``). Until then the config carries 16. ``16`` restores the old default;
    anything else is refused rather than guessed."""
    v = (value or "").strip().lower() or "auto"
    if v == "auto":
        return 16
    try:
        n = int(v)
    except ValueError:
        raise ValueError(f"E4B_PAGED_MAX_SEQS={value!r}: expected 'auto' or a positive int") from None
    if n < 1:
        raise ValueError("E4B_PAGED_MAX_SEQS must be >= 1")
    return n


def resolve_max_seqs(cfg: "PagedServeConfig", model_config, free_bytes) -> dict:
    """``E4B_PAGED_MAX_SEQS=auto``: set ``cfg.max_seqs`` by :func:`~.serve_recipe.choose_max_seqs` on the model's own
    topology and the device's free memory (measured before any weight is loaded), then re-derive the decode buckets
    for that width from ``E4B_PAGED_BUCKETS`` as given. An explicit width is left alone. Returns the record /health
    reports. Lane SC2e (#846) licensed it: on Qwen3-30B-A3B int4, one RTX 5090, 512-token prompts and 2,048 tokens a
    slot, 64 slots on the default bucket list held the SLO to 8 req/s against 4 at 16 slots, with identical serial
    output."""
    if str(cfg.max_seqs_requested).strip().lower() != "auto":
        return {}
    from .serve_recipe import MAX_SEQS_AUTO_WIDTHS, MAX_SEQS_AUTO_WIDTHS_AUTO_BUCKETS, ServeSetup, choose_max_seqs
    env = os.environ.get
    try:
        from .arch.topology import describe_moe
        topo = describe_moe(model_config)
    except Exception as e:                                 # a config describe_moe cannot read: today's default
        res = {"max_seqs": 16, "free_bytes": free_bytes, "candidates": [],
               "why": f"the model's topology could not be described ({type(e).__name__}: {str(e)[:160]}): 16"}
    else:
        req = cfg.buckets_requested if cfg.buckets_requested != "default" else ""
        b = _buckets_env(req)
        setup = ServeSetup(placement=cfg.placement, max_seqs=16, max_tokens_per_seq=cfg.max_tokens_per_seq,
                           chunk_tokens=cfg.chunk_tokens, graphs=cfg.graphs, buckets=b, kv_groups=cfg.kv_groups,
                           prefill_graph=cfg.prefill_graph, vram_gb=cfg.vram_gb, dram_gb=cfg.dram_gb,
                           hot_rows=cfg.hot_rows, bulk_kv=cfg.bulk_kv,
                           exp_int4="1" in (env("E4B_SERVE_EXP_INT4", "0"), env("E4B_SERVE_EXP_INT4_CALIB", "0")),
                           attn_int4="1" in (env("E4B_SERVE_ATTN_INT4", "0"), env("E4B_SERVE_ATTN_INT4_CALIB", "0")))
        widths = MAX_SEQS_AUTO_WIDTHS_AUTO_BUCKETS if str(b).strip().lower() == "auto" else MAX_SEQS_AUTO_WIDTHS
        res = choose_max_seqs(topo, setup, free_bytes, widths=widths)
    cfg.max_seqs = int(res["max_seqs"])
    cfg.buckets = _buckets_env(cfg.buckets_requested if cfg.buckets_requested != "default" else "")
    cfg.validate()
    cfg.max_seqs_resolution = res
    log(f"E4B_PAGED_MAX_SEQS=auto -> {cfg.max_seqs} ({res.get('why')}); buckets {list(cfg.buckets)}")
    return res


def _buckets_env(value: str):
    """``E4B_PAGED_BUCKETS``: ``auto`` (the default since lane P117, also when unset or empty): every power of two
    below ``max_seqs``, then ``max_seqs`` itself (:func:`~.serve_recipe.default_buckets`), so the widest decode step is
    one graph replay. Or a comma-separated list, trimmed to what ``max_seqs`` sequences can use
    (:func:`~.serve_recipe.usable_buckets`); ``1,2,4,8,16`` restores the old default, which runs a step above 16 rows
    as consecutive 16-row replays. Up to 16 sequences the two read the same buckets. Lane SC2e (#846) read the speed
    (64 slots: 12 req/s with ``auto``, 8 on the list) and lane P117 the quality (AT_PARITY at buckets 32 and 64), on
    Qwen3-30B-A3B int4 on an RTX 5090. Anything else is refused rather than guessed."""
    v = (value or "").strip()
    if not v:
        return "auto"
    if v.lower() == "auto":
        return "auto"
    try:
        return tuple(int(x) for x in v.split(",") if x.strip())
    except ValueError:
        raise ValueError(f"E4B_PAGED_BUCKETS={value!r}: expected 'auto' or comma-separated positive ints") from None


def _bulk_kv_env(value: str) -> bool:
    """``E4B_PAGED_BULK_KV``: ``1`` runs a request's KV bookkeeping in bulk (:class:`~.engines.paged_runner.PagedModelRunner`
    ``bulk_kv``): the slot reset at admission and at finish, the prompt's flush into the FP8 pool, and, with decode
    graphs, the claim of every block the slot can reach, each in a launch count independent of layers and blocks. The
    pool, tables and lengths it leaves are the per-layer path's. ``1`` is the default, also when unset or empty, since
    lanes SC2c (#1166) and SC2d (#1192); ``0`` keeps the per-layer path. The stall census (``bench/stall-census-2026-10-05``) counted ~13.5k host-issued launches of
    per-layer bookkeeping per request on Qwen3-30B-A3B at 2048 tokens a slot. Anything else is refused."""
    v = (value or "1").strip() or "1"
    if v in ("0", "1"):
        return v == "1"
    raise ValueError(f"E4B_PAGED_BULK_KV={value!r}: expected '0' or '1'")


def _lookahead_env(value: str) -> bool:
    """``E4B_PAGED_DECODE_LOOKAHEAD``: ``1`` issues each decode step before the previous one's tokens are read back
    (:class:`~.engines.scheduler.ContinuousScheduler` ``lookahead``), so the GPU has a step queued while the host works
    between steps; it needs decode graphs. ``0`` (the default, also when unset or empty) reads every step back before
    the next is issued. Opt-in until lane P118 (e4b#1313) reads it. Anything else is refused."""
    v = (value or "0").strip() or "0"
    if v in ("0", "1"):
        return v == "1"
    raise ValueError(f"E4B_PAGED_DECODE_LOOKAHEAD={value!r}: expected '0' or '1'")


def _fusion_knob_from_env(name: str, raw) -> str:
    """One fusion knob as ``from_env`` reads it: unset or empty -> :data:`FUSION_UNSET` (resolved per family at the
    build), otherwise :func:`_fusion_env`'s three settings."""
    return FUSION_UNSET if not (raw or "").strip() else _fusion_env(name, raw)


def resolve_fusion_modes(modes: dict, model_type) -> tuple:
    """``(resolved, sources)`` for the fusion knobs in ``modes``. :data:`FUSION_UNSET` resolves to ``auto`` on a
    ``model_type`` in :data:`FUSION_DEFAULT_FAMILIES` (source ``default-allowlisted``) and to ``0`` elsewhere
    (``default-off``); an explicit value passes through (``explicit``). Explicit ``auto`` off the list logs one warning
    per family per process, naming the read that family lacks or failed."""
    resolved, sources = {}, {}
    allowed = model_type in FUSION_DEFAULT_FAMILIES
    for k, v in modes.items():
        if v == FUSION_UNSET:
            resolved[k] = "auto" if allowed else "0"
            sources[k] = "default-allowlisted" if allowed else "default-off"
        else:
            resolved[k] = v
            sources[k] = "explicit"
    explicit_auto = [k for k, v in modes.items() if v == "auto"]
    if explicit_auto and not allowed and model_type not in _FUSION_WARNED:
        _FUSION_WARNED.add(model_type)
        why = FUSION_FAILED_READS.get(model_type, "no registered read")
        log(f"WARNING {', '.join(explicit_auto)}=auto on model_type {model_type!r}: {why}. Explicit auto is structural; "
            f"it is quality-licensed only on {sorted(FUSION_DEFAULT_FAMILIES)} (lane P115)")
    return resolved, sources


def _fuse_qkv_health(cfg, info: dict):
    """``/health``'s ``engine.fuse_qkv``: the q/k/v fusion as the build resolved it, not ``cfg.fuse_qkv`` (which an
    unset knob sets, to be resolved per family). Before the build an unset knob is unresolved (``None``); a config
    without modes keeps ``cfg.fuse_qkv``."""
    mode = (info.get("fusion_modes") or {}).get(FUSE_QKV_ENV) or (cfg.fusion_modes or {}).get(FUSE_QKV_ENV)
    if mode is None:
        return cfg.fuse_qkv
    return None if mode == FUSION_UNSET else mode != "0"


def _fusion_env(name: str, value: str) -> str:
    """One fusion knob (``E4B_PAGED_FUSE_QKV``, ``E4B_FUSE_T1_GLUE``, ``E4B_FUSE_T1_GLUE_R2``, ``E4B_FUSE_ROUTER_EPI``),
    one of three settings (:func:`~.engines.glue_fuse.fold_mode`):

    * ``0`` (the default, also when unset or empty) leaves the model's own modules;
    * ``1`` applies the fusion and refuses at startup if it patches nothing or the installed kernels lack it (the
      lanes' rule; the registered B=1 fused stack is all four at ``1``, P54 / P58 / P88);
    * ``auto`` applies it where the module structure and the installed kernels license it and patches nothing,
      without raising, where they do not -- another family, an older kernel cut. ``/health`` says what it skipped.

    Anything else is refused rather than read as one of these."""
    from .engines.glue_fuse import fold_mode

    return fold_mode(name, value or "")


def log(msg: str) -> None:
    print(f"[serve_paged] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PagedServeConfig:
    """Every knob, read once at startup, never at import."""

    model: str = ""                      # E4B_PAGED_MODEL (HF id or local snapshot)
    arena: str = ""                      # E4B_PAGED_ARENA (the NF4 arena; <arena>.index.json beside it)
    calib: str = ""                      # E4B_PAGED_CALIB (placement calibration JSON; the solver refuses to guess)
    revision: str = ""                   # E4B_PAGED_REVISION
    served_names: tuple = ()             # E4B_PAGED_SERVED_NAME: extra names accepted in `model` (comma-separated)
    max_seqs: int = 16                   # E4B_PAGED_MAX_SEQS: batch width == KV slots ("auto": resolve_max_seqs)
    max_seqs_requested: str = "16"       # E4B_PAGED_MAX_SEQS as given; from_env's default is "auto" (lane SC2e)
    max_seqs_resolution: dict = field(default_factory=dict)   # resolve_max_seqs's record, for /health
    max_tokens_per_seq: int = 4096       # E4B_PAGED_MAX_TOKENS_PER_SEQ: prompt + output per sequence
    chunk_tokens: int = 512              # E4B_PAGED_CHUNK_TOKENS
    max_prefill_tokens: int = 0          # E4B_PAGED_MAX_PREFILL_TOKENS: per-step budget; 0 -> chunk_tokens
    graphs: bool = False                 # E4B_PAGED_GRAPHS: from_env resolves auto (the default) / 1 / 0 (_graphs_env)
    prefill_graph: str = "auto"          # E4B_PAGED_PREFILL_GRAPH: auto (default) / 1 / 0 (_prefill_graph_env)
    last_logits: bool = False            # E4B_PAGED_LAST_LOGITS: opt-in prefill head at one position
    buckets: tuple = DEFAULT_BUCKETS     # in code: the list unless "auto" is passed; from_env: "auto" when unset (_buckets_env)
    buckets_requested: str = "default"   # E4B_PAGED_BUCKETS as given ("default" when unset or empty); /health reports it
    placement: str = "all-vram"          # E4B_PAGED_PLACEMENT: all-vram | solver
    vram_gb: float = 1.2                 # E4B_PAGED_VRAM_GB (solver budget; the harness default)
    dram_gb: float = 6.0                 # E4B_PAGED_DRAM_GB
    hot_rows: int = 64                   # E4B_PAGED_HOT_ROWS
    kv_groups: str = "auto"              # E4B_PAGED_KV_GROUPS: auto | <int>
    fuse_qkv: bool = False               # E4B_PAGED_FUSE_QKV not 0 (fuse_qkv applies the env-gated folds itself)
    fusion_modes: dict = field(default_factory=dict)   # FUSION_KNOBS -> auto | 0 | 1 (_fusion_env), or "default" when
                                                       # unset (resolved per family at the build); a knob not named keeps
                                                       # its old reading: fuse_qkv, or the fold's own env var
    torch_threads: int = 8               # E4B_PAGED_TORCH_THREADS
    max_tokens_cap: int = 0              # E4B_PAGED_MAX_TOKENS: 0 -> max_tokens_per_seq - 1 (refuses, never clamps)
    max_queue: int = 0                   # E4B_PAGED_MAX_QUEUE: in-flight cap, 0 = unbounded
    eos_ids: tuple = ()                  # E4B_PAGED_EOS_IDS: override the model's EOS set
    host: str = "127.0.0.1"              # E4B_HOST
    port: int = 8778                     # E4B_PORT (serve.py takes 8777)
    token: str = ""                      # E4B_TOKEN: bearer on /v1/*
    trace_path: str = ""                 # E4B_PAGED_TRACE: per-request JSONL
    step_trace_path: str = ""            # E4B_PAGED_STEP_TRACE: per-step JSONL (engines.step_trace)
    bulk_kv: bool = True                 # E4B_PAGED_BULK_KV: 1 (default since SC2c/SC2d) / 0 (_bulk_kv_env)
    decode_lookahead: bool = False       # E4B_PAGED_DECODE_LOOKAHEAD: 0 (default) / 1 (_lookahead_env)
    device: str = "cuda"

    @classmethod
    def from_env(cls) -> "PagedServeConfig":
        env = os.environ.get

        def _ints(s):
            return tuple(int(x) for x in s.split(",") if x.strip())

        modes = {k: _fusion_knob_from_env(k, env(k, "")) for k in FUSION_KNOBS}
        cfg = cls(
            model=env("E4B_PAGED_MODEL", ""),
            arena=env("E4B_PAGED_ARENA", ""),
            calib=env("E4B_PAGED_CALIB", ""),
            revision=env("E4B_PAGED_REVISION", ""),
            served_names=tuple(x.strip() for x in env("E4B_PAGED_SERVED_NAME", "").split(",") if x.strip()),
            max_seqs=_max_seqs_env(env("E4B_PAGED_MAX_SEQS", "")),
            max_seqs_requested=(env("E4B_PAGED_MAX_SEQS", "") or "").strip().lower() or "auto",
            max_tokens_per_seq=int(env("E4B_PAGED_MAX_TOKENS_PER_SEQ", "4096")),
            chunk_tokens=int(env("E4B_PAGED_CHUNK_TOKENS", "512")),
            max_prefill_tokens=int(env("E4B_PAGED_MAX_PREFILL_TOKENS", "0")),
            graphs=_graphs_env(env("E4B_PAGED_GRAPHS", "auto"), env("E4B_PAGED_DEVICE", "cuda"),
                               env("E4B_PAGED_PLACEMENT", "all-vram"), _capability(env("E4B_PAGED_DEVICE", "cuda"))),
            prefill_graph=_prefill_graph_env(env("E4B_PAGED_PREFILL_GRAPH", "auto")),
            last_logits=_last_logits_env(env("E4B_PAGED_LAST_LOGITS", "0")),
            buckets=_buckets_env(env("E4B_PAGED_BUCKETS", "")),
            buckets_requested=(env("E4B_PAGED_BUCKETS", "") or "").strip() or "default",
            placement=env("E4B_PAGED_PLACEMENT", "all-vram"),
            vram_gb=float(env("E4B_PAGED_VRAM_GB", "1.2")),
            dram_gb=float(env("E4B_PAGED_DRAM_GB", "6.0")),
            hot_rows=int(env("E4B_PAGED_HOT_ROWS", "64")),
            kv_groups=env("E4B_PAGED_KV_GROUPS", "auto"),
            fuse_qkv=modes[FUSE_QKV_ENV] != "0",
            fusion_modes=modes,
            torch_threads=int(env("E4B_PAGED_TORCH_THREADS", "8")),
            max_tokens_cap=int(env("E4B_PAGED_MAX_TOKENS", "0")),
            max_queue=int(env("E4B_PAGED_MAX_QUEUE", "0")),
            eos_ids=_ints(env("E4B_PAGED_EOS_IDS", "")),
            host=env("E4B_HOST", "127.0.0.1"),
            port=int(env("E4B_PORT", "8778")),
            token=env("E4B_TOKEN", ""),
            trace_path=env("E4B_PAGED_TRACE", ""),
            step_trace_path=env("E4B_PAGED_STEP_TRACE", ""),
            bulk_kv=_bulk_kv_env(env("E4B_PAGED_BULK_KV", "1")),
            decode_lookahead=_lookahead_env(env("E4B_PAGED_DECODE_LOOKAHEAD", "0")),
            device=env("E4B_PAGED_DEVICE", "cuda"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.max_seqs < 1:
            raise ValueError("E4B_PAGED_MAX_SEQS must be >= 1")
        if self.max_tokens_per_seq < 2:
            raise ValueError("E4B_PAGED_MAX_TOKENS_PER_SEQ must be >= 2 (one prompt token + one output token)")
        if self.chunk_tokens < 1:
            raise ValueError("E4B_PAGED_CHUNK_TOKENS must be >= 1")
        if self.max_prefill_tokens < 0:
            raise ValueError("E4B_PAGED_MAX_PREFILL_TOKENS must be >= 0 (0 = chunk_tokens)")
        from .serve_recipe import default_buckets, usable_buckets
        if isinstance(self.buckets, str):
            if self.buckets.strip().lower() != "auto":
                raise ValueError(f"E4B_PAGED_BUCKETS={self.buckets!r}: expected 'auto' or comma-separated positive ints")
            self.buckets = default_buckets(self.max_seqs)
        if not self.buckets or min(self.buckets) < 1:
            raise ValueError(f"E4B_PAGED_BUCKETS must be positive ints, got {self.buckets}")
        usable = usable_buckets(self.max_seqs, self.buckets)
        if tuple(self.buckets) != usable:
            # a bucket above max_seqs never runs; it costs a graph and scratch slots, and on a hybrid model served for
            # one sequence it failed to capture (lane SV3)
            log(f"buckets {list(self.buckets)} -> {list(usable)}: none above max_seqs={self.max_seqs}")
            self.buckets = usable
        if self.graphs and self.max_seqs > max(self.buckets):
            log(f"decode steps above {max(self.buckets)} rows run as consecutive {max(self.buckets)}-row replays "
                f"(max_seqs={self.max_seqs}); E4B_PAGED_BUCKETS=auto captures buckets up to max_seqs")
        if self.graphs and max(self.buckets) > 64:
            # glue_fuse / glue_r2 / router_epilogue fold only up to 64 rows: wider decode steps run the unfused glue
            log(f"bucket {max(self.buckets)}: decode rows above 64 leave the T=1 folds; nothing has read them there")
        if self.placement not in ("all-vram", "solver"):
            raise ValueError(f"E4B_PAGED_PLACEMENT must be all-vram or solver, got {self.placement!r}")
        if self.kv_groups != "auto":
            int(self.kv_groups)
        if self.max_queue < 0:
            raise ValueError("E4B_PAGED_MAX_QUEUE must be >= 0")
        if self.decode_lookahead and not self.graphs:
            raise ValueError("E4B_PAGED_DECODE_LOOKAHEAD=1 needs bucketed decode graphs, and E4B_PAGED_GRAPHS resolved "
                             "to eager decode here")

    @property
    def prefill_budget(self) -> int:
        return self.max_prefill_tokens or self.chunk_tokens

    @property
    def max_tokens_limit(self) -> int:
        return self.max_tokens_cap or (self.max_tokens_per_seq - 1)

    @property
    def model_names(self) -> tuple:
        names = [self.model] if self.model else []
        names += [n for n in self.served_names if n not in names]
        return tuple(names)


# ---------------------------------------------------------------------------
# Incremental detokenisation
# ---------------------------------------------------------------------------


class IncrementalDetokenizer:
    """Per-token text deltas that never split a code point.

    The vLLM detokenizer's scheme: decode a window from ``prefix_offset`` to the end, compare with
    the window decoded up to ``read_offset``, and emit the difference only when it does not end in
    U+FFFD (a byte-fallback token that is the first byte of a multi-byte character decodes to the
    replacement character until its continuation arrives). The window starts with the last few
    PROMPT tokens so a sentencepiece tokenizer's leading-space rule sees context rather than a
    sequence start. Decoding is windowed, not whole-output, so a long completion costs the same per
    token as a short one.
    """

    def __init__(self, tokenizer, prompt_ids: Sequence[int], *, skip_special_tokens: bool = True,
                 context: int = 5):
        self.tok = tokenizer
        self.ids: list[int] = [int(t) for t in list(prompt_ids)[-context:]] if context > 0 else []
        self.prefix_offset = 0
        self.read_offset = len(self.ids)
        self.skip = skip_special_tokens
        self.text = ""

    def _decode(self, ids) -> str:
        try:
            return self.tok.decode(ids, skip_special_tokens=self.skip, clean_up_tokenization_spaces=False)
        except TypeError:   # a tokenizer whose decode has no clean-up keyword
            return self.tok.decode(ids, skip_special_tokens=self.skip)

    def push(self, token_id: int) -> str:
        self.ids.append(int(token_id))
        prefix = self._decode(self.ids[self.prefix_offset:self.read_offset])
        full = self._decode(self.ids[self.prefix_offset:])
        if len(full) > len(prefix) and not full.endswith("�"):
            delta = full[len(prefix):]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.ids)
            self.text += delta
            return delta
        return ""

    def flush(self) -> str:
        """Whatever is still held (an incomplete trailing byte sequence decodes to U+FFFD)."""
        prefix = self._decode(self.ids[self.prefix_offset:self.read_offset])
        full = self._decode(self.ids[self.prefix_offset:])
        delta = full[len(prefix):] if len(full) > len(prefix) else ""
        self.prefix_offset = self.read_offset = len(self.ids)
        self.text += delta
        return delta


# ---------------------------------------------------------------------------
# Engine: one thread owns the GPU and steps the scheduler
# ---------------------------------------------------------------------------


class BusyError(Exception):
    pass


@dataclass
class EngineParts:
    """What :func:`build_engine` returns and what a test injects instead."""
    scheduler: ContinuousScheduler
    tokenizer: Any
    eos_ids: frozenset
    info: dict = field(default_factory=dict)
    runner: Any = None


_TOKENS, _DONE, _ERROR = "tokens", "done", "error"


class PagedStream:
    """One request's hand-off between the event loop and the engine thread."""

    def __init__(self, *, queue, request_id: str, prompt_ids: list, max_tokens: int, stop_ids,
                 min_tokens: int, arrival: float, arrival_epoch: float):
        self.queue = queue
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.stop_ids = stop_ids
        self.min_tokens = min_tokens
        self.arrival = arrival
        self.arrival_epoch = arrival_epoch
        self.rid: Optional[int] = None
        self.sent = 0
        self.finished = False


def _append_jsonl(path: str, record: dict) -> None:
    """Best-effort trace append; a trace that cannot be written must not take serving down."""
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception as e:  # noqa: BLE001
        log(f"trace append failed ({type(e).__name__}: {e})")


class PagedEngine:
    """The engine loop. ``parts=None`` builds the GPU stack on the engine thread at start (the HTTP
    layer answers 503 until it is ready); a test passes its own parts."""

    def __init__(self, cfg: PagedServeConfig, parts: Optional[EngineParts] = None, *, builder=None):
        self.cfg = cfg
        self.parts = parts
        self._builder = builder or build_engine
        self.state = "ready" if parts is not None else "loading"
        self.error: Optional[str] = None
        self.started_at = time.time()
        self._cv = threading.Condition()
        self._ops: collections.deque = collections.deque()
        self._lock = threading.Lock()            # scheduler + bookkeeping, held per step
        self._streams: dict = {}
        self._done_idx = 0
        self._abort_idx = 0
        self._inflight = 0
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._loop = None
        self.records: collections.deque = collections.deque(maxlen=256)
        self.n_records = 0

    # ------------------------------------------------------------ lifecycle --
    def start(self, loop) -> None:
        self._loop = loop
        self._thread = threading.Thread(target=self._run, name="e4b-paged-engine", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 10.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def queue_depth(self) -> int:
        return self._inflight

    # --------------------------------------------------------------- intake --
    def submit(self, stream: PagedStream) -> None:
        with self._cv:
            if self.cfg.max_queue and self._inflight >= self.cfg.max_queue:
                raise BusyError(f"{self._inflight} requests in flight (E4B_PAGED_MAX_QUEUE={self.cfg.max_queue})")
            self._inflight += 1
            self._ops.append(("submit", stream))
            self._cv.notify()

    def abort(self, stream: PagedStream) -> None:
        with self._cv:
            self._ops.append(("abort", stream))
            self._cv.notify()

    # ---------------------------------------------------------------- stats --
    def stats(self) -> dict:
        out = {"state": self.state, "error": self.error, "in_flight": self._inflight,
               "trace_path": self.cfg.trace_path or None, "records_total": self.n_records,
               "records_recent": list(self.records)}
        if self.parts is None:
            return out
        with self._lock:
            out["scheduler"] = self.parts.scheduler.stats()
            runner = self.parts.runner
            gs = getattr(runner, "graph_stats", None)
            out["graph_stats"] = ({str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None)
            st = getattr(runner, "graph_status", None)
            out["graph_status"] = ({str(k): v for k, v in st.items()} if isinstance(st, dict) else None)
        return out

    # ---------------------------------------------------------------- loop --
    def _run(self) -> None:
        if self.parts is None:
            try:
                self.parts = self._builder(self.cfg)
            except BaseException as e:  # noqa: BLE001 -- a failed build is the server's state, not a crash
                self.error = f"{type(e).__name__}: {e}"
                self.state = "error"
                log(f"engine build FAILED: {self.error}")
                self._fail_pending(self.error)
                return
        self.state = "ready"
        sched = self.parts.scheduler
        tr = self._attach_step_trace()
        while True:
            with self._cv:
                while not self._ops and not self._stop and not (sched.queue or sched.active):
                    self._cv.wait(timeout=1.0)
                if self._stop:
                    break
                ops = list(self._ops)
                self._ops.clear()
            with self._lock:
                if tr is not None:
                    tr.begin(ops=len(ops))
                for kind, stream in ops:
                    self._apply(kind, stream)
                if tr is not None:
                    tr.mark("ops")
                try:
                    plan = sched.step()
                except Exception as e:  # noqa: BLE001
                    self.error = f"{type(e).__name__}: {e}"
                    self.state = "error"
                    log(f"engine step FAILED: {self.error}")
                    for st in list(self._streams.values()):
                        self._push(st, (_ERROR, self.error))
                    self._streams.clear()
                    self._fail_pending(self.error)
                    if tr is not None:
                        tr.close()
                    return
                self._dispatch()
                if tr is not None:
                    if plan.is_empty:
                        tr.discard()
                    else:
                        tr.end()
        if tr is not None:
            tr.close()
        self._fail_pending("server shutting down")
        for st in list(self._streams.values()):
            self._push(st, (_ERROR, "server shutting down"))
        self._streams.clear()

    def _attach_step_trace(self):
        """``E4B_PAGED_STEP_TRACE``: hand one :class:`~.engines.step_trace.StepTrace` to the scheduler and the runner,
        which mark their own steps (see that module for the row)."""
        if not self.cfg.step_trace_path:
            return None
        from .engines.step_trace import StepTrace
        tr = StepTrace(self.cfg.step_trace_path, cuda=str(self.cfg.device).startswith("cuda"))
        self.parts.scheduler.tracer = tr
        if self.parts.runner is not None and hasattr(self.parts.runner, "tracer"):
            self.parts.runner.tracer = tr
        log(f"STEP_TRACE -> {self.cfg.step_trace_path}")
        return tr

    def _fail_pending(self, msg: str) -> None:
        with self._cv:
            ops = list(self._ops)
            self._ops.clear()
        for kind, stream in ops:
            if kind == "submit":
                self._push(stream, (_ERROR, msg))

    def _apply(self, kind: str, stream: PagedStream) -> None:
        sched = self.parts.scheduler
        if kind == "submit":
            try:
                rid = sched.add_request(stream.prompt_ids, max_new_tokens=stream.max_tokens, now=stream.arrival,
                                        stop_ids=stream.stop_ids, min_tokens=stream.min_tokens)
            except ValueError as e:
                self._push(stream, (_ERROR, str(e)))
                return
            stream.rid = rid
            self._streams[rid] = stream
        elif kind == "abort":
            if stream.rid is not None and stream.rid in self._streams:
                sched.abort(stream.rid)

    def _push(self, stream: PagedStream, item) -> None:
        if item[0] in (_DONE, _ERROR):
            if stream.finished:
                return
            stream.finished = True
            with self._cv:
                self._inflight = max(0, self._inflight - 1)
        loop = self._loop
        try:
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(stream.queue.put_nowait, item)
        except RuntimeError:     # loop closed under us (shutdown): nobody is listening
            pass

    def _dispatch(self) -> None:
        sched = self.parts.scheduler
        for rid, req in sched.active.items():
            st = self._streams.get(rid)
            if st is not None and len(req.out) > st.sent:
                self._push(st, (_TOKENS, req.out[st.sent:]))
                st.sent = len(req.out)
        for lst, idx_attr in ((sched.done, "_done_idx"), (sched.aborted, "_abort_idx")):
            i = getattr(self, idx_attr)
            while i < len(lst):
                req = lst[i]
                i += 1
                st = self._streams.pop(req.rid, None)
                if st is None:
                    continue
                rest = req.out[st.sent:]
                st.sent = len(req.out)
                rec = self._record(req, st)
                self._push(st, (_DONE, (rest, req.finish_reason or "length", rec)))
            setattr(self, idx_attr, i)

    def _record(self, req: Request, st: PagedStream) -> dict:
        rec = {
            "request_id": st.request_id, "rid": req.rid,
            "arrival": req.arrival, "arrival_epoch": st.arrival_epoch,
            "admitted_at": req.admitted_at, "first_token_at": req.first_token_at,
            "finished_at": req.finished_at,
            "prompt_len": req.prompt_len, "out_len": len(req.out),
            "finish_reason": req.finish_reason,
            "ttft": req.ttft, "queue_wait": req.queue_wait,
            "decode_s": ((req.finished_at - req.first_token_at)
                         if req.finished_at is not None and req.first_token_at is not None else None),
        }
        # the scheduler keeps every finished Request for stats(); its stats never read the prompt
        # again, so drop the ids -- a long benchmark would otherwise hold every prompt it ever saw
        req.prompt = ()
        self.records.append(rec)
        self.n_records += 1
        if self.cfg.trace_path:
            _append_jsonl(self.cfg.trace_path, rec)
        return rec


# ---------------------------------------------------------------------------
# GPU construction (the harness's sequence; needs a CUDA box)
# ---------------------------------------------------------------------------


def _routed_topk(cfg) -> int:
    """step_decomp._routed_topk: the routed top-k under whatever name this family uses
    (:func:`experts4bit_qlora.arch.topology.routed_top_k`, the one alias list)."""
    from .arch.topology import routed_top_k

    v = routed_top_k(cfg)
    if v is None:
        raise ValueError("cannot find the routed top-k in this config")
    return v


def _kv_geometry(cfg):
    """step_decomp._kv_geometry: (kv heads, head_dim), scalars or per-layer lists.

    A composite (vision-language) config, e.g. Qwen3.5 / Qwen3.6 MoE, keeps the geometry on ``text_config``, which is
    therefore read FIRST. Reading ``num_key_value_heads`` first broke Gemma-4: on a per-layer config a global read
    raises transformers' ``AmbiguousGlobalPerLayerAttributeError``, a ``RuntimeError`` that ``getattr``'s default does
    not catch, so ``build_engine`` died before reaching the per-layer branch below."""
    def _one(c):
        heads = getattr(c, "num_key_value_heads")
        hd = getattr(c, "head_dim", None) or (c.hidden_size // getattr(c, "num_attention_heads"))
        return int(heads), int(hd)
    if getattr(cfg, "text_config", None) is not None:
        cfg = cfg.text_config            # a composite (vision-language) config, e.g. Qwen3.5 / Qwen3.6 MoE, Gemma-4
    try:
        return _one(cfg)
    except Exception as e:  # noqa: BLE001  (transformers' AmbiguousGlobalPerLayerAttributeError)
        if "per-layer attribute" not in str(e):
            raise
    per = [_one(lc) for lc in cfg.per_layer_config]
    heads, dims = [h for h, _ in per], [d for _, d in per]
    if len(set(heads)) == 1 and len(set(dims)) == 1:
        return heads[0], dims[0]
    log(f"KV geometry varies per layer: {sorted(set(zip(heads, dims)))}")
    return heads, dims


def arena_layer_ids(arena: str, n_moe_layers: int) -> list:
    """The arena's own layer ids, in order, for a model with ``n_moe_layers`` MoE modules.

    A bake keys its rows by the checkpoint's layer numbers. When a model's leading layers are dense (ERNIE-4.5's
    layer 0, DeepSeek-V2's first_k_dense_replace, LFM2's dense layers) those ids are not the MoE modules' ordinals 0..L-1.
    Served by ordinal, the first module asked for row (0, 0), which such an arena does not have. Refuses an arena whose
    layer count is not the model's."""
    idx = json.loads(open(arena + ".index.json", encoding="utf-8").read())
    ids = sorted({int(row[0]) for row in idx["rows"]})
    if len(ids) != n_moe_layers:
        raise RuntimeError(f"the arena {arena} holds {len(ids)} layers {ids[:4]}...; the model has {n_moe_layers} MoE "
                           "layers -- refusing to serve one layer's experts as another's")
    return ids


def _bytes_per_expert(arena: str) -> int:
    idx = json.loads(open(arena + ".index.json", encoding="utf-8").read())
    bpe = 0
    for seg in idx["segments"]:
        n = 1
        for d in seg["shape_per_expert"]:
            n *= d
        bpe += n * (4 if seg["dtype"] == "F32" else 1)
    return bpe


def _calib_batches(tok, n_seq=None, seq_len=512, bsz=4):
    """The lane hook's ``_calib_batches`` (hook v5/v7), same defaults: 32 x 512 tokens of C4 validation."""
    import torch
    from datasets import load_dataset
    n_seq = int(os.environ.get("E4B_CALIB_NSEQ", "32")) if n_seq is None else n_seq
    src = os.environ.get("E4B_CALIB_SOURCE", "c4")
    if src == "c4":
        ds = load_dataset("allenai/c4", data_files={"v": "en/c4-validation.00000-of-00008.json.gz"}, split="v")
        text = "\n\n".join(ds["text"][:4000])
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
    text = text[:6_000_000]
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = max(1, (ids.numel() - seq_len) // n_seq)
    rows = [ids[i * step:i * step + seq_len] for i in range(n_seq)]
    return [torch.stack(rows[i:i + bsz]) for i in range(0, n_seq, bsz)]


def prefill_routes() -> dict:
    """The prefill routes this process resolves, read where the forward reads them (for /health).

    A box's own environment is not evidence of what the server ran: SC2's e4b arm inherited the box script's
    ``E4B_INT4_PREFILL=loop`` / ``E4B_PAGED_PREFILL_ATTN=math`` while its registration said ``k19`` + ``flash``. So the
    server reports the routes itself, from the same functions the forward calls. ``int4_prefill`` is the resolved
    ``E4B_INT4_PREFILL`` (``auto`` -> ``k19`` where K19 can run, else ``loop``). With ``device_grouping`` on (a server
    capturing batched decode graphs), prefill calls above 256 rows take K19 only under ``k19``, and otherwise the
    grouped M-tile GEMM. ``prefill_attn`` is the resolved ``E4B_PAGED_PREFILL_ATTN``. A route the environment makes
    invalid is reported as the error the forward would raise.

    Those fields are the ENVIRONMENT's resolution, and they name what an int4-b32 store and a layer without sinks or a
    window would take. They do not say what this model ran. On gpt-oss (sc2g-prove-2) they read ``k19`` and ``flash``
    while no call took either: the MXFP4 store's rows up to 256 take K21 and rows above take the kept NF4 stacks, and
    every layer has sinks, so it keeps the explicit mask. ``seen`` is what ran: ``moe`` is
    :data:`~.engines.hot_residency.ROUTE_SEEN` (route and row class per expert-GEMM call) and ``prefill_attn`` is
    :data:`~.engines.paged_attention.ATTN_SEEN` (path per prefill attention call), both counted in the Python
    forward, so eager calls and graph captures count and graph replays do not. Cite ``seen`` as engagement evidence."""
    from .engines import hot_residency, paged_attention

    out: dict = {"int4_prefill_env": os.environ.get("E4B_INT4_PREFILL", "") or None,
                 "prefill_attn_env": os.environ.get("E4B_PAGED_PREFILL_ATTN", "") or None,
                 "device_grouping": bool(hot_residency.DEVICE_GROUPING[0])}
    try:
        mode = hot_residency._int4_prefill_mode_env()
        out["int4_prefill"] = mode
        out["int4_prefill_above_256_rows"] = (("k19" if mode == "k19" else "mtile") if out["device_grouping"]
                                              else mode)
    except ValueError as e:
        out["int4_prefill"] = f"invalid: {e}"
    try:
        out["prefill_attn"] = paged_attention._prefill_attn_mode_env()
    except ValueError as e:
        out["prefill_attn"] = f"invalid: {e}"
    out["seen"] = {"moe": dict(sorted(hot_residency.ROUTE_SEEN.items())),
                   "prefill_attn": dict(sorted(paged_attention.ATTN_SEEN.items()))}
    return out


def engage_prefill_graph(runner, cfg: PagedServeConfig) -> None:
    """Apply ``E4B_PAGED_PREFILL_GRAPH`` to a built runner (see :func:`_prefill_graph_env`). ``1`` turns a refusal
    into a startup error naming the reason; ``auto`` records it on the runner (prefill stays eager) and also stands
    down on too little free memory; ``0`` does nothing."""
    if cfg.prefill_graph not in ("1", "auto"):
        return
    from .engines.paged_runner import PrefillGraphRefused
    try:
        st = runner.enable_prefill_graph(cfg.chunk_tokens, require_headroom=(cfg.prefill_graph == "auto"))
    except PrefillGraphRefused as e:
        if cfg.prefill_graph == "1":
            raise RuntimeError(f"E4B_PAGED_PREFILL_GRAPH=1 refused: {e.why}") from e
        runner.note_prefill_graph_refused(e.why)
        log(f"PREFILL_GRAPH auto: refused, prefill stays eager ({e.why})")
        return
    log(f"PREFILL_GRAPH on ({cfg.prefill_graph}): first chunks of {cfg.chunk_tokens} tokens replay one graph "
        f"(startup check bitwise on two prompts; pool {st.get('pool_mib')} MiB, {st.get('free_after_mib')} MiB free)")


def prefill_graph_report(cfg: PagedServeConfig, engine) -> dict:
    """``/health``'s ``prefill_graph`` block. ``status`` is ``off``, ``on`` (engaged: ``T``, ``replays``,
    ``eager_chunks``, ``eager_reasons``, ``pool_mib`` and ``free_after_mib`` follow, from
    :meth:`~.engines.paged_runner.PagedModelRunner.prefill_graph_stats`), ``refused`` (``why`` names the reason:
    under ``auto`` the server runs with eager prefill; under ``1`` the engine stopped at startup), ``loading``, or
    ``error``. ``requested`` is the setting, ``auto``, ``1`` or ``0``."""
    if engine.parts is not None:
        runner = getattr(engine.parts, "runner", None)
        rep = runner.prefill_graph_stats() if hasattr(runner, "prefill_graph_stats") else {"status": "off"}
    elif cfg.prefill_graph == "0":
        rep = {"status": "off"}
    elif engine.state == "error":
        refused = "E4B_PAGED_PREFILL_GRAPH=1 refused" in (engine.error or "")
        rep = {"status": "refused" if refused else "error", "why": engine.error}
    else:
        rep = {"status": engine.state}
    return dict(rep, requested=cfg.prefill_graph)


def kv_bookkeeping_report(cfg: PagedServeConfig, engine) -> dict:
    """``/health``'s ``kv_bookkeeping`` block: ``requested`` (``E4B_PAGED_BULK_KV``) and, once the engine is built,
    :meth:`~.engines.paged_runner.PagedModelRunner.kv_bookkeeping_stats` -- how many requests' prompt flushes and
    first-decode block claims took the per-layer path and how many the bulk one."""
    runner = getattr(engine.parts, "runner", None) if engine.parts is not None else None
    rep = runner.kv_bookkeeping_stats() if hasattr(runner, "kv_bookkeeping_stats") else {}
    return dict(rep, requested=cfg.bulk_kv)


def _apply_levers(model, cfg: PagedServeConfig, tok) -> dict:
    """``bench/p42/hook/usercustomize.py::_apply_lanes``, called where the hook calls it (right after
    ``enable_hybrid_tier``), reading the same environment. A refusal raises, as the hook re-raises."""
    env = os.environ.get
    out = {"exp_int4_layers_enabled": 0, "attn_int4_rtn_projections": 0, "attn_int4_calib_projections": 0}
    mt = getattr(getattr(model, "config", None), "model_type", "?")
    if env("E4B_SERVE_EXP_INT4", "0") == "1":
        from huggingface_hub import snapshot_download
        from .engines.int4_experts import enable_serve_experts_int4
        src = snapshot_download(cfg.model, allow_patterns=["*.json", "*.safetensors"],
                                revision=cfg.revision or None)
        if env("E4B_SERVE_EXP_INT4_CALIB", "0") == "1":
            from .engines.int4_experts import enable_serve_experts_int4_calibrated
            batches = _calib_batches(tok)
            art_dir = env("E4B_INT4_ARTIFACT_DIR") or None
            art_fp = env("E4B_INT4_EXPECTED_FINGERPRINT") or None
            dump_dir = env("E4B_INT4_DUMP_ARTIFACT_DIR") or None
            lpp = env("E4B_CALIB_LAYERS_PER_PASS")
            log(f"INT4EXP {'loading licensed artifact ' + str(art_dir) if (art_dir or art_fp) else 'calibrating (streamed)'}"
                f": {len(batches)} batches of {env('E4B_CALIB_SOURCE', 'c4')}, layers_per_pass={lpp or 'auto'}")
            n = enable_serve_experts_int4_calibrated(model, src, batches, artifact_dir=art_dir,
                                                     expected_fingerprint=art_fp, dump_artifact_dir=dump_dir,
                                                     layers_per_pass=int(lpp) if lpp else None)
        else:
            n = enable_serve_experts_int4(model, src)
        log(f"INT4EXP enabled: {n} layers (model_type={mt})")
        if n <= 0:
            raise RuntimeError("E4B_SERVE_EXP_INT4=1 but enable_serve_experts_int4 patched 0 layers")
        out["exp_int4_layers_enabled"] = int(n)
    if env("E4B_SERVE_ATTN_INT4", "0") == "1":
        from .engines.int4_attn import enable_serve_attn_int4
        n = enable_serve_attn_int4(model)
        log(f"ATTNINT4 rtn: {n} projections (uncalibrated; model_type={mt})")
        out["attn_int4_rtn_projections"] = int(n)
    _attn = env("E4B_SERVE_ATTN_INT4_CALIB", "0") == "1"
    _head = env("E4B_SERVE_LMHEAD_INT4_CALIB", "0") == "1"
    _dense = env("E4B_SERVE_DENSE_INT4_CALIB", "0") == "1"
    if _attn or _head or _dense:
        from .engines.int4_attn_calib import calibrate_attention_hessians, enable_serve_attn_int4_calib
        batches = _calib_batches(tok)
        hs = calibrate_attention_hessians(model, batches, include_attention=_attn, include_head=_head,
                                          include_dense_mlp=_dense)
        n = enable_serve_attn_int4_calib(model, hs, include_attention=_attn, include_head=_head,
                                         include_dense_mlp=_dense)
        log(f"ATTNINT4 calibrated: {n} projections (attn={int(_attn)} head={int(_head)} dense={int(_dense)}; "
            f"{len(batches)} batches of {env('E4B_CALIB_SOURCE', 'c4')}; model_type={mt})")
        out["attn_int4_calib_projections"] = int(n)
    return out


def _count(v):
    return [int(x) for x in v] if isinstance(v, (tuple, list)) else int(v)


def _apply_fusions(model, cfg: PagedServeConfig, report: dict | None = None) -> dict:
    """One assembly point, as the harness: ``qkv_fuse.fuse_qkv`` imports and calls ``fuse_t1_glue``,
    ``fuse_t1_glue_r2`` and ``fuse_router_epilogue`` itself after fusing (so the env flags are live on the
    fused path -- the registered B=1 stack is ``--fuse-qkv`` WITH the fold flags set); the unfused branch
    calls the three directly. The census carries what each fold RETURNED, never a literal 0: in the fused
    branch the fold functions are wrapped on their modules for the duration of the call -- ``fuse_qkv``
    imports them inside its body, so the wrapper is what it calls -- and restored afterwards.

    Modes come from ``cfg.fusion_modes`` (``auto`` / ``0`` / ``1``, :func:`_fusion_env`; ``default`` resolved per
    family by :func:`resolve_fusion_modes` from the model's ``config.model_type``). A knob it does not name keeps
    its old reading -- ``cfg.fuse_qkv`` for the q/k/v fusion, the fold's own environment variable for a fold -- and is
    called exactly as before, so a config built without modes behaves as it always did. Fused q/k/v at ``1`` refuses a
    model with no Qwen3-MoE attention; at ``auto`` it fuses what matches and the folds run either way. ``report`` (a
    dict, when given) receives the resolved modes and each fold's report."""
    from .engines import glue_fuse, glue_r2, router_epilogue
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    modes, sources = resolve_fusion_modes(dict(cfg.fusion_modes or {}), model_type)
    qkv_mode = modes.get(FUSE_QKV_ENV) or ("1" if cfg.fuse_qkv else "0")
    fold_modes = {k: modes.get(k) for k in FUSION_ENV}
    fold_reports = {} if report is not None else None
    resolved = {k: glue_fuse.fold_mode(k, fold_modes[k]) for k in FUSION_ENV}
    set_folds = [k for k in FUSION_ENV if resolved[k] != "0"]
    folds = ((glue_fuse, "fuse_t1_glue", "fuse_t1_glue_n"),
             (glue_r2, "fuse_t1_glue_r2", "fuse_t1_glue_r2_n"),
             (router_epilogue, "fuse_router_epilogue", "fuse_router_epilogue_n"))

    def _fold_kw(name):
        kw = {}
        if fold_modes[name] is not None:
            kw["mode"] = fold_modes[name]
        if fold_reports is not None:
            kw["report"] = fold_reports.setdefault(name, {})
        return kw

    if qkv_mode != "0":
        from .engines.qkv_fuse import fuse_qkv
        captured: dict = {}
        saved = []
        for mod, fname, key in folds:
            orig = getattr(mod, fname)

            def _recording(*a, _orig=orig, _key=key, **k):
                r = _orig(*a, **k)
                captured[_key] = r
                return r

            saved.append((mod, fname, orig))
            setattr(mod, fname, _recording)
        qkv_kw = {}
        if any(v is not None for v in fold_modes.values()):
            qkv_kw["fold_modes"] = fold_modes
        if fold_reports is not None:
            qkv_kw["fold_reports"] = fold_reports
        try:
            n = fuse_qkv(model, **qkv_kw)
        finally:
            for mod, fname, orig in saved:
                setattr(mod, fname, orig)
        if n == 0 and qkv_mode == "1":
            raise RuntimeError("E4B_PAGED_FUSE_QKV=1 matched no attention module -- refusing a vacuous fusion")
        missing = [key for _, _, key in folds if key not in captured]
        if missing:
            raise RuntimeError(
                f"fuse_qkv returned without calling {missing}: qkv_fuse no longer applies the folds at its assembly "
                "point, so this census cannot be reported -- update _apply_fusions rather than guessing")
        out = {"fuse_qkv_n": int(n), **{key: _count(captured[key]) for _, _, key in folds}}
    else:
        out = {"fuse_qkv_n": 0, **{key: _count(getattr(mod, fname)(model, **_fold_kw(env_name)))
                                   for (mod, fname, key), env_name in zip(folds, FUSION_ENV)}}
    if report is not None:
        report.update(modes={FUSE_QKV_ENV: qkv_mode, **resolved}, folds=fold_reports, sources=sources,
                      model_type=model_type)
    log(f"fusions (q/k/v {qkv_mode}; folds set: {set_folds or 'none'}): {out}")
    return out


def _eos_ids(model, tok, cfg: PagedServeConfig) -> frozenset:
    if cfg.eos_ids:
        return frozenset(int(t) for t in cfg.eos_ids)
    ids = set()
    v = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(v, int):
        ids.add(v)
    elif isinstance(v, (list, tuple)):
        ids.update(int(x) for x in v)
    t = getattr(tok, "eos_token_id", None)
    if isinstance(t, int):
        ids.add(t)
    if not ids:
        raise RuntimeError("no EOS id on the model's generation_config or tokenizer; set E4B_PAGED_EOS_IDS")
    return frozenset(ids)


def _batched_graph_grouping(cfg: PagedServeConfig) -> dict:
    """The batched lane's capture-safety switches, as ``bench/p39/step_decomp.py``'s ``_bv3_stage`` sets them.

    A ``[b, 1]`` decode step with ``b > 1`` routes ``T = b`` MoE rows, and the library's default at ``T > 1`` is EAGER
    grouping, whose host-size sync invalidates a CUDA-graph capture. The harness's batched lane therefore sets
    ``hot_residency.DEVICE_GROUPING`` (sync-free device grouping) and clears ``FORCE_SINGLETON_GROUPS`` before it
    captures. Its B=1 lane and its eager runs leave both at their defaults. Mirrored here: the switches move only when
    graphs are on and more than one sequence can decode, on the all-resident placement the harness asserts. Without
    this, the first GPU run (lane SC1 proof ``sc1a-prove-7``) captured bucket 1 and failed every bucket above it.
    Returns the two flags as set, for the census."""
    from .engines import hot_residency as _hr
    if cfg.graphs and cfg.max_seqs > 1 and max(cfg.buckets) > 1:
        if cfg.placement != "all-vram":
            raise ValueError("batched decode graphs bind to the all-resident point (step_decomp's batched lane asserts "
                             f"placement all-vram); got E4B_PAGED_PLACEMENT={cfg.placement!r} with max_seqs={cfg.max_seqs}")
        _hr.DEVICE_GROUPING[0] = True
        _hr.FORCE_SINGLETON_GROUPS[0] = False
    return {"device_grouping": bool(_hr.DEVICE_GROUPING[0]), "force_singleton_groups": bool(_hr.FORCE_SINGLETON_GROUPS[0])}


def _kv_pool_mib(kv, cfg: PagedServeConfig):
    """MiB the FP8 paged KV pool holds: ``serve_recipe.paged_kv_pool_bytes`` on the geometry the pool was built with
    (the tests assert that arithmetic equals a constructed pool's), so ``/health`` names what a ``max_seqs`` costs."""
    from .serve_recipe import paged_kv_pool_bytes

    try:
        b = paged_kv_pool_bytes(kv.L, kv.Hs, kv.Ds, batch=kv.B, max_tokens_per_seq=cfg.max_tokens_per_seq,
                                k_groups=None if cfg.kv_groups == "auto" else int(cfg.kv_groups),
                                scratch_slots=kv.n_scratch)
    except (AttributeError, ImportError, TypeError, ValueError):
        return None
    return round(b / 2**20, 1)


def build_engine(cfg: PagedServeConfig) -> EngineParts:
    """The harness's construction, in its order (see the module docstring). GPU only."""
    cfg.validate()
    for name, val in (("E4B_PAGED_MODEL", cfg.model), ("E4B_PAGED_ARENA", cfg.arena), ("E4B_PAGED_CALIB", cfg.calib)):
        if not val:
            raise ValueError(f"{name} is required")
    for path in (cfg.arena, cfg.arena + ".index.json", cfg.calib):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    import torch
    from transformers import AutoTokenizer

    from . import load_moe_4bit_streaming
    from .engines.fp8_paged_kv import Fp8PagedKV
    from .engines.host_heap import release_cached_pinned_memory, release_freed_host_heap
    from .engines.hot_residency import target_modules
    from .engines.hybrid import enable_hybrid_tier
    from .engines.paged_attention import register
    from .engines.paged_runner import PagedModelRunner, decoder_layers, kv_layers
    from .engines.placement import solve_placement

    torch.manual_seed(1689)
    tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision or None)
    from transformers import AutoConfig

    from .engines.paged_runner import kv_layout_refusal
    try:
        model_config = AutoConfig.from_pretrained(cfg.model, revision=cfg.revision or None)
        why = kv_layout_refusal(model_config)
    except (OSError, ValueError, KeyError):           # a config only the loader reads: its own refusals stand
        model_config, why = None, None
    if why:
        raise RuntimeError(f"serve_paged refuses {cfg.model}: {why}")   # before any weight is read
    if str(cfg.max_seqs_requested).strip().lower() == "auto":
        # sized before any weight is read: the estimate's device total includes the weights
        free = (torch.cuda.mem_get_info(torch.device(cfg.device))[0]
                if str(cfg.device).startswith("cuda") and torch.cuda.is_available() else None)
        resolve_max_seqs(cfg, model_config, free)
    model, _ = load_moe_4bit_streaming(cfg.model, cfg.device, torch.bfloat16, r=8, alpha=16, quant_type="nf4",
                                       arena=cfg.arena, revision=cfg.revision or None)
    model.eval()
    mods = target_modules(model)
    L, E = len(mods), mods[0].num_experts
    k = _routed_topk(model.config)
    torch.set_num_threads(cfg.torch_threads)
    man = solve_placement(n_layers=L, n_experts=E, bytes_per_expert=_bytes_per_expert(cfg.arena),
                          vram_budget_bytes=int(cfg.vram_gb * 2**30), dram_budget_bytes=int(cfg.dram_gb * 2**30),
                          calibration=json.loads(open(cfg.calib, encoding="utf-8").read()), profile_path=None,
                          batch=1, top_k=k, cpu_us_fixed=None, cpu_us_per_row=None)
    if cfg.placement == "all-vram":
        # step_decomp --placement-override all-vram: the solver ran, then every expert is in the VRAM tier
        pairs = sorted(tuple(pp) for t in ("vram", "dram", "nvme") for pp in man["tiers"][t])
        man["tiers"] = {"vram": [list(pp) for pp in pairs], "dram": [], "nvme": []}
        man["masses"] = {"vram_frac": 1.0, "dram_frac": 0.0, "nvme_frac": 0.0}
    # the arena and the placement are keyed by the arena's layer ids; the solver numbered its layers 0..L-1
    ids = arena_layer_ids(cfg.arena, L)
    if ids != list(range(L)):
        man["tiers"] = {t: [[ids[int(lay)], int(e)] for lay, e in pairs] for t, pairs in man["tiers"].items()}
    n = enable_hybrid_tier(model, cfg.arena, man, hot_rows=cfg.hot_rows, threads=0, pool=True,
                           dispatch_diet=False, collapse_resident=True, layers=ids)
    if n != L:
        raise RuntimeError(f"enable_hybrid_tier patched {n}/{L} MoE layers")
    levers = _apply_levers(model, cfg, tok)
    for m in mods:
        m._hot_residency.arm_amortization(False)          # --amort off: the production shape
    register(model)
    fusion_report: dict = {}
    fusions = _apply_fusions(model, cfg, report=fusion_report)

    # proof of execution, the lanes' census (serve_stack.build_served_model) + their refusal rule
    try:
        from .engines.int4_attn import Int4Linear
        int4_attn = sum(1 for m in model.modules() if isinstance(m, Int4Linear))
    except ImportError:
        int4_attn = 0
    stores = [getattr(getattr(m, "_hot_residency", None), "_int4_stores", None) for m in mods]
    int4_layers = sum(1 for s in stores if s)
    kinds = sorted({str(s.get("kind", "int4_b32")) if isinstance(s, dict) else "int4_b32" for s in stores if s})
    env = os.environ.get
    if env("E4B_SERVE_EXP_INT4", "0") == "1" and int4_layers == 0:
        raise RuntimeError("E4B_SERVE_EXP_INT4=1 but no expert layer carries an int4 store")
    if (env("E4B_SERVE_ATTN_INT4", "0") == "1" or env("E4B_SERVE_ATTN_INT4_CALIB", "0") == "1") and int4_attn == 0:
        raise RuntimeError("int4 attention requested but no projection is Int4Linear")
    if 0 < int4_layers < L:
        log(f"WARNING int4 expert stores on {int4_layers}/{L} MoE layers -- a partial stack; see /health")

    hkv, hd = _kv_geometry(model.config)
    scratch = max(cfg.buckets) if cfg.graphs else 0
    # a hybrid model's linear-attention layers keep no K/V: the pool holds its attention layers only
    kv = Fp8PagedKV(kv_layers(model, decoder_layers(model.config)), hkv, hd, batch=cfg.max_seqs, max_tokens_per_seq=cfg.max_tokens_per_seq,
                    k_groups=(None if cfg.kv_groups == "auto" else int(cfg.kv_groups)),
                    batched_append=True, device=cfg.device, scratch_slots=scratch)
    runner = PagedModelRunner(model, kv, device=cfg.device, bulk_kv=cfg.bulk_kv, last_logits=cfg.last_logits)
    grouping = _batched_graph_grouping(cfg)          # before capture: the batched lane's sync-free grouping
    graph_status = runner.enable_decode_graphs(cfg.buckets) if cfg.graphs else None
    engage_prefill_graph(runner, cfg)
    sched = ContinuousScheduler(runner=runner, max_seqs=cfg.max_seqs, kv_slots=cfg.max_seqs,
                                chunk_tokens=cfg.chunk_tokens, max_prefill_tokens_per_step=cfg.prefill_budget,
                                lookahead=cfg.decode_lookahead)
    info = {"moe_layers": L, "experts": E, "top_k": k, "model_type": getattr(model.config, "model_type", None),
            "int4_expert_layers": int4_layers, "int4_store_kinds": kinds, "int4_attn_projections": int4_attn,
            "kv": {"n_kv_heads": hkv, "head_dim": hd, "k_groups": cfg.kv_groups, "scratch_slots": scratch,
                   "blocks_per_seq": getattr(kv, "blocks_per_seq", None), "pool_mib": _kv_pool_mib(kv, cfg)},
            "graph_status": graph_status, "grouping": grouping, "prefill_graph": cfg.prefill_graph, "levers_env": {k_: env(k_) for k_ in LEVER_ENV if env(k_) is not None}}
    info.update(levers)
    info.update(fusions)
    info["fusion_modes"] = fusion_report.get("modes")
    info["fusion_sources"] = fusion_report.get("sources")
    info["fusion_report"] = fusion_report.get("folds")
    # The build churns through host buffers it frees (the hybrid tier's setup tier, the stacks' one-shot reads), and
    # glibc keeps freed blocks under its mmap threshold resident for the life of the server: 0.34 GB on OLMoE-1B-7B
    # (RTX A2000 host, the one place it was measured). The loader alone leaves ~4 MB.
    # likewise torch's caching host allocator keeps the loader's freed pinned staging (1.09 GB after loading Qwen3-30B-A3B)
    info["pinned_cache_released"] = release_cached_pinned_memory()
    info["host_heap_trimmed"] = release_freed_host_heap()
    log(f"ready: {json.dumps(info, default=str)}")
    return EngineParts(scheduler=sched, tokenizer=tok, eos_ids=_eos_ids(model, tok, cfg), info=info, runner=runner)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _graph_stats(parts):
    """Per decode bucket, what the runner did with it (``replays``, ``eager_steps``, ``rows``, ``pad_rows``): a copy of
    ``PagedModelRunner.graph_stats``, None for a runner without decode graphs. ``/stats`` carries the same."""
    gs = getattr(getattr(parts, "runner", None), "graph_stats", None)
    return {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'), ensure_ascii=False)}\n\n"


def _usage(prompt_len: int, out_len: int) -> dict:
    return {"prompt_tokens": prompt_len, "completion_tokens": out_len, "total_tokens": prompt_len + out_len}


async def stream_events(engine: PagedEngine, stream: PagedStream, detok: IncrementalDetokenizer, chunk, usage_only,
                        include_usage: bool):
    """The SSE body: one chunk per token; finish_reason (and usage) on the last token's chunk; an optional
    usage-only chunk; ``[DONE]``. Starlette closes this generator when the client disconnects, and the
    ``finally`` then aborts the request so its KV slot is freed instead of decoding to ``max_tokens``."""
    try:
        while True:
            kind, payload = await stream.queue.get()
            if kind == _TOKENS:
                for tid in payload:
                    yield _sse(chunk(detok.push(tid), None, None))
            elif kind == _DONE:
                rest, finish, rec = payload
                texts = [detok.push(tid) for tid in rest]
                tail = detok.flush()
                if texts:
                    texts[-1] += tail
                else:
                    texts = [tail]
                usage = _usage(rec["prompt_len"], rec["out_len"])
                for i, text in enumerate(texts):
                    last = i == len(texts) - 1
                    yield _sse(chunk(text, finish if last else None, usage if last else None))
                if include_usage:
                    yield _sse(usage_only(usage))
                yield "data: [DONE]\n\n"
                return
            else:
                yield _sse({"error": {"message": str(payload), "type": "engine_error"}})
                return
    finally:
        if not stream.finished:
            engine.abort(stream)          # the client went away: free the slot now


def create_app(cfg: Optional[PagedServeConfig] = None, engine: Optional[PagedEngine] = None):
    """App factory. ``engine`` injection is for tests (a :class:`PagedEngine` built on fake parts)."""
    from contextlib import asynccontextmanager

    from fastapi import Depends, FastAPI, Header, HTTPException, Request as HttpRequest
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    cfg = cfg or PagedServeConfig.from_env()
    engine = engine or PagedEngine(cfg)

    def _auth(authorization: Optional[str] = Header(None)):
        if cfg.token and authorization != f"Bearer {cfg.token}":
            raise HTTPException(401, "missing or invalid bearer token (set 'Authorization: Bearer <E4B_TOKEN>')")

    @asynccontextmanager
    async def lifespan(app):
        engine.start(asyncio.get_running_loop())
        yield
        engine.shutdown()

    app = FastAPI(title="experts4bit-qlora serve_paged", lifespan=lifespan)
    app.state.engine = engine

    class CompletionRequest(BaseModel):
        model: str
        prompt: Any
        max_tokens: Optional[int] = 16
        min_tokens: int = 0
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        top_k: Optional[int] = None
        seed: Optional[int] = None
        n: int = 1
        best_of: Optional[int] = None
        stream: bool = False
        stream_options: Optional[dict] = None
        ignore_eos: bool = False
        logprobs: Optional[int] = None
        echo: bool = False
        stop: Any = None
        stop_token_ids: Optional[list] = None
        repetition_penalty: float = 1.0
        presence_penalty: float = 0.0
        frequency_penalty: float = 0.0
        suffix: Optional[str] = None
        add_special_tokens: bool = True
        skip_special_tokens: bool = True

    class ChatRequest(BaseModel):
        model: str
        messages: list
        max_tokens: Optional[int] = None
        max_completion_tokens: Optional[int] = None
        min_tokens: int = 0
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        top_k: Optional[int] = None
        seed: Optional[int] = None
        n: int = 1
        stream: bool = False
        stream_options: Optional[dict] = None
        ignore_eos: bool = False
        logprobs: Any = None
        top_logprobs: Optional[int] = None
        stop: Any = None
        stop_token_ids: Optional[list] = None
        repetition_penalty: float = 1.0
        presence_penalty: float = 0.0
        frequency_penalty: float = 0.0
        skip_special_tokens: bool = True

    def _ready() -> EngineParts:
        if engine.state == "error":
            raise HTTPException(500, f"engine unavailable: {engine.error}")
        if engine.state != "ready" or engine.parts is None:
            raise HTTPException(503, "engine is loading", headers={"Retry-After": "30"})
        return engine.parts

    def _names(parts: EngineParts) -> list:
        names = list(cfg.model_names)
        mid = parts.info.get("model_id")
        if mid and mid not in names:
            names.append(mid)
        return names or ["experts4bit-qlora"]

    def _common_checks(req, parts: EngineParts) -> None:
        if req.model not in _names(parts):
            raise HTTPException(404, f"unknown model {req.model!r}; this server serves {_names(parts)}")
        if req.temperature is not None and req.temperature != 0:
            raise HTTPException(400, "this server is greedy-only (the engine argmaxes): send temperature 0 or omit "
                                     "it (vllm bench serve: --temperature 0; sglang.bench_serving sends 0.0 by default)")
        if req.n != 1:
            raise HTTPException(400, "n must be 1 (greedy decoding has one completion)")
        if getattr(req, "best_of", None) not in (None, 1):
            raise HTTPException(400, "best_of must be 1 or absent")
        if getattr(req, "logprobs", None) not in (None, False) or getattr(req, "top_logprobs", None):
            raise HTTPException(400, "logprobs are not supported in v1 (the runner returns argmax ids, not logits)")
        if getattr(req, "echo", False):
            raise HTTPException(400, "echo is not supported")
        if getattr(req, "suffix", None):
            raise HTTPException(400, "suffix is not supported")
        if req.stop not in (None, "", []):
            raise HTTPException(400, "stop strings are not supported in v1; stop_token_ids are")
        if req.repetition_penalty != 1.0 or req.presence_penalty != 0.0 or req.frequency_penalty != 0.0:
            raise HTTPException(400, "penalties are not supported (they would change the greedy argmax)")

    def _budget(prompt_len: int, max_tokens, min_tokens: int) -> int:
        if max_tokens is None:
            max_tokens = 16
        if max_tokens < 1:
            raise HTTPException(400, "max_tokens must be >= 1")
        if max_tokens > cfg.max_tokens_limit:
            raise HTTPException(400, f"max_tokens {max_tokens} exceeds this server's limit {cfg.max_tokens_limit} "
                                     "(E4B_PAGED_MAX_TOKENS; requests are refused, not clamped)")
        if min_tokens < 0 or min_tokens > max_tokens:
            raise HTTPException(400, "min_tokens must be in [0, max_tokens]")
        if prompt_len < 1:
            raise HTTPException(400, "empty prompt")
        if prompt_len + max_tokens > cfg.max_tokens_per_seq:
            raise HTTPException(400, f"prompt_len {prompt_len} + max_tokens {max_tokens} exceeds "
                                     f"E4B_PAGED_MAX_TOKENS_PER_SEQ={cfg.max_tokens_per_seq} (the KV a sequence owns)")
        return max_tokens

    def _stop_set(parts: EngineParts, ignore_eos: bool, extra) -> Optional[frozenset]:
        ids = set() if ignore_eos else set(parts.eos_ids)
        if extra:
            ids.update(int(t) for t in extra)
        return frozenset(ids) if ids else None

    def _encode(tok, text: str, add_special: bool) -> list:
        return [int(t) for t in tok.encode(text, add_special_tokens=add_special)]

    def _prompt_ids(parts: EngineParts, prompt, add_special: bool) -> list:
        tok = parts.tokenizer
        if isinstance(prompt, str):
            return _encode(tok, prompt, add_special)
        if isinstance(prompt, list) and prompt:
            if all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
                return [int(t) for t in prompt]
            if len(prompt) == 1 and isinstance(prompt[0], str):
                return _encode(tok, prompt[0], add_special)
            if len(prompt) == 1 and isinstance(prompt[0], list) and all(isinstance(t, int) for t in prompt[0]):
                return [int(t) for t in prompt[0]]
            raise HTTPException(400, "prompt must be one string or one list of token ids (batched prompts are not "
                                     "supported; send one request per prompt)")
        raise HTTPException(400, "prompt must be a non-empty string or a non-empty list of token ids")

    def _submit(parts: EngineParts, prompt_ids: list, max_tokens: int, min_tokens: int, stop_ids,
                request_id: str, arrival: float, arrival_epoch: float) -> PagedStream:
        stream = PagedStream(queue=asyncio.Queue(), request_id=request_id, prompt_ids=prompt_ids,
                             max_tokens=max_tokens, stop_ids=stop_ids, min_tokens=min_tokens,
                             arrival=arrival, arrival_epoch=arrival_epoch)
        try:
            engine.submit(stream)
        except BusyError as e:
            raise HTTPException(503, str(e), headers={"Retry-After": "1"})
        return stream

    async def _collect(stream: PagedStream):
        """Non-streaming: every token id, the finish reason and the trace record."""
        ids: list = []
        while True:
            kind, payload = await stream.queue.get()
            if kind == _TOKENS:
                ids.extend(payload)
            elif kind == _DONE:
                rest, finish, rec = payload
                ids.extend(rest)
                return ids, finish, rec
            else:
                raise HTTPException(500, f"engine error: {payload}")

    def _include_usage(opts) -> bool:
        return bool(opts and opts.get("include_usage"))

    # ------------------------------------------------------------- routes --
    @app.get("/health")
    async def health():
        parts = engine.parts
        info = dict(parts.info) if parts is not None else {}
        return {
            "status": "busy" if (engine.state == "ready" and engine.queue_depth > 0) else engine.state,
            "error": engine.error,
            "model": cfg.model,
            "served_model_names": list(cfg.model_names),
            "engine": {
                "max_seqs": cfg.max_seqs, "kv_slots": cfg.max_seqs, "max_tokens_per_seq": cfg.max_tokens_per_seq,
                "max_seqs_requested": cfg.max_seqs_requested, "max_seqs_resolution": cfg.max_seqs_resolution or None,
                "chunk_tokens": cfg.chunk_tokens, "max_prefill_tokens_per_step": cfg.prefill_budget,
                "graphs": cfg.graphs, "buckets": list(cfg.buckets), "buckets_requested": cfg.buckets_requested,
                "graph_status": info.pop("graph_status", None), "graph_stats": _graph_stats(parts),
                "placement": cfg.placement, "fuse_qkv": _fuse_qkv_health(cfg, info),
                "max_tokens_limit": cfg.max_tokens_limit,
                "max_queue": cfg.max_queue or None, "bulk_kv": cfg.bulk_kv,
                "decode_lookahead": cfg.decode_lookahead,
            },
            "levers": info,
            "prefill_routes": prefill_routes(),
            "prefill_graph": prefill_graph_report(cfg, engine),
            "last_logits": dict(
                engine.parts.runner.last_logits_stats()
                if engine.parts is not None and hasattr(engine.parts.runner, "last_logits_stats")
                else {"status": "off" if not cfg.last_logits else engine.state}, requested=cfg.last_logits),
            "kv_bookkeeping": kv_bookkeeping_report(cfg, engine),
            "eos_token_ids": sorted(parts.eos_ids) if parts is not None else None,
            "sampling": {"greedy_only": True, "logprobs": False, "stop_strings": False},
            "queue_depth": engine.queue_depth,
            "trace_path": cfg.trace_path or None,
            "step_trace_path": cfg.step_trace_path or None,
            "uptime_s": round(time.time() - engine.started_at, 1),
        }

    @app.get("/stats")
    async def stats():
        return await asyncio.to_thread(engine.stats)

    @app.get("/v1/models")
    async def v1_models():
        names = _names(engine.parts) if engine.parts is not None else list(cfg.model_names)
        return {"object": "list",
                "data": [{"id": n, "object": "model", "owned_by": "experts4bit-qlora",
                          "max_model_len": cfg.max_tokens_per_seq} for n in names]}

    @app.post("/v1/completions", dependencies=[Depends(_auth)])
    async def v1_completions(req: CompletionRequest, http: HttpRequest):
        arrival, arrival_epoch = time.monotonic(), time.time()
        parts = _ready()
        _common_checks(req, parts)
        prompt_ids = _prompt_ids(parts, req.prompt, req.add_special_tokens)
        max_tokens = _budget(len(prompt_ids), req.max_tokens, req.min_tokens)
        stop_ids = _stop_set(parts, req.ignore_eos, req.stop_token_ids)
        cid = http.headers.get("x-request-id") or f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        stream = _submit(parts, prompt_ids, max_tokens, req.min_tokens, stop_ids, cid, arrival, arrival_epoch)

        def chunk(text, finish, usage):
            body = {"id": cid, "object": "text_completion", "created": created, "model": req.model,
                    "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]}
            if usage is not None:
                body["usage"] = usage
            return body

        def usage_only(usage):
            return {"id": cid, "object": "text_completion", "created": created, "model": req.model,
                    "choices": [], "usage": usage}

        detok = IncrementalDetokenizer(parts.tokenizer, prompt_ids, skip_special_tokens=req.skip_special_tokens)
        if req.stream:
            return StreamingResponse(stream_events(engine, stream, detok, chunk, usage_only,
                                             _include_usage(req.stream_options)),
                                     media_type="text/event-stream")
        ids, finish, rec = await _collect(stream)
        for tid in ids:
            detok.push(tid)
        detok.flush()
        return chunk(detok.text, finish, _usage(rec["prompt_len"], rec["out_len"]))

    @app.post("/v1/chat/completions", dependencies=[Depends(_auth)])
    async def v1_chat_completions(req: ChatRequest, http: HttpRequest):
        arrival, arrival_epoch = time.monotonic(), time.time()
        parts = _ready()
        _common_checks(req, parts)
        tok = parts.tokenizer
        if not getattr(tok, "chat_template", None):
            raise HTTPException(400, "this model's tokenizer has no chat template; use /v1/completions")
        try:
            text = tok.apply_chat_template(req.messages, add_generation_prompt=True, tokenize=False)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"chat template refused these messages: {type(e).__name__}: {e}")
        prompt_ids = _encode(tok, text, False)       # the template already placed the special tokens
        max_tokens = req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens
        max_tokens = _budget(len(prompt_ids), max_tokens, req.min_tokens)
        stop_ids = _stop_set(parts, req.ignore_eos, req.stop_token_ids)
        cid = http.headers.get("x-request-id") or f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        stream = _submit(parts, prompt_ids, max_tokens, req.min_tokens, stop_ids, cid, arrival, arrival_epoch)
        first = {"sent": False}

        def chunk(text, finish, usage):
            delta = {"content": text}
            if not first["sent"]:
                delta = {"role": "assistant", "content": text}
                first["sent"] = True
            body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                    "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}
            if usage is not None:
                body["usage"] = usage
            return body

        def usage_only(usage):
            return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                    "choices": [], "usage": usage}

        detok = IncrementalDetokenizer(tok, prompt_ids, skip_special_tokens=req.skip_special_tokens)
        if req.stream:
            return StreamingResponse(stream_events(engine, stream, detok, chunk, usage_only,
                                             _include_usage(req.stream_options)),
                                     media_type="text/event-stream")
        ids, finish, rec = await _collect(stream)
        for tid in ids:
            detok.push(tid)
        detok.flush()
        return {"id": cid, "object": "chat.completion", "created": created, "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": detok.text},
                             "logprobs": None, "finish_reason": finish}],
                "usage": _usage(rec["prompt_len"], rec["out_len"])}

    return app


def main() -> None:
    import uvicorn

    cfg = PagedServeConfig.from_env()
    if not cfg.model:
        raise SystemExit("E4B_PAGED_MODEL is required (plus E4B_PAGED_ARENA and E4B_PAGED_CALIB)")
    exposure = "localhost" if cfg.host in ("127.0.0.1", "localhost", "::1") else f"LAN ({cfg.host})"
    log(f"listening on {cfg.host}:{cfg.port} [{exposure}, {'token-gated' if cfg.token else 'no auth'}] "
        f"max_seqs={cfg.max_seqs if cfg.max_seqs_requested != 'auto' else 'auto (resolved when the engine builds)'} "
        f"max_tokens_per_seq={cfg.max_tokens_per_seq} chunk={cfg.chunk_tokens} "
        f"graphs={int(cfg.graphs)} buckets={list(cfg.buckets)} placement={cfg.placement}; "
        f"the engine builds on its own thread -- /health reports 'loading' until it is ready")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
