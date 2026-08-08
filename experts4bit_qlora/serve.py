"""HTTP serving for a QLoRA-tuned fused-MoE: one NF4 base, many adapters, tiny VRAM.

This wraps the package's inference path (:mod:`.infer`) in a FastAPI app so the fine-tune can be
*shared* — by other containers, cron jobs, and agents — instead of each caller paying its own
model load. The design goal is coexistence on a small shared GPU: with ``OFFLOAD_EXPERTS=1``
(the default here, unlike :mod:`.infer`) the frozen experts live in pinned CPU RAM and the
GPU-resident footprint is ~1.7 GB for OLMoE — an LLM endpoint that leaves the card to everyone
else. This is capability, not throughput: decode is batch-1 (~1.4 tok/s offloaded on an RTX
A2000), so responses stream by default-capable SSE and requests queue behind a single worker.

Why a single GPU worker thread (not just an asyncio lock): the offload machinery keeps
class-level residency state (``_ExpertOffload._resident`` / ``_staged_now``) and a per-device
prefetch stream — two forwards racing would evict each other's staged experts mid-kernel. All GPU
work (generation *and* adapter swaps) is serialized onto one thread, and one process serves one
base model.

Multi-adapter serving: adapters are per-expert LoRA state dicts (``train.py`` saves every
``"lora"`` tensor), a few hundred MB at most, and the LoRA parameters stay GPU-resident even
under offload. Each registered adapter is held in (pinned) CPU RAM and copied over the live LoRA
parameters when a request names it — tens of milliseconds against a multi-second generation — so
N fine-tunes cost the VRAM of one. Two invariants make the swap safe (see ``_complete_adapter``):
adapter dicts must contain only ``lora`` keys (a full state dict would hit the offload
placeholders), and every adapter is completed against the model's initial LoRA state so keys one
adapter has and another lacks (e.g. attention LoRA) can never leak between them — a missing pair
falls back to init, where ``lora_B == 0`` makes the delta exactly zero.

Env-configured like :mod:`.train` / :mod:`.infer`::

    E4B_ADAPTERS="alpaca=./out/adapter_best.pt" python -m experts4bit_qlora.serve

Variables: ``MODEL``, ``R``/``ALPHA`` (must match the adapters), ``QUANT_TYPE``,
``OFFLOAD_EXPERTS`` (default **1** here), ``OFFLOAD_PIN``, ``PREFETCH``, ``E4B_ADAPTERS``
(``name=path`` pairs, comma-separated; ``base`` = the un-tuned base model is always available),
``E4B_HOST`` / ``E4B_PORT`` (default 127.0.0.1:8777 — localhost; set ``E4B_HOST=0.0.0.0`` to expose on
the LAN), ``E4B_TOKEN`` (when set, generation routes require ``Authorization: Bearer <token>``),
``E4B_QUEUE_MAX`` (default 2 waiting),
``E4B_MAX_INPUT_TOKENS`` (default 2048 -> 413), ``E4B_MAX_NEW_TOKENS`` (cap, default 256),
``E4B_REQUEST_TIMEOUT_S`` (default 600; partial text with ``stopped:"timeout"``),
``E4B_EMPTY_CACHE`` (default 1: release allocator blocks to the driver when idle),
``E4B_VRAM_FRACTION`` (optional hard cap), ``E4B_WARMUP_TOKENS`` (default 8),
``E4B_RECEIPTS_PATH`` (append a one-line JSON receipt per generation — token counts,
peak VRAM, wall time, versions; empty/unset = off).

``E4B_EXPERT_PROFILE=<path>`` attaches the routing profiler (no-op when unset): the
aggregated JSONL is written once at clean shutdown, and is exactly the input the
residency dial below consumes. Profile with residency OFF, on real traffic.

Residency (spare VRAM -> decode speed), off by default::

    E4B_EXPERT_PROFILE=/tmp/prof.jsonl python -m experts4bit_qlora.serve   # 1. profile a real workload
    E4B_RESIDENCY=pipelined E4B_HOT_PROFILE=/tmp/prof.jsonl E4B_HOT_PER_LAYER=8 \
      python -m experts4bit_qlora.serve                                     # 2. serve with hot experts resident

``E4B_RESIDENCY=pipelined`` keeps the ``E4B_HOT_PER_LAYER`` most-routed experts of each
layer in VRAM and streams the cold tail through the pipelined engine. The hot sets are
**frequency-ranked from a profile, never by index** — an index-ordered set on a
256-expert top-6 layer serves ~6% of routed slots, so there is deliberately no by-index
fallback: without a profile this raises rather than serving a dial that does nothing.
``E4B_K_SLOTS`` overrides the routed top-k if the config lacks ``num_experts_per_tok``.
``/health`` reports the module count and the profile's predicted coverage.

**Residency and trained adapters are mutually exclusive.** The engine patches the frozen
base, and ``ExpertsLoRA`` only delegates to it while the adapter is provably zero — so
activating a trained adapter turns residency off *silently*. Serving ``base`` (the judge
/ evaluation case) is what this is for; a swap to a non-zero adapter logs a warning.

Endpoints: ``POST /generate`` (JSON or SSE), ``GET /health``, and OpenAI-compatible
``/v1/completions`` + ``/v1/models`` (``model`` selects the adapter). There is deliberately no
``/v1/chat/completions``: OLMoE-1B-7B has no chat template and the shipped fine-tunes are
Alpaca-format — send ``### Instruction:\\n...\\n\\n### Response:\\n`` prompts instead.
"""

import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from .util import log

_SENTINEL = object()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ServeConfig:
    """All knobs, read once at startup — never at import (the :mod:`.infer` module-level-env
    pattern makes import order load-bearing; a server must be constructible from code)."""

    model: str = "allenai/OLMoE-1B-7B-0924"
    r: int = 8
    alpha: int = 16
    quant_type: str = "nf4"
    offload: bool = True  # the point of this deployment; .infer defaults off
    pin: bool = True
    prefetch: bool = True
    adapters: Dict[str, str] = field(default_factory=dict)  # name -> path
    host: str = "127.0.0.1"  # localhost by default; LAN exposure is opt-in (E4B_HOST=0.0.0.0)
    port: int = 8777
    token: str = ""  # E4B_TOKEN: when set, require "Authorization: Bearer <token>" on generate routes
    queue_max: int = 2  # requests allowed to WAIT behind the running one
    max_input_tokens: int = 2048
    max_new_tokens: int = 256  # hard cap; requests are clamped, not rejected
    request_timeout_s: float = 600.0
    empty_cache: bool = True
    vram_fraction: float = 0.0  # 0 = off
    warmup_tokens: int = 8
    device: str = "cuda"
    receipts_path: str = ""  # "" = receipts off
    # Residency: trade spare VRAM for decode speed. "" = off (stream every expert),
    # "pipelined" = enable_pipelined_residency. Hot sets come from a PROFILE, never by
    # index — a by-index set on a 256-expert top-6 layer serves ~6% of routed slots, so
    # the dial only works when it is frequency-ranked (see hot_sets_from_profile).
    residency: str = ""
    hot_profile: str = ""      # JSONL from E4B_EXPERT_PROFILE
    hot_per_layer: int = 0     # experts kept resident per layer; K is the dial
    k_slots: int = 0           # routed top-k; 0 = read it off the model config

    @classmethod
    def from_env(cls) -> "ServeConfig":
        return cls(
            model=os.environ.get("MODEL", cls.model),
            r=int(os.environ.get("R", "8")),
            alpha=int(os.environ.get("ALPHA", "16")),
            quant_type=os.environ.get("QUANT_TYPE", "nf4"),
            offload=os.environ.get("OFFLOAD_EXPERTS", "1") == "1",
            pin=os.environ.get("OFFLOAD_PIN", "1") == "1",
            prefetch=os.environ.get("PREFETCH", "1") == "1",
            adapters=parse_adapter_spec(os.environ.get("E4B_ADAPTERS", "")),
            host=os.environ.get("E4B_HOST", "127.0.0.1"),
            port=int(os.environ.get("E4B_PORT", "8777")),
            token=os.environ.get("E4B_TOKEN", ""),
            queue_max=int(os.environ.get("E4B_QUEUE_MAX", "2")),
            max_input_tokens=int(os.environ.get("E4B_MAX_INPUT_TOKENS", "2048")),
            max_new_tokens=int(os.environ.get("E4B_MAX_NEW_TOKENS", "256")),
            request_timeout_s=float(os.environ.get("E4B_REQUEST_TIMEOUT_S", "600")),
            empty_cache=os.environ.get("E4B_EMPTY_CACHE", "1") == "1",
            vram_fraction=float(os.environ.get("E4B_VRAM_FRACTION", "0")),
            warmup_tokens=int(os.environ.get("E4B_WARMUP_TOKENS", "8")),
            receipts_path=os.environ.get("E4B_RECEIPTS_PATH", ""),
            residency=os.environ.get("E4B_RESIDENCY", ""),
            hot_profile=os.environ.get("E4B_HOT_PROFILE", ""),
            hot_per_layer=int(os.environ.get("E4B_HOT_PER_LAYER", "0")),
            k_slots=int(os.environ.get("E4B_K_SLOTS", "0")),
        )


def parse_adapter_spec(spec: str) -> Dict[str, str]:
    """``"alpaca=/adapters/a.pt, helpdesk=/adapters/h.pt"`` -> ``{name: path}``. ``base`` is
    reserved (it is always served: the un-tuned model, i.e. the initial LoRA state)."""
    out: Dict[str, str] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, sep, path = item.partition("=")
        name, path = name.strip(), path.strip()
        if not sep or not name or not path:
            raise ValueError(f"E4B_ADAPTERS entry {item!r}: expected name=path")
        if name == "base":
            raise ValueError("E4B_ADAPTERS: 'base' is reserved for the un-tuned base model")
        if name in out:
            raise ValueError(f"E4B_ADAPTERS: duplicate adapter name {name!r}")
        out[name] = path
    return out


# ---------------------------------------------------------------------------
# Adapter registry (pure tensor-dict logic; unit-testable without a model)
# ---------------------------------------------------------------------------


def validate_adapter(name: str, sd: Dict[str, torch.Tensor], init_state: Dict[str, torch.Tensor]) -> None:
    """Fail fast, before the adapter can crash a request mid-generation.

    Rejects non-``lora`` keys (a full state dict would hit the offload 0-element placeholders —
    ``load_state_dict`` onto an offloaded model was never supported), unknown lora keys, and shape
    mismatches (an R mismatch shows up here structurally; ALPHA is invisible in the file and must
    match by convention — it is baked into ``scaling`` at construction)."""
    non_lora = [k for k in sd if "lora" not in k]
    if non_lora:
        raise ValueError(
            f"adapter {name!r}: {len(non_lora)} non-lora keys (first: {non_lora[0]!r}) — expected a "
            "train.py adapter file (lora tensors only), not a full model state dict"
        )
    unknown = [k for k in sd if k not in init_state]
    if unknown:
        raise ValueError(
            f"adapter {name!r}: {len(unknown)} keys not in this model's LoRA parameter set "
            f"(first: {unknown[0]!r}) — check MODEL and R against the training run"
        )
    for k, t in sd.items():
        if tuple(t.shape) != tuple(init_state[k].shape):
            raise ValueError(
                f"adapter {name!r}: shape mismatch for {k!r}: adapter {tuple(t.shape)} vs model "
                f"{tuple(init_state[k].shape)} — check R against the training run"
            )


def _complete_adapter(
    sd: Dict[str, torch.Tensor], init_state: Dict[str, torch.Tensor], pin: bool
) -> Dict[str, torch.Tensor]:
    """Extend ``sd`` to the model's FULL LoRA key-set so a swap overwrites every LoRA tensor.

    Swaps are plain ``copy_`` over live parameters; a partial dict would leave the previous
    adapter's values in whatever keys this one lacks (classic case: adapter A trained attention
    LoRA, adapter B did not — A's attention deltas would silently color B's outputs). Missing keys
    fall back to the *initial* state, where ``lora_B == 0`` guarantees a zero delta regardless of
    ``lora_A``. Adapter tensors are cast to the init dtype (bf16 params vs an fp32-saved file) and
    pinned best-effort so the H2D swap copy can be async."""
    out: Dict[str, torch.Tensor] = {}
    for k, ref in init_state.items():
        t = sd.get(k, ref).detach().to("cpu", dtype=ref.dtype).contiguous()
        if pin:
            try:
                t = t.pin_memory()
            except (RuntimeError, AssertionError):
                pass  # pageable fallback is correct, the swap copy just blocks the host thread
        out[k] = t
    return out


def build_registry(
    files: Dict[str, Dict[str, torch.Tensor]],
    init_state: Dict[str, torch.Tensor],
    pin: bool = True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Validate + complete every adapter; ``base`` (= the initial LoRA state, delta zero) is
    always present. ``files`` maps adapter name -> loaded state dict."""
    registry = {"base": _complete_adapter({}, init_state, pin)}
    for name, sd in files.items():
        validate_adapter(name, sd, init_state)
        registry[name] = _complete_adapter(sd, init_state, pin)
    return registry


# ---------------------------------------------------------------------------
# Engine: owns the model and the single GPU worker thread
# ---------------------------------------------------------------------------


class BusyError(Exception):
    """Queue full — surfaced as 503 + Retry-After."""


class _StopSignal:
    """Deadline + client-cancel as a transformers StoppingCriteria (duck-typed: the base class is
    imported lazily with the rest of transformers, but the criteria protocol is just __call__)."""

    def __init__(self, stop_event: Optional[threading.Event], deadline: float):
        self.stop_event, self.deadline = stop_event, deadline
        self.reason: Optional[str] = None

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            self.reason = "cancelled"
            return True
        if time.monotonic() > self.deadline:
            self.reason = "timeout"
            return True
        return False


class Engine:
    """One base model, one GPU worker thread, N hot-swappable adapters.

    ``state``: ``loading`` -> ``ready`` (or ``error``). ``/health`` reads attributes only and must
    never touch the worker — a wedged generation should not wedge the healthcheck.
    """

    def __init__(self, cfg: ServeConfig):
        self.cfg = cfg
        self.state = "loading"
        self.error: Optional[str] = None
        self.tokenizer = None
        self.model = None
        self.registry: Dict[str, Dict[str, torch.Tensor]] = {}
        self.active_adapter: Optional[str] = None
        self.pinned_offload: Optional[bool] = None
        self.residency_n: int = 0
        self.residency_coverage: Optional[float] = None
        self.started_at = time.time()
        self._lora_params: Dict[str, torch.nn.Parameter] = {}
        self._pending = 0  # running + queued; mutated only on the event loop thread
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="e4b-gpu")
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # -- lifecycle ----------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        fut = loop.run_in_executor(self._worker, self._load)
        fut.add_done_callback(self._on_load_done)

    def _on_load_done(self, fut) -> None:
        exc = fut.exception()
        if exc is not None:
            self.state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            log(f"serve: model load FAILED — {self.error}")

    def shutdown(self) -> None:
        self._worker.shutdown(wait=False, cancel_futures=True)

    def _load(self) -> None:
        """Runs on the worker thread. Cheap validation first (bad adapter paths must not cost a
        14 GB download), then the streaming load, adapter registry, and a warmup generation (the
        4-bit GEMV probe and allocator pools are lazy — without warmup the first request pays)."""
        from transformers import AutoTokenizer

        from .lora import ExpertsLoRA, add_attention_lora
        from .loader import load_moe_4bit_streaming

        cfg = self.cfg
        files: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, path in cfg.adapters.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"adapter {name!r}: no file at {path}")
            files[name] = torch.load(path, map_location="cpu", weights_only=True)

        if cfg.vram_fraction:
            torch.cuda.set_per_process_memory_fraction(cfg.vram_fraction)

        log(
            f"serve: loading {cfg.model} ({cfg.quant_type}) | offload={'on' if cfg.offload else 'off'}"
            f" prefetch={'on' if (cfg.prefetch and cfg.offload) else 'off'} | adapters: "
            f"{', '.join(files) or '(none)'} + base"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        model, model_cfg = load_moe_4bit_streaming(
            cfg.model,
            cfg.device,
            torch.bfloat16,
            cfg.r,
            cfg.alpha,
            offload=cfg.offload,
            pin=cfg.pin,
            prefetch=cfg.prefetch and cfg.offload,
            quant_type=cfg.quant_type,
        )
        if not cfg.offload:
            model.to(cfg.device)  # under offload a blanket .to() would undo the CPU homes

        # Wrap attention BEFORE snapshotting the init state, so attention lora keys exist for
        # every adapter's zero-fill completion whenever ANY adapter carries them.
        if any(any("self_attn" in k for k in sd) for sd in files.values()):
            n = add_attention_lora(model, cfg.r, cfg.alpha, torch.bfloat16)
            log(f"serve: wrapped {n} attention projections with LoRA (an adapter carries self_attn keys)")

        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        model.config.use_cache = True

        self._lora_params = {k: p for k, p in model.named_parameters() if "lora" in k}
        init_state = {k: p.detach().to("cpu") for k, p in self._lora_params.items()}
        self.registry = build_registry(files, init_state, pin=cfg.pin)
        self.active_adapter = "base"  # the live params ARE the init state right now

        handles = [m._offload for m in model.modules() if isinstance(m, ExpertsLoRA) and hasattr(m, "_offload")]
        self.pinned_offload = all(h.pinned for h in handles) if handles else None

        # Routing profiler — no-op unless E4B_EXPERT_PROFILE names an output path. train.py
        # has attached it since the feature existed; serve never did, so profiling a SERVING
        # workload (the input `hot_sets_from_profile` needs for the residency dial below)
        # silently wrote nothing. Attach before residency: the probes hook the wrapper's
        # forward, which still runs when it delegates to the patched base. The JSONL lands
        # once, at clean process exit (atexit) — docker stop's SIGTERM is enough.
        from . import expert_profile
        expert_profile.attach(model)

        # After eval()/requires_grad_(False) (the delegation preconditions), before warmup.
        lm_cfg = getattr(model_cfg, "text_config", None) or model_cfg
        self._enable_residency(model, lm_cfg)
        self.model = model

        if cfg.warmup_tokens > 0:
            self._generate_once(
                prompt="Hello.",
                adapter="base",
                max_new_tokens=cfg.warmup_tokens,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                seed=None,
                stop_event=None,
                streamer=None,
                record_receipt=False,  # synthetic startup traffic — keep the audit log real
            )
        torch.cuda.synchronize()
        log(f"serve: ready. GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        self.state = "ready"

    # -- request path -------------------------------------------------------

    def count_tokens(self, prompt: str) -> int:
        return len(self.tokenizer(prompt).input_ids)

    def admit(self) -> None:
        """Called on the event loop before submitting a job; capacity = 1 running + queue_max."""
        if self._pending >= 1 + self.cfg.queue_max:
            raise BusyError(f"{self._pending} requests in flight (capacity {1 + self.cfg.queue_max})")
        self._pending += 1

    def release(self) -> None:
        self._pending = max(0, self._pending - 1)

    @property
    def queue_depth(self) -> int:
        return self._pending

    def submit(self, streamer, **job_kwargs):
        """Submit a generation to the GPU worker (call :meth:`admit` first). Returns an awaitable
        resolving to the result dict."""
        return self._loop.run_in_executor(self._worker, lambda: self._generate_once(streamer=streamer, **job_kwargs))

    def _enable_residency(self, model, lm_config) -> None:
        """Attach the pipelined residency engine after load, before warmup.

        Ordering is load-bearing: the engine patches the frozen ``ExpertsNbit`` bases and
        ``ExpertsLoRA`` only delegates to them under eval + no-grad + a provably-zero
        adapter, so this must run after ``model.eval()`` / ``requires_grad_(False)`` and
        before the warmup generation (which should exercise the patched path, not the
        reference one).
        """
        cfg = self.cfg
        if not cfg.residency:
            return
        if cfg.residency != "pipelined":
            raise ValueError(
                f"E4B_RESIDENCY={cfg.residency!r} is not supported; use 'pipelined' or unset it")
        if not cfg.hot_profile or cfg.hot_per_layer <= 0:
            # Refusing beats defaulting. Serving a by-index hot set would look like the
            # feature working while buying ~nothing: on a 256-expert top-6 layer an
            # index-ordered set of 16 catches a routed expert ~6% of the time.
            raise ValueError(
                "E4B_RESIDENCY needs BOTH E4B_HOT_PROFILE (a JSONL from E4B_EXPERT_PROFILE) "
                "and E4B_HOT_PER_LAYER>0. Hot sets must be frequency-ranked; this path "
                "deliberately has no by-index fallback.")
        if not os.path.isfile(cfg.hot_profile):
            raise FileNotFoundError(f"E4B_HOT_PROFILE: no file at {cfg.hot_profile}")

        from .expert_profile import coverage_from_profile, hot_sets_from_profile
        from .hot_residency import target_modules
        from .pipelined import enable_pipelined_residency

        k = cfg.k_slots or int(getattr(lm_config, "num_experts_per_tok", 0) or 0)
        if k <= 0:
            raise ValueError(
                "routed top-k unknown: set E4B_K_SLOTS (the model config has no "
                "num_experts_per_tok). It sizes the slot store — a forward with a "
                "different k silently falls back to the reference path.")

        hot_sets = hot_sets_from_profile(cfg.hot_profile, cfg.hot_per_layer)
        if cfg.offload:
            # Residency and offload are mutually exclusive BY CONSTRUCTION: offload moves the
            # 4-bit expert weights into pinned CPU RAM and leaves 0-element placeholders on the
            # module, while _PipelinedResidency fills its arena by copying those very tensors.
            # Enabling both made the arena copy read from nothing and died deep in the engine
            # as "The size of tensor a (N) must match the size of tensor b (0)" — a shape error
            # that names neither offload nor residency. Reproduced 2026-08-08 on granite
            # (gate_up numel 16777216 with offload off vs 0 with it on) and on Qwen3-30B-A3B
            # via this server, whose default OFFLOAD_EXPERTS=1 made the dial unusable.
            # Residency IS an offload mechanism (hot resident + cold streamed from its own
            # pinned arena), so the fix is to pick one, not to order them.
            raise ValueError(
                "E4B_RESIDENCY=pipelined requires OFFLOAD_EXPERTS=0. Offload leaves 0-element "
                "expert placeholders on the module and the residency arena is built by copying "
                "those tensors, so the combination cannot work — residency already provides the "
                "hot-resident/cold-streamed split that offload is doing. Set OFFLOAD_EXPERTS=0 "
                "and size E4B_HOT_PER_LAYER to the VRAM you actually have.")
        n_targets = len(target_modules(model))
        if len(hot_sets) != n_targets:
            # The engine takes one entry per targeted module IN MODULE ORDER; a profile
            # from a different model (or one missing trailing layers) would silently
            # shift every set onto the wrong layer.
            raise ValueError(
                f"hot_sets has {len(hot_sets)} entries but the model has {n_targets} "
                f"targetable expert modules — profile/model mismatch. Re-profile "
                f"{cfg.model} rather than padding.")

        cov = coverage_from_profile(cfg.hot_profile, hot_sets)
        n = enable_pipelined_residency(model, hot_sets, device=cfg.device, k_slots=k)
        if n <= 0:
            raise RuntimeError(
                "enable_pipelined_residency patched 0 modules — residency was requested "
                "and is not running. Refusing to serve a silently-unaccelerated model.")
        self.residency_n = n
        self.residency_coverage = cov
        log(f"serve: pipelined residency on {n} module(s) | hot={cfg.hot_per_layer}/layer "
            f"k_slots={k} | profile coverage {cov:.1%} of routed token-slots")

    def _swap_adapter(self, name: str) -> float:
        """Copy an adapter's tensors over the live LoRA parameters (worker thread only). The
        copies are enqueued on the compute stream, so a following generate is ordered after them;
        the sync is only to keep the reported swap_ms honest."""
        if name == self.active_adapter:
            return 0.0
        t0 = time.perf_counter()
        sd = self.registry[name]
        with torch.no_grad():
            for k, src in sd.items():
                self._lora_params[k].data.copy_(src, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.active_adapter = name

        # ``copy_`` onto the live LoRA params fires NEITHER of ExpertsLoRA's cache
        # invalidations (train() / load_state_dict), so ``_delegate_ok`` keeps the
        # PREVIOUS adapter's verdict. With residency enabled that is not a perf bug but a
        # correctness one: a stale "adapter is zero" keeps delegating to the patched base
        # and serves BASE outputs under the new adapter's name (caught by Bugbot on the
        # PR that added this method; reproduced before fixing). Invalidate first, so the
        # warning below also reads truth rather than the same stale cache.
        from .lora import ExpertsLoRA
        wrapped = [m for m in self.model.modules() if isinstance(m, ExpertsLoRA)]
        for m in wrapped:
            m._delegate_ok = None

        if self.residency_n:
            # The engine patches the frozen base; ExpertsLoRA only delegates to it while
            # the adapter is provably zero. A trained adapter therefore turns residency
            # OFF silently — same speed as never enabling it, no error, and the /health
            # counters would still read "residency: 48 modules". Say it once per swap.
            nonzero = sum(1 for m in wrapped if not m._adapter_is_zero())
            if nonzero:
                log(f"serve: WARNING adapter {name!r} is non-zero on {nonzero} expert "
                    f"module(s) — pipelined residency cannot run under it (the wrapper "
                    f"stops delegating), so decode falls back to the streaming path.")
        return (time.perf_counter() - t0) * 1000.0

    def _generate_once(
        self,
        prompt,
        adapter,
        max_new_tokens,
        temperature,
        top_p,
        repetition_penalty,
        seed,
        stop_event,
        streamer,
        record_receipt=True,
    ) -> dict:
        """The one function that touches the GPU (worker thread only): swap, generate, account."""
        from transformers import StoppingCriteriaList

        tok, model, cfg = self.tokenizer, self.model, self.cfg
        write_receipt = bool(cfg.receipts_path) and record_receipt
        if write_receipt and torch.cuda.is_available():
            # Single worker thread: the peak window is exactly this request (swap + generate).
            torch.cuda.reset_peak_memory_stats()
        swap_ms = self._swap_adapter(adapter)
        if seed is not None:
            torch.manual_seed(seed)

        enc = tok(prompt, return_tensors="pt").to(cfg.device)
        n_prompt = enc.input_ids.shape[1]
        stop = _StopSignal(stop_event, time.monotonic() + cfg.request_timeout_s)
        do_sample = temperature is not None and temperature > 0.0

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([stop]),
                streamer=streamer,
            )
        dt = time.perf_counter() - t0
        n_new = out.shape[1] - n_prompt
        text = tok.decode(out[0][n_prompt:], skip_special_tokens=True)
        stopped = stop.reason or ("length" if n_new >= max_new_tokens else "eos")

        if write_receipt:
            _append_receipt(
                cfg.receipts_path,
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "adapter": adapter,
                    "input_tokens": n_prompt,
                    "output_tokens": n_new,
                    # 1e9 divisor to stay comparable with /health's *_gb fields.
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)
                    if torch.cuda.is_available()
                    else None,
                    "wall_s": round(dt, 1),
                    "tok_per_s": round(n_new / dt, 3) if dt > 0 else 0.0,
                    "stopped": stopped,
                    "e4b_version": _E4B_VERSION,
                    "torch": torch.__version__,
                },
            )
        # Return freed allocator blocks to the driver when nothing is waiting, so bursty
        # neighbors (TTS/SDXL) see the memory. Benign race on _pending: worst case we skip once.
        if cfg.empty_cache and self._pending <= 1 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "text": text,
            "adapter": adapter,
            "prompt_tokens": n_prompt,
            "tokens": n_new,
            "tok_per_s": round(n_new / dt, 3) if dt > 0 else 0.0,
            "swap_ms": round(swap_ms, 1),
            "stopped": stopped,
        }


try:
    import importlib.metadata

    _E4B_VERSION = importlib.metadata.version("experts4bit-qlora")
except Exception:  # not installed as a dist (e.g. PYTHONPATH use)
    from . import __version__ as _E4B_VERSION


def _append_receipt(path: str, record: dict) -> None:
    """Append one JSON line to the receipts log. Never raises — a receipts problem
    (read-only mount, full disk) must not break serving."""
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        log(f"receipts: dropped record ({e})")


def _gpu_stats() -> Optional[dict]:
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        "free_gb": round(free / 1e9, 3),
        "total_gb": round(total / 1e9, 3),
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app(cfg: Optional[ServeConfig] = None, engine: Optional[Engine] = None):
    """App factory. ``engine`` injection exists for tests (a fake with the same surface)."""
    from contextlib import asynccontextmanager

    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    cfg = cfg or ServeConfig.from_env()
    engine = engine or Engine(cfg)

    def _auth(authorization: Optional[str] = Header(None)):
        # Off by default (localhost tool). When E4B_TOKEN is set, the generation
        # routes require "Authorization: Bearer <token>"; /health stays open so
        # monitors can poll it unauthenticated.
        if cfg.token and authorization != f"Bearer {cfg.token}":
            raise HTTPException(401, "missing or invalid bearer token (set 'Authorization: Bearer <E4B_TOKEN>')")

    @asynccontextmanager
    async def lifespan(app):
        engine.start(asyncio.get_running_loop())
        yield
        engine.shutdown()

    app = FastAPI(title="experts4bit-qlora serve", lifespan=lifespan)
    app.state.engine = engine

    class GenerateRequest(BaseModel):
        prompt: str
        adapter: str = "base"
        max_new_tokens: int = 64
        temperature: float = 0.0
        top_p: float = 1.0
        repetition_penalty: float = 1.3
        stream: bool = False
        seed: Optional[int] = None

    class CompletionRequest(BaseModel):
        prompt: str
        model: str = "base"
        max_tokens: int = 64
        temperature: float = 0.0
        top_p: float = 1.0
        stream: bool = False
        seed: Optional[int] = None

    def _check(req_adapter: str, prompt: str, max_new: int) -> int:
        """Shared admission checks; returns the clamped max_new_tokens."""
        if engine.state == "error":
            raise HTTPException(500, f"model failed to load: {engine.error}")
        if engine.state != "ready":
            raise HTTPException(503, "model is loading", headers={"Retry-After": "60"})
        if req_adapter not in engine.registry:
            raise HTTPException(404, f"unknown adapter {req_adapter!r}; available: {sorted(engine.registry)}")
        if engine.count_tokens(prompt) > cfg.max_input_tokens:
            raise HTTPException(413, f"prompt exceeds E4B_MAX_INPUT_TOKENS={cfg.max_input_tokens}")
        return max(1, min(max_new, cfg.max_new_tokens))

    def _job_kwargs(prompt, adapter, max_new, temperature, top_p, repetition_penalty, seed) -> dict:
        return dict(
            prompt=prompt,
            adapter=adapter,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            stop_event=None,
        )

    async def _run(job: dict) -> dict:
        try:
            engine.admit()
        except BusyError as e:
            raise HTTPException(503, str(e), headers={"Retry-After": "60"})
        try:
            return await engine.submit(streamer=None, **job)
        finally:
            engine.release()

    def _sse_stream(job: dict, token_to_event, meta_to_event) -> StreamingResponse:
        """Shared SSE plumbing: the generation runs on the GPU worker; tokens cross to the event
        loop through the streamer's queue (a plain thread-safe iterator). The finally block covers
        client disconnects — the stop event ends the generation at its next token."""
        from transformers import TextIteratorStreamer

        try:
            engine.admit()
        except BusyError as e:
            raise HTTPException(503, str(e), headers={"Retry-After": "60"})
        stop_event = threading.Event()
        job = dict(job, stop_event=stop_event)
        streamer = TextIteratorStreamer(
            engine.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=cfg.request_timeout_s + 60
        )
        fut = engine.submit(streamer=streamer, **job)

        async def gen():
            try:
                it = iter(streamer)
                while True:
                    piece = await asyncio.to_thread(next, it, _SENTINEL)
                    if piece is _SENTINEL:
                        break
                    if piece:
                        yield token_to_event(piece)
                meta = await fut
                for line in meta_to_event(meta):
                    yield line
            except Exception as e:  # the response already started; surface the failure in-band
                yield f"data: {json.dumps({'error': f'{type(e).__name__}: {e}'})}\n\n"
            finally:
                stop_event.set()
                engine.release()

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/health")
    async def health():
        return {
            "status": "busy" if (engine.state == "ready" and engine.queue_depth > 0) else engine.state,
            "error": engine.error,
            "model": cfg.model,
            "adapters": sorted(engine.registry),
            "active_adapter": engine.active_adapter,
            "queue_depth": engine.queue_depth,
            "gpu": _gpu_stats(),
            "offload": {"enabled": cfg.offload, "pinned": engine.pinned_offload},
            # Observable on purpose: "did the residency dial actually engage" is not
            # answerable from tok/s alone, and a patched-but-never-running engine is the
            # failure this reports (see _swap_adapter's non-zero-adapter warning).
            "residency": {
                "mode": cfg.residency or None,
                "modules": engine.residency_n,
                "hot_per_layer": cfg.hot_per_layer or None,
                "profile_coverage": (round(engine.residency_coverage, 4)
                                     if engine.residency_coverage is not None else None),
            },
            "uptime_s": round(time.time() - engine.started_at, 1),
        }

    @app.post("/generate", dependencies=[Depends(_auth)])
    async def generate(req: GenerateRequest):
        max_new = _check(req.adapter, req.prompt, req.max_new_tokens)
        job = _job_kwargs(req.prompt, req.adapter, max_new, req.temperature, req.top_p, req.repetition_penalty, req.seed)
        if not req.stream:
            return await _run(job)
        return _sse_stream(
            job,
            lambda piece: f"data: {json.dumps({'token': piece})}\n\n",
            lambda meta: [f"data: {json.dumps(dict(meta, done=True))}\n\n"],
        )

    @app.get("/v1/models")
    async def v1_models():
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "experts4bit-qlora"} for name in sorted(engine.registry)
            ],
        }

    @app.post("/v1/completions", dependencies=[Depends(_auth)])
    async def v1_completions(req: CompletionRequest):
        max_new = _check(req.model, req.prompt, req.max_tokens)
        job = _job_kwargs(req.prompt, req.model, max_new, req.temperature, req.top_p, 1.3, req.seed)
        rid, created = f"cmpl-{int(time.time() * 1000):x}", int(time.time())

        def _oai(meta: dict, text: str, finish: Optional[str]) -> dict:
            return {
                "id": rid,
                "object": "text_completion",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "text": text, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": meta.get("prompt_tokens", 0),
                    "completion_tokens": meta.get("tokens", 0),
                    "total_tokens": meta.get("prompt_tokens", 0) + meta.get("tokens", 0),
                },
            }

        if not req.stream:
            meta = await _run(job)
            finish = "stop" if meta["stopped"] == "eos" else "length"
            return _oai(meta, meta["text"], finish)

        def token_to_event(piece: str) -> str:
            chunk = _oai({}, piece, None)
            return f"data: {json.dumps(chunk)}\n\n"

        def meta_to_event(meta):
            finish = "stop" if meta["stopped"] == "eos" else "length"
            yield f"data: {json.dumps(_oai(meta, '', finish))}\n\n"
            yield "data: [DONE]\n\n"

        return _sse_stream(job, token_to_event, meta_to_event)

    return app


def main() -> None:
    import uvicorn

    cfg = ServeConfig.from_env()
    exposure = "localhost" if cfg.host in ("127.0.0.1", "localhost", "::1") else f"LAN ({cfg.host})"
    auth = "token-gated" if cfg.token else "no auth"
    log(f"serve: listening on {cfg.host}:{cfg.port} [{exposure}, {auth}] (docs at /docs)")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
