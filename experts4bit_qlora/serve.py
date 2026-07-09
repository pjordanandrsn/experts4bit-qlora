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
``E4B_HOST`` / ``E4B_PORT`` (default 0.0.0.0:8777), ``E4B_QUEUE_MAX`` (default 2 waiting),
``E4B_MAX_INPUT_TOKENS`` (default 2048 -> 413), ``E4B_MAX_NEW_TOKENS`` (cap, default 256),
``E4B_REQUEST_TIMEOUT_S`` (default 600; partial text with ``stopped:"timeout"``),
``E4B_EMPTY_CACHE`` (default 1: release allocator blocks to the driver when idle),
``E4B_VRAM_FRACTION`` (optional hard cap), ``E4B_WARMUP_TOKENS`` (default 8),
``E4B_RECEIPTS_PATH`` (append a one-line JSON receipt per generation — token counts,
peak VRAM, wall time, versions; empty/unset = off).

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
from typing import Dict, List, Optional, Union

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
    host: str = "0.0.0.0"
    port: int = 8777
    queue_max: int = 2  # requests allowed to WAIT behind the running one
    max_input_tokens: int = 2048
    max_new_tokens: int = 256  # hard cap; requests are clamped, not rejected
    request_timeout_s: float = 600.0
    empty_cache: bool = True
    vram_fraction: float = 0.0  # 0 = off
    warmup_tokens: int = 8
    device: str = "cuda"
    receipts_path: str = ""  # "" = receipts off

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
            host=os.environ.get("E4B_HOST", "0.0.0.0"),
            port=int(os.environ.get("E4B_PORT", "8777")),
            queue_max=int(os.environ.get("E4B_QUEUE_MAX", "2")),
            max_input_tokens=int(os.environ.get("E4B_MAX_INPUT_TOKENS", "2048")),
            max_new_tokens=int(os.environ.get("E4B_MAX_NEW_TOKENS", "256")),
            request_timeout_s=float(os.environ.get("E4B_REQUEST_TIMEOUT_S", "600")),
            empty_cache=os.environ.get("E4B_EMPTY_CACHE", "1") == "1",
            vram_fraction=float(os.environ.get("E4B_VRAM_FRACTION", "0")),
            warmup_tokens=int(os.environ.get("E4B_WARMUP_TOKENS", "8")),
            receipts_path=os.environ.get("E4B_RECEIPTS_PATH", ""),
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
        model, _ = load_moe_4bit_streaming(
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

    def _swap_adapter(self, name: str) -> float:
        """Copy an adapter's tensors over the live LoRA parameters (worker thread only). The
        copies are enqueued on the compute stream, so a following generate is ordered after them;
        the sync is only to keep the reported swap_ms honest.

        ``active_adapter`` is cleared BEFORE the copies and only restored after they all land: a
        mid-swap failure otherwise leaves the live weights a mix of two adapters while the name
        still claims the old one — and every later same-adapter request would skip the swap and
        generate silently wrong output. With the clear, the next request re-swaps from scratch
        (the copies are idempotent full overwrites, so a re-swap always converges)."""
        if name == self.active_adapter:
            return 0.0
        t0 = time.perf_counter()
        sd = self.registry[name]
        self.active_adapter = None
        with torch.no_grad():
            for k, src in sd.items():
                self._lora_params[k].data.copy_(src, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.active_adapter = name
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
        stop_strings=None,
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
                # transformers' native stop-sequence support (needs the tokenizer to bind them).
                stop_strings=stop_strings or None,
                tokenizer=tok if stop_strings else None,
            )
        dt = time.perf_counter() - t0
        n_new = out.shape[1] - n_prompt
        text = tok.decode(out[0][n_prompt:], skip_special_tokens=True)
        # generate() includes the matched stop string in the output — trim it (and anything a
        # multi-token stop dragged in past it) so callers can parse the text directly.
        hit_stop = False
        if stop_strings:
            cuts = [i for i in (text.find(s) for s in stop_strings) if i != -1]
            if cuts:
                text = text[: min(cuts)]
                hit_stop = True
        stopped = stop.reason or ("stop" if hit_stop else "length" if n_new >= max_new_tokens else "eos")

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

    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    cfg = cfg or ServeConfig.from_env()
    engine = engine or Engine(cfg)

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
        stop: Optional[Union[str, List[str]]] = None  # stop sequence(s), trimmed from the output

    class CompletionRequest(BaseModel):
        prompt: str
        model: str = "base"
        max_tokens: int = 64
        temperature: float = 0.0
        top_p: float = 1.0
        stream: bool = False
        seed: Optional[int] = None
        stop: Optional[Union[str, List[str]]] = None  # OpenAI-compat: string or list

    def _stop_list(stop) -> Optional[List[str]]:
        if not stop:
            return None
        return [stop] if isinstance(stop, str) else [s for s in stop if s]

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

    def _job_kwargs(prompt, adapter, max_new, temperature, top_p, repetition_penalty, seed, stop=None) -> dict:
        return dict(
            prompt=prompt,
            adapter=adapter,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            stop_event=None,
            stop_strings=_stop_list(stop),
        )

    def _release_when_done(fut) -> None:
        """Free the admission slot when the GPU job actually finishes — not when the HTTP
        handler unwinds. A client disconnect cancels the handler coroutine, but the worker keeps
        generating until the stop event lands; releasing early would let admits outrun the GPU
        and queue real work behind a phantom-free slot. Runs on the event loop thread (the only
        mutator of the pending counter)."""
        engine.release()
        if not fut.cancelled():
            fut.exception()  # retrieve, so an abandoned failure doesn't warn at GC

    async def _run(job: dict) -> dict:
        try:
            engine.admit()
        except BusyError as e:
            raise HTTPException(503, str(e), headers={"Retry-After": "60"})
        stop_event = threading.Event()
        fut = engine.submit(streamer=None, **dict(job, stop_event=stop_event))
        fut.add_done_callback(_release_when_done)
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            stop_event.set()  # client went away: end the generation at its next token
            raise

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
        fut.add_done_callback(_release_when_done)

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
                stop_event.set()  # the slot itself frees when the worker finishes (done callback)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/health")
    async def health(response: Response):
        # A failed load must FAIL the healthcheck: 200-with-status-"error" reads as healthy to
        # docker/compose and leaves a dead endpoint looking green. "loading" stays 200 — the
        # compose start_period covers boot, and a 5xx there would just flap the check early.
        if engine.state == "error":
            response.status_code = 503
        return {
            "status": "busy" if (engine.state == "ready" and engine.queue_depth > 0) else engine.state,
            "error": engine.error,
            "model": cfg.model,
            "adapters": sorted(engine.registry),
            "active_adapter": engine.active_adapter,
            "queue_depth": engine.queue_depth,
            "gpu": _gpu_stats(),
            "offload": {"enabled": cfg.offload, "pinned": engine.pinned_offload},
            "uptime_s": round(time.time() - engine.started_at, 1),
        }

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        max_new = _check(req.adapter, req.prompt, req.max_new_tokens)
        job = _job_kwargs(
            req.prompt, req.adapter, max_new, req.temperature, req.top_p, req.repetition_penalty, req.seed, req.stop
        )
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

    @app.post("/v1/completions")
    async def v1_completions(req: CompletionRequest):
        max_new = _check(req.model, req.prompt, req.max_tokens)
        job = _job_kwargs(req.prompt, req.model, max_new, req.temperature, req.top_p, 1.3, req.seed, req.stop)
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
            finish = {"eos": "stop", "stop": "stop", "length": "length"}.get(meta["stopped"], meta["stopped"])
            return _oai(meta, meta["text"], finish)

        def token_to_event(piece: str) -> str:
            chunk = _oai({}, piece, None)
            return f"data: {json.dumps(chunk)}\n\n"

        def meta_to_event(meta):
            finish = {"eos": "stop", "stop": "stop", "length": "length"}.get(meta["stopped"], meta["stopped"])
            yield f"data: {json.dumps(_oai(meta, '', finish))}\n\n"
            yield "data: [DONE]\n\n"

        return _sse_stream(job, token_to_event, meta_to_event)

    return app


def main() -> None:
    import uvicorn

    cfg = ServeConfig.from_env()
    log(f"serve: listening on {cfg.host}:{cfg.port} (docs at /docs)")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
