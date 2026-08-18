# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Continuous-batching scheduler — hybrid Stage 2, Phase 9.

The engine this replaces (``serve.py``) admits ONE generation at a time
and holds the GPU until it finishes, because two concurrent forwards
would evict each other's staged experts mid-kernel. That is correct and
it is also why aggregate throughput cannot rise with load: a batch of 8
requests costs 8× one request. Continuous batching fixes the shape —
every step carries whatever work is ready, sequences join and leave
between steps, and nobody waits for the slowest.

What this module owns is the DECISION layer, deliberately separated from
the model:

* which sequences run this step, and in what mode;
* how a long prompt is split so its prefill does not starve decode;
* when a new request may be admitted (KV capacity, not optimism);
* the timing facts the gate demands — TTFT including queue wait, and
  per-stream as well as aggregate rates.

A :class:`StepRunner` supplies the execution. That seam is what makes
scheduling testable without a GPU (the tests drive a deterministic fake),
and it is where the mixed-mode split lands: PREFILL chunks are
compute-bound and run GPU-only with expert weights streamed once per
chunk and amortized over its many tokens; DECODE steps are
bandwidth-bound and run the hybrid tier. The measured crossover behind
that split is G8's — the DRAM tier leaves the bandwidth-bound regime near
~8 tokens per expert, and a prefill chunk is far past it.

Fairness is FIFO by arrival with prefill work preferred over decode when
both are ready, which is what keeps TTFT bounded under load; the
alternative (decode-first) starves arrivals and produces exactly the
throughput-looks-great-latency-is-terrible number the gate's protocol
exists to prevent.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence


class Phase(Enum):
    WAITING = "waiting"        # admitted to the queue, no KV yet
    PREFILL = "prefill"        # ingesting prompt chunks
    DECODE = "decode"          # emitting tokens
    DONE = "done"


@dataclass
class Request:
    """One sequence's whole life, including the clocks the gate reads."""
    rid: int
    prompt: Sequence[int]
    max_new_tokens: int
    arrival: float
    phase: Phase = Phase.WAITING
    prompt_pos: int = 0                 # prompt tokens already ingested
    out: list[int] = field(default_factory=list)
    slot: int | None = None             # KV slot while resident
    first_token_at: float | None = None
    finished_at: float | None = None
    admitted_at: float | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt)

    @property
    def ttft(self) -> float | None:
        """Time to first token INCLUDING queue wait — measured from
        arrival, not from admission. Measuring from admission is how a
        loaded server reports a flattering TTFT while callers wait."""
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.arrival

    @property
    def queue_wait(self) -> float | None:
        if self.admitted_at is None:
            return None
        return self.admitted_at - self.arrival


@dataclass
class StepPlan:
    """What one engine step should execute. ``prefill`` carries
    (rid, start, length) chunk descriptors; ``decode`` carries rids that
    need exactly one token each."""
    prefill: list[tuple[int, int, int]] = field(default_factory=list)
    decode: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill and not self.decode

    @property
    def prefill_tokens(self) -> int:
        return sum(n for _, _, n in self.prefill)


class StepRunner(Protocol):
    """Execution seam. The scheduler never imports torch."""

    def run_prefill(self, chunks: list[tuple[int, int, int]]) -> dict[int, int]:
        """Ingest the given (rid, start, length) chunks. Returns
        {rid: sampled_token} ONLY for sequences whose prompt completed in
        this call (that token is their first output)."""

    def run_decode(self, rids: list[int]) -> dict[int, int]:
        """One token for each rid."""

    def bind(self, rid: int, slot: int, prompt) -> None:
        """Optional: told when a sequence is admitted to ``slot``. A
        runner that owns KV needs this to clear the slot's history — a
        recycled slot whose previous tenant is still readable produces
        fluent nonsense rather than an error."""

    def free_slot(self, rid: int) -> None:
        """Release a finished sequence's KV."""


class ContinuousScheduler:
    """Admit/evict per step; no wait-for-slowest.

    ``max_seqs`` and ``kv_slots`` are separate on purpose: the first is a
    batch-width choice, the second is physical KV capacity. Admission
    checks BOTH, because admitting past KV capacity is how a scheduler
    turns a queue delay into a mid-generation eviction.
    """

    def __init__(self, *, runner: StepRunner, max_seqs: int = 8,
                 kv_slots: int | None = None, chunk_tokens: int = 512,
                 max_prefill_tokens_per_step: int | None = None,
                 clock=time.monotonic):
        if max_seqs < 1:
            raise ValueError("max_seqs must be >= 1")
        if chunk_tokens < 1:
            raise ValueError("chunk_tokens must be >= 1")
        self.runner = runner
        self.max_seqs = max_seqs
        self.kv_slots = max_seqs if kv_slots is None else kv_slots
        if self.kv_slots < 1:
            raise ValueError("kv_slots must be >= 1")
        self.chunk_tokens = chunk_tokens
        # a step's prefill budget: one chunk by default. Larger budgets
        # raise prefill throughput and delay every resident decode by the
        # same amount — the tradeoff the gate wants swept, not chosen
        # silently.
        self.max_prefill_tokens = (max_prefill_tokens_per_step
                                   or chunk_tokens)
        self.clock = clock
        self._ids = itertools.count()
        self.queue: list[Request] = []          # arrived, not yet admitted
        self.active: dict[int, Request] = {}
        self.done: list[Request] = []
        self._free_slots = list(range(self.kv_slots))
        self.steps = 0
        self.tokens_emitted = 0
        self.prefill_tokens = 0

    # ------------------------------------------------------------ intake --
    def add_request(self, prompt: Sequence[int], max_new_tokens: int = 16,
                    now: float | None = None) -> int:
        if not len(prompt):
            raise ValueError("empty prompt")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        rid = next(self._ids)
        self.queue.append(Request(rid=rid, prompt=list(prompt),
                                  max_new_tokens=max_new_tokens,
                                  arrival=self.clock() if now is None else now))
        return rid

    def _admit(self) -> None:
        """FIFO admission, bounded by batch width AND KV slots."""
        while (self.queue and len(self.active) < self.max_seqs
               and self._free_slots):
            req = self.queue.pop(0)
            req.slot = self._free_slots.pop(0)
            req.phase = Phase.PREFILL
            req.admitted_at = self.clock()
            self.active[req.rid] = req
            bind = getattr(self.runner, "bind", None)
            if bind is not None:
                bind(req.rid, req.slot, req.prompt)

    # ------------------------------------------------------------- plan --
    def plan(self) -> StepPlan:
        """Compose the next step. Prefill is preferred while any prompt
        is unfinished — decode-first would let a busy stream of long
        generations starve arrivals indefinitely, which shows up as
        excellent aggregate throughput and unbounded TTFT."""
        self._admit()
        plan = StepPlan()
        budget = self.max_prefill_tokens
        for req in self.active.values():
            if req.phase is not Phase.PREFILL or budget <= 0:
                continue
            take = min(self.chunk_tokens, budget,
                       req.prompt_len - req.prompt_pos)
            if take > 0:
                plan.prefill.append((req.rid, req.prompt_pos, take))
                budget -= take
        for req in self.active.values():
            if req.phase is Phase.DECODE:
                plan.decode.append(req.rid)
        return plan

    # ------------------------------------------------------------- step --
    def step(self) -> StepPlan:
        """Run exactly one engine step. Returns the plan that executed
        (empty plan = nothing was ready, which the caller may treat as
        idle rather than as an error)."""
        plan = self.plan()
        if plan.is_empty:
            return plan
        self.steps += 1

        if plan.prefill:
            first = self.runner.run_prefill(plan.prefill)
            self.prefill_tokens += plan.prefill_tokens
            for rid, start, take in plan.prefill:
                req = self.active[rid]
                req.prompt_pos = start + take
                if req.prompt_pos >= req.prompt_len:
                    # prompt fully ingested: the runner hands back this
                    # sequence's FIRST token, and the clock stops here
                    tok = first.get(rid)
                    if tok is None:
                        raise RuntimeError(
                            f"runner completed prompt for rid {rid} without "
                            f"returning its first token")
                    self._emit(req, tok)
        if plan.decode:
            for rid, tok in self.runner.run_decode(plan.decode).items():
                self._emit(self.active[rid], tok)
        self._retire()
        return plan

    def _emit(self, req: Request, token: int) -> None:
        if req.first_token_at is None:
            req.first_token_at = self.clock()
        req.out.append(token)
        self.tokens_emitted += 1
        req.phase = (Phase.DONE if len(req.out) >= req.max_new_tokens
                     else Phase.DECODE)

    def _retire(self) -> None:
        for rid in [r for r, q in self.active.items() if q.phase is Phase.DONE]:
            req = self.active.pop(rid)
            req.finished_at = self.clock()
            self.runner.free_slot(rid)
            self._free_slots.append(req.slot)
            req.slot = None
            self.done.append(req)

    # ------------------------------------------------------------ drive --
    def run_until_idle(self, max_steps: int = 1_000_000) -> int:
        """Step until nothing is queued or active. Returns steps taken."""
        taken = 0
        while (self.queue or self.active) and taken < max_steps:
            if self.step().is_empty:
                break
            taken += 1
        return taken

    # ---------------------------------------------------------- metrics --
    def stats(self) -> dict:
        """Everything the Stage-2 benchmark protocol demands together:
        aggregate AND per-stream rates, with TTFT beside them. Aggregate
        throughput without per-stream latency is not a result."""
        ttfts = sorted(r.ttft for r in self.done if r.ttft is not None)
        waits = sorted(r.queue_wait for r in self.done
                       if r.queue_wait is not None)

        def _pct(xs, p):
            if not xs:
                return None
            i = min(len(xs) - 1, int(round((len(xs) - 1) * p)))
            return xs[i]

        spans = [(r.finished_at - r.arrival) for r in self.done
                 if r.finished_at is not None]
        per_stream = [len(r.out) / s for r, s in zip(self.done, spans)
                      if s and s > 0]
        return {
            "steps": self.steps,
            "completed": len(self.done),
            "in_flight": len(self.active),
            "queued": len(self.queue),
            "tokens_emitted": self.tokens_emitted,
            "prefill_tokens": self.prefill_tokens,
            "ttft_p50": _pct(ttfts, 0.50), "ttft_p99": _pct(ttfts, 0.99),
            "queue_wait_p50": _pct(waits, 0.50),
            "per_stream_tok_s_mean": (sum(per_stream) / len(per_stream)
                                      if per_stream else None),
            "kv_slots_free": len(self._free_slots),
        }
