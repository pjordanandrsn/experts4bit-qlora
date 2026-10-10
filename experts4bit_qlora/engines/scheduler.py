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
    # Optional stop set (the serving layer's EOS ids). ``None`` keeps the
    # original contract -- the sequence runs to ``max_new_tokens`` -- which
    # is what every registered serving measurement did. A token in the set
    # ends the sequence ONLY once ``min_tokens`` have been emitted; the stop
    # token itself is kept in ``out`` (it was computed and it counts), its
    # text is the caller's business.
    stop_ids: frozenset | None = None
    min_tokens: int = 0
    finish_reason: str | None = None    # "length" | "stop" | "abort" once DONE

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
    need exactly one token each. Under the decode lookahead ``decode`` is
    the step ISSUED and ``collected`` the rids whose previously issued
    step's tokens were read back (and emitted, unless already finished)."""
    prefill: list[tuple[int, int, int]] = field(default_factory=list)
    decode: list[int] = field(default_factory=list)
    collected: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill and not self.decode and not self.collected

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
        """One token for each rid. A speculative runner (``speculative = True``, ``E4B_PAGED_SPEC``) may return a
        list of tokens for a rid instead, emitted in order (:meth:`ContinuousScheduler._emit_tokens`)."""

    def bind(self, rid: int, slot: int, prompt) -> None:
        """Optional: told when a sequence is admitted to ``slot``. A
        runner that owns KV needs this to clear the slot's history — a
        recycled slot whose previous tenant is still readable produces
        fluent nonsense rather than an error."""

    def free_slot(self, rid: int) -> None:
        """Release a finished sequence's KV."""

    # Optional, for ContinuousScheduler(lookahead=True):
    # issue_decode(rids) -> handle   enqueue one token for each rid, without waiting
    # collect_decode(handle) -> {rid: token}   wait for that step; steps are collected in issue order
    #
    # Optional, for a speculative runner (``speculative = True``):
    # decode_budgets({rid: tokens left})   told before each run_decode how many tokens each rid may still emit, so a
    #                                       verify never runs past a request's length


class ContinuousScheduler:
    """Admit/evict per step; no wait-for-slowest.

    ``max_seqs`` and ``kv_slots`` are separate on purpose: the first is a
    batch-width choice, the second is physical KV capacity. Admission
    checks BOTH, because admitting past KV capacity is how a scheduler
    turns a queue delay into a mid-generation eviction.

    ``lookahead=True`` (``E4B_PAGED_DECODE_LOOKAHEAD``, lane P118) runs decode
    through the runner's ``issue_decode`` / ``collect_decode``: each step issues
    the next decode step and then collects the one issued before it, so the
    GPU has a step queued while the host emits, retires and plans. A
    sequence is not issued past ``max_new_tokens``; one that stops on a stop
    id has had one more step issued, whose token is discarded
    (``lookahead_discarded``), and its slot is freed once that step is
    collected. When requests wait for a slot and the queued step carries a
    sequence that ends with it, that step is collected before planning, so
    admission is the synchronous path's. Sequences, tokens and finish
    reasons are the synchronous path's; a sequence that stops on a stop id
    frees its slot one step later.
    """

    def __init__(self, *, runner: StepRunner, max_seqs: int = 8,
                 kv_slots: int | None = None, chunk_tokens: int = 512,
                 max_prefill_tokens_per_step: int | None = None,
                 clock=time.monotonic, lookahead: bool = False):
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
        self.aborted: list[Request] = []        # see abort(); kept out of done
        self._free_slots = list(range(self.kv_slots))
        self.steps = 0
        self.tokens_emitted = 0
        self.prefill_tokens = 0
        # optional per-step timeline (engines.step_trace.StepTrace; serve_paged's E4B_PAGED_STEP_TRACE)
        self.tracer = None
        self.lookahead = bool(lookahead)
        if self.lookahead and not (callable(getattr(runner, "issue_decode", None))
                                   and callable(getattr(runner, "collect_decode", None))):
            raise ValueError("lookahead needs a runner with issue_decode and collect_decode")
        self._pending = None                    # lookahead: the issued, uncollected decode step
        self._pending_rids: frozenset = frozenset()
        self._inflight: dict[int, int] = {}     # rid -> tokens issued, not yet collected
        self.lookahead_discarded = 0            # tokens collected for a sequence that had already finished
        # a speculative runner (E4B_PAGED_SPEC, lane SD2) returns several tokens a step; the ones after the token that
        # finishes a sequence (a stop id, or its length) are dropped and counted here
        self.speculative = bool(getattr(runner, "speculative", False))
        if self.speculative and not callable(getattr(runner, "decode_budgets", None)):
            raise ValueError("a speculative runner needs decode_budgets: a verify must not run past a request's length")
        if self.speculative and self.lookahead:
            raise ValueError("the decode lookahead and a speculative runner do not combine: each speculative step "
                             "reads its accepted tokens back before the next is planned")
        self.spec_dropped = 0

    # ------------------------------------------------------------ intake --
    def add_request(self, prompt: Sequence[int], max_new_tokens: int = 16,
                    now: float | None = None, *,
                    stop_ids=None, min_tokens: int = 0) -> int:
        if not len(prompt):
            raise ValueError("empty prompt")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if min_tokens < 0:
            raise ValueError("min_tokens must be >= 0")
        rid = next(self._ids)
        self.queue.append(Request(rid=rid, prompt=list(prompt),
                                  max_new_tokens=max_new_tokens,
                                  arrival=self.clock() if now is None else now,
                                  stop_ids=(None if stop_ids is None
                                            else frozenset(int(t) for t in stop_ids)),
                                  min_tokens=int(min_tokens)))
        return rid

    def abort(self, rid: int) -> bool:
        """Drop a request the caller no longer wants (a disconnected
        client). Queued: removed before it ever takes a slot. Active: its
        slot is freed NOW, not at the end of the step it would have
        finished in. Aborted requests go to ``aborted``, never ``done``,
        so they cannot pull the gate's percentiles either way. Returns
        whether anything was found; must be called between steps by the
        thread that steps (the scheduler has no locks of its own)."""
        for i, req in enumerate(self.queue):
            if req.rid == rid:
                self.queue.pop(i)
                req.phase = Phase.DONE
                req.finish_reason = "abort"
                req.finished_at = self.clock()
                self.aborted.append(req)
                return True
        req = self.active.get(rid)
        if req is None:
            return False
        req.phase = Phase.DONE
        req.finish_reason = "abort"
        self._retire()
        return True

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
        if self.lookahead:
            return self._step_lookahead()
        tr = self.tracer
        queued = len(self.queue)
        plan = self.plan()
        if plan.is_empty:
            return plan
        self.steps += 1
        if tr is not None:
            tr.mark("plan")
            tr.note(admitted=queued - len(self.queue), active=len(self.active), queued=len(self.queue))

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
            if tr is not None:
                tr.mark("pf_emit")
        if plan.decode:
            if self.speculative:
                self.runner.decode_budgets({rid: self.active[rid].max_new_tokens - len(self.active[rid].out)
                                            for rid in plan.decode})
            for rid, got in self.runner.run_decode(plan.decode).items():
                self._emit_tokens(self.active[rid], got)
            if tr is not None:
                tr.mark("dec_emit")
        self._retire()
        if tr is not None:
            tr.mark("retire")
        return plan

    def _step_lookahead(self) -> StepPlan:
        """:meth:`step` under the decode lookahead: prefill as usual, issue
        the next decode step, then collect the previous one."""
        tr = self.tracer
        queued = len(self.queue)
        early = []
        if (self._pending is not None and self.queue
                and not (len(self.active) < self.max_seqs and self._free_slots)
                and any(self._ends_with_pending(self.active[r]) for r in self._pending_rids)):
            # a request waits for a slot that the queued step's last token frees: collect it before admitting
            early = self._collect_pending()
            self._retire()
        plan = self.plan()
        plan.collected = early
        # a sequence whose issued steps already reach its length is not issued again; it waits for them
        plan.decode = [rid for rid in plan.decode
                       if len(self.active[rid].out) + self._inflight.get(rid, 0)
                       < self.active[rid].max_new_tokens]
        if plan.is_empty and self._pending is None:
            return plan
        self.steps += 1
        if tr is not None:
            tr.mark("plan")
            tr.note(admitted=queued - len(self.queue), active=len(self.active), queued=len(self.queue))

        if plan.prefill:
            first = self.runner.run_prefill(plan.prefill)
            self.prefill_tokens += plan.prefill_tokens
            for rid, start, take in plan.prefill:
                req = self.active[rid]
                req.prompt_pos = start + take
                if req.prompt_pos >= req.prompt_len:
                    tok = first.get(rid)
                    if tok is None:
                        raise RuntimeError(
                            f"runner completed prompt for rid {rid} without "
                            f"returning its first token")
                    self._emit(req, tok)
            if tr is not None:
                tr.mark("pf_emit")
        issued = self.runner.issue_decode(plan.decode) if plan.decode else None
        for rid in plan.decode:
            self._inflight[rid] = self._inflight.get(rid, 0) + 1
        if self._pending is not None:
            plan.collected = self._collect_pending()
        self._pending = issued
        self._pending_rids = frozenset(plan.decode) if issued is not None else frozenset()
        self._retire()
        if tr is not None:
            tr.mark("retire")
        return plan

    def _ends_with_pending(self, req: Request) -> bool:
        """Whether the queued decode step is the last ``req`` needs: it has
        finished already, or that step's token reaches its length."""
        return (req.phase is Phase.DONE
                or len(req.out) + self._inflight.get(req.rid, 0) >= req.max_new_tokens)

    def _collect_pending(self) -> list[int]:
        """Read the queued decode step back and emit its tokens (a sequence
        that finished meanwhile has its token discarded). The caller retires
        what finished."""
        got = self.runner.collect_decode(self._pending)
        self._pending, self._pending_rids = None, frozenset()
        for rid, tok in got.items():
            left = self._inflight[rid] - 1
            if left:
                self._inflight[rid] = left
            else:
                del self._inflight[rid]
            req = self.active[rid]
            if req.phase is Phase.DECODE:
                self._emit(req, tok)
                if req.phase is Phase.DONE:
                    req.finished_at = self.clock()
            else:
                self.lookahead_discarded += 1
        if self.tracer is not None:
            self.tracer.mark("dec_emit")
        return list(got)

    def _emit(self, req: Request, token: int) -> None:
        if req.first_token_at is None:
            req.first_token_at = self.clock()
        req.out.append(token)
        self.tokens_emitted += 1
        # a stop token at the length boundary reports "stop", as vLLM does:
        # the model ended the sequence, the budget merely coincided
        if (req.stop_ids is not None and token in req.stop_ids
                and len(req.out) >= req.min_tokens):
            req.phase = Phase.DONE
            req.finish_reason = "stop"
        elif len(req.out) >= req.max_new_tokens:
            req.phase = Phase.DONE
            req.finish_reason = "length"
        else:
            req.phase = Phase.DECODE

    def _emit_tokens(self, req: Request, got) -> None:
        """One decode step's output for ``req``: a token, or a speculative runner's list of tokens, emitted in order.
        The tokens after the one that finishes the sequence are dropped and counted in ``spec_dropped``."""
        if not isinstance(got, (list, tuple)):
            self._emit(req, got)
            return
        if not got:
            raise RuntimeError(f"runner returned no token for rid {req.rid}")
        for i, tok in enumerate(got):
            self._emit(req, int(tok))
            if req.phase is Phase.DONE:
                self.spec_dropped += len(got) - i - 1
                return

    def _retire(self) -> None:
        # under the lookahead a finished sequence keeps its slot while an issued step still carries it
        for rid in [r for r, q in self.active.items()
                    if q.phase is Phase.DONE and r not in self._pending_rids]:
            req = self.active.pop(rid)
            if req.finished_at is None:
                req.finished_at = self.clock()
            self.runner.free_slot(rid)
            self._free_slots.append(req.slot)
            req.slot = None
            (self.aborted if req.finish_reason == "abort" else self.done).append(req)

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
            "aborted": len(self.aborted),
            "in_flight": len(self.active),
            "queued": len(self.queue),
            "tokens_emitted": self.tokens_emitted,
            "prefill_tokens": self.prefill_tokens,
            "ttft_p50": _pct(ttfts, 0.50), "ttft_p99": _pct(ttfts, 0.99),
            "queue_wait_p50": _pct(waits, 0.50),
            "per_stream_tok_s_mean": (sum(per_stream) / len(per_stream)
                                      if per_stream else None),
            "kv_slots_free": len(self._free_slots),
            **({"lookahead_discarded": self.lookahead_discarded} if self.lookahead else {}),
            **({"spec_dropped": self.spec_dropped} if self.speculative else {}),
        }
