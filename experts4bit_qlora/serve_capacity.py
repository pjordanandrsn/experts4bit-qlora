# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The paged server's capacity, before serving: :class:`StepCosts`, :func:`simulate` and :func:`ceiling`.

:mod:`experts4bit_qlora.serve_recipe` prices what :func:`experts4bit_qlora.serve_paged.build_engine` allocates. This
predicts what the server delivers: TTFT, TPOT, SLO attainment and the capacity ceiling for a request workload. It is
a model of :class:`~experts4bit_qlora.engines.scheduler.ContinuousScheduler` as ``serve_paged`` drives it, with the
step costs as inputs:

* admission is FIFO into ``max_seqs`` slots, and a request past them waits;
* a step runs at most one prefill chunk (the default per-step budget is one chunk), BEFORE that step's decode;
* a prompt of ``L`` tokens takes ``ceil(L / chunk_tokens)`` such steps, and its first token comes at the end of the last;
* every decoding slot emits one token per step, and the step's decode is padded to a bucket. An active set larger than
  the largest bucket runs as consecutive replays of at most that bucket;
* a slot's first decode adds ``first_decode_s``: the block claims of ``_ensure_graph_ready``, ~0 under
  ``E4B_PAGED_BULK_KV=1``, where the flush already made them.

**The costs are measured, not derived.** :meth:`StepCosts.from_step_trace` reads them from a server's own
``E4B_PAGED_STEP_TRACE`` rows:
* the prefill step: host time from the step's start through the first token;
* the first-decode claims per prompt;
* a least-squares fit of decode-only step time on the bucket.

They hold for the box, model, stack and prompt length they were measured on. ``StepCosts.source`` says which.

**Checked against receipts.** With the step costs of SC2b's two ON servers (lane SC2b of #846; prefill step from the
request trace's admission-to-first-token, decode and first-decode terms from the bucket-controlled fit), the model
reproduces their measured attainment at 1 / 2 / 4 / 8 req/s within 0.10 on every rate and draw
(``tests/test_serve_capacity.py``). The worst gap is +0.093, at 4 req/s, where the model is OPTIMISTIC: the knee is
where effects it does not represent bite (GIL contention with the HTTP side, allocator stalls, thermal). Read its
ceiling as an upper bound near the knee, not a promise.

Request plans are ``bench/sc2/sc2_driver.plan`` exactly (same RNG calls), so a seed reproduces an SC2 workload.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

DEFAULT_BUCKETS = (1, 2, 4, 8, 16)
SLO_TTFT_S, SLO_TPOT_S = 1.0, 0.100          # SC2's SLO (sc2_driver.SLO_TTFT_S / SLO_TPOT_S)


def bucket_for(n: int, buckets=DEFAULT_BUCKETS) -> int:
    for b in buckets:
        if n <= b:
            return b
    return buckets[-1]


@dataclass(frozen=True)
class StepCosts:
    """What one engine step costs, in seconds.

    * ``prefill_step_s``: a step that completes a prompt's first chunk, from the step's start to the first token. The
      admission reset, the forward or its graph replay, the K/V flush and the first-token sync are all in it.
    * ``later_chunk_s``: a prefill chunk after the first (eager; None = ``prefill_step_s``).
    * ``first_decode_s``: what a slot's first decode adds to its step.
    * ``decode_base_s`` + ``decode_per_row_s`` x bucket: a decode step's replay.
    """
    prefill_step_s: float
    first_decode_s: float
    decode_base_s: float
    decode_per_row_s: float
    later_chunk_s: float | None = None
    buckets: tuple = DEFAULT_BUCKETS
    source: str = "assumed"

    def __post_init__(self):
        for name in ("prefill_step_s", "first_decode_s", "decode_base_s", "decode_per_row_s"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"StepCosts.{name} must be >= 0, got {getattr(self, name)!r}")
        if not self.buckets or min(self.buckets) < 1 or tuple(sorted(self.buckets)) != tuple(self.buckets):
            raise ValueError(f"buckets must be ascending positive ints, got {self.buckets}")

    def decode_step_s(self, n_rows: int) -> float:
        """A decode step over ``n_rows`` active slots: one padded replay per largest-bucket chunk."""
        t, k = 0.0, n_rows
        while k > 0:
            c = min(k, self.buckets[-1])
            t += self.decode_base_s + self.decode_per_row_s * bucket_for(c, self.buckets)
            k -= c
        return t

    @classmethod
    def from_step_trace(cls, rows, *, buckets=DEFAULT_BUCKETS, source: str = "step trace") -> "StepCosts":
        """Fit the costs from ``E4B_PAGED_STEP_TRACE`` rows (``engines/step_trace.py``).

        * ``prefill_step_s``: the median, over steps that replayed the first-chunk graph, of the host segments up to
          the first token (``ops``, ``plan``, ``pf_prep``, ``pf_forward``, ``pf_flush``, ``pf_sync``, ``pf_emit``).
        * ``first_decode_s``: total ``dec_ready`` ÷ prompts completed.
        * ``decode_base_s`` / ``decode_per_row_s``: least squares of decode-only steps' ``step_ms`` on their bucket.
          With a single bucket the slope is 0 and the base is its median.

        Refuses (ValueError) a trace with no prefill step or no decode-only step: no number is invented."""
        rows = list(rows)
        pf = [r for r in rows if r.get("prefill_replays")]
        dec = [r for r in rows if r.get("decode_rows") and not r.get("prefill_tokens") and r.get("bucket")]
        if not pf:
            raise ValueError("the step trace has no prefill step that replayed the first-chunk graph")
        if not dec:
            raise ValueError("the step trace has no decode-only step with a bucket")
        pre = ("ops", "plan", "pf_prep", "pf_forward", "pf_flush", "pf_sync", "pf_emit")
        p_ms = statistics.median(sum(r["seg"].get(k, 0.0) for k in pre) for r in pf)
        prompts = sum(r.get("prefill_replays", 0) for r in rows)
        ready_ms = sum(r["seg"].get("dec_ready", 0.0) for r in rows) / prompts
        xs = [float(r["bucket"]) for r in dec]
        ys = [float(r["step_ms"]) for r in dec]
        if len(set(xs)) < 2:
            base, slope = statistics.median(ys), 0.0
        else:
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            base = my - slope * mx
        return cls(prefill_step_s=p_ms / 1e3, first_decode_s=ready_ms / 1e3, decode_base_s=max(0.0, base) / 1e3,
                   decode_per_row_s=max(0.0, slope) / 1e3, buckets=tuple(buckets),
                   source=f"{source}: {len(pf)} prefill steps, {len(dec)} decode-only steps, {prompts} prompts")


@dataclass(frozen=True)
class Workload:
    """A request plan as ``sc2_driver.plan`` draws it. ``rate`` None is SC2's serial mode: each request starts when the
    previous one finishes."""
    n: int
    rate: float | None = None
    seed: int = 0
    max_tokens: tuple = (64, 256)
    prompt_tokens: int = 512
    n_prompts: int = 64

    def plan(self) -> list:
        """``[(arrival_offset_s, prompt_index, max_tokens)]``, identical to ``sc2_driver.plan``."""
        mode = "serial" if self.rate is None else "poisson"
        rng = random.Random(self.seed)
        out, t = [], 0.0
        lo, hi = self.max_tokens
        for _ in range(self.n):
            if mode == "poisson":
                t += rng.expovariate(self.rate)
            out.append((round(t, 6) if mode == "poisson" else 0.0, rng.randrange(self.n_prompts), rng.randint(lo, hi)))
        return out


@dataclass
class Result:
    ttft_s: list = field(default_factory=list)
    tpot_s: list = field(default_factory=list)
    attainment: float = 0.0

    def summary(self) -> dict:
        def pct(xs, p):
            s = sorted(xs)
            return s[min(len(s) - 1, int(round((len(s) - 1) * p)))] if s else None
        return {"n": len(self.ttft_s), "attainment": round(self.attainment, 4),
                "ttft_p50_s": pct(self.ttft_s, 0.5), "ttft_p99_s": pct(self.ttft_s, 0.99),
                "tpot_p50_s": pct([x for x in self.tpot_s if x is not None], 0.5)}


def simulate(costs: StepCosts, workload: Workload, *, max_seqs: int = 16, chunk_tokens: int = 512,
             slo=(SLO_TTFT_S, SLO_TPOT_S), http_s: float = 0.003) -> Result:
    """Run ``workload`` through the scheduler model (module docstring). ``http_s`` is added to every TTFT for the
    client's side of the request. Returns per-request TTFT and TPOT and the share meeting ``slo``
    (TTFT <= slo[0] and TPOT <= slo[1], a 1-token request on TTFT alone)."""
    if max_seqs < 1 or chunk_tokens < 1:
        raise ValueError("max_seqs and chunk_tokens must be >= 1")
    later = costs.prefill_step_s if costs.later_chunk_s is None else costs.later_chunk_s
    n_chunks = max(1, math.ceil(workload.prompt_tokens / chunk_tokens))
    reqs = [{"arr": a, "max": m, "out": 0, "chunks": 0} for a, _, m in workload.plan()]
    serial = workload.rate is None
    n, i, t, done = len(reqs), 0, 0.0, 0
    queue, active = [], []
    while done < n:
        if serial:
            if not active and not queue and i < n:
                reqs[i]["arr"] = t
                queue.append(reqs[i])
                i += 1
        else:
            while i < n and reqs[i]["arr"] <= t:
                queue.append(reqs[i])
                i += 1
        while queue and len(active) < max_seqs:
            r = queue.pop(0)
            r["ph"] = "pf"
            active.append(r)
        pfs = [r for r in active if r["ph"] == "pf"]
        decs = [r for r in active if r["ph"] == "dec"]
        if not pfs and not decs:
            t = reqs[i]["arr"]
            continue
        step = 0.0
        if pfs:
            r = pfs[0]
            r["chunks"] += 1
            if r["chunks"] < n_chunks:
                step += costs.prefill_step_s if r["chunks"] == 1 else later
            else:
                step += costs.prefill_step_s if n_chunks == 1 else later
                r["first"] = t + step
                r["out"] = 1
                r["ph"] = "dec" if r["out"] < r["max"] else "end"
                if r["ph"] == "end":
                    r["fin"] = t + step
        if decs:
            step += costs.first_decode_s * sum(1 for r in decs if r["out"] == 1)
            step += costs.decode_step_s(len(decs))
            for r in decs:
                r["out"] += 1
                if r["out"] >= r["max"]:
                    r["ph"] = "end"
                    r["fin"] = t + step
        t += step
        for r in [r for r in active if r["ph"] == "end"]:
            active.remove(r)
            done += 1
    res = Result()
    good = 0
    for r in reqs:
        ttft = r["first"] - r["arr"] + http_s
        tpot = (r["fin"] - r["first"]) / (r["max"] - 1) if r["max"] > 1 else None
        res.ttft_s.append(ttft)
        res.tpot_s.append(tpot)
        good += ttft <= slo[0] and (tpot is None or tpot <= slo[1])
    res.attainment = good / n if n else 0.0
    return res


def ceiling(costs: StepCosts, rates=(1, 2, 4, 8), *, seeds=lambda draw, rate: draw * 100 + int(rate), draws=(1, 2),
            n: int = 120, min_attainment: float = 0.95, **kw) -> dict:
    """SC2's capacity ceiling under the model: the largest rate whose attainment is >= ``min_attainment`` in every
    draw (seeds as SC2b drew them: draw x 100 + rate), together with the attainment table. ``kw`` goes to
    :func:`simulate`, e.g. ``max_seqs``."""
    table, best = {}, 0
    for r in rates:
        att = [simulate(costs, Workload(n=n, rate=r, seed=seeds(d, r)), **kw).attainment for d in draws]
        table[r] = [round(a, 4) for a in att]
        if min(att) >= min_attainment:
            best = r
    return {"ceiling": best, "attainment": table, "source": costs.source}
