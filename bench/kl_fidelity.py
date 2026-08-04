"""kl_fidelity.py — KL-from-reference as a fidelity instrument.

MOTIVATION (external, not ours): Quesma, 2026-08-03, "Quantization hurts knowledge
nonlinearly" — across 55 GGUF quantizations of Qwen3.6-27B, IKP knowledge-benchmark
accuracy correlates with mean KL divergence from the BF16 model at r = -0.981, and the
most obscure facts (their tiers 5-7) degrade first. That makes KL-from-reference a
validated cheap proxy for downstream knowledge loss. Their scope is dense GGUF k-quants;
ours is fused-MoE NF4/MXFP4. The correlation is THEIRS until we test it on our formats.
Benchmark origin: IKP, Bojie Li (1,400 questions, 7 obscurity tiers).

WHY KL RATHER THAN PERPLEXITY: perplexity is a scalar summary and cannot express
*identity*. KL can. Our streaming paths are bit-identical across tiers, so they should
measure exactly 0.000 — in the same units everyone else reports degradation in.

THIS FILE IS AN INSTRUMENT. It registers no performance claim. Any claim intended for
publication needs its own prereg, stamped pre-data.

--------------------------------------------------------------------------------
METRIC DEFINITION (fixed before any measurement; do not quietly change)

    KL(P_ref || P_test), mean per token, over the FULL vocabulary,
    computed on TEACHER-FORCED logits over a fixed, committed prompt set.

  * Teacher-forced only — never sampled generations. Sampling would inject decode
    randomness into a fidelity measurement.
  * Full vocabulary. No top-k. A KL implementation that silently drops tail mass reads
    LOW and flatters us — the exact direction that needs guarding — so truncation is
    refused, not merely discouraged (`assert_full_vocab`).
  * fp64 from log_softmax. Never exp() then log(); never fp32 accumulation over a
    50k-wide sum.
  * Masked where P_ref == 0: those terms are 0 by the p*log(p/q) convention, but
    0 * (-inf) is NaN in floating point, so they are masked explicitly.
  * Aggregation is TOKEN-WEIGHTED across prompts of unequal length (a prompt-weighted
    mean would let short prompts dominate). Stated in every summary dict.

REPORTED: mean, median, p95, max-per-token (primary) + top-1 agreement rate
(secondary — the quantity readers intuit, but a weaker instrument: two models can agree
on argmax while differing materially in the tail that carries obscure knowledge).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Iterable

import torch

# Bump when the metric definition changes in a way that makes old numbers
# incomparable. Receipts record it, so a stale comparison is detectable.
METRIC_VERSION = "kl-fidelity/1.0.0"


# --------------------------------------------------------------------------- core


def assert_full_vocab(ref_logits: torch.Tensor, test_logits: torch.Tensor,
                      declared_vocab: int | None = None) -> int:
    """Refuse to measure on truncated distributions. Returns the vocab size.

    Truncation biases KL DOWNWARD (dropped tail mass contributes nothing), i.e. it
    would make every path look more faithful than it is. That is the one direction a
    fidelity instrument must never fail in, so this is an assert, not a warning.
    """
    if ref_logits.shape != test_logits.shape:
        raise ValueError(f"shape mismatch: ref {tuple(ref_logits.shape)} vs "
                         f"test {tuple(test_logits.shape)} — logits must be aligned "
                         "token-for-token (same tokenizer, same teacher forcing)")
    v = int(ref_logits.shape[-1])
    if declared_vocab is not None and v != declared_vocab:
        raise ValueError(f"vocab {v} != declared {declared_vocab}: refusing to measure "
                         "on a truncated or resized distribution")
    return v


def kl_per_token(ref_logits: torch.Tensor, test_logits: torch.Tensor) -> torch.Tensor:
    """KL(P_ref || P_test) per position, fp64, full vocab, masked at P_ref==0.

    Shapes: (..., vocab) -> (...). No reduction over positions here; the caller
    aggregates token-weighted.
    """
    assert_full_vocab(ref_logits, test_logits)
    # fp64 BEFORE the softmax: a bf16/fp32 logit difference of ~1e-3 is the signal
    # we are trying to resolve at the 1e-6 level.
    log_p = torch.log_softmax(ref_logits.to(torch.float64), dim=-1)
    log_q = torch.log_softmax(test_logits.to(torch.float64), dim=-1)
    p = log_p.exp()
    term = p * (log_p - log_q)
    # p == 0  ->  contribute exactly 0 (limit of p log p), never 0 * inf = NaN.
    term = torch.where(p > 0, term, torch.zeros((), dtype=term.dtype, device=term.device))
    kl = term.sum(dim=-1)
    # KL is non-negative analytically; tiny negatives are fp noise near identity.
    # Clamp only the noise band and surface anything larger as a real error.
    if bool((kl < -1e-9).any()):
        raise FloatingPointError(f"negative KL {float(kl.min()):.3e} — numerically "
                                 "impossible; the harness or the alignment is wrong")
    return kl.clamp_min(0.0)


def top1_agreement(ref_logits: torch.Tensor, test_logits: torch.Tensor) -> torch.Tensor:
    """Per-position argmax agreement (bool). SECONDARY metric — see module docstring."""
    return ref_logits.argmax(dim=-1) == test_logits.argmax(dim=-1)


@dataclass
class KLAccumulator:
    """Token-weighted accumulator over many prompts of unequal length."""

    metric_version: str = METRIC_VERSION
    vocab_size: int | None = None
    _kls: list[torch.Tensor] = field(default_factory=list)
    _agree: list[torch.Tensor] = field(default_factory=list)
    n_prompts: int = 0

    def add(self, ref_logits: torch.Tensor, test_logits: torch.Tensor) -> None:
        v = assert_full_vocab(ref_logits, test_logits, self.vocab_size)
        self.vocab_size = v
        self._kls.append(kl_per_token(ref_logits, test_logits).flatten().cpu())
        self._agree.append(top1_agreement(ref_logits, test_logits).flatten().cpu())
        self.n_prompts += 1

    def summary(self) -> dict:
        if not self._kls:
            raise RuntimeError("nothing accumulated")
        kl = torch.cat(self._kls)            # token-weighted by construction
        agree = torch.cat(self._agree)
        return {
            "metric_version": self.metric_version,
            "metric": "KL(P_ref||P_test), teacher-forced, full vocab, fp64",
            "aggregation": "token-weighted",
            "vocab_size": self.vocab_size,
            "truncation": "none (full vocabulary)",
            "n_prompts": self.n_prompts,
            "n_tokens_scored": int(kl.numel()),
            "kl_mean": float(kl.mean()),
            "kl_median": float(kl.median()),
            "kl_p95": float(kl.quantile(0.95)) if kl.numel() > 1 else float(kl.mean()),
            "kl_max_per_token": float(kl.max()),
            "top1_agreement": float(agree.to(torch.float64).mean()),
            "exactly_zero": bool((kl == 0).all()),
        }


def teacher_forced_logits(model, input_ids: torch.Tensor, attention_mask=None):
    """Logits for a single un-padded sequence. Batch of 1 by design: padding a batch
    invites a mask bug that silently scores pad positions, and pad positions are
    exactly where two paths trivially agree — inflating apparent fidelity."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[0] != 1:
        raise ValueError("teacher_forced_logits expects batch=1 (no padding)")
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return (out.logits if hasattr(out, "logits") else out)[0]


def decode_teacher_forced_logits(model, input_ids: torch.Tensor):
    """Same teacher forcing, one token per forward, carrying a KV cache.

    Identical conditioning to :func:`teacher_forced_logits` — position ``i`` is scored
    having seen tokens ``0..i`` and nothing later — and the same ``[T, vocab]`` result.
    The difference is the SHAPE OF THE FORWARD, and for some serving paths that is the
    whole measurement: the pipelined residency engine (and the [fast] decode path) only
    engage at ``T == 1`` and hand a multi-token prefill straight back to the reference
    forward. Scored with the prefill function, such a path reports a perfect 0.000 that
    means "the engine never ran", not "the engine is faithful".

    Not a drop-in replacement for the prefill scorer: it is ~T times more forwards, and
    it is NOT numerically identical to it (different kernels, different accumulation).
    So a row must use ONE of the two on BOTH sides of its comparison, and say which.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[0] != 1:
        raise ValueError("decode_teacher_forced_logits expects batch=1 (no padding)")
    steps, past = [], None
    with torch.no_grad():
        for i in range(input_ids.shape[1]):
            out = model(input_ids=input_ids[:, i:i + 1],
                        past_key_values=past, use_cache=True)
            past = out.past_key_values if hasattr(out, "past_key_values") else out[1]
            logits = out.logits if hasattr(out, "logits") else out[0]
            steps.append(logits[0, -1])
    return torch.stack(steps)


# ----------------------------------------------------------------- prompt set (K1)


def prompt_set_digest(prompts: Iterable[str]) -> str:
    """sha256 over the prompt set, order-sensitive. Committed BEFORE measurement:
    a prompt set chosen after seeing results is a fitted metric, and KL is
    prompt-distribution dependent."""
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ------------------------------------------------------------------- controls (K0)
#
# All three must pass before ANY path in K2 is measured. They validate the
# INSTRUMENT, not the stack, and they run on CPU with small deterministic modules so
# they gate cheaply and everywhere.


def _toy_lm(vocab: int = 512, hidden: int = 64, seed: int = 1689):
    """A tiny deterministic 'model': embedding -> MLP -> vocab logits.

    Deliberately not a real LM. K0 asks whether the HARNESS detects perturbations;
    that question is answered by any differentiable map from weights to logits, and a
    toy keeps the gate free and CPU-only. Real models are K2's job.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.nn.ModuleDict({
        "emb": torch.nn.Embedding(vocab, hidden),
        "mlp": torch.nn.Linear(hidden, hidden),
        "head": torch.nn.Linear(hidden, vocab),
    }).to(torch.float32).eval().requires_grad_(False), g


def _toy_logits(m, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        h = m["emb"](ids)
        h = torch.nn.functional.silu(m["mlp"](h))
        return m["head"](h)


def _blockwise_int8(w: torch.Tensor, block: int = 64) -> torch.Tensor:
    """Deliberately coarse quantize->dequantize, for the known-nonzero control."""
    flat = w.reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, block)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    deq = (blocks / scale).round().clamp(-127, 127) * scale
    out = deq.reshape(-1)[: w.numel()]
    return out.reshape(w.shape)


def control_self_kl(n_tokens: int = 64) -> dict:
    """CONTROL 1 — reference against ITSELF must be exactly 0.0.

    Not 'small': exactly 0.0. Identical logits give identical log-softmax, so every
    term is p*(x-x)=0 in fp64. Anything else means the harness is wrong (a stray
    dropout, a nondeterministic kernel, an fp32 downcast) and nothing below it is
    trustworthy."""
    m, g = _toy_lm()
    ids = torch.randint(0, 512, (n_tokens,), generator=g)
    lg = _toy_logits(m, ids)
    acc = KLAccumulator()
    acc.add(lg, lg.clone())
    s = acc.summary()
    s["control"] = "self-KL"
    s["expected"] = "exactly 0.0"
    s["passed"] = (s["kl_mean"] == 0.0 and s["kl_max_per_token"] == 0.0
                   and s["exactly_zero"] and s["top1_agreement"] == 1.0)
    return s


def control_perturbation(n_tokens: int = 64) -> dict:
    """CONTROL 2 — flip ONE byte of ONE weight; KL must rise measurably.

    Guards the opposite failure from control 1: a harness that always reports ~0
    would 'pass' self-KL and silently certify every broken path as identical. We
    flip a low mantissa bit of a single fp32 weight — about the smallest physically
    meaningful corruption — and require the instrument to see it.
    """
    m, g = _toy_lm()
    ids = torch.randint(0, 512, (n_tokens,), generator=g)
    ref = _toy_logits(m, ids)

    import copy
    m2 = copy.deepcopy(m)
    w = m2["mlp"].weight
    flat = w.detach().reshape(-1)
    raw = flat[0].view(torch.int32)                 # flip one bit of one weight's bytes
    flat[0] = (raw ^ torch.tensor(1 << 13, dtype=torch.int32)).view(torch.float32)
    test = _toy_logits(m2, ids)

    acc = KLAccumulator()
    acc.add(ref, test)
    s = acc.summary()
    s["control"] = "single-byte perturbation (one weight, one mantissa bit)"
    s["expected"] = "KL > 0 and detected"
    s["delta_weight"] = float((w.detach().reshape(-1)[0] - m["mlp"].weight.detach().reshape(-1)[0]).abs())
    s["passed"] = s["kl_mean"] > 0.0 and math.isfinite(s["kl_mean"])
    return s


def control_known_nonzero(n_tokens: int = 64) -> dict:
    """CONTROL 3 — a deliberately coarse int8 blockwise quantization must agree with
    the ANALYTIC prediction for its own logit perturbation.

    Controls 1+2 can both pass on a harness that is merely monotone; this one asserts
    MAGNITUDE. The first version of this control used a hand-picked band [1e-6, 1e1]
    and int8 measured 4.94e-7 — a FAIL. Investigation: the number was right and the
    band was guessed. The toy's output distribution is near-uniform, and KL between
    near-uniform distributions under a small logit shift is genuinely ~1e-7; a real LM
    has peaked logits where the same relative weight error costs far more. An arbitrary
    threshold cannot tell "instrument compressing the scale" from "this model is flat".

    So the band is replaced by a self-calibrating check. For a small logit perturbation
    Delta, expanding log-sum-exp to second order gives

        KL(P_ref || P_test)  ~=  1/2 * Var_{P_ref}(Delta)

    which depends only on the measured logit deltas — not on any guess. An instrument
    that silently drops tail mass or accumulates in fp32 breaks this identity, so
    agreement with it is a much stronger statement than landing inside a wide band.
    The absolute value is still reported (and required non-trivial), but it is NOT
    comparable to K2 rows: different model, different logit scale.
    """
    m, g = _toy_lm()
    ids = torch.randint(0, 512, (n_tokens,), generator=g)
    ref = _toy_logits(m, ids)

    import copy
    m2 = copy.deepcopy(m)
    with torch.no_grad():
        for k in ("mlp", "head"):
            m2[k].weight.copy_(_blockwise_int8(m2[k].weight))
    test = _toy_logits(m2, ids)

    acc = KLAccumulator()
    acc.add(ref, test)
    s = acc.summary()

    # analytic second-order prediction from the perturbation the quantizer actually made
    p = torch.log_softmax(ref.to(torch.float64), dim=-1).exp()
    delta = (test.to(torch.float64) - ref.to(torch.float64))
    mean_d = (p * delta).sum(-1, keepdim=True)
    predicted = float((0.5 * (p * (delta - mean_d) ** 2).sum(-1)).mean())
    measured = s["kl_mean"]
    ratio = measured / predicted if predicted > 0 else float("inf")

    s["control"] = "known-nonzero (int8 blockwise, block=64, vs fp32 reference)"
    s["expected"] = "KL agrees with 1/2*Var_p(delta-logit) to within 2x, and is non-trivial"
    s["predicted_2nd_order"] = predicted
    s["measured_over_predicted"] = ratio
    s["mean_abs_delta_logit"] = float(delta.abs().mean())
    # non-trivial floor: must be far above fp64 noise (~1e-16), i.e. a real signal
    s["passed"] = bool(0.5 <= ratio <= 2.0 and measured > 1e-12)
    return s


def run_controls() -> dict:
    """All three K0 controls. Returns a receipt; `all_passed` gates K2."""
    res = {
        "metric_version": METRIC_VERSION,
        "torch": torch.__version__,
        "controls": [control_self_kl(), control_perturbation(), control_known_nonzero()],
    }
    res["all_passed"] = all(c["passed"] for c in res["controls"])
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="KL-from-reference fidelity instrument")
    ap.add_argument("--controls", action="store_true", help="run the three K0 controls")
    ap.add_argument("--out", default=None, help="write the receipt JSON here")
    a = ap.parse_args()
    if not a.controls:
        ap.error("--controls (K0 gate). Path measurement (K2) has its own driver.")
    r = run_controls()
    for c in r["controls"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['control']}")
        print(f"        expected: {c['expected']}")
        print(f"        kl mean={c['kl_mean']:.6e} median={c['kl_median']:.6e} "
              f"p95={c['kl_p95']:.6e} max={c['kl_max_per_token']:.6e}")
        print(f"        top1_agreement={c['top1_agreement']:.4f}  "
              f"tokens={c['n_tokens_scored']}  vocab={c['vocab_size']}")
    print(f"\nALL CONTROLS {'PASSED' if r['all_passed'] else 'FAILED'} "
          f"({r['metric_version']}, torch {r['torch']})")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
        print(f"receipt -> {a.out}")
    raise SystemExit(0 if r["all_passed"] else 1)
