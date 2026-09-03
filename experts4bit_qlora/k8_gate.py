# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The K8 perplexity gate, as one function, so every lane applies the same
rule.

Two regimes, decided 2026-09-03 (receipts INT4B16/P22, P22b):

* **uncalibrated** formats (round-to-nearest int4, fp8, ...): the candidate
  must stay within ``|delta| <= budget`` of base. A large move in EITHER
  direction means a broken instrument, not a result.
* **calibrated** packs (GPTQ-style, Hessian-compensated): the documented
  direction is a small IMPROVEMENT over base, because the calibration is a
  light fit of the projections to the model's own activations. The gate is
  one-sided, ``delta <= +budget``, but a negative delta is only trusted when
  it holds with the SAME SIGN on at least two scoring texts, one of them
  outside the calibration domain. Two wikitext-train-calibrated packs that
  differed by 0.06 ppl from the choice of calibration windows alone are
  what this clause refuses: an improvement that moves with the windows is
  fitting the scored text.

Every arm carries ``text_sha``, ``steps`` and ``ppl_source``; arms that
scored different text, or different step counts, never compare.
"""
from __future__ import annotations

from dataclasses import dataclass

BUDGET = 0.05


@dataclass(frozen=True)
class Arm:
    ppl: float
    text_sha: str
    steps: int
    ppl_source: str = "wikitext"


def _paired(base: Arm, cand: Arm) -> float:
    if base.text_sha != cand.text_sha:
        raise ValueError(f"arms scored different text ({base.ppl_source}: "
                         f"{base.text_sha[:12]} vs {cand.text_sha[:12]})")
    if base.steps != cand.steps:
        raise ValueError(f"arms scored different step counts "
                         f"({base.steps} vs {cand.steps})")
    return cand.ppl - base.ppl


def verdict(pairs: list[tuple[Arm, Arm]], calibrated: bool,
            budget: float = BUDGET, calibration_domain: str | None = None
            ) -> tuple[bool, list[str]]:
    """``pairs`` = ``[(base, candidate), ...]``, one per scoring text.
    Returns ``(passed, lines)``; the lines are the receipt."""
    if not pairs:
        raise ValueError("no arms to gate")
    deltas = [_paired(b, c) for b, c in pairs]
    lines = [f"K8 {c.ppl_source}: base={b.ppl:.5f} cand={c.ppl:.5f} "
             f"delta={d:+.5f} sha={b.text_sha[:12]} steps={b.steps}"
             for (b, c), d in zip(pairs, deltas)]
    if not calibrated:
        ok = all(abs(d) <= budget for d in deltas)
        lines.append(f"K8 VERDICT {'PASS' if ok else 'FAIL'} "
                     f"(uncalibrated: |delta| <= {budget} on every text)")
        return ok, lines
    within = all(d <= budget for d in deltas)
    improving = [d < 0 for d in deltas]
    if any(improving):
        # an improvement must be corroborated: same sign on >= 2 texts,
        # one outside the calibration domain
        srcs = {c.ppl_source for (_b, c), d in zip(pairs, deltas) if d < 0}
        outside = (calibration_domain is None
                   or any(s != calibration_domain for s in srcs))
        corroborated = all(improving) and len(pairs) >= 2 and outside
        if not corroborated:
            lines.append("K8 VERDICT FAIL (calibrated: an improvement needs "
                         "the same sign on >= 2 scoring texts, one outside "
                         "the calibration domain -- an improvement that "
                         "moves with the calibration text is fitting it)")
            return False, lines
    lines.append(f"K8 VERDICT {'PASS' if within else 'FAIL'} "
                 f"(calibrated: delta <= +{budget}"
                 + ("; improvement corroborated on "
                    f"{len(pairs)} texts" if any(improving) else "") + ")")
    return within, lines
