"""Speculative decoding loop for the DFlash drafter.

The drafter (:mod:`experts4bit_qlora.arch.glimmer_draft`) proposes a short run of
tokens cheaply; the target verifies them in ONE forward and keeps the longest
prefix it agrees with. Correctness here is exact: greedy speculative decoding
must produce **the identical token stream** the target would have produced on
its own — the drafter only changes the SPEED, never the OUTPUT. A loop that
accepts a token the target would not have chosen is a silent correctness bug,
so the accept rule below is the strict one (target's argmax must equal the
drafter's proposal) and the tests assert stream-identity against plain greedy.

This module is deliberately model-agnostic: it takes two callables, so it is
unit-testable on CPU with toy models and reused unchanged once the real
drafter-load path lands. Only the drafter-load half of the DFlash task needs
the released Glimmer weights and a GPU; the loop is complete here.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence


def speculative_greedy_decode(
    prompt: Sequence[int],
    draft_k: Callable[[Sequence[int], int], list[int]],
    target_argmax: Callable[[Sequence[int]], list[int]],
    *,
    max_new_tokens: int,
    k: int = 4,
    eos_id: int | None = None,
) -> dict:
    """Greedy speculative decode. Returns the generated ids plus accounting.

    ``draft_k(context, k) -> [k proposed ids]`` runs the drafter autoregressively
    for ``k`` steps from ``context``.

    ``target_argmax(context) -> [argmax id per position of context]`` runs the
    target ONCE over ``context`` and returns, for each position ``i``, the token
    the target would greedily emit *after* ``context[:i+1]``. So
    ``target_argmax(context)[-1]`` is the target's next token given the whole
    context, and the earlier entries verify the drafter's proposals in parallel.

    The accept rule is exact-match greedy: walk the ``k`` proposals against the
    target's argmax at the matching positions; accept while they agree, and on
    the first disagreement take the TARGET's token there and discard the rest.
    Because the target's argmax is always appended even when every proposal is
    accepted, at least one token is committed per round — the loop cannot stall.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not prompt:
        # The drafter conditions on the last context token and the verifier
        # indexes position len(ctx)-1, so an empty prompt has nothing to
        # condition on. Say so, rather than surfacing an IndexError from inside
        # the loop.
        raise ValueError("prompt must contain at least one token")
    out: list[int] = []
    ctx = list(prompt)
    proposed = accepted = rounds = 0

    while len(out) < max_new_tokens:
        rounds += 1
        draft = draft_k(ctx, k)
        if len(draft) != k:
            raise ValueError(
                f"draft_k returned {len(draft)} ids, expected k={k}")
        proposed += k
        # One target pass over the context PLUS the k proposals. Position i of
        # this extended context is verified by argmax[i]; argmax over the last
        # k+1 positions covers "after each accepted proposal" and "the bonus
        # token after all k".
        verify_ctx = ctx + draft
        argmax = target_argmax(verify_ctx)
        base = len(ctx)                       # first verified position

        committed: list[int] = []
        for j in range(k):
            want = argmax[base - 1 + j]       # target's token after ctx+draft[:j]
            if want == draft[j]:
                committed.append(want)        # proposal matched: accept it
                accepted += 1
                if eos_id is not None and want == eos_id:
                    break
            else:
                committed.append(want)        # mismatch: take the target's token
                break
        else:
            # every proposal accepted -> also take the free bonus token the
            # target produces after the last accepted one.
            bonus = argmax[base - 1 + k]
            committed.append(bonus)

        for t in committed:
            if len(out) >= max_new_tokens:
                break
            out.append(t)
            ctx.append(t)
            if eos_id is not None and t == eos_id:
                return _result(out, proposed, accepted, rounds, eos=True)

    return _result(out, proposed, accepted, rounds, eos=False)


def _result(out, proposed, accepted, rounds, *, eos):
    return {
        "token_ids": out,
        "proposed": proposed,
        "accepted": accepted,
        # accepted / proposed: the fraction of drafted tokens the target kept.
        # Higher means the drafter tracks the target better and decoding is
        # faster; it never changes the OUTPUT, only the speed.
        "acceptance_rate": (accepted / proposed) if proposed else 0.0,
        "rounds": rounds,
        "hit_eos": eos,
    }
