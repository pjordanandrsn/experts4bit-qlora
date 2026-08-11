"""Speculative decode must equal plain greedy, exactly — only faster."""
from experts4bit_qlora.speculative import speculative_greedy_decode


# Toy deterministic target: the next token after t is (t + 1) % V. Greedy
# decoding from any prompt is then a simple counter, which makes the correct
# output obvious and independent of the drafter.
V = 17


def target_argmax(ctx):
    return [(t + 1) % V for t in ctx]


def plain_greedy(prompt, n):
    out, ctx = [], list(prompt)
    for _ in range(n):
        nxt = (ctx[-1] + 1) % V
        out.append(nxt)
        ctx.append(nxt)
    return out


def perfect_drafter(ctx, k):
    """Predicts the target exactly -> every proposal should be accepted."""
    last, run = ctx[-1], []
    for _ in range(k):
        last = (last + 1) % V
        run.append(last)
    return run


def blind_drafter(ctx, k):
    """Always proposes 0 -> almost never matches, but output must stay correct."""
    return [0] * k


def test_output_is_identical_to_plain_greedy_regardless_of_drafter():
    prompt = [3]
    ref = plain_greedy(prompt, 20)
    for drafter, name in ((perfect_drafter, "perfect"), (blind_drafter, "blind")):
        r = speculative_greedy_decode(prompt, drafter, target_argmax,
                                      max_new_tokens=20, k=4)
        assert r["token_ids"] == ref, f"{name} drafter changed the output stream"


def test_perfect_drafter_accepts_everything():
    r = speculative_greedy_decode([1], perfect_drafter, target_argmax,
                                  max_new_tokens=16, k=4)
    assert r["acceptance_rate"] == 1.0
    # every proposal accepted -> each round commits k+1 tokens, so far fewer
    # target passes than tokens produced.
    assert r["rounds"] < len(r["token_ids"])


def test_blind_drafter_is_correct_but_low_acceptance():
    r = speculative_greedy_decode([1], blind_drafter, target_argmax,
                                  max_new_tokens=16, k=4)
    assert r["token_ids"] == plain_greedy([1], 16)
    assert r["acceptance_rate"] < 0.2          # ~never matches (token 0 is rare)
    # a mismatch still commits the target's token, so it never stalls
    assert len(r["token_ids"]) == 16


def test_eos_stops_immediately():
    # target emits (t+1)%V; make EOS the token after 4, i.e. 5.
    r = speculative_greedy_decode([3], perfect_drafter, target_argmax,
                                  max_new_tokens=50, k=4, eos_id=5)
    assert r["hit_eos"] is True
    assert r["token_ids"][-1] == 5
    assert 5 not in r["token_ids"][:-1]        # nothing generated past EOS


def test_never_exceeds_max_new_tokens_even_mid_accept():
    # k=8 but a cap of 3 must truncate inside a committed run.
    r = speculative_greedy_decode([0], perfect_drafter, target_argmax,
                                  max_new_tokens=3, k=8)
    assert len(r["token_ids"]) == 3
    assert r["token_ids"] == plain_greedy([0], 3)


def test_k_must_be_positive():
    import pytest
    with pytest.raises(ValueError, match="k must be"):
        speculative_greedy_decode([0], perfect_drafter, target_argmax,
                                  max_new_tokens=4, k=0)
