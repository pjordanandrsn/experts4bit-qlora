# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""CUDA-graph decode capture: the gate is token-stream equality, not "it ran".

A captured graph that reads a stale pointer or writes the wrong cache slot
replays without raising and returns plausible tokens. The first version of this
harness did exactly that -- an off-by-one advanced `cache_position` BEFORE the
replay, and the warmup/capture forwards left their writes in the cache -- and it
produced a stream that looked fine and diverged on the very first replayed token.
So every assertion here compares against eager generation.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _tiny():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        dtype=torch.bfloat16).cuda().eval()
    return m


def test_replay_matches_eager_token_for_token():
    from experts4bit_qlora.capture import probe_capture
    model = _tiny()
    ids = torch.tensor([[1, 450, 7483, 310, 3444, 338]], device="cuda")
    rep = probe_capture(model, ids, max_new_tokens=12)
    if not rep["captured"]:
        pytest.skip(f"capture unavailable here: {rep['error']}")
    assert rep["matches_eager"], (
        f"replay diverged from eager:\n  eager   ={rep['eager']}\n  replayed={rep['replayed']}")


def test_probe_reports_capture_failure_instead_of_raising():
    """A model that cannot be captured must come back as a report, not an exception —
    the probe exists to build a support matrix, so an unsupported model is data."""
    from experts4bit_qlora.capture import probe_capture
    model = _tiny()
    ids = torch.tensor([[1, 450, 7483]], device="cuda")
    rep = probe_capture(model, ids, max_new_tokens=4)
    assert set(rep) >= {"captured", "matches_eager", "error"}
    assert rep["captured"] or rep["error"], "a failed capture must say why"


def test_cache_is_reset_after_capture():
    """Warmup and capture each write into the cache. If those writes survive, the
    first generated token reads polluted slots — which is precisely the bug this
    guards. Two decoders built from the same prompt must agree."""
    from experts4bit_qlora.capture import capture_decode
    model = _tiny()
    ids = torch.tensor([[1, 450, 7483, 310, 3444, 338]], device="cuda")
    try:
        d1, t1 = capture_decode(model, ids, max_length=ids.shape[-1] + 8)
        d2, t2 = capture_decode(model, ids, max_length=ids.shape[-1] + 8)
    except RuntimeError as e:
        pytest.skip(f"capture unavailable: {e}")
    assert int(t1) == int(t2), "prefill token differs between builds — cache not reset"
    a = [int(t1)] + [int(d1.step(t1)) for _ in range(3)]
    b = [int(t2)] + [int(d2.step(t2)) for _ in range(3)]
    assert a == b, f"two decoders from one prompt disagree: {a} vs {b}"


def test_free_running_tie_is_not_reported_as_a_capture_defect():
    """A tie must be told from a real divergence by measurement, not by judgement.

    Low-confidence logits make greedy argmax close to a coin flip, so free-running
    token streams can drift apart even when replay is exact. Teacher-forcing
    removes the compounding -- same tokens down both paths, compare logits. This
    pins the classification, because an earlier version of the probe called such a
    tie `verdict="real"` off a 1.7-ulp gap while replay was bit-identical.
    """
    from experts4bit_qlora.capture import probe_capture
    model = _tiny()
    ids = torch.tensor([[1, 450, 7483, 310, 3444, 338]], device="cuda")
    rep = probe_capture(model, ids, max_new_tokens=8)
    if not rep["captured"]:
        pytest.skip(f"capture unavailable here: {rep['error']}")
    if rep["matches_eager"]:
        assert "teacher_forced" not in rep          # nothing needed explaining
        return
    tf = rep["teacher_forced"]
    assert tf["steps"] > 0, tf
    assert tf["max_abs_delta"] == 0.0, f"replay is NOT bit-identical to eager: {tf}"
    assert rep["divergence"]["verdict"].startswith("tie"), rep["divergence"]


def test_reset_allows_a_second_generation():
    """`step()` never rewinds, so a reused decoder runs off the end of the cache.

    Without `reset()` a second generation walks `cache_position` past `max_length`
    and trips `index_copy_(): index out of bounds` inside StaticCache -- which is
    how the first benchmark of this harness died, after the probe had already
    passed.
    """
    from experts4bit_qlora.capture import capture_decode
    model = _tiny()
    ids = torch.tensor([[1, 450, 7483, 310, 3444, 338]], device="cuda")
    dec, _ = capture_decode(model, ids, max_length=ids.shape[-1] + 6)
    runs = []
    for _ in range(3):                    # more generations than the cache could hold
        t = dec.reset(ids)
        run = [int(t)]
        for _ in range(3):
            t = dec.step(t)
            run.append(int(t))
        runs.append(run)
    assert runs[0] == runs[1] == runs[2], runs
