"""Lane P70's cast proof compares routings as maps expert -> weight, at the same row count as upstream (amendment 1,
bench/p70/P70-PREREG.md). p70-prove-2 compared slot by slot against an 80-row upstream call and failed on the GEMM's
row-count variance, not on the router; these pin the comparison the corrected proof uses."""
import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("experts4bit_qlora")
pytest.importorskip("transformers")

REPO = pathlib.Path(__file__).resolve().parents[1]
for sub in ("bench", "bench/p59", "bench/p64", "bench/p70"):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
pytest.importorskip("kl_b16")
_spec = importlib.util.spec_from_file_location("kl_router", REPO / "bench" / "p70" / "kl_router.py")
KR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(KR)

W = torch.tensor([[0.40, 0.30, 0.20, 0.10], [0.25, 0.25, 0.30, 0.20]])
I = torch.tensor([[7, 2, 9, 0], [5, 1, 3, 8]])


def test_a_slot_permutation_is_the_same_routing():
    perm = torch.tensor([[2, 0, 3, 1], [1, 0, 3, 2]])
    a, b = (W, I), (W.gather(-1, perm), I.gather(-1, perm))
    assert not torch.equal(a[1], b[1])                       # slot by slot they differ...
    assert KR._same_experts(a, b)                            # ...as routings they do not
    assert KR._map_bit_equal_frac(a, b) == 1.0


def test_a_different_expert_is_a_different_routing():
    i2 = I.clone()
    i2[1, 3] = 11
    assert not KR._same_experts((W, I), (W, i2))
    assert KR._map_bit_equal_frac((W, I), (W, i2)) == 0.0


def test_the_weight_follows_its_expert_not_its_slot():
    """Swapping two experts' weights while keeping the slots is a different function; the map sees it."""
    w2 = W.clone()
    w2[0, [0, 1]] = w2[0, [1, 0]]
    assert KR._same_experts((W, I), (w2, I))
    assert KR._map_bit_equal_frac((W, I), (w2, I)) == pytest.approx(6 / 8)


def test_dtypes_that_differ_are_not_bit_equal():
    assert KR._map_bit_equal_frac((W, I), (W.to(torch.bfloat16), I)) == 0.0


def test_the_proof_compares_at_matching_row_counts():
    """The source of the corrected proof: every fused call is compared with upstream at its own row count."""
    import inspect
    src = inspect.getsource(KR._prove_cast)
    assert "up = {m: ref(x[:m])" in src and "out[cast] = {m: holder.gate(x[:m])" in src
    assert "ri[:64]" not in src and "rw[:64]" not in src
