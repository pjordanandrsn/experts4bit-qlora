# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_INT8_HEAD_ROWS`` (K2b, bench/dense-small-m/DESIGN-K2.md) through the real kernels: K36's ``gemv_i8_bw`` at one
row and K37's ``dense_i8.linear_i8`` at 2..value rows, over the head's own codes and scales. Under Triton's interpreter
on CPU (tests/test_int8_head_bw.py runs this file in its own process), or compiled on a card with
``TRITON_INTERPRET=0``, where the CUDA-graph test also runs. Skipped when the installed grouped-nf4-gemm has no
``dense_i8`` (K37)."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
nn = torch.nn
# triton decorates its own @jit library at its first import, so the interpreter env must precede that import (as in
# tests/test_int8_attn_interp.py)
if os.environ["TRITON_INTERPRET"] == "1" and type(triton.language.zeros).__name__ != "InterpretedFunction":
    pytest.skip("triton was imported compiled earlier in this process; tests/test_int8_head_bw.py runs this file in a "
                "fresh interpreted process", allow_module_level=True)
DEV = "cuda" if os.environ["TRITON_INTERPRET"] == "0" and torch.cuda.is_available() else "cpu"
if os.environ["TRITON_INTERPRET"] != "1" and DEV == "cpu":
    pytest.skip("compiled kernels need a CUDA card", allow_module_level=True)
pytest.importorskip("dense_i8", reason="grouped-nf4-gemm without dense_i8 (K37)")

from experts4bit_qlora.engines import int8_head_bw as ih  # noqa: E402

VOCAB, HIDDEN = 200, 96            # ragged against K37's 32-wide N tile and its 256-wide K chunk


@pytest.fixture(autouse=True)
def routed(monkeypatch):
    monkeypatch.setattr(ih, "_on_device", lambda x: True)   # interpreted CPU calls take the kernel path too


class _Toy(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, HIDDEN, dtype=torch.bfloat16, device=DEV)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False, dtype=torch.bfloat16, device=DEV)
        with torch.no_grad():
            self.lm_head.weight.copy_((torch.randn(VOCAB, HIDDEN, generator=g) / HIDDEN ** 0.5).to(torch.bfloat16))

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, m):
        self.lm_head = m


def _head(value):
    m = _Toy()
    ih.apply_int8_head_bw(m, "1")                            # K36's gemv and quantize_rows_int8: the served grid
    rec = ih.apply_int8_head_rows(m, value)                  # K37's linear_i8 and its plan
    return m.lm_head, rec


def test_the_record_names_k37_and_the_plan_it_runs_at_this_shape():
    import dense_i8
    _, rec = _head(8)
    assert rec == {"rows_max": 8, "kernel": "dense_i8.linear_i8 (K37)", "plan": list(dense_i8.plan(VOCAB, HIDDEN, 2))}


@pytest.mark.parametrize("rows", [1, 2, 3, 8])
def test_every_routed_row_reads_the_grid_within_one_bf16_step(rows):
    head, _ = _head(8)
    x = torch.randn(rows, HIDDEN, device=DEV).to(torch.bfloat16)
    out = head(x).double()
    ref = x.double() @ (head.q.double() * head.s.double()[:, None]).T
    assert head.routed == {rows: 1} and head.passed == {}
    assert bool(((out - ref).abs() <= 2 ** -7 * ref.abs() + 1e-6).all())   # one bf16 step (the interpreter truncates)


def test_past_the_bound_the_head_keeps_f_linear_on_the_grid():
    head, _ = _head(4)
    x = torch.randn(5, HIDDEN, device=DEV).to(torch.bfloat16)
    assert torch.equal(head(x), torch.nn.functional.linear(x, head.weight))
    assert head.routed == {} and head.passed == {5: 1}


def test_the_k37_rows_track_f_linear_on_the_grid():
    """The served change at 2..8 rows (design note section 4): K37's fp32-scaled sum against cuBLAS (or CPU) on the
    bf16 rounding of the same grid, both within bf16's output rounding of the grid."""
    head, _ = _head(8)
    x = torch.randn(8, HIDDEN, device=DEV).to(torch.bfloat16)
    a, b = head(x).float(), torch.nn.functional.linear(x, head.weight).float()
    assert (a - b).abs().max().item() <= 2 ** -6 * b.abs().max().item()


@pytest.mark.skipif(DEV != "cuda", reason="CUDA graphs need a card (TRITON_INTERPRET=0)")
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_a_captured_bucket_replays_its_eager_call_bit_for_bit(rows):
    head, _ = _head(8)
    x = torch.randn(rows, HIDDEN, device=DEV).to(torch.bfloat16)
    eager = head(x).clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        head(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = head(x)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
