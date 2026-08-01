"""Token-budget batching: the trainer used to run ONE row per forward.

A fused-MoE step's cost is largely fixed per active expert — the reference path
dequantizes each routed expert once, the fused path launches one grouped GEMM per expert
group — so a forward carrying 150 tokens pays nearly what one carrying 4000 does. The
trainer paid that per example.

These gate the two things a batcher can silently get wrong: contaminating examples with
each other's padding, and losing or duplicating rows. Pure python — no model, no CUDA.
"""
import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.train import _pad_batch, _token_budget_batches  # noqa: E402

PAD = 7


def _rows(lengths):
    """Encoded examples whose ids are their own 1-based index, so provenance is visible."""
    out = []
    for i, n in enumerate(lengths, start=1):
        out.append({"input_ids": [i] * n, "labels": [i] * n})
    return out


def test_padding_is_inert():
    """Pad positions must carry label -100 and attention 0 — that is the whole reason this
    is safe to do without touching the model: no row can see another's tokens."""
    ids, lbl, att = _pad_batch(_rows([3, 5, 2]), PAD, "cpu")
    assert ids.shape == lbl.shape == att.shape == (3, 5)
    for r, n in enumerate((3, 5, 2)):
        assert (ids[r, :n] != PAD).all() or n == 0
        assert (att[r, :n] == 1).all()
        assert (lbl[r, :n] != -100).all()
        assert (ids[r, n:] == PAD).all(), "pad id not written"
        assert (att[r, n:] == 0).all(), "padding is attended to"
        assert (lbl[r, n:] == -100).all(), "padding contributes loss"


def test_real_labels_survive_untouched():
    rows = _rows([4, 2])
    ids, lbl, _ = _pad_batch(rows, PAD, "cpu")
    for r, row in enumerate(rows):
        n = len(row["input_ids"])
        assert ids[r, :n].tolist() == row["input_ids"]
        assert lbl[r, :n].tolist() == row["labels"]


@pytest.mark.parametrize("budget", [16, 64, 256])
def test_every_row_emitted_exactly_once(budget):
    """Losing a row silently shrinks the epoch; duplicating one silently reweights it."""
    lengths = [3, 17, 5, 9, 1, 12, 8, 2, 30, 6, 4, 11]
    seen = []
    for ids, _lbl, att in _token_budget_batches(_rows(lengths), budget, PAD, "cpu", bucket=5):
        for r in range(ids.shape[0]):
            n = int(att[r].sum())
            seen.append((int(ids[r, 0]), n))            # (row id, true length)
    assert sorted(n for _, n in seen) == sorted(lengths)
    assert len(set(i for i, _ in seen)) == len(lengths), "a row was duplicated"


@pytest.mark.parametrize("budget", [16, 64, 256])
def test_padded_cost_stays_within_budget(budget):
    """The budget is `rows * width` — the work the GPU actually does — not the sum of true
    lengths. Budgeting on true lengths lets one long row blow up a batch that looked
    affordable. A single row wider than the whole budget is emitted alone, by necessity."""
    for ids, _lbl, _att in _token_budget_batches(
            _rows([3, 17, 5, 9, 1, 12, 8, 2, 30, 6]), budget, PAD, "cpu", bucket=5):
        rows, width = ids.shape
        assert rows * width <= budget or rows == 1, (rows, width, budget)


def test_length_bucketing_limits_padding_waste():
    """Rows are drawn length-sorted within a bucket, so a batch's rows are similar lengths.
    Without it, pairing a 1-token row with a 30-token row pads 29 dead positions."""
    lengths = [1, 30, 2, 29, 3, 28, 4, 27] * 3
    waste = used = 0
    for ids, _lbl, att in _token_budget_batches(_rows(lengths), 256, PAD, "cpu", bucket=24):
        used += int(att.sum())
        waste += ids.numel() - int(att.sum())
    assert used > 0
    assert waste / (waste + used) < 0.35, f"padding waste {waste / (waste + used):.0%}"


def test_batches_actually_carry_more_than_one_row():
    """The point of the change. With a budget well above the row width, a batch must hold
    many rows — otherwise this is still one-row-per-forward with extra machinery."""
    batches = list(_token_budget_batches(_rows([8] * 40), 256, PAD, "cpu", bucket=16))
    assert max(b[0].shape[0] for b in batches) >= 8, [b[0].shape for b in batches]


def test_empty_input_yields_nothing():
    assert list(_token_budget_batches([], 128, PAD, "cpu")) == []
