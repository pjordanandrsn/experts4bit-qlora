"""P44-b run 3: a decode-shaped scorer whose model returns no KV cache scores every position on a single token and
reads as a large, lever-independent KL. The scorer must refuse that, not report it."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from kl_fidelity import decode_teacher_forced_logits  # noqa: E402


class _Out:
    def __init__(self, logits, past):
        self.logits, self.past_key_values = logits, past


class _NoCache(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"use_cache": False})()

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        return _Out(torch.zeros(1, input_ids.shape[1], 16), None)


class _Cache(torch.nn.Module):
    def forward(self, input_ids, past_key_values=None, use_cache=True):
        n = (past_key_values or 0) + input_ids.shape[1]
        return _Out(torch.full((1, input_ids.shape[1], 16), float(n)), n)


def test_no_cache_refuses_with_the_config_in_the_message():
    with pytest.raises(RuntimeError, match="returned no past_key_values after step 0 .*use_cache=False"):
        decode_teacher_forced_logits(_NoCache(), torch.tensor([[1, 2, 3]]))


def test_cache_is_carried_and_every_position_is_scored():
    out = decode_teacher_forced_logits(_Cache(), torch.tensor([[1, 2, 3, 4]]))
    assert out.shape == (4, 16) and out[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
