"""hot_sets_from_profile / coverage_from_profile — the frequency-ranked residency dial.

Pinning experts BY INDEX is close to worthless: measured on DeepSeek-V4-Flash it bought
7% for 2x the VRAM, because a routed expert lands in an index-ordered hot set at roughly
the uniform rate. These helpers turn a routing profile into the hot set that is actually
worth paying VRAM for, so what they must guarantee is *ranking by routed volume* and
*never silently falling back to index order*.
"""
import json

import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.expert_profile import (  # noqa: E402
    coverage_from_profile,
    hot_sets_from_profile,
)


def _write(tmp_path, layers, counts):
    """layers: {layer_id: num_experts}; counts: {layer_id: {expert_id: tokens}}"""
    p = tmp_path / "profile.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"row": "meta"}) + "\n")
        for lid, n in layers.items():
            f.write(json.dumps({"row": "layer", "layer_id": lid, "num_experts": n}) + "\n")
        for lid, c in counts.items():
            for e, n in c.items():
                if n == 0:
                    continue  # cold experts are omitted by the writer, by design
                f.write(json.dumps({"row": "expert", "layer_id": lid, "expert_id": e,
                                    "hits": 1, "tokens_routed": n}) + "\n")
    return str(p)


def test_ranks_by_routed_volume_not_index(tmp_path):
    path = _write(tmp_path, {0: 8}, {0: {0: 1, 1: 1, 5: 100, 7: 50, 3: 10}})
    assert hot_sets_from_profile(path, 3) == [[5, 7, 3]]


def test_ties_break_by_expert_id_so_the_result_is_deterministic(tmp_path):
    path = _write(tmp_path, {0: 8}, {0: {4: 10, 1: 10, 6: 10}})
    assert hot_sets_from_profile(path, 3) == [[1, 4, 6]]


def test_never_pads_with_unrouted_experts(tmp_path):
    """Asking for more than were routed must NOT top up with index order -- that is
    exactly the by-index behaviour this function replaces."""
    path = _write(tmp_path, {0: 64}, {0: {9: 5, 2: 3}})
    assert hot_sets_from_profile(path, 8) == [[9, 2]]


def test_unprofiled_layer_yields_empty_not_a_guess(tmp_path):
    path = _write(tmp_path, {0: 8, 1: 8}, {0: {3: 7}})
    assert hot_sets_from_profile(path, 4) == [[3], []]


def test_hits_key_is_selectable(tmp_path):
    p = tmp_path / "p.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"row": "layer", "layer_id": 0, "num_experts": 4}) + "\n")
        # expert 1 gets many tokens in few forwards; expert 2 few tokens in many forwards
        f.write(json.dumps({"row": "expert", "layer_id": 0, "expert_id": 1,
                            "hits": 1, "tokens_routed": 100}) + "\n")
        f.write(json.dumps({"row": "expert", "layer_id": 0, "expert_id": 2,
                            "hits": 50, "tokens_routed": 10}) + "\n")
    assert hot_sets_from_profile(str(p), 1) == [[1]]
    assert hot_sets_from_profile(str(p), 1, key="hits") == [[2]]


def test_empty_profile_raises_rather_than_returning_nothing(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text(json.dumps({"row": "meta"}) + "\n")
    with pytest.raises(ValueError, match="no layer rows"):
        hot_sets_from_profile(str(p), 4)


def test_coverage_scores_informed_above_by_index(tmp_path):
    """The whole claim, in one assertion."""
    counts = {0: {e: 1 for e in range(32)}}
    counts[0].update({20: 200, 21: 150, 22: 120, 23: 90})   # concentrated tail
    path = _write(tmp_path, {0: 32}, counts)

    informed = hot_sets_from_profile(path, 4)
    by_index = [[0, 1, 2, 3]]
    ci = coverage_from_profile(path, informed)
    cx = coverage_from_profile(path, by_index)

    assert informed == [[20, 21, 22, 23]]
    assert ci > 0.85, ci                    # the four hot experts carry nearly everything
    assert cx < 0.05, cx                    # index order catches almost none of it
    assert ci / cx > 10


def test_coverage_of_everything_is_one(tmp_path):
    path = _write(tmp_path, {0: 4}, {0: {0: 5, 1: 5, 2: 5, 3: 5}})
    assert coverage_from_profile(path, [[0, 1, 2, 3]]) == pytest.approx(1.0)
    assert coverage_from_profile(path, [[]]) == 0.0
