"""PGX's fit profile (bench/pgx/pgx_fit_profile.py): hand-counted on synthetic traces, refused when partial or out of
range, and read back by the placement solver's own profile reader. The committed bench/pgx/fit_profile.jsonl is
checked for shape and coverage. CPU only."""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PGX = os.path.join(ROOT, "bench", "pgx")
sys.path.insert(0, PGX)
import pgx_fit_profile as fp  # noqa: E402


def _placement():
    # stdlib-only module; loaded by path so the package __init__ (torch, bitsandbytes) is not needed
    spec = importlib.util.spec_from_file_location(
        "pgx_placement", os.path.join(ROOT, "experts4bit_qlora", "engines", "placement.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trace(tmp_path, name, steps):
    p = tmp_path / name
    lines = [{"meta": {"model": "toy"}}] + [{"step": i, "routed": r} for i, r in enumerate(steps)]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return str(p)


def test_hand_counted(tmp_path):
    a = _trace(tmp_path, "a.jsonl", [{"0": [1, 2], "1": [0, 2]}, {"0": [1, 3], "1": [2, 3]}])
    b = _trace(tmp_path, "b.jsonl", [{"0": [1, 0], "1": [3, 3]}])
    counts, steps = fp.fit([a, b], layers=2, experts=4)
    assert counts == {(0, 1): 3, (0, 2): 1, (0, 3): 1, (0, 0): 1, (1, 0): 1, (1, 2): 2, (1, 3): 3}
    assert steps == {"a.jsonl": 2, "b.jsonl": 1}


def test_output_reads_back_through_the_solvers_reader(tmp_path):
    a = _trace(tmp_path, "a.jsonl", [{"0": [1, 2], "1": [0, 2]}])
    out = tmp_path / "prof.jsonl"
    assert fp.main([a, "--out", str(out), "--layers", "2", "--experts", "4"]) == 0
    mass, sha = _placement().load_routing_mass(str(out), 2, 4)
    assert mass == {(0, 1): 1.0, (0, 2): 1.0, (1, 0): 1.0, (1, 2): 1.0}
    assert sha == fp.sha256(str(out))


def test_refuses_a_partial_profile(tmp_path):
    a = _trace(tmp_path, "a.jsonl", [{"0": [1, 2]}])            # layer 1 never routed
    with pytest.raises(ValueError, match="partial profile"):
        fp.fit([a], layers=2, experts=4)
    assert fp.main([a, "--out", str(tmp_path / "o"), "--layers", "2", "--experts", "4"]) == 2


@pytest.mark.parametrize("routed", [{"0": [4], "1": [0]}, {"2": [0], "0": [1], "1": [1]}, {"0": [-1], "1": [0]}])
def test_refuses_out_of_range(tmp_path, routed):
    a = _trace(tmp_path, "a.jsonl", [routed])
    with pytest.raises(ValueError, match="outside 2 x 4"):
        fp.fit([a], layers=2, experts=4)


def test_refuses_an_unexpected_trace(tmp_path):
    a = _trace(tmp_path, "a.jsonl", [{"0": [1], "1": [0]}])
    assert fp.main([a, "--out", str(tmp_path / "o"), "--layers", "2", "--experts", "4",
                    "--expect-sha", "00ff"]) == 2
    assert fp.main([a, "--out", str(tmp_path / "o"), "--layers", "2", "--experts", "4",
                    "--expect-sha", fp.sha256(a)[:12]]) == 0


def test_the_committed_profile():
    """bench/pgx/fit_profile.jsonl, as registered: the four #1469 Qwen3 decode traces, 512 steps each, 48 x 128, top-8,
    every layer carrying exactly its share of the picks."""
    path = os.path.join(PGX, "fit_profile.jsonl")
    with open(path, encoding="utf-8") as f:
        meta = json.loads(f.readline())
    assert meta["row"] == "meta" and meta["layers"] == 48 and meta["experts"] == 128
    assert sorted(meta["traces"]) == [f"qwen3_{k}.jsonl" for k in ("code", "dialogue", "math", "prose")]
    assert set(meta["steps"].values()) == {512}
    mass, _ = _placement().load_routing_mass(path, 48, 128)
    assert all(0 <= lay < 48 and 0 <= e < 128 for lay, e in mass)
    per_layer = {}
    for (lay, _e), v in mass.items():
        per_layer[lay] = per_layer.get(lay, 0) + v
    assert sorted(per_layer) == list(range(48))
    assert set(per_layer.values()) == {4 * 512 * 8}
    assert meta["picks"] == 48 * 4 * 512 * 8
