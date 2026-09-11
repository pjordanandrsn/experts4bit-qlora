"""The P42 census reducer reads ms/STEP, and never turns a missing census into a zero.

The profiler table is the only machine-readable record of where a decode step goes, and
its own percentages are percentages of the CENSUS, not of the step. Getting the
denominator wrong is the whole way to misread it, so the arithmetic is pinned here, on
a table in the exact shape torch writes.
"""
import importlib.util
import json
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "bench" / "p42" / "p42_reduce.py"
_spec = importlib.util.spec_from_file_location("p42_reduce", _SRC)
p42 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p42)

TABLE = """profiled replay steps: 8 (batched B=16 graph replay)
-------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
_gemv_int4_b32         0.00%       0.000us         0.00%       0.000us       0.000us      16.000ms        50.00%      16.000ms       5.775us          2304
 a_reduce_kern         0.00%       0.000us         0.00%       0.000us       0.000us       8.000ms        25.00%       8.000ms       2.110us          2688
   an_eltwise         0.00%       0.000us         0.00%       0.000us       0.000us     800.000us         2.50%     800.000us       0.801us           768
-------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CUDA time total: 24.800ms
"""


def test_parses_replays_and_rows():
    replays, rows = p42.parse_census(TABLE)
    assert replays == 8
    assert [r["name"] for r in rows] == ["_gemv_int4_b32", "a_reduce_kern", "an_eltwise"]
    assert [r["self_ms"] for r in rows] == [16.0, 8.0, 0.8]
    assert [r["calls"] for r in rows] == [2304, 2688, 768]


def test_per_step_divides_by_the_replay_count(tmp_path):
    """16 ms of self-CUDA over 8 replays is 2 ms of ONE step -- not 16, and not 50 %."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "census_int4_b16.txt").write_text(TABLE)
    (tmp_path / "e4b_b16_int4_b16.json").write_text(json.dumps({"step_ms_clean": 12.39}))
    a = p42.reduce_run(tmp_path)["arms"]["int4_b16"]
    assert a["step_ms_clean"] == 12.39
    top = a["top"][0]
    assert top["name"] == "_gemv_int4_b32"
    assert abs(top["ms_per_step"] - 2.0) < 1e-9
    assert abs(top["calls_per_step"] - 288) < 1e-9
    assert abs(a["accounted_ms_per_step"] - 3.1) < 1e-9
    assert abs(a["accounted_fraction"] - 3.1 / 12.39) < 1e-9


def test_missing_census_is_missing_not_zero(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "e4b_b16_int4_b16.json").write_text(json.dumps({"step_ms_clean": 12.39}))
    a = p42.reduce_run(tmp_path)["arms"]["int4_b16"]
    assert a["census"] == "MISSING"
    assert a["step_ms_clean"] == 12.39
    assert "accounted_ms_per_step" not in a


def test_over_100_percent_is_called_out_as_overlap(tmp_path):
    """Self-CUDA sums across streams; a total above the step is overlap, not an error."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "census_int4_b16.txt").write_text(TABLE)
    (tmp_path / "e4b_b16_int4_b16.json").write_text(json.dumps({"step_ms_clean": 1.0}))
    md = p42.to_md(p42.reduce_run(tmp_path))
    assert "Over 100 %" in md and "overlap" in md
