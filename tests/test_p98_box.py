# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P98's measurement helpers (``bench/p98/p98_box.py``) on CPU, before any GPU is rented: the staggered workload
walks the active set through every bucket, the decode timer records one (rows, ms) per call and passes the runner's
result through, and the throughput summary drops the warm-up calls and counts rows, not calls."""
import importlib.util
from pathlib import Path

BOX = Path(__file__).resolve().parents[1] / "bench" / "p98" / "p98_box.py"


def _box():
    spec = importlib.util.spec_from_file_location("p98_box", BOX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_staggered_workload_steps_every_bucket_as_sequences_finish():
    box = _box()
    lens = box.new_tokens(16, 48, 4)
    assert lens == [48 + 4 * i for i in range(16)] and len(set(lens)) == 16
    # with every sequence decoding from the same step, one finishes per 4 steps: 16 active, then 15, ..., then 1
    active = [sum(1 for m in lens if m > t) for t in range(max(lens))]
    buckets = (1, 2, 4, 8, 16)
    hit = {min(b for b in buckets if n <= b) for n in active if n}
    assert hit == set(buckets)


def test_the_decode_timer_records_rows_and_passes_the_result_through():
    box = _box()

    class Runner:
        def run_decode(self, rids):
            return {r: r * 10 for r in rids}

    runner = Runner()
    timer = box.DecodeTimer(runner, sync=False)
    assert runner.run_decode is timer
    assert runner.run_decode([1, 2, 3]) == {1: 10, 2: 20, 3: 30}
    runner.run_decode([4])
    assert [r for r, _ in timer.calls] == [3, 1] and all(ms >= 0 for _, ms in timer.calls)


def test_the_summary_drops_the_warm_up_and_counts_rows():
    box = _box()
    calls = [(16, 1000.0), (16, 1000.0), (16, 100.0), (16, 100.0), (8, 50.0)]
    s = box.decode_summary(calls, warm=2)
    assert s["steps"] == 5 and s["timed_steps"] == 3 and s["rows"] == 40 and s["ms"] == 250.0
    assert s["tok_per_s"] == 160.0                                  # 40 rows over 0.25 s
    assert s["ms_per_step_by_rows"] == {"8": 50.0, "16": 100.0}
    assert box.decode_summary([], warm=3)["tok_per_s"] is None
