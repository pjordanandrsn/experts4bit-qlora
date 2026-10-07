"""bench/sv4/sv4_measure.py's driver sampler (SV7 amendment 1): one ``nvidia-smi --query-compute-apps`` sample matched to
this process. Lane SV7's sv7-4090-1 read zero samples because its container listed host-namespace PIDs."""
from __future__ import annotations

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _measure():
    spec = importlib.util.spec_from_file_location("sv4_measure", REPO / "bench" / "sv4" / "sv4_measure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_this_process_s_row_wins():
    m = _measure()
    assert m.driver_sample("4242, 21000\n77, 500\n", "4242") == (21000 << 20, "pid")


def test_a_sole_unmatched_process_is_taken_and_says_so():
    m = _measure()
    assert m.driver_sample("918273, 21869\n", "117") == (21869 << 20, "sole-process")


def test_several_unmatched_processes_are_not_guessed():
    m = _measure()
    assert m.driver_sample("918273, 21869\n918274, 300\n", "117") == (None, "no-match")


def test_no_rows_and_unparsable_rows_read_nothing():
    m = _measure()
    assert m.driver_sample("", "117") == (None, "no-rows")
    assert m.driver_sample("117, [N/A]\n", "117") == (None, "no-rows")
