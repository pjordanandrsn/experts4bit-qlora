"""Lane SC2's trace tool (``bench/sc2/sc2_trace.py``, #846): its self-test, and the read's quoted fit reproduced from the
committed ``sc2-5090-1`` e4b traces (``bench/h2h-2026-10-02/sc2/README.md`` quotes these numbers)."""
import importlib.util
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "bench" / "sc2" / "sc2_trace.py"
TRACES = REPO / "bench" / "h2h-2026-10-02" / "sc2" / "receipts" / "sc2-5090-1" / "sc2"


def _mod():
    spec = importlib.util.spec_from_file_location("sc2_trace", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_self_test_recovers_a_known_model():
    out = subprocess.run([sys.executable, str(TOOL), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (5/5 cases)" in out.stdout, out.stdout + out.stderr


def test_the_read_quotes_the_fit_of_the_committed_traces():
    t = _mod()
    for mode, a_ms, b_s, r2 in (("int4", 6.146, 0.3561, 0.9845), ("nf4", 19.758, 0.4835, 0.9708)):
        rows = [json.loads(line) for line in open(TRACES / f"trace_e4b_{mode}.jsonl") if line.strip()]
        f = t.analyse(rows)["fit"]
        assert (f["decode_ms_per_token"], f["stall_s_per_prefill"], f["r2"], f["n"]) == (a_ms, b_s, r2, 1008), (mode, f)


def test_the_sc2b_read_quotes_the_fit_of_its_committed_traces():
    """SC2b's read (bench/h2h-2026-10-02/sc2b/README.md) quotes the per-prefill stall with the prefill graph OFF and ON,
    from each server's own trace in its own order (``--plan``)."""
    t = _mod()
    d = REPO / "bench" / "h2h-2026-10-02" / "sc2b" / "receipts" / "sc2b-5090-1" / "sc2"
    base = [("warm", 4), ("serial", 24)]
    rates = [(f"r{r}", 120) for r in (1, 2, 4, 8)]
    want = {"e4b_off_d1": (5.681, 0.3265, 0.9868, 528, base + [("serial_repeat", 24)] + rates),
            "e4b_on_d1": (6.064, 0.2617, 0.9845, 504, base + rates),
            "e4b_on_d2": (6.127, 0.2692, 0.9904, 504, base + rates),
            "e4b_off_d2": (6.101, 0.3131, 0.9879, 504, base + rates)}
    for tag, (a_ms, b_s, r2, n, plan) in want.items():
        rows = [json.loads(line) for line in open(d / f"trace_{tag}.jsonl") if line.strip()]
        f = t.analyse(rows, plan=plan)["fit"]
        assert (f["decode_ms_per_token"], f["stall_s_per_prefill"], f["r2"], f["n"]) == (a_ms, b_s, r2, n), (tag, f)


def test_the_sc2g_read_quotes_the_fit_of_its_committed_trace():
    """SC2g's read (bench/h2h-2026-10-02/sc2g/README.md) quotes Q4's registered fit on gpt-oss-20b: b/a 27.4 >= 10, R² >= 0.9.
    The default plan is box G's (4 warm, then serial and the four rates, twice); the reducer reads it the same way."""
    t = _mod()
    p = REPO / "bench" / "h2h-2026-10-02" / "sc2g" / "receipts" / "sc2g-5090-2" / "sc2" / "trace_e4b_gptoss.jsonl"
    rows = [json.loads(line) for line in open(p) if line.strip()]
    f = t.analyse(rows)["fit"]
    assert (f["decode_ms_per_token"], f["stall_s_per_prefill"], f["r2"], f["n"]) == (8.939, 0.2452, 0.9841, 1008), f
