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
