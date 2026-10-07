"""Lane SV7 (bench/sv7/SV7-PREREG.md): the reducer's self-test, the runner's exit codes and the registered numbers the
runner, the reducer and the PREREG must agree on."""
from __future__ import annotations

import importlib.util
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
SV7 = REPO / "bench" / "sv7"


def _reducer():
    spec = importlib.util.spec_from_file_location("sv7_reduce", SV7 / "sv7_reduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reducer_self_test_passes():
    assert _reducer().self_test() == 0


def test_runner_uses_host_codes_only_for_host_floors():
    """13 is the disk floor and 18 the RAM and driver floors (as SV6's amendment 1); no other refusal or failure takes
    13, 14, 17 or 18."""
    run = (SV7 / "sv7_run.sh").read_text()
    codes = re.findall(r"finish (\d+)", run)
    lines = {c: [ln for ln in run.splitlines() if f"finish {c}" in ln] for c in set(codes)}
    assert set(codes) <= {"0", "9", "10", "11", "12", "13", "16", "18"}, codes
    assert len(lines["13"]) == 1 and "disk" in lines["13"][0]
    assert len(lines["18"]) == 2 and ["ram" in lines["18"][0].lower(), "driver" in lines["18"][1]] == [True, True]
    assert "BAKE FAIL" in lines["12"][0]


def test_runner_reducer_and_prereg_agree_on_the_registered_numbers():
    run, prereg, red = (SV7 / "sv7_run.sh").read_text(), (SV7 / "SV7-PREREG.md").read_text(), _reducer()
    assert f"dev == {red.EST_BYTES}" in run
    assert f"{red.EST_BYTES:,}" in prereg and f"{red.PLAN_BYTES:,}" in prereg and f"{red.RESERVE_BYTES:,}" in prereg
    s = red.SETUP
    assert (f"--vram-gb {s['vram_gb']} --dram-gb {s['dram_gb']} --hot-rows {s['hot_rows']} --max-seqs {s['max_seqs']} "
            f"--context {s['max_tokens_per_seq']} --graphs 0 --prefill-graph {s['prefill_graph']}") in run
    for tag, prompts in red.ARMS.items():
        assert f"arm {tag} {prompts}" in run
