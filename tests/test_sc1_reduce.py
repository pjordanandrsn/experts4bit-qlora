"""CI wrapper for ``bench/sc1/sc1_reduce.py`` (lane SC1, experts4bit-qlora#846): the reducer's readings on hand-built
receipt sets, plus the arithmetic the registration names, pinned at its boundaries.

The selftest writes synthetic receipt directories from code (no fixtures) and drives every reading: a clean VALID
position, each VOID reason firing once, UNSTABLE, the third-draw rule, a QUALITY_FAIL licence, NOT_COMPARABLE, the
substitution licence failing, the SAMEPROMPT label, a degraded-control FAIL, a missing oracle (d_bf16 UNREAD, not 0)
and the cross-box comparison, printing each predicate's failing case. A reducer whose VOID cannot fire quotes everything,
so the negative cases are the point. The unit tests below pin the K8-gate pairing (and that the local copy of the rule
is byte-identical to the package's), the band classification at 0.0095 / 0.02 exactly, the interval arithmetic and the
energy integral on a synthetic csv.
"""
import importlib.util
import inspect
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REDUCE = REPO / "bench" / "sc1" / "sc1_reduce.py"
GATE = REPO / "experts4bit_qlora" / "k8_gate.py"


def _mod():
    spec = importlib.util.spec_from_file_location("sc1_reduce_under_test", REDUCE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _gate_file():
    spec = importlib.util.spec_from_file_location("k8_gate_file_under_test", GATE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_sc1_reduce_selftest_runs_and_names_at_least_20_cases():
    p = subprocess.run([sys.executable, str(REDUCE), "--selftest"], capture_output=True, text=True, timeout=600, cwd=REPO)
    assert p.returncode == 0, (p.stdout + p.stderr)[-4000:]
    m = re.search(r"REDUCE SELFTEST OK cases=(\d+)", p.stdout)
    assert m and int(m.group(1)) >= 20, p.stdout[-800:]
    names = re.findall(r"^CASE \d+ \[([^\]]+)\]", p.stdout, flags=re.M)
    assert len(set(names)) >= 20, names
    for needed in ("clean VALID position", "UNSTABLE", "third draw", "QUALITY_FAIL", "NOT_COMPARABLE", "substitution", "SAMEPROMPT",
                   "degraded FAIL", "missing oracle", "cross-box"):
        assert any(needed.lower() in n.lower() for n in names), (needed, names)


def test_vocabulary_and_status_map():
    R = _mod()
    assert set(R.VERDICTS) == {"VALID", "VOID", "OOM", "UNSUPPORTED", "UNSTABLE", "HARNESS_ERROR", "ALARM", "NOT_RUN"}
    assert R.status_of(None) == ("NOT_RUN", "no receipt")
    assert R.status_of({"status": "ok"}) == ("VALID", "")
    assert R.status_of({})[0] == "VALID"
    assert R.status_of({"status": "oom"})[0] == "OOM"
    for st in ("refused", "unsupported", "install_failed", "import_failed", "load_fault"):
        assert R.status_of({"status": st})[0] == "UNSUPPORTED", st
    assert R.status_of({"status": "harness_error"})[0] == "HARNESS_ERROR"
    assert R.status_of({"status": "alarm"})[0] == "ALARM"
    assert R.status_of({"status": "not_run"})[0] == "NOT_RUN"
    assert R.status_of({"status": "void", "void_reason": "x"}) == ("VOID", "x")
    assert R.status_of({"verdict": "VOID", "void_reason": "row stopped early"}) == ("VOID", "row stopped early")


def test_band_classification_at_the_boundaries():
    R = _mod()
    assert R.band(0.0) == "CLOSE"
    assert R.band(0.0095) == "CLOSE" and R.band(-0.0095) == "CLOSE"
    assert R.band(0.0095 + 1e-9) == "COMPARABLE"
    assert R.band(0.02) == "COMPARABLE" and R.band(-0.02) == "COMPARABLE"
    assert R.band(0.02 + 1e-9) == "NOT_COMPARABLE" and R.band(-0.5) == "NOT_COMPARABLE"
    assert R.band(None) == "UNREAD"


def test_interval_arithmetic_point_is_median_over_median_and_bounds_are_cross_draw_extremes():
    R = _mod()
    iv = R.interval([300.0, 306.0], [100.0, 102.0])
    assert iv["n_cross"] == 4
    assert abs(iv["point"] - 303.0 / 101.0) < 1e-12
    assert abs(iv["min"] - 300.0 / 102.0) < 1e-12 and abs(iv["max"] - 306.0 / 100.0) < 1e-12
    iv3 = R.interval([300.0, 306.0, 303.0], [100.0])
    assert iv3["n_cross"] == 3 and abs(iv3["point"] - 3.03) < 1e-12
    assert R.interval([], [1.0]) is None


def test_draw_rules_at_the_boundaries():
    R = _mod()
    assert R.draws_summary([100.0, 103.0])["status"] == "STABLE"                       # exactly 3 %
    s = R.draws_summary([100.0, 103.1])
    assert s["status"] == "THIRD_DRAW_REQUIRED" and not s["usable"]
    assert R.draws_summary([100.0, 103.1, 101.0])["status"] == "STABLE"               # third landed, spread 3.1 % <= 5 %
    assert R.draws_summary([100.0, 105.0])["status"] == "STABLE" or R.draws_summary([100.0, 105.0])["status"] == "THIRD_DRAW_REQUIRED"
    u = R.draws_summary([100.0, 105.1])
    assert u["status"] == "UNSTABLE" and not u["usable"]
    assert R.draws_summary([100.0, 101.0, 106.0])["status"] == "UNSTABLE"
    assert R.draws_summary([100.0])["status"] == "SINGLE" and not R.draws_summary([100.0])["usable"]
    assert R.draws_summary([])["status"] == "NONE"
    assert R.draws_summary([100.0, 102.0, 101.0])["point"] == 101.0                   # median of three


def test_k8_gate_pairing_and_local_copy_is_byte_identical_to_the_package():
    R = _mod()
    G = _gate_file()
    for name in ("Arm", "_paired", "verdict"):
        assert textwrap.dedent(inspect.getsource(getattr(R, name))) == textwrap.dedent(inspect.getsource(getattr(G, name))), name
    assert R.BUDGET == G.BUDGET == 0.05
    base_w = R.Arm(ppl=6.3576, text_sha="a" * 64, steps=2048, ppl_source="wikitext")
    base_c = R.Arm(ppl=12.0, text_sha="b" * 64, steps=2048, ppl_source="c4val1")
    ok, lines = R.verdict([(base_w, R.Arm(6.40, "a" * 64, 2048, "wikitext")), (base_c, R.Arm(12.04, "b" * 64, 2048, "c4val1"))],
                          calibrated=True, budget=0.05, calibration_domain="c4val1")
    assert ok and lines[-1].startswith("K8 VERDICT PASS")
    ok, _ = R.verdict([(base_w, R.Arm(6.3576 + 0.0501, "a" * 64, 2048, "wikitext")), (base_c, R.Arm(12.0, "b" * 64, 2048, "c4val1"))],
                      calibrated=True, calibration_domain="c4val1")
    assert not ok
    # an improvement on the calibration text only is NOT corroborated
    ok, lines = R.verdict([(base_w, R.Arm(6.36, "a" * 64, 2048, "wikitext")), (base_c, R.Arm(11.99, "b" * 64, 2048, "c4val1"))],
                          calibrated=True, calibration_domain="c4val1")
    assert not ok and "improvement needs" in lines[-1]
    with pytest.raises(ValueError):
        R.verdict([(base_w, R.Arm(6.36, "c" * 64, 2048, "wikitext"))], calibrated=True)
    with pytest.raises(ValueError):
        R.verdict([(base_w, R.Arm(6.36, "a" * 64, 1024, "wikitext"))], calibrated=True)
    # the package's own function agrees with the local copy on the same inputs
    pairs = [(G.Arm(6.3576, "a" * 64, 2048, "wikitext"), G.Arm(6.40, "a" * 64, 2048, "wikitext")), (G.Arm(12.0, "b" * 64, 2048, "c4val1"), G.Arm(12.04, "b" * 64, 2048, "c4val1"))]
    assert G.verdict(pairs, calibrated=True, calibration_domain="c4val1")[0] is True
    assert R._load_k8_gate()[1] in ("package", "file")


def test_k8_row_pairs_by_window_sha_and_steps(tmp_path):
    R = _mod()
    import json
    (tmp_path / "k8_window_wikitext.json").write_text(json.dumps({"ids": [1, 2, 3], "text_sha": "w" * 64}))
    (tmp_path / "k8_nf4_wikitext.json").write_text(json.dumps({"k8": "ppl", "mean_nll": math.log(6.0), "ppl": 6.0, "text_sha": "w" * 64, "steps": 2048, "tokens_scored": 2048, "ppl_source": "wikitext"}))
    (tmp_path / "k8_lic_auto_wikitext.json").write_text(json.dumps({"k8": "ppl", "mean_nll": math.log(6.01), "ppl": 6.01, "text_sha": "x" * 64, "steps": 2048, "ppl_source": "wikitext"}))
    (tmp_path / "k8_lic_1_wikitext.json").write_text(json.dumps({"k8": "ppl", "mean_nll": math.log(6.01), "ppl": 6.01, "text_sha": "w" * 64, "steps": 1024, "ppl_source": "wikitext"}))
    arm, why, _ = R.k8_row(str(tmp_path), "nf4_wikitext", "w" * 64)
    assert arm is not None and arm.ppl == 6.0 and arm.steps == 2048
    arm, why, _ = R.k8_row(str(tmp_path), "lic_auto_wikitext", "w" * 64)
    assert arm is None and "text_sha" in why
    arm, why, _ = R.k8_row(str(tmp_path), "lic_1_wikitext", "w" * 64)
    assert arm is None and "steps 1024" in why
    arm, why, _ = R.k8_row(str(tmp_path), "nf4_wikitext", None)
    assert arm is None and "unpinned" in why


def test_energy_integral_on_a_synthetic_csv():
    R = _mod()
    # 11 samples at 50 ms, constant 400 W -> 0.5 s x 400 W = 200 J; with timestamps
    rows = ["timestamp, power.draw.instant [W], memory.used [MiB], utilization.gpu [%]"]
    t0 = 1_700_000_000.0
    for i in range(11):
        rows.append(f"{t0 + 0.05 * i:.3f}, 400.00 W, 20000 MiB, 95 %")
    e = R.energy_integral("\n".join(rows), t0, t0 + 0.5)
    assert e["n_samples"] == 11 and abs(e["joules"] - 200.0) < 1e-6 and abs(e["mean_w"] - 400.0) < 1e-6 and e["basis"] == "csv timestamps"
    # the window clips: only the first 6 samples (0.25 s) -> 100 J
    e = R.energy_integral("\n".join(rows), t0, t0 + 0.25)
    assert e["n_samples"] == 6 and abs(e["joules"] - 100.0) < 1e-6
    # a ramp 0..400 W over 0.5 s (trapezoid = 100 J), no timestamps, sampler_start given
    rows = ["power.draw.instant [W], memory.used [MiB]"] + [f"{40.0 * i:.2f} W, 1 MiB" for i in range(11)]
    e = R.energy_integral("\n".join(rows), t0, t0 + 0.5, interval_ms=50, sampler_start=t0)
    assert abs(e["joules"] - 100.0) < 1e-6 and "sampler_start_epoch" in e["basis"]
    # neither timestamps nor sampler_start: rows assumed to span the window, stated in the basis
    e = R.energy_integral("\n".join(rows), t0, t0 + 1.0, interval_ms=50)
    assert abs(e["joules"] - 200.0) < 1e-6 and "assumed" in e["basis"]
    assert R.energy_integral("power.draw.instant [W]\n100 W\n", t0, t0 + 1)["joules"] is None
    assert R.energy_integral("", t0, t0 + 1)["joules"] is None
    # nvidia-smi's own timestamp format parses
    assert abs(R._parse_ts("2026/10/02 01:13:43.500") - (R.calendar.timegm((2026, 10, 2, 1, 13, 43, 0, 0, 0)) + 0.5)) < 1e-6


def test_census_parse_and_kernel_lookup(tmp_path):
    R = _mod()
    table = ("profiled replay steps: 8 (batched B=16 graph replay)\n"
             "----  ----\n"
             "   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls\n"
             "----  ----\n"
             "   _gemm_int4_b32_grouped_smallm   0.00%   0.000us   0.00%   0.000us   0.000us   40.398ms   51.63%   40.398ms   52.601us   768\n"
             "   _tile_table_r1   0.00%   0.000us   0.00%   0.000us   0.000us   1.0ms   1.0%   1.0ms   1.0us   384\n"
             "Self CUDA time total: 78.2ms\n")
    names = R.parse_census_names(table)
    assert names == {"_gemm_int4_b32_grouped_smallm": 768, "_tile_table_r1": 384}
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "census_k8_lic_1_wikitext.txt").write_text(table)
    assert R.census_kernels({}, str(tmp_path), "k8_lic_1_wikitext") == set(names)
    assert R.census_kernels({}, str(tmp_path), "nothing") is None                                   # missing is None, never empty
    assert R.census_kernels({"census": {"k19_engaged": True, "k23_engaged": False}}, str(tmp_path), "x") == {R.K19_KERNEL}
    assert R.census_kernels({"census": {"kernels": ["a", "b"]}}, str(tmp_path), "x") == {"a", "b"}


# --- A5 (sc1a-5090-1): energy from the sampler's headerless CSV; SAMEPROMPT checks its effective rows -------------------

def test_energy_reads_a_headerless_sampler_csv_with_its_field_list():
    """sc1a-5090-1: every energy row read "no power.draw column" -- the sampler writes `--format=csv,noheader,nounits` and
    records the field list in the receipt (`sampler_fields`); the reducer looked for a header row."""
    import calendar
    import time
    m = _mod()
    fields = "timestamp,memory.used,utilization.gpu,clocks.sm,power.draw.instant,pcie.link.gen.current"
    t0 = calendar.timegm(time.strptime("2026/10/02 11:22:00", "%Y/%m/%d %H:%M:%S"))
    lines = [time.strftime("%Y/%m/%d %H:%M:%S", time.gmtime(t0 + i // 2)) + (".500" if i % 2 else ".000")
             + ", 28000, 99, 2542, 300.00, 4" for i in range(21)]                          # 10 s at 300 W, 0.5 s apart
    csv_text = "\n".join(lines) + "\n"
    e = m.energy_integral(csv_text, t0, t0 + 10.0, 50, None, fields=fields)
    assert e["basis"] == "csv timestamps" and e["n_samples"] == 21, e
    assert e["joules"] == pytest.approx(3000.0, rel=1e-3) and e["mean_w"] == pytest.approx(300.0, rel=1e-3)
    assert m.energy_integral(csv_text, t0, t0 + 10.0, 50, None)["basis"] == "no power.draw column"   # the registered reading
    assert m.energy_integral("power.draw.instant [W]\n300 W\n300 W\n", t0, t0 + 1.0, 50, t0)["joules"] is not None  # header form


def test_sameprompt_arm_checks_its_effective_rows_against_the_same_file():
    """sc1a-5090-1: vLLM's SAMEPROMPT arms read VOID "prompts_sha differs" -- the reducer compared the DISTINCT file's sha the
    arm recorded (`file_prompts_sha256`) with prompts_b16_same.json's; the arm's EFFECTIVE rows (`effective_prompts_sha256`)
    are what that file holds."""
    m = _mod()
    distinct, same = "f67e" * 16, "1e45" * 16
    pf_same = {"prompts_sha256": same, "batch": 16}
    sp = {"source_row": 0, "rows_identical": True, "copies": 16, "file_prompts_sha256": distinct, "effective_prompts_sha256": same}
    vllm_rec = {"prompts_sha256": distinct, "sameprompt": sp, "prompt_tokens": [512] * 16}          # the vLLM driver's shape
    e4b_rec = {"prompts_sha256": same, "sameprompt": sp, "prompt_tokens": [512] * 16}               # sc1_e4b_sched's shape
    for rec in (vllm_rec, e4b_rec):
        why = m.check_prompts(rec, pf_same, 16, same=True)
        assert not any("prompts_sha differs" in w for w in why), why
    bad = dict(vllm_rec, sameprompt=dict(sp, effective_prompts_sha256="dead" * 16))
    assert any("prompts_sha differs" in w for w in m.check_prompts(bad, pf_same, 16, same=True))   # still refuses a wrong file
