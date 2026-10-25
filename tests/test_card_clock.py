# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""bench/common/card_clock.py: the loaded-SM-clock host floor, read on REAL stored per-second clock logs.

The capped card is TC1 amendment 79's box (tc1-5090-pos79-r1: a flat 2400 MHz with NVML's Reliability reason on every sample). The
power-capped one is tc1-5090-136 (SwPowerCap on about half its busy samples, 2377 MHz). The clean one is amendment 69's box
(tc1-5090-137: 2895 MHz, no reason). A test written only on hand-made lines would encode a belief about the log format, so every
verdict here is read from a committed receipt.
"""
import importlib.util
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "bench" / "common" / "card_clock.py"
RECEIPTS = REPO / "bench" / "h2h-2026-10-02" / "tc1" / "receipts"
CAPPED = RECEIPTS / "tc1-5090-pos79-r1" / "tc1" / "gpuclk_qwen3pos79_e4b_fused_attn4_m_fd.txt"
POWER_CAPPED = RECEIPTS / "tc1-5090-136" / "gpuclk_qwen3pos68_e4b_fused_attn4_m_pd.txt"
CLEAN = RECEIPTS / "tc1-5090-137" / "gpuclk_qwen3pos69_e4b_fused_attn4_m_fd.txt"
FLOOR = 2700           # TC1 amendment 80's RTX 5090 floor


def _cc():
    spec = importlib.util.spec_from_file_location("card_clock", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_reason_table_is_nvmls():
    """NVML's values as nvidia-ml-py 13.615.71 defines them (pynvml.py 2190-2199, 2215-2226), pinned here so an edit shows."""
    cc = _cc()
    assert cc.REASON_BITS == {0x1: "GpuIdle", 0x2: "ApplicationsClocksSetting", 0x4: "SwPowerCap", 0x8: "HwSlowdown",
                              0x10: "SyncBoost", 0x20: "SwThermalSlowdown", 0x40: "HwThermalSlowdown", 0x80: "HwPowerBrakeSlowdown",
                              0x100: "DisplayClockSetting", 0x200: "BoardLimit", 0x400: "Reliability"}
    assert cc.REFUSE_BITS == 0x4 | 0x8 | 0x20 | 0x40 | 0x80 | 0x200 | 0x400
    assert cc.decode(0x400) == ["Reliability"]
    assert cc.decode(0x5) == ["GpuIdle", "SwPowerCap"]
    assert cc.decode(0x800) == ["0x800"]              # an unknown bit is named by value, never dropped


def test_the_capped_card_refuses():
    cc = _cc()
    s = cc.summarize_log(str(CAPPED))
    assert (s["sm_median"], s["sm_max"], s["reasons_seen"], s["busy_clean_fraction"]) == (2400, 2415, ["Reliability"], 0.0)
    ok, why = cc.passes(s, FLOOR, 0.95)
    assert not ok and "2400 MHz < floor" in why and "Reliability" in why


def test_the_power_capped_card_refuses():
    cc = _cc()
    s = cc.summarize_log(str(POWER_CAPPED))
    assert s["reasons_seen"] == ["SwPowerCap"] and s["sm_max"] <= 2400
    assert not cc.passes(s, FLOOR, 0.95)[0]


def test_the_clean_card_passes():
    cc = _cc()
    s = cc.summarize_log(str(CLEAN))
    assert (s["sm_median"], s["reasons_seen"], s["busy_clean_fraction"]) == (2895, [], 1.0)
    assert cc.passes(s, FLOOR, 0.95) == (True, "")


def _rewrite(src, dst, fn):
    lines = []
    for ln in src.read_text().splitlines():
        r = _cc().parse_gpuclk_line(ln)
        lines.append(fn(ln, r) if r else ln)
    dst.write_text("\n".join(lines) + "\n")


def test_mutant_a_clean_card_with_a_cap_bit_on_ten_percent_refuses(tmp_path):
    """The clean card's own log with SwPowerCap set on every 10th busy sample: 90 % clean < 95 %, so it refuses."""
    n = {"i": 0}

    def flip(ln, r):
        if r[1] > 1000:
            n["i"] += 1
            if n["i"] % 10 == 0:
                return ln.replace("0x0000000000000000", "0x0000000000000004", 1)
        return ln
    out = tmp_path / "gpuclk_mutant.txt"
    _rewrite(CLEAN, out, flip)
    s = _cc().summarize_log(str(out))
    assert s["sm_median"] == 2895 and 0.89 < s["busy_clean_fraction"] < 0.91
    ok, why = _cc().passes(s, FLOOR, 0.95)
    assert not ok and "SwPowerCap" in why


def test_mutant_a_clean_card_clocked_down_refuses(tmp_path):
    """The clean card's log with every busy SM clock scaled by 2400/2895: the bits stay clean and the floor alone refuses."""
    def scale(ln, r):
        head, rest = ln.split(" ", 1)
        return f"{head} {int(r[1] * 2400 / 2895)}," + rest.split(",", 1)[1] if r[1] > 1000 else ln
    out = tmp_path / "gpuclk_slow.txt"
    _rewrite(CLEAN, out, scale)
    s = _cc().summarize_log(str(out))
    assert s["busy_clean_fraction"] == 1.0 and s["sm_median"] < FLOOR
    assert not _cc().passes(s, FLOOR, 0.95)[0]


def test_the_burn_rule_refuses_the_power_capped_card_on_its_clock():
    """The burn holds SwPowerCap to 95 % of busy samples, not to every one. tc1-5090-136 (SwPowerCap on about half its samples, 2377 MHz)
    still refuses, on its median and on its SwPowerCap share."""
    s = _cc().summarize_log(str(POWER_CAPPED))
    ok, why = _cc().passes(s, FLOOR, 0.95, burn=True)
    assert not ok and "2377 MHz < floor" in why and "SwPowerCap on" in why


def test_the_burn_rule_passes_a_brief_power_cap_and_refuses_any_hard_reason(tmp_path):
    """SwPowerCap on 3 % of busy samples passes the burn. A single Reliability sample refuses it."""
    n = {"i": 0}

    def sprinkle(bit, every):
        def f(ln, r):
            if r[1] > 1000:
                n["i"] += 1
                if n["i"] % every == 0:
                    return ln.replace("0x0000000000000000", f"0x{bit:016x}", 1)
            return ln
        return f
    soft = tmp_path / "soft.txt"
    _rewrite(CLEAN, soft, sprinkle(0x4, 33))
    s = _cc().summarize_log(str(soft))
    assert 0.96 < s["busy_soft_clean_fraction"] < 0.98 and _cc().passes(s, FLOOR, 0.95, burn=True) == (True, "")
    n["i"] = 0
    hard = tmp_path / "hard.txt"
    _rewrite(CLEAN, hard, sprinkle(0x400, 10_000))                          # one busy sample at most
    lines = hard.read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if (_cc().parse_gpuclk_line(ln) or (0, 0))[1] > 1000)
    lines[i] = lines[i].replace("0x0000000000000000", "0x0000000000000400", 1)
    hard.write_text("\n".join(lines) + "\n")
    s = _cc().summarize_log(str(hard))
    ok, why = _cc().passes(s, FLOOR, 0.95, burn=True)
    assert not ok and "Reliability" in why
    assert _cc().passes(s, FLOOR, 0.95)[0] is True                          # on a run's log one sample in 345 is inside the 95 %


def test_an_unread_card_never_passes():
    cc = _cc()
    assert cc.passes(cc.summarize([]), FLOOR, 0.95)[0] is False
    assert cc.passes(cc.summarize([(2900, 13801, 50, 0)] * 5), FLOOR, 0.95)[0] is False     # under 10 busy samples


def test_the_check_log_cli_exit_codes():
    ok = subprocess.run([sys.executable, str(TOOL), "check-log", str(CLEAN), "--floor", str(FLOOR)], capture_output=True, text=True)
    bad = subprocess.run([sys.executable, str(TOOL), "check-log", str(CAPPED), "--floor", str(FLOOR)], capture_output=True, text=True)
    assert ok.returncode == 0 and json.loads(ok.stdout)["pass"] is True
    assert bad.returncode == 1 and json.loads(bad.stdout)["pass"] is False


def test_a_burn_that_cannot_read_the_card_refuses_with_the_host_floor_code(tmp_path):
    """No CUDA here (CI, or a box whose driver is gone): the burn cannot run, so it refuses with rc 18, never a pass."""
    import importlib.util as iu
    if iu.find_spec("torch") is not None:
        import torch
        if torch.cuda.is_available():
            import pytest
            pytest.skip("a CUDA device is present: the burn would really run")
    out = tmp_path / "card_clock.json"
    r = subprocess.run([sys.executable, str(TOOL), "burn", "--floor", str(FLOOR), "--seconds", "1", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 18
    assert json.loads(out.read_text())["pass"] is False
