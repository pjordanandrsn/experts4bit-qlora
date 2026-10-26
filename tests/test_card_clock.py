# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""bench/common/card_clock.py: the loaded-SM-clock host floor, read on REAL stored clock records (TC1 amendment 83).

The rule (the ``clock`` mode, the default): the clock binds; slowdown bits refuse on any sample; Reliability refuses when PERSISTENT
(>= 95 % of busy samples); SwPowerCap is recorded only. Every verdict here is read from a committed record, never a hand-made line:
- tc1-5090-pos79-r1: a flat 2400 MHz with NVML's Reliability reason on every busy sample (the capped card amendment 80 was written for);
- tc1-5090-136: SwPowerCap on about half its busy samples at 2377 MHz (a power-capped card, refused on its clock);
- tc1-5090-137: 2895 MHz, no reason (the clean card);
- axo1767b-5090-1's hp1shake log (tests/fixtures/card_clock/, sha-pinned): the boundary case, the HEALTHIEST-clock card with the
  HIGHEST Reliability share on record (busy median 3007 MHz, Reliability on 237 of 294 busy samples, 81 %, alternating with
  SwPowerCap on a 475 W card). It must pass: Reliability there is a power-limit reason, not a slow card;
- p238-shake-1 and p238-shake-r1: the only two card_clock burns ever run (both P238's shakedown, 2026-10-26), as their burn summaries.
The synthetic SwPowerCap-sprinkle cases amendment 80 tested with are gone: they encoded a belief ("a healthy card may touch it
briefly") that the first real burn on a healthy card disproved (SwPowerCap on 98.6 % of its busy samples).
"""
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "bench" / "common" / "card_clock.py"
RECEIPTS = REPO / "bench" / "h2h-2026-10-02" / "tc1" / "receipts"
CAPPED = RECEIPTS / "tc1-5090-pos79-r1" / "tc1" / "gpuclk_qwen3pos79_e4b_fused_attn4_m_fd.txt"
POWER_CAPPED = RECEIPTS / "tc1-5090-136" / "gpuclk_qwen3pos68_e4b_fused_attn4_m_pd.txt"
CLEAN = RECEIPTS / "tc1-5090-137" / "gpuclk_qwen3pos69_e4b_fused_attn4_m_fd.txt"
FX = REPO / "tests" / "fixtures" / "card_clock"
BOUNDARY = FX / "axo1767b-5090-1_gpuclk_hp1shake_axolotl.txt"        # store: receipts/.../2026-10-19/axo1767b-5090-1/tc1/gpuclk_qwen3hp1shake_axolotl_hp1shake_axolotl.txt
SHAKE_1 = FX / "p238-shake-1_card_clock_burn.json"                     # store: receipts/.../2026-10-26/p238-shake-1/p238/card_clock_burn.json
SHAKE_R1 = FX / "p238-shake-r1_card_clock_burn.json"                   # store: receipts/.../2026-10-26/p238-shake-r1/p238/card_clock_burn.json
FIXTURE_SHA256 = {BOUNDARY.name: "9a8216173f187ecc586ab351f23a8b2370deec17a54b568670cf3aa5c43db42e",
                  SHAKE_1.name: "b65d02d2c9eb8b7ac263516a87c71af32787fdf2084355fee160a14fbf3630b9",
                  SHAKE_R1.name: "d8b02836665bfc2b8729e895bbdd895ab4831f151e365ee5824bff0ed55c9bdf"}
FLOOR = 2700           # TC1 amendment 80's RTX 5090 floor


def _cc(src=None):
    """card_clock.py as committed, or a MUTANT of its source text (an exact edit of the real code, executed)."""
    if src is None:
        spec = importlib.util.spec_from_file_location("card_clock", TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    mod = types.ModuleType("card_clock_mutant")
    exec(compile(src, str(TOOL), "exec"), mod.__dict__)
    return mod


def _mutant(old, new):
    src = TOOL.read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    return _cc(src.replace(old, new))


def test_the_fixtures_are_the_stored_bytes():
    for name, sha in FIXTURE_SHA256.items():
        assert hashlib.sha256((FX / name).read_bytes()).hexdigest() == sha, name


def test_the_reason_table_is_nvmls_and_amendment_80s_names_are_unchanged():
    """NVML's values as nvidia-ml-py 13.615.71 defines them (pynvml.py 2190-2199, 2215-2226). Amendment 80's names keep their values:
    SC2f's judge-run reads HARD_BITS / SOFT_BITS / BUSY_MHZ by module, and amendment 83 only adds its own sets beside them."""
    cc = _cc()
    assert cc.REASON_BITS == {0x1: "GpuIdle", 0x2: "ApplicationsClocksSetting", 0x4: "SwPowerCap", 0x8: "HwSlowdown",
                              0x10: "SyncBoost", 0x20: "SwThermalSlowdown", 0x40: "HwThermalSlowdown", 0x80: "HwPowerBrakeSlowdown",
                              0x100: "DisplayClockSetting", 0x200: "BoardLimit", 0x400: "Reliability"}
    assert (cc.REFUSE_BITS, cc.SOFT_BITS, cc.HARD_BITS, cc.BUSY_MHZ) == (0x6EC, 0x4, 0x6E8, 1000)
    assert (cc.ANY_SAMPLE_BITS, cc.PERSISTENT_BITS, cc.RECORDED_BITS, cc.PERSISTENT_SHARE) == (0x2E8, 0x400, 0x4, 0.95)
    assert cc.ANY_SAMPLE_BITS | cc.PERSISTENT_BITS | cc.RECORDED_BITS == cc.REFUSE_BITS     # every refusing bit has one rule
    assert cc.decode(0x400) == ["Reliability"] and cc.decode(0x5) == ["GpuIdle", "SwPowerCap"]
    assert cc.decode(0x800) == ["0x800"]              # an unknown bit is named by value, never dropped


# -- the review's order: the two burns, then the committed logs, then the boundary -------------------------------------------
def test_shake_r1_record_mutant_restore_95_bar_fails_r1():
    """p238-shake-r1 (a stock 575 W card): each clause asserted separately. The clock passes (2827 >= 2700); SwPowerCap (98.6 % of
    busy samples) no longer refuses; amendment 80's burn rule (the restored 0.95 bar: the required mutant) refused it on BOTH SwPowerCap
    and Reliability-any; under amendment 83 its LEGACY summary holds no per-sample Reliability share, so that clause is UNREAD and
    refuses, saying so (its share is anywhere in 1/72..72/72: assuming a pass would be the trap amendment 83 fixes)."""
    cc, s = _cc(), json.loads(SHAKE_R1.read_text())
    assert (s["sm_median"], s["busy_samples"], s["busy_soft_clean_fraction"], s["reasons_seen_mask"]) == (2827.0, 72, 0.0139, "0x404")
    ok, why = cc.passes(s, FLOOR, burn=True, mode="clock")
    assert not ok
    assert "MHz < floor" not in why                                     # the clock clause passes
    assert "SwPowerCap" not in why                                      # SwPowerCap is recorded, never refuses
    assert "legacy burn summary: no per-sample Reliability share" in why and "UNREAD" in why
    ok80, why80 = cc.passes(s, FLOOR, 0.95, burn=True, mode="pct_clean")
    assert not ok80 and "SwPowerCap on 98.6%" in why80 and "Reliability" in why80


def test_shake_1_refuses_on_its_clock_under_both_rules():
    """p238-shake-1 (a 430 W card): busy median 2493.5 MHz < 2700. Amendment 83 refuses it on the clock; so did amendment 80."""
    cc, s = _cc(), json.loads(SHAKE_1.read_text())
    for mode in ("clock", "pct_clean"):
        ok, why = cc.passes(s, FLOOR, 0.95, burn=True, mode=mode)
        assert not ok and "2493.5 MHz < floor" in why, mode


def test_the_capped_card_refuses():
    cc = _cc()
    s = cc.summarize_log(str(CAPPED))
    assert (s["sm_median"], s["sm_max"], s["reasons_seen"], s["busy_clean_fraction"]) == (2400, 2415, ["Reliability"], 0.0)
    assert (s["busy_reliability_samples"], s["busy_samples"]) == (505, 505)
    ok, why = cc.passes(s, FLOOR, mode="clock")
    assert not ok and "2400 MHz < floor" in why and "Reliability on 505 of 505" in why


def test_the_power_capped_card_refuses_on_its_clock():
    cc = _cc()
    s = cc.summarize_log(str(POWER_CAPPED))
    assert s["reasons_seen"] == ["SwPowerCap"] and s["sm_max"] <= 2400
    ok, why = cc.passes(s, FLOOR, mode="clock")
    assert not ok and "2377 MHz < floor" in why and "SwPowerCap" not in why


def test_the_clean_card_passes():
    cc = _cc()
    s = cc.summarize_log(str(CLEAN))
    assert (s["sm_median"], s["reasons_seen"], s["busy_clean_fraction"]) == (2895, [], 1.0)
    assert cc.passes(s, FLOOR, mode="clock") == (True, "") and cc.passes(s, FLOOR, 0.95, mode="pct_clean") == (True, "")


def test_the_boundary_card_passes_the_healthiest_clock_with_the_highest_reliability_share():
    """axo1767b-5090-1's hp1shake log: busy median 3007 MHz, Reliability on 237 of 294 busy samples (81 %) and SwPowerCap on 19 %.
    The persistence bar (95 %) sits in the record's gap: no healthy-clock log above 81 %, every sub-floor one at 100 %."""
    cc = _cc()
    s = cc.summarize_log(str(BOUNDARY))
    assert (s["sm_median"], s["busy_reliability_samples"], s["busy_samples"], s["busy_swpowercap_fraction"]) == (3007.0, 237, 294, 0.1939)
    assert cc.passes(s, FLOOR, mode="clock") == (True, "")
    assert cc.passes(s, FLOOR, 0.95, mode="pct_clean")[0] is False      # amendment 80's per-draw bar refused this healthy card


# -- the clock floor and the persistence bar, on REAL bytes ------------------------------------------------------------------------
def _rewrite(src, dst, fn):
    lines = []
    for ln in src.read_text().splitlines():
        r = _cc().parse_gpuclk_line(ln)
        lines.append(fn(ln, r) if r else ln)
    dst.write_text("\n".join(lines) + "\n")


def _lift(ln, r, to=2895):
    """One busy line of a log with its SM clock set to ``to`` (everything else, the reason mask included, kept byte for byte)."""
    head, rest = ln.split(" ", 1)
    return f"{head} {to}," + rest.split(",", 1)[1]


def test_mutant_pos79_with_its_clock_lifted_still_refuses_on_persistent_reliability(tmp_path):
    """The pos79 log with every busy clock lifted to 2895 MHz: the clock passes, Reliability is still on 100 % of busy samples."""
    out = tmp_path / "gpuclk_lifted.txt"
    _rewrite(CAPPED, out, lambda ln, r: _lift(ln, r) if r[1] > 1000 else ln)
    s = _cc().summarize_log(str(out))
    assert s["sm_median"] == 2895 and s["busy_reliability_samples"] == s["busy_samples"] == 505
    ok, why = _cc().passes(s, FLOOR, mode="clock")
    assert not ok and "MHz < floor" not in why and "Reliability on 505 of 505" in why


def _persistent_pair(tmp_path, keep_busy, rel_on):
    """The pos79 log cut to its first ``keep_busy`` busy lines, clock lifted, Reliability KEPT on the first ``rel_on`` of them and
    CLEARED on the rest (its mask set to 0): real lines, one bit moved."""
    out = tmp_path / f"gpuclk_{keep_busy}_{rel_on}.txt"
    n, lines = 0, []
    for ln in CAPPED.read_text().splitlines():
        r = _cc().parse_gpuclk_line(ln)
        if r and r[1] > 1000:
            n += 1
            if n > keep_busy:
                break
            ln = _lift(ln, r)
            if n > rel_on:
                ln = ln.replace("0x0000000000000400", "0x0000000000000000", 1)
        lines.append(ln)
    out.write_text("\n".join(lines) + "\n")
    return _cc().summarize_log(str(out))


def test_the_persistence_bar_is_95_percent_of_busy_samples_inclusive(tmp_path):
    """500 busy samples: Reliability on 475 (exactly 95 %) refuses, on 474 (94.8 %) passes, so a '>' for '>=' is caught; and on
    the whole 505: 480 (95.05 %) refuses, 479 (94.85 %) passes. The denominator is the BUSY samples (SM > 1000 MHz)."""
    for keep, at, below in ((500, 475, 474), (505, 480, 479)):
        hi, lo = _persistent_pair(tmp_path, keep, at), _persistent_pair(tmp_path, keep, below)
        assert (hi["busy_samples"], hi["busy_reliability_samples"], lo["busy_reliability_samples"]) == (keep, at, below)
        assert _cc().passes(hi, FLOOR, mode="clock")[0] is False and _cc().passes(lo, FLOOR, mode="clock") == (True, ""), keep
    hi = _persistent_pair(tmp_path, 500, 475)
    gt = _mutant("if rel * 100 >= round(PERSISTENT_SHARE * 100) * n:", "if rel * 100 > round(PERSISTENT_SHARE * 100) * n:")
    assert gt.passes(hi, FLOOR, mode="clock")[0] is True                              # the off-by-one mutant would pass exactly 95 %: caught above


def test_mutant_drop_the_clock_floor_passes_the_power_capped_card():
    """The required mutant: with the clock clause removed, tc1-5090-136 (2377 MHz, no Reliability) would pass. The real rule refuses it."""
    m = _mutant('            why.append(f"busy SM median {summary[\'sm_median\']} MHz < floor {floor_mhz}")', "            pass")
    s = m.summarize_log(str(POWER_CAPPED))
    assert m.passes(s, FLOOR, mode="clock") == (True, "") and _cc().passes(s, FLOOR, mode="clock")[0] is False


def test_mutant_swpowercap_refusing_again_fails_the_healthy_boundary_card():
    """Put SwPowerCap back among the any-sample bits: the healthy 475 W card (SwPowerCap on 19 % of its busy samples) would refuse."""
    m = _mutant("ANY_SAMPLE_BITS = 0x8 | 0x20 | 0x40 | 0x80 | 0x200", "ANY_SAMPLE_BITS = 0x4 | 0x8 | 0x20 | 0x40 | 0x80 | 0x200")
    s = m.summarize_log(str(BOUNDARY))
    assert m.passes(s, FLOOR, mode="clock")[0] is False and _cc().passes(_cc().summarize_log(str(BOUNDARY)), FLOOR, mode="clock") == (True, "")


def test_a_slowdown_bit_refuses_on_a_single_sample(tmp_path):
    """The clean card's log with HwSlowdown (0x8) on ONE busy sample: refused. No record above the floor has carried one (a guard)."""
    done = {"x": False}

    def one(ln, r):
        if r[1] > 1000 and not done["x"]:
            done["x"] = True
            return ln.replace("0x0000000000000000", "0x0000000000000008", 1)
        return ln
    out = tmp_path / "gpuclk_hw.txt"
    _rewrite(CLEAN, out, one)
    ok, why = _cc().passes(_cc().summarize_log(str(out)), FLOOR, mode="clock")
    assert not ok and "HwSlowdown" in why


# -- the summary's fields and the named mode ---------------------------------------------------------------------------------------
def test_amendment_83_adds_fields_and_renames_none():
    """A burn summary written before amendment 83 (p238-shake-r1's) has amendment 80's keys; summarize() still writes every one of
    them, with the same values on the same samples, and only adds amendment 83's beside them."""
    old_keys = set(json.loads(SHAKE_R1.read_text())) - {"burn_seconds", "matmul_n", "dtype", "sample_period_s", "floor_mhz", "min_clean",
                                                         "hard_bits_every_sample", "pass", "why", "refusal"}
    cc = _cc()
    s = cc.summarize_log(str(CAPPED))
    assert old_keys <= set(s), old_keys - set(s)
    assert {k: s[k] for k in ("sm_median", "sm_max", "reasons_seen", "busy_clean_fraction", "busy_soft_clean_fraction",
                              "hard_reasons_any_sample")} == {"sm_median": 2400, "sm_max": 2415, "reasons_seen": ["Reliability"],
                                                              "busy_clean_fraction": 0.0, "busy_soft_clean_fraction": 1.0,
                                                              "hard_reasons_any_sample": ["Reliability"]}
    assert {"busy_swpowercap_samples", "busy_reliability_samples", "busy_swpowercap_fraction", "busy_reliability_fraction",
            "any_sample_bits_seen"} <= set(s)


@pytest.mark.parametrize("mode", ["clock", "pct_clean"])
def test_the_burn_line_keeps_summarize_fields_and_names_its_rule_fields_by_mode(tmp_path, monkeypatch, mode):
    """The burn line, written by main() for a burn over a REAL log's samples (the clean card's; no GPU needed): every summarize() field
    amendment 80 wrote is kept with the same value; the clock mode REPLACES amendment 80's two rule fields (min_clean,
    hard_bits_every_sample) with mode / persistent_share / any_sample_bits_every_sample, and pct_clean writes the two as before."""
    cc = _cc()
    samples = [r[1:] for r in (cc.parse_gpuclk_line(ln) for ln in CLEAN.read_text().splitlines()) if r is not None]
    monkeypatch.setattr(cc, "burn", lambda seconds: dict(cc.summarize(samples), burn_seconds=seconds, per_sample=[[s[0], hex(s[3])] for s in samples]))
    out = tmp_path / "card_clock.json"
    rc = cc.main(["burn", "--floor", str(FLOOR), "--mode", mode, "--min-clean", "0.95", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rc == 0 and rec["pass"] is True and rec["mode"] == mode
    kept = ("samples", "busy_samples", "sm_median", "sm_p10", "sm_max", "mem_median", "reasons_seen", "reasons_seen_mask",
            "busy_clean_fraction", "busy_soft_clean_fraction", "hard_reasons_any_sample")
    want = cc.summarize_log(str(CLEAN))
    assert {k: rec[k] for k in kept} == {k: want[k] for k in kept}
    assert len(rec["per_sample"]) == rec["samples"]
    if mode == "clock":
        assert "min_clean" not in rec and "hard_bits_every_sample" not in rec
        assert rec["persistent_share"] == 0.95 and rec["any_sample_bits_every_sample"] is True
    else:
        assert rec["min_clean"] == 0.95 and rec["hard_bits_every_sample"] is True
        assert "persistent_share" not in rec and "any_sample_bits_every_sample" not in rec


def test_the_pct_clean_mode_keeps_amendment_80s_bars_by_name(tmp_path):
    """The clean card with SwPowerCap on every 10th busy sample: pct_clean refuses (90 % clean < 95 %), clock passes."""
    n = {"i": 0}

    def flip(ln, r):
        if r[1] > 1000:
            n["i"] += 1
            if n["i"] % 10 == 0:
                return ln.replace("0x0000000000000000", "0x0000000000000004", 1)
        return ln
    out = tmp_path / "gpuclk_cap10.txt"
    _rewrite(CLEAN, out, flip)
    s = _cc().summarize_log(str(out))
    ok, why = _cc().passes(s, FLOOR, 0.95, mode="pct_clean")
    assert not ok and "SwPowerCap" in why
    assert _cc().passes(s, FLOOR, mode="clock") == (True, "")
    with pytest.raises(ValueError):
        _cc().passes(s, FLOOR, mode="lenient")


def test_mutant_a_clean_card_clocked_down_refuses(tmp_path):
    """The clean card's log with every busy SM clock scaled by 2400/2895: the bits stay clean and the floor alone refuses."""
    def scale(ln, r):
        head, rest = ln.split(" ", 1)
        return f"{head} {int(r[1] * 2400 / 2895)}," + rest.split(",", 1)[1] if r[1] > 1000 else ln
    out = tmp_path / "gpuclk_slow.txt"
    _rewrite(CLEAN, out, scale)
    s = _cc().summarize_log(str(out))
    assert s["busy_clean_fraction"] == 1.0 and s["sm_median"] < FLOOR
    assert not _cc().passes(s, FLOOR, mode="clock")[0]


def test_an_unread_card_never_passes():
    cc = _cc()
    for mode in ("clock", "pct_clean"):
        assert cc.passes(cc.summarize([]), FLOOR, 0.95, mode=mode)[0] is False
        assert cc.passes(cc.summarize([(2900, 13801, 50, 0)] * 5), FLOOR, 0.95, mode=mode)[0] is False     # under 10 busy samples


def test_a_caller_that_names_no_mode_fails_loudly():
    """An amendment 80-style call (floor and min_clean, no mode) raises TypeError rather than silently taking either rule; mode is
    keyword-only, so it cannot ride in positionally either. The mutant that gives mode a default again accepts the old call."""
    cc, s = _cc(), _cc().summarize_log(str(CLEAN))
    with pytest.raises(TypeError):
        cc.passes(s, FLOOR, 0.95)
    with pytest.raises(TypeError):
        cc.passes(s, FLOOR, 0.95, 10, False, "clock")
    m = _mutant("burn: bool = False, *, mode: str):", 'burn: bool = False, *, mode: str = "clock"):')
    assert m.passes(s, FLOOR, 0.95) == (True, "")


def test_the_check_log_cli_exit_codes():
    def run(*a):
        return subprocess.run([sys.executable, str(TOOL), "check-log", *a, "--floor", str(FLOOR)], capture_output=True, text=True)
    ok, bad, edge = run(str(CLEAN)), run(str(CAPPED)), run(str(BOUNDARY))
    assert ok.returncode == 0 and json.loads(ok.stdout)["pass"] is True and json.loads(ok.stdout)["mode"] == "clock"
    assert bad.returncode == 1 and json.loads(bad.stdout)["pass"] is False
    assert edge.returncode == 0                                          # the healthy boundary card passes the default mode
    old = run(str(BOUNDARY), "--mode", "pct_clean", "--min-clean", "0.95")
    assert old.returncode == 1 and json.loads(old.stdout)["min_clean"] == 0.95


def test_a_burn_that_cannot_read_the_card_refuses_with_the_host_floor_code(tmp_path):
    """No CUDA here (CI, or a box whose driver is gone): the burn cannot run, so it refuses with rc 18, never a pass. --min-clean is
    still accepted (every staged kit passes it), and the line names its mode."""
    import importlib.util as iu
    if iu.find_spec("torch") is not None:
        import torch
        if torch.cuda.is_available():
            pytest.skip("a CUDA device is present: the burn would really run")
    out = tmp_path / "card_clock.json"
    r = subprocess.run([sys.executable, str(TOOL), "burn", "--floor", str(FLOOR), "--seconds", "1", "--min-clean", "0.95", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 18
    rec = json.loads(out.read_text())
    assert rec["pass"] is False and rec["refusal"] == "probe_error"          # told apart from a card read below its floor
    assert rec["mode"] == "clock" and rec["persistent_share"] == 0.95 and "min_clean" not in rec
