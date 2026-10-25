"""TC1 amendment 81: device time is kernels-only AT THE SOURCE (bench/tc1/tc1_arm.py summarize_profile).

Tested on a REAL stored profile (tc1-5090-110, q28, three profiled steps, one ``Optimizer.step#AdamW8bit.step`` annotation row). The
profiler's ``key_averages()`` is rebuilt from that profile's own tables: every stored device and CPU row, plus ONE remainder kernel row
carrying the device time of the rows past the stored top 80, so the rebuilt rows sum to the recorded ``device_ms`` exactly. Each row
carries torch's ``is_user_annotation`` flag, True only on the annotation row, as torch 2.8.0's key_averages() sets it.

The contract checked:
- the at-source figure equals amendment 80's by-name correction (bench/tc1/f15v2.kernels_only_device_ms) on the same real profile;
- the profile it writes is stamped ``kernels_only_at_source``, and f15v2 REFUSES to correct it again;
- the annotation row is kept, flagged, and its time recorded as ``user_annotation_ms``; event counts do not change;
- the family fractions lose the annotation from the optimizer family only;
- a torch without the flag on any device row gives the legacy all-CUDA sum, stamped ``all_cuda``, never a partial correction;
- MUTANT: the same rows with the annotation's flag off give the legacy figure, so the equality above can fail.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.autograd import DeviceType  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ARM = REPO / "bench" / "tc1" / "tc1_arm.py"
F15V2 = REPO / "bench" / "tc1" / "f15v2.py"
PROFILE = REPO / "bench/h2h-2026-10-02/tc1/receipts/tc1-5090-110/qwen3prof28_e4b_fused_attn4_m_q28_profile.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def arm():
    return _load("tc1_arm_am81", ARM)


@pytest.fixture(scope="module")
def f15v2():
    return _load("f15v2_am81", F15V2)


@pytest.fixture(scope="module")
def real():
    return json.loads(PROFILE.read_text())


class _Avg:
    """One key_averages() row, with the attributes summarize_profile reads."""

    def __init__(self, key, device, count, self_cpu_ms, self_device_ms, ann=False, flag=True):
        self.key, self.count = key, count
        self.device_type = DeviceType.CUDA if device else DeviceType.CPU
        self.self_cpu_time_total = self_cpu_ms * 1e3
        self.self_device_time_total = self_device_ms * 1e3
        if flag:
            self.is_user_annotation = ann


class _Prof:
    def __init__(self, rows):
        self.rows = rows

    def key_averages(self):
        return list(self.rows)


def _rows(real, f15v2, ann_flag=True, flag=True):
    """The real profile's rows as key_averages() would give them. ``ann_flag`` False is the MUTANT (the annotation not flagged);
    ``flag`` False is a torch without the attribute on any row."""
    out = []
    for r in real["top_device"]:
        is_ann = bool(f15v2.ANNOTATION_RE.match(r["name"]))
        out.append(_Avg(r["name"], True, r["count"], 0.0, r["self_device_ms"], ann=is_ann and ann_flag, flag=flag))
    rest = real["summary"]["device_ms"] - sum(r["self_device_ms"] for r in real["top_device"])
    assert rest >= 0
    out.append(_Avg("remainder_kernels_past_top80", True, 1, 0.0, rest, ann=False, flag=flag))
    for r in real["top_cpu"]:
        out.append(_Avg(r["name"], False, r["count"], r["self_cpu_ms"], 0.0, flag=flag))
    return out


def _summ(arm, rows, real, tmp_path, name="p.json"):
    s = real["summary"]
    return arm.summarize_profile(_Prof(rows), s["wall_ms"] / 1e3, s["profiled_steps"], str(tmp_path / name))


def test_the_real_profile_is_the_registered_fixture(real, f15v2):
    """The fixture is what the test claims: legacy (no method field), one Optimizer.step# row, count == profiled_steps."""
    assert "device_ms_method" not in real["summary"]
    ann = [r for r in real["top_device"] if r["name"].startswith("Optimizer.step#")]
    assert len(ann) == 1 and ann[0]["count"] == real["summary"]["profiled_steps"]
    k, a, names = f15v2.kernels_only_device_ms(real)
    assert a > 0 and names == [ann[0]["name"]]


def test_at_source_equals_amendment_80s_correction_on_the_real_profile(arm, f15v2, real, tmp_path):
    summ = _summ(arm, _rows(real, f15v2), real, tmp_path)
    k_per_step, a_per_step, _ = f15v2.kernels_only_device_ms(real)
    ps = real["summary"]["profiled_steps"]
    assert summ["device_ms_method"] == arm.DEVICE_MS_METHOD == "kernels_only_at_source"
    assert summ["device_ms"] == pytest.approx(round(k_per_step * ps, 1), abs=0.11)
    assert summ["user_annotation_ms"] == pytest.approx(round(a_per_step * ps, 1), abs=0.11)
    assert summ["device_busy_fraction"] == pytest.approx(summ["device_ms"] / real["summary"]["wall_ms"], abs=1e-4)


def test_f15v2_refuses_to_correct_an_at_source_profile_again(arm, f15v2, real, tmp_path):
    _summ(arm, _rows(real, f15v2), real, tmp_path, "written.json")
    written = json.loads((tmp_path / "written.json").read_text())
    assert written["summary"]["device_ms_method"] == "kernels_only_at_source"
    with pytest.raises(f15v2.AnnotationError, match="corrected twice"):
        f15v2.kernels_only_device_ms(written)


def test_the_annotation_row_is_kept_flagged_and_counts_do_not_move(arm, f15v2, real, tmp_path):
    new = _summ(arm, _rows(real, f15v2), real, tmp_path, "new.json")
    old = _summ(arm, _rows(real, f15v2, flag=False), real, tmp_path, "old.json")
    written = json.loads((tmp_path / "new.json").read_text())
    flagged = [r for r in written["top_device"] if r.get("user_annotation")]
    assert [r["name"] for r in flagged] == ["Optimizer.step#AdamW8bit.step"]
    for k in ("device_events_per_step", "cpu_ops_per_step", "launches_per_step", "cpu_self_ms", "wall_ms"):
        assert new[k] == old[k], k


def test_only_the_optimizer_family_loses_the_annotation(arm, f15v2, real, tmp_path):
    """Exact re-attribution: in absolute ms, every family but the annotation's own is unchanged."""
    new = _summ(arm, _rows(real, f15v2), real, tmp_path, "n.json")
    old = _summ(arm, _rows(real, f15v2, flag=False), real, tmp_path, "o.json")
    ann_fam = arm._family("Optimizer.step#AdamW8bit.step")
    n_abs = {f: v * new["device_ms"] for f, v in new["device_by_family_fraction"].items()}
    o_abs = {f: v * old["device_ms"] for f, v in old["device_by_family_fraction"].items()}
    tol = 1e-4 * max(new["device_ms"], old["device_ms"])      # the fractions are rounded to 4 decimals: one ulp each side
    moved = {f for f in set(n_abs) | set(o_abs) if abs(n_abs.get(f, 0.0) - o_abs.get(f, 0.0)) > tol}
    assert moved <= {ann_fam}, moved
    # the optimizer family drops by the annotation's time, to within the same 4-decimal rounding of the two fractions
    assert o_abs[ann_fam] - n_abs.get(ann_fam, 0.0) == pytest.approx(new["user_annotation_ms"], abs=tol)


def test_a_torch_without_the_flag_gives_the_legacy_sum_stamped_all_cuda(arm, f15v2, real, tmp_path):
    summ = _summ(arm, _rows(real, f15v2, flag=False), real, tmp_path, "legacy.json")
    assert summ["device_ms_method"] == "all_cuda" and summ["user_annotation_ms"] == 0.0
    assert summ["device_ms"] == pytest.approx(real["summary"]["device_ms"], abs=0.11)
    written = json.loads((tmp_path / "legacy.json").read_text())
    f15v2.kernels_only_device_ms(written)          # legacy: amendment 80's correction still applies, exactly once


def test_a_partly_flagged_torch_is_never_half_corrected(arm, f15v2, real, tmp_path):
    rows = _rows(real, f15v2)
    plain = next(r for r in rows if r.device_type == DeviceType.CUDA and not r.is_user_annotation)
    del plain.is_user_annotation                  # one device row without the flag
    summ = _summ(arm, rows, real, tmp_path, "mixed.json")
    assert summ["device_ms_method"] == "all_cuda" and summ["user_annotation_ms"] == 0.0
    assert summ["device_ms"] == pytest.approx(real["summary"]["device_ms"], abs=0.11)


def test_mutant_annotation_not_flagged_reads_the_legacy_figure(arm, f15v2, real, tmp_path):
    """The equality test above can FAIL: with the annotation's flag off, the figure is the legacy one, not kernels-only."""
    summ = _summ(arm, _rows(real, f15v2, ann_flag=False), real, tmp_path, "mut.json")
    k_per_step, a_per_step, _ = f15v2.kernels_only_device_ms(real)
    ps = real["summary"]["profiled_steps"]
    assert summ["device_ms"] == pytest.approx(real["summary"]["device_ms"], abs=0.11)
    assert abs(summ["device_ms"] - k_per_step * ps) > a_per_step * ps * 0.9
