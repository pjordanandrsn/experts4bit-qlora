"""TC1 amendment 81: tc1_arm.summarize_profile counts device time from KERNEL rows only, tested on a second real profile and with a
SOURCE mutant. (The cases and the fixture design are from a draft of this amendment by another contributor; tests/test_tc1_am81.py holds
the at-source = f15v2 equality on tc1-5090-110.)

A user-annotation row (record_function's GPU-side span, ``Optimizer.step#AdamW8bit.step`` in every TC1 profile) covers the kernels
launched in its range. Those kernels are rows of their own, so the old sum counted them twice (e4b#2129). The rule is torch's own,
``evt.is_user_annotation``, never a name. The events here are built from amendment 79's stored e4b profile (tc1-5090-pos79-r1), with the
attribute set on the annotation row as torch sets it, and fed to the real summarize_profile. No GPU is needed.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
ARM = REPO / "bench" / "tc1" / "tc1_arm.py"
PROFILE = REPO / "bench" / "h2h-2026-10-02" / "tc1" / "receipts" / "tc1-5090-pos79-r1" / "tc1" / "qwen3pos79_e4b_fused_attn4_m_fd_profile.json"
ANNOTATION = "Optimizer.step#AdamW8bit.step"
RULE = 'is_ann = bool(flag_seen and is_dev and getattr(e, "is_user_annotation", False))'


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ARM_MOD = _load(ARM, "tc1_arm_device_time")
F2 = _load(REPO / "bench" / "tc1" / "f15v2.py", "f15v2_device_time")


class _Ev:
    """The FunctionEventAvg fields summarize_profile reads; times in microseconds, as torch keeps them."""

    def __init__(self, key, cpu_ms, dev_ms, count, cuda, annotation=None):
        self.key, self.count = key, count
        self.self_cpu_time_total, self.self_device_time_total = cpu_ms * 1e3, dev_ms * 1e3
        self.device_type = torch.autograd.DeviceType.CUDA if cuda else torch.autograd.DeviceType.CPU
        if annotation is not None:
            self.is_user_annotation = annotation


class _Prof:
    def __init__(self, events):
        self.events = events

    def key_averages(self):
        return self.events


def _events(extra=(), attribute=True):
    prof = json.loads(PROFILE.read_text())
    ev = [_Ev(r["name"], 0.0, r["self_device_ms"], r["count"], True, (r["name"] == ANNOTATION) if attribute else None)
          for r in prof["top_device"]]
    ev += [_Ev(r["name"], r["self_cpu_ms"], 0.0, r["count"], False, False if attribute else None) for r in prof["top_cpu"]]
    return ev + list(extra), prof


def _kernel_ms(prof):
    return sum(r["self_device_ms"] for r in prof["top_device"] if r["name"] != ANNOTATION)


def _summarize(tmp_path, events, fn=None):
    out = tmp_path / "x_profile.json"
    s = (fn or ARM_MOD.summarize_profile)(_Prof(events), wall_s=17.95, n_steps=3, out_path=str(out))
    return s, json.loads(out.read_text())


def _counts_kernels_only(s, prof):
    return abs(s["device_ms"] - round(_kernel_ms(prof), 1)) < 0.2


def test_device_time_is_the_kernels_and_the_span_is_recorded_apart(tmp_path):
    ev, prof = _events()
    s, out = _summarize(tmp_path, ev)
    span = next(r["self_device_ms"] for r in prof["top_device"] if r["name"] == ANNOTATION)
    assert s["device_ms_method"] == "kernels_only_at_source" and _counts_kernels_only(s, prof)
    assert abs(s["user_annotation_ms"] - span) < 0.2 and round(span / 3, 1) == 85.3      # the real row: 85.3 ms a profiled step
    row = next(r for r in out["top_device"] if r["name"] == ANNOTATION)
    assert row.get("user_annotation") is True and not any(r.get("user_annotation") for r in out["top_device"] if r["name"] != ANNOTATION)
    # event counts are unchanged by the rule (the annotation row stays a counted event), so no other figure moves
    assert s["device_events_per_step"] == round(sum(r["count"] for r in prof["top_device"]) / 3)


def test_an_injected_annotation_never_moves_device_time(tmp_path):
    ev, _ = _events()
    base, _ = _summarize(tmp_path, ev)
    more, _ = _summarize(tmp_path, ev + [_Ev("ProfilerStep#7", 0.0, 400.0, 3, True, True)])
    assert more["device_ms"] == base["device_ms"] and more["user_annotation_ms"] > base["user_annotation_ms"]


def test_a_torch_that_cannot_tell_keeps_the_legacy_sum_and_says_so(tmp_path):
    """No is_user_annotation attribute: the profile is marked all_cuda (the old rule), never reported as corrected."""
    ev, prof = _events(attribute=False)
    s, _ = _summarize(tmp_path, ev)
    assert s["device_ms_method"] == "all_cuda"
    assert abs(s["device_ms"] - round(sum(r["self_device_ms"] for r in prof["top_device"]), 1)) < 0.2


def test_the_mutant_that_sums_annotations_is_caught(tmp_path):
    """The rule disabled in a copy of summarize_profile's own source: device time takes the span back, and the kernels-only check
    fails. Only a source mutant shows the filter is in the code path; a fixture mutant cannot."""
    src = inspect.getsource(ARM_MOD.summarize_profile)
    assert src.count(RULE) == 1
    ns = dict(vars(ARM_MOD))
    exec(compile(src.replace(RULE, "is_ann = False"), str(ARM), "exec"), ns)
    ev, prof = _events()
    s, _ = _summarize(tmp_path, ev, fn=ns["summarize_profile"])
    assert not _counts_kernels_only(s, prof)


def test_amendment_80s_by_name_correction_refuses_a_profile_corrected_at_source(tmp_path):
    """The two instruments never stack: f15v2 refuses a profile this summary marks kernels_only_at_source."""
    ev, _ = _events()
    _, out = _summarize(tmp_path, ev)
    with pytest.raises(F2.AnnotationError, match="corrected twice"):
        F2.kernels_only_device_ms(out)
