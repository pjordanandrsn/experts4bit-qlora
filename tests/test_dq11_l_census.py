"""Census falsification: correlation, interval budget and prospective decisions."""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import pytest

LANE = Path(__file__).parents[1] / "bench/dq11"
sys.path.insert(0, str(LANE))
import dq11_l_census_reduce as census
from dq11_l_census_capture import Capture


def cpu(name, start, duration, external, *, cat="cpu_op", seq=None, fwd=0, dims=None):
    args = {"External id": external}
    if seq is not None:
        args.update({"Sequence number": seq, "Fwd thread id": fwd})
    if dims is not None:
        args["Input Dims"] = dims
    return {"ph": "X", "cat": cat, "name": name, "ts": start, "dur": duration, "pid": 1, "tid": 1, "args": args}


def fixture():
    events, external, seq, correlation = [], 1, 0, 1
    for step in census.STEPS:
        start = step * 2000000
        events.append(cpu(census.STEP_PREFIX + str(step), start, 1000000, external, cat="user_annotation"))
        external += 1
        for i, cls in enumerate(census.CLASSES):
            pos = start + 10 + i * 50
            events.append(cpu(census.PREFIX + cls + "|" + cls, pos, 40, external, cat="user_annotation"))
            external += 1
            op = cpu("aten::mm", pos + 1, 5, external, seq=seq, dims=[[2048, 16], [16, 4096]])
            if cls in ("attention", "optimizer", "other", "base NF4 GEMM"):
                op["args"]["Input Dims"] = [[2048, 4096], [4096, 4096]]
            events.append(op)
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "clipped-same-name",
                    "pid": 0,
                    "tid": 7,
                    "ts": start + 1000 + i * 100000,
                    "dur": 100000,
                    "args": {"External id": external, "correlation": correlation, "stream": 7},
                }
            )
            external += 1
            seq += 1
            correlation += 1
        # Backward MLP is launched outside forward ranges but bound by sequence.
        events.append(
            cpu("autograd::engine::evaluate_function: MmBackward0", start + 500, 20, external, seq=seq - 7, fwd=1)
        )
        external += 1
        events.append(cpu("aten::mm", start + 501, 5, external, dims=[[4096, 2048], [2048, 16]]))
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "clipped-same-name",
                "pid": 0,
                "tid": 7,
                "ts": start + 800000,
                "dur": 20000,
                "args": {"External id": external, "correlation": correlation, "stream": 7},
            }
        )
        external += 1
        correlation += 1
    return {"schemaVersion": 1, "traceEvents": events}


def test_actual_correlation_backward_mapping_and_clipped_names_not_deduplicated():
    normalized = census.normalize_trace(fixture())
    assert len(normalized["events"]) == 24
    assert [e["class"] for e in normalized["events"] if e["phase"] == "backward"] == ["MLP-adapter"] * 3
    result = census.reduce_events(normalized, [1.0] * 40)
    assert result["verdict"] == "BUILD_MLP"
    assert result["candidate_min_union_seconds"]["MLP-adapter"] == pytest.approx(0.12)
    assert all(r["closure_residual_ns"] == 0 and r["event_count"] == 8 for r in result["updates"])
    assert (
        sum(result["mean_exclusive_ns"].values())
        + result["mean_mixed_overlap_ns"]
        + result["mean_host_or_no_device_ns"]
        == result["mean_profile_wall_ns"]
    )


def test_backward_creator_uses_forward_thread_and_exact_op_under_nested_base():
    step = 6
    start = step * 2000000
    events = [cpu(census.STEP_PREFIX + str(step), start, 1000, 1, cat="user_annotation")]
    events += [cpu(census.PREFIX + "qkv-adapter|q", start + 10, 190, 2, cat="user_annotation")]
    events += [cpu("aten::to", start + 20, 10, 3, seq=5)]
    events += [cpu(census.PREFIX + "base NF4 GEMM|q.base", start + 40, 100, 4, cat="user_annotation")]
    events += [cpu("aten::clone", start + 50, 10, 5, seq=5)]
    events += [
        {
            "ph": "X",
            "cat": "kernel",
            "name": "forward-base",
            "pid": 0,
            "tid": 7,
            "ts": start + 55,
            "dur": 5,
            "args": {"External id": 5},
        }
    ]
    recompute = cpu(census.PREFIX + "MLP-adapter|recompute", start + 300, 100, 6, cat="user_annotation")
    recompute["tid"] = 15
    events.append(recompute)
    replay = cpu("aten::clone", start + 310, 10, 7, seq=5)
    replay["tid"] = 15
    events.append(replay)
    backward = cpu("autograd::engine::evaluate_function: CloneBackward0", start + 500, 100, 8, seq=5, fwd=1)
    backward["tid"] = 15
    events.append(backward)
    launch = cpu("aten::mm", start + 510, 10, 9)
    launch["tid"] = 15
    events.append(launch)
    events += [
        {
            "ph": "X",
            "cat": "kernel",
            "name": "backward-base",
            "pid": 0,
            "tid": 7,
            "ts": start + 520,
            "dur": 5,
            "args": {"External id": 9},
        }
    ]
    trace = {"schemaVersion": 1, "traceEvents": events}
    normalized = census.normalize_trace(trace, expected_steps=(step,))
    assert [(e["class"], e["path"]) for e in normalized["events"]] == [
        ("base NF4 GEMM", "q.base"),
        ("base NF4 GEMM", "q.base"),
    ]
    trace["traceEvents"].append(cpu("aten::clone", start + 150, 10, 10, seq=5))
    with pytest.raises(ValueError, match="ambiguous or missing backward creator"):
        census.normalize_trace(trace, expected_steps=(step,))


@pytest.mark.parametrize(
    "backward,outer,creator,scope,expected",
    [
        (
            "ScaledDotProductFlashAttentionBackward0",
            "aten::scaled_dot_product_attention",
            "aten::_scaled_dot_product_flash_attention",
            True,
            ("attention", "layer0"),
        ),
        ("DivBackward0", "aten::ne", "aten::div", False, ("other", "outside-registered-class")),
    ],
)
def test_real_l_capture_backward_creators_select_exact_nested_op(backward, outer, creator, scope, expected):
    start = 6 * 2000000
    events = [cpu(census.STEP_PREFIX + "6", start, 1000, 1, cat="user_annotation")]
    if scope:
        events.append(cpu(census.PREFIX + "attention|layer0", start + 10, 190, 2, cat="user_annotation"))
    events += [cpu(outer, start + 20, 100, 3, seq=5), cpu(creator, start + 30, 50, 4, seq=5)]
    events += [cpu(f"autograd::engine::evaluate_function: {backward}", start + 300, 100, 5, seq=5, fwd=1)]
    events += [cpu("aten::mm", start + 310, 10, 6)]
    events += [
        {
            "ph": "X",
            "cat": "kernel",
            "name": "backward-attention",
            "pid": 0,
            "tid": 7,
            "ts": start + 320,
            "dur": 5,
            "args": {"External id": 6},
        }
    ]
    normalized = census.normalize_trace({"schemaVersion": 1, "traceEvents": events}, expected_steps=(6,))
    assert [(e["class"], e["path"]) for e in normalized["events"]] == [expected]
    if backward == "DivBackward0":
        events.append(cpu(census.PREFIX + "attention|unexpected", start + 10, 190, 2, cat="user_annotation"))
        with pytest.raises(ValueError, match="changed its registered class scope"):
            census.normalize_trace({"schemaVersion": 1, "traceEvents": events}, expected_steps=(6,))


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_step", "missing captured"),
        ("unbound_kernel", "unbound device"),
        ("duplicate_cpu", "duplicate or malformed CPU"),
        ("duplicate_kernel", "duplicate physical"),
        ("cross_step", "crosses a synchronized"),
        ("ambiguous_class", "ambiguous equally"),
        ("missing_sequence", "unbound backward"),
        ("unscoped_rank", "unscoped rank16"),
        ("bad_schema", "unsupported Kineto"),
        ("negative_duration", "invalid duration"),
        ("boolean_timestamp", "invalid timestamp"),
        ("duplicate_step", "duplicate or unexpected"),
        ("bad_class", "malformed class"),
    ],
)
def test_corrupted_trace_refuses_instead_of_a_favorable_budget(mutation, match):
    d = fixture()
    events = d["traceEvents"]
    if mutation == "missing_step":
        events[:] = [e for e in events if e["name"] != census.STEP_PREFIX + "7"]
    elif mutation == "unbound_kernel":
        next(e for e in events if e["cat"] == "kernel")["args"]["External id"] = 999999
    elif mutation == "duplicate_cpu":
        events.append(copy.deepcopy(events[2]))
    elif mutation == "duplicate_kernel":
        events.append(copy.deepcopy(next(e for e in events if e["cat"] == "kernel")))
    elif mutation == "cross_step":
        e = next(e for e in events if e["cat"] == "kernel")
        e.update(ts=12000001, dur=1000000)
    elif mutation == "ambiguous_class":
        e = copy.deepcopy(events[1])
        e["name"] = census.PREFIX + "o-adapter|conflict"
        e["args"]["External id"] = 999
        events.append(e)
    elif mutation == "missing_sequence":
        next(e for e in events if e["name"].startswith("autograd::"))["args"]["Sequence number"] = 999
    elif mutation == "unscoped_rank":
        events[:] = [e for e in events if not e["name"].startswith(census.PREFIX + "MLP-adapter")]
    elif mutation == "bad_schema":
        d["schemaVersion"] = 2
    elif mutation == "negative_duration":
        events[1]["dur"] = -1
    elif mutation == "boolean_timestamp":
        events[1]["ts"] = True
    elif mutation == "duplicate_step":
        e = copy.deepcopy(events[0])
        e["args"]["External id"] = 999
        events.append(e)
    elif mutation == "bad_class":
        events[1]["name"] = census.PREFIX + "unknown|x"
    with pytest.raises(ValueError, match=match):
        census.normalize_trace(d)


def test_nested_base_not_charged_to_adapter_and_parent_operator_view_not_summed():
    d = fixture()
    events = d["traceEvents"]
    tag = events[1]
    inner = copy.deepcopy(tag)
    inner.update(ts=tag["ts"] + 0.5, dur=10, name=census.PREFIX + "base NF4 GEMM|nested-base")
    inner["args"]["External id"] = 999
    events.append(inner)
    # Extra op parent carrying CUDA-total in args must not count as a kernel.
    parent = copy.deepcopy(events[2])
    parent.update(ts=tag["ts"] + 0.6, dur=8, name="aten::matmul")
    parent["args"] = {"External id": 1000, "CUDA total": 1000000}
    events.append(parent)
    n = census.normalize_trace(d)
    assert n["events"][0]["class"] == "base NF4 GEMM"
    assert len(n["events"]) == 24


def test_overlapping_streams_same_and_mixed_classes_are_unioned_once():
    step = {"step": 6, "start_ns": 0, "end_ns": 100}

    def e(i, c, a, b):
        return dict(
            id=i, step=6, **{"class": c}, start_ns=a, end_ns=b, phase="forward", path="p", device_category="kernel"
        )

    row = census.partition(step, [e(1, "MLP-adapter", 10, 60), e(2, "MLP-adapter", 30, 80), e(3, "attention", 50, 90)])
    assert row["class_union_ns"]["MLP-adapter"] == 70
    assert row["kernel_duration_sum_ns"]["MLP-adapter"] == 100
    assert row["exclusive_ns"]["MLP-adapter"] == 40
    assert row["mixed_overlap_ns"] == 30
    assert row["host_or_no_device_ns"] == 20
    assert row["closure_residual_ns"] == 0


def test_cpu_op_parent_cuda_total_and_flow_metadata_are_never_device_time():
    d = fixture()
    d["traceEvents"].extend(
        [
            {"ph": "s", "cat": "ac2g", "name": "flow", "ts": 0, "args": {}},
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 12000020,
                "dur": 10,
                "pid": 1,
                "tid": 1,
                "args": {"correlation": 1},
            },
        ]
    )
    assert len(census.normalize_trace(d)["events"]) == 24


@pytest.mark.parametrize(
    "kind,expected", [("small", "CLOSE_BUILD"), ("qkv", "RANK_CANDIDATE_REVIEW"), ("tie", "TIE_REVIEW")]
)
def test_fixed_threshold_rank_or_close_and_minimum_of_three(kind, expected):
    n = census.normalize_trace(fixture())
    for e in n["events"]:
        if e["class"] == "MLP-adapter":
            e["end_ns"] = e["start_ns"] + 1000000
        if e["class"] == "qkv-adapter":
            e["end_ns"] = e["start_ns"] + (200000000 if kind in ("qkv", "tie") else 1000000)
        if e["class"] == "o-adapter":
            e["end_ns"] = e["start_ns"] + (200000000 if kind == "tie" else 1000000)
    result = census.reduce_events(n, [1.0] * 40)
    assert result["verdict"] == expected
    assert result["threshold_seconds"] == 0.1
    # One small third update must defeat a tempting pair of large readings.
    for e in n["events"]:
        if e["step"] == 8:
            e["end_ns"] = e["start_ns"] + 1000000
    assert census.reduce_events(n, [1.0] * 40)["verdict"] == "CLOSE_BUILD"


@pytest.mark.parametrize("bad", [[1.0] * 39, [1.0] * 41, [True] * 40, [float("nan")] * 40, [0.0] * 40])
def test_reference_missing_nonfinite_or_nonpositive_steps_refuse(bad):
    with pytest.raises(ValueError):
        census.reduce_events(census.normalize_trace(fixture()), bad)


def test_empty_or_duplicate_normalized_updates_refuse():
    n = census.normalize_trace(fixture())
    n["events"].append(n["events"][0])
    with pytest.raises(ValueError, match="duplicate"):
        census.reduce_events(n, [1.0] * 40)
    n = census.normalize_trace(fixture())
    n["events"] = [e for e in n["events"] if e["step"] != 7]
    with pytest.raises(ValueError, match="empty"):
        census.reduce_events(n, [1.0] * 40)


def test_cpu_real_autograd_scopes_restore_callables_and_are_gradient_neutral(tmp_path):
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 4)
            self.lora_A = torch.nn.Linear(4, 16, bias=False)
            self.lora_B = torch.nn.Linear(16, 4, bias=False)
            for p in self.base_layer.parameters():
                p.requires_grad_(False)

        def get_base_layer(self):
            return self.base_layer

        def forward(self, x):
            return self.base_layer(x) + self.lora_B(self.lora_A(x))

    model = Toy()
    key = "layers.0.self_attn.q_proj"
    slots = {key + ".A": model.lora_A.weight, key + ".B": model.lora_B.weight}
    modules = {key: model}
    x = torch.randn(3, 4, requires_grad=True)
    original = torch.nn.functional.scaled_dot_product_attention
    observer = Capture(model, slots, modules, tmp_path / "trace.json", cuda=False)
    with observer.installed():
        observer.begin_profile()
        observer.step = 0
        with torch.profiler.record_function(census.STEP_PREFIX + "0"):
            model(x).sum().backward()
        observer.end_profile()
    gradients = {k: p.grad.clone() for k, p in slots.items()}
    model.zero_grad(set_to_none=True)
    x.grad = None
    model(x).sum().backward()
    assert all(torch.equal(gradients[k], p.grad) for k, p in slots.items())
    assert observer.removed and not observer.handles and not observer.open_scopes
    assert torch.nn.functional.scaled_dot_product_attention is original
    assert observer.gradients["0"] == dict.fromkeys(slots, "torch.float32")
    # Actual2.12CPU trace proves sequence export shape; no scientific CUDA reading.
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert any(e.get("args", {}).get("Sequence number", -1) >= 0 for e in trace["traceEvents"])
    with pytest.raises(ValueError, match="no captured device events"):
        census.normalize_trace(trace, expected_steps=(0,))


def test_exception_removes_hooks_and_reporting_patches(tmp_path):
    import torch

    model = torch.nn.Linear(4, 4)
    original = torch.optim.AdamW.step
    capture = Capture(model, {}, {}, tmp_path / "unused", cuda=False)
    with pytest.raises(RuntimeError, match="fixture"), capture.installed():
        raise RuntimeError("fixture")
    assert torch.optim.AdamW.step is original and capture.removed


def identity_fixture():
    initial = dict.fromkeys(census.expected_parameters(), "a" * 64)
    seal = {n: "b" * 64 for n in ("dq11_l_census_run.py", "dq11_l_census_capture.py", "dq11_l_census_reduce.py")}
    runtime = {"experts4bit-qlora": census.SCIENCE, "loggetta": census.LOGGETTA}
    row = {
        "schema": "dq11-l-census-run/1",
        "kind": "reference",
        "science": census.SCIENCE,
        "loggetta": census.LOGGETTA,
        "torch": "2.12.1+cu130",
        "science_eligible": True,
        "profiler_removed": True,
        "frozen_unchanged": True,
        "status": "OK",
        "quality_pass": True,
        "input_sha256": census.INPUT_SHA,
        "model_sha256": census.MODEL_SHA,
        "science_manifest_sha256": census.SCIENCE_SHA,
        "nonce": "fixture",
        "source_authority_sha256": "c" * 64,
        "proof_sha256": "d" * 64,
        "initial": initial,
        "tokens": {},
        "runtime": runtime,
        "identity_sha256": census.object_sha({"initial": initial, "tokens": {}, "runtime": runtime}),
        "instrument": seal,
        "instrument_sha256": census.object_sha(seal),
        "path_before": {"fixture": True},
        "path_after": {"fixture": True},
        "capture_steps": [],
    }
    reference = copy.deepcopy(row)
    capture = copy.deepcopy(row)
    capture.update(kind="capture", capture_steps=[6, 7, 8])
    capture["gradients"] = {str(i): dict.fromkeys(initial, "torch.float32") for i in (6, 7, 8)}
    capture["calls"] = {str(i): dict.fromkeys({k.rsplit(".", 1)[0] for k in initial}, 2) for i in (6, 7, 8)}
    return reference, capture


def test_complete_source_identity_and_gradient_checkpoint_coverage():
    census.validate_identity(*identity_fixture())


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_source",
        "wrong_input_both",
        "wrong_model_both",
        "wrong_instrument",
        "initializer",
        "missing_gradient",
        "wrong_dtype",
        "missing_recompute",
        "capture_window",
        "reference_profiled",
        "rehearsal_in_science",
        "changed_binding",
    ],
)
def test_identity_and_full_engagement_mutants_refuse(mutation):
    reference, capture = identity_fixture()
    if mutation == "wrong_source":
        capture["science"] = "d" * 40
    elif mutation == "wrong_input_both":
        reference["input_sha256"] = capture["input_sha256"] = "d" * 64
    elif mutation == "wrong_model_both":
        reference["model_sha256"] = capture["model_sha256"] = "d" * 64
    elif mutation == "wrong_instrument":
        capture["instrument"]["dq11_l_census_run.py"] = "d" * 64
    elif mutation == "initializer":
        capture["initial"].pop(next(iter(capture["initial"])))
    elif mutation == "missing_gradient":
        capture["gradients"]["7"].pop(next(iter(capture["gradients"]["7"])))
    elif mutation == "wrong_dtype":
        capture["gradients"]["8"][next(iter(capture["gradients"]["8"]))] = "torch.bfloat16"
    elif mutation == "missing_recompute":
        capture["calls"]["6"][next(iter(capture["calls"]["6"]))] = 1
    elif mutation == "capture_window":
        capture["capture_steps"] = [6, 7, 9]
    elif mutation == "reference_profiled":
        reference["capture_steps"] = [6]
    elif mutation == "rehearsal_in_science":
        capture["science_eligible"] = False
        capture["torch"] = "2.11.0+cu128"
    elif mutation == "changed_binding":
        capture["path_after"] = {"changed": True}
    with pytest.raises(ValueError):
        census.validate_identity(reference, capture)


def test_rehearsal_cannot_implicitly_enter_scientific_reducer():
    reference, capture = identity_fixture()
    for row in (reference, capture):
        row.update(science_eligible=False, torch="2.11.0+cu128")
    with pytest.raises(ValueError):
        census.validate_identity(reference, capture)
    census.validate_identity(reference, capture, rehearsal=True)


def test_clock_captures_only_registered_updates_and_restores_trainer_global(tmp_path):
    import torch

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 4)
            self.lora_A = torch.nn.Linear(4, 16, bias=False)
            self.lora_B = torch.nn.Linear(16, 4, bias=False)
            for p in self.base_layer.parameters():
                p.requires_grad_(False)

        def get_base_layer(self):
            return self.base_layer

        def forward(self, x):
            return self.base_layer(x) + self.lora_B(self.lora_A(x))

    model = Adapter()
    key = "layers.0.mlp.down_proj"
    slots = {key + ".A": model.lora_A.weight, key + ".B": model.lora_B.weight}
    modules = {key: model}

    capture = Capture(model, slots, modules, tmp_path / "clock.json", cuda=False)
    original_time = time
    try:
        with capture.installed(), capture.trainer_clock(_clock_fixture_loop):
            _clock_fixture_loop(model, slots)
        assert _clock_fixture_loop.__globals__["time"] is original_time
    finally:
        _clock_fixture_loop.__globals__["time"] = original_time
    assert set(capture.gradients) == {"6", "7", "8"}
    assert all(v == {key: 2} for v in capture.calls.values())
    assert capture.removed and not capture.open_scopes
    trace = json.loads((tmp_path / "clock.json").read_text())
    assert {e["name"] for e in trace["traceEvents"] if e.get("name", "").startswith(census.STEP_PREFIX)} == {
        census.STEP_PREFIX + str(i) for i in (6, 7, 8)
    }


def _clock_fixture_loop(model, slots):
    import torch
    from torch.utils.checkpoint import checkpoint

    opt = torch.optim.AdamW(list(slots.values()), lr=2e-4)
    for _ in range(40):
        start = time.time()
        opt.zero_grad(set_to_none=True)
        x = torch.ones(3, 4, requires_grad=True)
        checkpoint(model, x, use_reentrant=False).sum().backward()
        torch.nn.utils.clip_grad_norm_(list(slots.values()), 1.0, error_if_nonfinite=True)
        opt.step()
        _elapsed = time.time() - start
