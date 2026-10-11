"""CPU-only, kernel-view census: correlated classes, interval unions, fixed decision."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

SCIENCE = "13b55990433f357d63232f75926ce19f5bda5b42"
LOGGETTA = "34ecb6cec6f43a6f8607ff9f192749fdc7b587e9"
CLASSES = ("MLP-adapter", "qkv-adapter", "o-adapter", "base NF4 GEMM", "attention", "optimizer", "other")
INPUT_SHA = "9937024a50d3d44b28016d47b85e3e2feba159ce9d50f82403120e17581a4144"
MODEL_SHA = "c21aac15939dd1c6a8c8c367b2887c44668335927f81368afa996690470a0ff5"
SCIENCE_SHA = "4691e7329f12ad2f4b7518b75c4d3e351207fc4e5220aee514fdd42c429fef01"
STEPS = (6, 7, 8)
PREFIX = "dq11.census.class:"
STEP_PREFIX = "dq11.census.step:"
DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
MAX_TRACE_BYTES = 512 * 1024 * 1024
MAX_EVENTS = 2000000
# Kineto's sequence counter also appears on nested operators and on
# checkpoint recomputation in the autograd worker thread. Resolve the one
# forward operator that CREATED each backward node, on its recorded forward
# thread. The table includes observed real-L A2000 nodes and prospective
# native SDPA variants for the 5090, all fixed before deciding data. An
# unrecognized node or multiple matching creators refuse certification.
BACKWARD_CREATOR = {
    "AddBackward0": ("aten::add", "aten::add_"),
    "AliasBackward0": ("aten::alias",),
    "BackwardHookFunctionBackward": ("BackwardHookFunction",),
    "BmmBackward0": ("aten::bmm",),
    "CatBackward0": ("aten::cat",),
    "CloneBackward0": ("aten::clone",),
    "DivBackward0": ("aten::div",),
    "ExpandBackward0": ("aten::expand",),
    "LogSoftmaxBackward0": ("aten::_log_softmax",),
    "MeanBackward1": ("aten::mean",),
    "MmBackward0": ("aten::mm",),
    "MulBackward0": ("aten::mul",),
    "MulBackward1": ("aten::mul",),
    "NegBackward0": ("aten::neg",),
    "NllLossBackward0": ("aten::nll_loss_forward",),
    "PowBackward0": ("aten::pow",),
    "ReshapeAliasBackward0": ("aten::_reshape_alias",),
    "RsqrtBackward0": ("aten::rsqrt",),
    "ScaledDotProductFlashAttentionBackward0": ("aten::_scaled_dot_product_flash_attention",),
    "ScaledDotProductEfficientAttentionBackward0": ("aten::_scaled_dot_product_efficient_attention",),
    "ScaledDotProductCudnnAttentionBackward0": ("aten::_scaled_dot_product_cudnn_attention",),
    "SafeSoftmaxBackward0": ("aten::_safe_softmax",),
    "SiluBackward0": ("aten::silu",),
    "SliceBackward0": ("aten::slice",),
    "TBackward0": ("aten::t",),
    "ToCopyBackward0": ("aten::_to_copy",),
    "TransposeBackward0": ("aten::transpose",),
    "UnsafeViewBackward0": ("aten::_unsafe_view",),
    "UnsqueezeBackward0": ("aten::unsqueeze",),
    "ViewBackward0": ("aten::view",),
    "_LateBoundMatMul4BitBackward": ("_LateBoundMatMul4Bit",),
}
BACKWARD_EXPECTED_CLASS = {
    "ScaledDotProductFlashAttentionBackward0": "attention",
    "ScaledDotProductEfficientAttentionBackward0": "attention",
    "ScaledDotProductCudnnAttentionBackward0": "attention",
    "BmmBackward0": "attention",
    "SafeSoftmaxBackward0": "attention",
    "DivBackward0": "other",
}


def fail(message):
    raise ValueError("CENSUS_REFUSAL: " + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        fail("invalid " + name)
    return value


def interval(event, *, allow_zero=False):
    # Kineto emits microseconds; normalize individual timestamps to integer ns.
    start = finite(event.get("ts"), "timestamp")
    duration = finite(event.get("dur"), "duration", positive=not allow_zero)
    if duration < 0:
        fail("invalid duration")
    ns = lambda x: int((Decimal(str(x)) * 1000).to_integral_value(rounding=ROUND_HALF_EVEN))
    a, width = ns(start), ns(duration)
    if width < 0 or (width == 0 and not allow_zero):
        fail("event rounds to an empty interval")
    return a, a + width


def class_range(name):
    if not name.startswith(PREFIX):
        return None
    fields = name[len(PREFIX) :].split("|", 1)
    if len(fields) != 2 or fields[0] not in CLASSES or not fields[1]:
        fail("malformed class range")
    return tuple(fields)


def contains(outer, inner):
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def normalize_trace(trace, *, expected_steps=STEPS):
    """External-id kernel binding and autograd sequence mapping; no name classifier."""
    if trace.get("schemaVersion") != 1 or not isinstance(trace.get("traceEvents"), list):
        fail("unsupported Kineto schema")
    events = trace["traceEvents"]
    if len(events) > MAX_EVENTS:
        fail("event cap exceeded")
    cpu, device, ranges, steps = {}, [], [], {}
    cpu_list = []
    for index, event in enumerate(events):
        if event.get("ph") != "X":
            continue
        cat, name = event.get("cat"), event.get("name", "")
        if cat not in DEVICE_CATEGORIES | {"cpu_op", "user_annotation"}:
            # Runtime/flow/metadata are not a second view of the device budget.
            continue
        bounds = interval(event, allow_zero=cat not in DEVICE_CATEGORIES)
        args = event.get("args", {})
        if not isinstance(args, dict):
            fail("malformed event arguments")
        if cat in DEVICE_CATEGORIES:
            device.append((index, event, bounds))
            continue
        row = {"event": event, "bounds": bounds, "index": index}
        cpu_list.append(row)
        external = args.get("External id")
        if external is not None:
            if type(external) is not int or external <= 0 or external in cpu:
                fail("duplicate or malformed CPU external id")
            cpu[external] = row
        tag = class_range(name)
        if tag:
            ranges.append({**row, "tag": tag})
        if name.startswith(STEP_PREFIX):
            try:
                number = int(name[len(STEP_PREFIX) :])
            except ValueError:
                fail("malformed step range")
            if number not in expected_steps or number in steps:
                fail("duplicate or unexpected captured step")
            steps[number] = bounds
    if set(steps) != set(expected_steps):
        fail("missing captured step")
    if not device:
        fail("no captured device events")
    ordered = sorted(steps.items())
    if any(a[1][1] > b[1][0] for a, b in itertools.pairwise(ordered)):
        fail("overlapping or out-of-order step brackets")

    # Build parent chains once; scanning every CPU event for every kernel is
    # quadratic at full-model trace sizes. Kineto CPU ranges nest per thread.
    by_thread = defaultdict(list)
    for row in cpu_list:
        event = row["event"]
        by_thread[(event["pid"], event["tid"])].append(row)
    parents = {}
    for rows in by_thread.values():
        stack = []
        for row in sorted(rows, key=lambda r: (r["bounds"][0], -r["bounds"][1], r["index"])):
            while stack and stack[-1]["bounds"][1] <= row["bounds"][0]:
                stack.pop()
            if stack and not contains(stack[-1]["bounds"], row["bounds"]):
                fail("crossing CPU ranges")
            parents[row["index"]] = stack[-1] if stack else None
            stack.append(row)

    def ancestors(row):
        while row is not None:
            yield row
            row = parents[row["index"]]

    def direct(row):
        tags = [(r["bounds"], class_range(r["event"]["name"])) for r in ancestors(row)]
        tags = [(bounds, tag) for bounds, tag in tags if tag is not None]
        if not tags:
            return None
        if len({tag for bounds, tag in tags if bounds == tags[0][0]}) != 1:
            fail("ambiguous equally nested class ranges")
        return tags[0][1]

    def backward(row):
        for candidate in ancestors(row):
            event = candidate["event"]
            if (
                event["name"].startswith("autograd::engine::evaluate_function:")
                or event.get("args", {}).get("Fwd thread id", 0) > 0
            ):
                return candidate
        return None

    # The same sequence is visible on unrelated nested ops and on recompute
    # in a different thread. A backward node's Fwd thread id plus its exact
    # creator op distinguishes those views without assigning by kernel name.
    forward, backward_nodes, sequence = defaultdict(list), [], {}
    for row in cpu_list:
        event = row["event"]
        seq = event.get("args", {}).get("Sequence number")
        if type(seq) is not int or seq < 0:
            continue
        if event["name"].startswith("autograd::engine::evaluate_function: "):
            backward_nodes.append(row)
        elif event.get("args", {}).get("Fwd thread id", 0) == 0:
            forward[(event["pid"], event["tid"], seq)].append(row)
    if not backward_nodes:
        fail("missing backward sequence nodes")
    for node in backward_nodes:
        event = node["event"]
        name = event["name"].split(": ", 1)[1]
        seq = event["args"]["Sequence number"]
        fwd_thread = event["args"].get("Fwd thread id")
        if type(fwd_thread) is not int or fwd_thread <= 0:
            fail("backward node lacks forward thread identity")
        key = (event["pid"], fwd_thread, seq)
        if key in sequence:
            fail("duplicate backward sequence node")
        if name == "torch::autograd::AccumulateGrad":
            # Parameter-gradient hooks provide the direct class scope;
            # AccumulateGrad has no forward creator operator.
            sequence[key] = None
            continue
        if name not in BACKWARD_CREATOR:
            fail("unregistered backward node kind")
        if not forward[key]:
            fail("unbound backward sequence")
        matches = [row for row in forward[key] if row["event"]["name"] in BACKWARD_CREATOR[name]]
        if len(matches) != 1:
            fail("ambiguous or missing backward creator")
        tag = direct(matches[0]) or ("other", "outside-registered-class")
        if name in BACKWARD_EXPECTED_CLASS and tag[0] != BACKWARD_EXPECTED_CLASS[name]:
            fail("backward creator changed its registered class scope")
        sequence[key] = tag
    output, device_keys = [], set()
    for index, event, bounds in device:
        matches = [n for n, b in steps.items() if contains(b, bounds)]
        if not matches:
            if any(bounds[0] < b[1] and b[0] < bounds[1] for b in steps.values()):
                fail("device event crosses a synchronized step bracket")
            continue  # profiler start/stop bookkeeping outside captured updates
        if len(matches) != 1:
            fail("device interval belongs to multiple steps")
        external = event.get("args", {}).get("External id")
        if type(external) is not int or external not in cpu:
            fail("unbound device external id")
        origin = cpu[external]
        if not contains(steps[matches[0]], origin["bounds"]):
            fail("launch operation outside its device event update")
        key = (
            event.get("pid"),
            event.get("tid"),
            bounds,
            event["cat"],
            event.get("args", {}).get("correlation"),
            external,
        )
        if key in device_keys:
            fail("duplicate physical device interval")
        device_keys.add(key)
        tag = direct(origin)
        phase = "forward-or-other"
        node = backward(origin)
        if tag is None and node is not None:
            seq = node["event"].get("args", {}).get("Sequence number")
            fwd_thread = node["event"].get("args", {}).get("Fwd thread id")
            key = (node["event"]["pid"], fwd_thread, seq)
            if type(seq) is not int or key not in sequence or sequence[key] is None:
                fail("unbound backward sequence")
            tag = sequence[key]
            phase = "backward"
        elif node is not None:
            phase = "backward-or-recompute"
        if tag is None:
            tag = ("other", "outside-registered-class")
        # Unscoped rank16 GEMMs must never quietly disappear into 'other'.
        dims = origin["event"].get("args", {}).get("Input Dims", [])
        if (
            tag[0] == "other"
            and origin["event"]["name"] in ("aten::mm", "aten::matmul", "aten::linear", "aten::addmm")
            and any(isinstance(shape, list) and len(shape) == 2 and 16 in shape for shape in dims)
        ):
            fail("unscoped rank16 adapter operation")
        output.append(
            {
                "id": index,
                "step": matches[0],
                "class": tag[0],
                "path": tag[1],
                "phase": phase,
                "start_ns": bounds[0],
                "end_ns": bounds[1],
                "external_id": external,
                "device_category": event["cat"],
                "kernel_name": event["name"],
                "launch_operator": origin["event"]["name"],
                "stream": event.get("args", {}).get("stream"),
            }
        )
    if not output:
        fail("no captured device events")
    return {
        "schema": "dq11-l-census-events/1",
        "steps": [{"step": n, "start_ns": b[0], "end_ns": b[1]} for n, b in ordered],
        "events": output,
        "sequence_classes": {"|".join(map(str, k)): list(v) if v else None for k, v in sequence.items()},
        "raw_device_events": len(device),
        "captured_device_events": len(output),
    }


def partition(step, events):
    a, b = step["start_ns"], step["end_ns"]
    if type(a) is not int or type(b) is not int or b <= a:
        fail("invalid normalized step bounds")
    changes = defaultdict(list)
    seen = set()
    sums = dict.fromkeys(CLASSES, 0)
    detail = defaultdict(lambda: {"duration_ns": 0, "count": 0})
    for event in events:
        if event["id"] in seen or event["class"] not in CLASSES:
            fail("duplicate device event or unknown class")
        seen.add(event["id"])
        x, y, cls = event["start_ns"], event["end_ns"], event["class"]
        if type(x) is not int or type(y) is not int or not (a <= x < y <= b):
            fail("normalized device event outside step")
        changes[x].append((cls, 1))
        changes[y].append((cls, -1))
        sums[cls] += y - x
        key = (cls, event["phase"], event["path"], event["device_category"])
        detail[key]["duration_ns"] += y - x
        detail[key]["count"] += 1
    exclusive, unions = dict.fromkeys(CLASSES, 0), dict.fromkeys(CLASSES, 0)
    overlap, idle, previous = 0, 0, a
    counts = defaultdict(int)
    for point in sorted(set(changes) | {a, b}):
        duration = point - previous
        active = {k for k, v in counts.items() if v > 0}
        if not active:
            idle += duration
        elif len(active) == 1:
            exclusive[next(iter(active))] += duration
        else:
            overlap += duration
        for cls in active:
            unions[cls] += duration
        for cls, delta in changes[point]:
            counts[cls] += delta
            if counts[cls] < 0:
                fail("negative interval sweep count")
        previous = point
    total = sum(exclusive.values()) + overlap + idle
    if total != b - a or any(counts.values()):
        fail("interval partition does not close")
    return {
        "step": step["step"],
        "wall_ns": b - a,
        "exclusive_ns": exclusive,
        "mixed_overlap_ns": overlap,
        "host_or_no_device_ns": idle,
        "closure_residual_ns": (b - a) - total,
        "class_union_ns": unions,
        "kernel_duration_sum_ns": sums,
        "event_count": len(events),
        "detail": [
            {"class": k[0], "phase": k[1], "path": k[2], "device_category": k[3], **v}
            for k, v in sorted(detail.items())
        ],
    }


def reduce_events(normalized, reference_seconds):
    if normalized.get("schema") != "dq11-l-census-events/1" or [s["step"] for s in normalized["steps"]] != list(STEPS):
        fail("normalized captured updates changed")
    if len(reference_seconds) != 40:
        fail("reference does not contain forty updates")
    for value in reference_seconds:
        finite(value, "reference step", positive=True)
    T = statistics.median(reference_seconds[5:])
    ids = [e["id"] for e in normalized["events"]]
    if len(ids) != len(set(ids)) or any(e["step"] not in STEPS for e in normalized["events"]):
        fail("duplicate or unexpected normalized event")
    rows = [partition(s, [e for e in normalized["events"] if e["step"] == s["step"]]) for s in normalized["steps"]]
    if any(r["event_count"] == 0 for r in rows):
        fail("empty captured update")
    ceilings = {c: min(r["class_union_ns"][c] for r in rows) / 1e9 for c in CLASSES[:3]}
    candidates = sorted(ceilings, key=lambda c: (-ceilings[c], c))
    threshold = 0.10 * T
    if ceilings["MLP-adapter"] >= threshold:
        verdict, selected = "BUILD_MLP", "MLP-adapter"
    elif ceilings[candidates[0]] >= threshold:
        tied = len(candidates) > 1 and ceilings[candidates[0]] == ceilings[candidates[1]]
        verdict, selected = ("TIE_REVIEW", None) if tied else ("RANK_CANDIDATE_REVIEW", candidates[0])
    else:
        verdict, selected = "CLOSE_BUILD", None
    return {
        "schema": "dq11-l-census-decision/1",
        "verdict": verdict,
        "selected": selected,
        "reference_median_seconds": T,
        "threshold_seconds": threshold,
        "candidate_min_union_seconds": ceilings,
        "candidate_rank": candidates,
        "updates": rows,
        "scope": "addressable device time only; no end-to-end speed claim",
        "mean_profile_wall_ns": statistics.mean(r["wall_ns"] for r in rows),
        "mean_exclusive_ns": {c: statistics.mean(r["exclusive_ns"][c] for r in rows) for c in CLASSES},
        "mean_mixed_overlap_ns": statistics.mean(r["mixed_overlap_ns"] for r in rows),
        "mean_host_or_no_device_ns": statistics.mean(r["host_or_no_device_ns"] for r in rows),
    }


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expected_parameters():
    return {
        f"layers.{i}.{part}.{proj}.{factor}"
        for i in range(32)
        for part, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for proj in projections
        for factor in ("A", "B")
    }


def validate_identity(reference, capture, *, rehearsal=False):
    for row, kind in ((reference, "reference"), (capture, "capture")):
        if (
            row.get("schema") != "dq11-l-census-run/1"
            or row.get("kind") != kind
            or row.get("science") != SCIENCE
            or row.get("loggetta") != LOGGETTA
            or row.get("torch") != ("2.11.0+cu128" if rehearsal else "2.12.1+cu130")
            or row.get("science_eligible") is not (not rehearsal)
            or row.get("profiler_removed") is not True
            or row.get("frozen_unchanged") is not True
            or row.get("status") != "OK"
            or row.get("quality_pass") is not True
            or row.get("input_sha256") != INPUT_SHA
            or row.get("model_sha256") != MODEL_SHA
            or row.get("science_manifest_sha256") != SCIENCE_SHA
            or row.get("path_before") != row.get("path_after")
        ):
            fail("unregistered or incomplete census identity")
        for field in (
            "input_sha256",
            "instrument_sha256",
            "source_authority_sha256",
            "identity_sha256",
            "proof_sha256",
        ):
            if not isinstance(row.get(field), str) or not re.fullmatch(r"[a-f0-9]{64}", row[field]):
                fail("invalid identity field " + field)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", row.get("nonce", "")):
            fail("invalid fresh nonce")
        names = {"dq11_l_census_run.py", "dq11_l_census_capture.py", "dq11_l_census_reduce.py"}
        if (
            set(row.get("instrument", {})) != names
            or any(not re.fullmatch(r"[a-f0-9]{64}", v) for v in row["instrument"].values())
            or object_sha(row["instrument"]) != row["instrument_sha256"]
        ):
            fail("instrument seal mismatch")
        if (
            set(row.get("initial", {})) != expected_parameters()
            or row["identity_sha256"]
            != object_sha({"initial": row["initial"], "tokens": row["tokens"], "runtime": row["runtime"]})
            or row["runtime"].get("experts4bit-qlora") != SCIENCE
            or row["runtime"].get("loggetta") != LOGGETTA
        ):
            fail("canonical source/initializer/runtime identity mismatch")
    for field in (
        "nonce",
        "input_sha256",
        "instrument_sha256",
        "source_authority_sha256",
        "identity_sha256",
        "path_before",
        "proof_sha256",
    ):
        if reference[field] != capture[field]:
            fail("reference/capture identity differs: " + field)
    expected = expected_parameters()
    projections = {key.rsplit(".", 1)[0] for key in expected}
    for step in STEPS:
        gradients = capture.get("gradients", {}).get(str(step), {})
        if set(gradients) != expected or any(v != "torch.float32" for v in gradients.values()):
            fail("missing/non-FP32 captured gradient")
        calls = capture.get("calls", {}).get(str(step), {})
        if set(calls) != projections or any(type(v) is not int or v != 2 for v in calls.values()):
            fail("missing checkpoint forward/recompute call")
    if capture.get("capture_steps") != list(STEPS) or reference.get("capture_steps") != []:
        fail("capture window or reference observer changed")


def report_files(args):
    if args.trace.stat().st_size > MAX_TRACE_BYTES or args.proof_trace.stat().st_size > MAX_TRACE_BYTES:
        fail("trace byte cap exceeded")
    reference, capture, proof = (json.loads(p.read_text()) for p in (args.reference, args.capture, args.proof))
    validate_identity(reference, capture, rehearsal=args.rehearsal)
    from dq11_reduce import quality, validate_observer_policy

    if (
        proof.get("kind") != "proof"
        or proof.get("observer_same_arm_bitwise") is not True
        or proof.get("cuda_canary_device_events", 0) <= 0
        or proof.get("profiler_removed") is not True
        or proof.get("science_eligible") is not (not args.rehearsal)
        or reference.get("proof_sha256") != sha(args.proof)
        or proof.get("trace_sha256") != sha(args.proof_trace)
        or any(
            proof.get(k) != reference[k]
            for k in ("nonce", "identity_sha256", "instrument_sha256", "source_authority_sha256", "path_before")
        )
    ):
        fail("fresh GPU correlation/C bitwise proof missing or mismatched")
    validate_observer_policy(proof["observer_policy"])
    canary = normalize_trace(json.loads(args.proof_trace.read_text()), expected_steps=(0,))
    if canary["captured_device_events"] != proof["cuda_canary_device_events"]:
        fail("GPU correlation canary event count changed")
    if not all(quality(reference, capture, field) for field in ("initial_quality", "final_quality")):
        fail("unchanged K8 refused reference/capture quality")
    if capture.get("trace_sha256") != sha(args.trace):
        fail("raw trace hash differs from capture")
    normalized = normalize_trace(json.loads(args.trace.read_text()))
    result = reduce_events(normalized, reference["training"]["measured"]["step_seconds"])
    if args.rehearsal:
        result.update(
            verdict="CORRECTNESS_REHEARSAL_ONLY",
            selected=None,
            science_eligible=False,
            interpretation="A2000 real-L capture/correlation/closure; no BUILD or timing inference",
        )
    result["raw_sha256"] = {
        "trace": sha(args.trace),
        "reference": sha(args.reference),
        "capture": sha(args.capture),
        "proof": sha(args.proof),
        "proof_trace": sha(args.proof_trace),
    }
    return result, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "reference", "capture", "proof", "proof-trace", "normalized-out", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rehearsal", action="store_true")
    args = parser.parse_args()
    if args.out.exists() or args.normalized_out.exists():
        fail("refuse overwrite of census evidence")
    try:
        result, normalized = report_files(args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = {
            "schema": "dq11-l-census-decision/1",
            "verdict": "CENSUS_REFUSAL",
            "selected": None,
            "reason": str(exc),
            "science_eligible": not args.rehearsal,
        }
        with args.out.open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        raise SystemExit(1) from exc
    with args.normalized_out.open("x") as output:
        output.write(json.dumps(normalized, indent=2) + "\n")
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
