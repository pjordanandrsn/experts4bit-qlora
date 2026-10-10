"""Quality-before-decision DQ11 reducer; no verdict is final without guard teardown."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics

from dq11_common import HERE, file_sha, object_sha, value_identity

SCOPE = "against Unsloth in this matched, eager, full-logits configuration, not Unsloth's default"
ORDER = ((1, "L"), (1, "U"), (1, "U0"), (2, "U0"), (2, "U"), (2, "L"))
TEXTS = ("alpaca-heldout", "wikitext-test")


def registered_identity(row, *, rehearsal=False):
    if rehearsal:
        from dq11_rehearsal import mode

        if not mode(HERE) or row.get("schema") != "dq11-rehearsal-arm/1" or row.get("science_eligible") is not False:
            raise ValueError("unmarked or unauthorized correctness rehearsal")
    elif row.get("schema", "dq11-arm/1") != "dq11-arm/1" or row.get("science_eligible") is False:
        raise ValueError("rehearsal receipts cannot enter science reduction")
    locked = json.loads((HERE / "locked_inputs.json").read_text())
    wheels = json.loads((HERE / "wheels.json").read_text())["packages"]
    if (row["tokens"] != locked["tokens"] or row["input_seal_sha256"] != file_sha(HERE / "locked_inputs.json")
            or any(row["runtime"].get(w["name"]) != w["version"] for w in wheels)
            or row["runtime"].get("loggetta") != "34ecb6cec6f43a6f8607ff9f192749fdc7b587e9"
            or not re.fullmatch("[0-9a-f]{40}", row["runtime"].get("experts4bit-qlora", ""))):
        raise ValueError("registered input/runtime identity changed")
    for name in ("science.sha256", "wheels.json", "model_files.json"):
        if row["provenance"][name] != file_sha(HERE / name):
            raise ValueError("registered closure changed")
    if not re.fullmatch("[0-9a-f]{64}", row["provenance"]["source_authority.json"]):
        raise ValueError("source authority is missing")
    expected = object_sha({"initial": row["initial"], "tokens": row["tokens"], "runtime": row["runtime"]})
    if row["identity_sha256"] != expected:
        raise ValueError("receipt identity hash is inconsistent")


def initial_gate(proofs):
    if len(proofs) != 3 or {row["arm"] for row in proofs} != {"L", "U", "U0"}:
        raise ValueError("missing or duplicate initial proof")
    by_arm = {row["arm"]: row for row in proofs}
    return quality(by_arm["L"], by_arm["U"], "initial_quality") and quality(by_arm["U0"], by_arm["U"], "initial_quality")


def quality(base, candidate, field):
    from experts4bit_qlora.k8_gate import Arm, verdict

    pairs = []
    for text in TEXTS:
        b, c = base[field][text], candidate[field][text]
        if b["targets"] != 16376 or c["targets"] != 16376:
            raise ValueError("quality target count changed")
        if base["tokens"][text] != candidate["tokens"][text]:
            raise ValueError("quality tokens differ")
        for row in (b, c):
            if not all(isinstance(row[key], (float, int)) and not isinstance(row[key], bool)
                       and math.isfinite(row[key]) for key in ("ppl", "nll")):
                raise ValueError("nonfinite/malformed quality")
            if row["ppl"] <= 0 or row["nll"] < 0 or not math.isclose(row["ppl"], math.exp(row["nll"] / row["targets"]), rel_tol=1e-12):
                raise ValueError("quality PPL does not follow NLL/target count")
        steps = 0 if field == "initial_quality" else 40
        sha = base["tokens"][text]["sha256"]
        pairs.append((Arm(b["ppl"], sha, steps, text), Arm(c["ppl"], sha, steps, text)))
    return verdict(pairs, calibrated=False)[0]


def validate_observer_policy(policy):
    from dq11_proof_policy import CONFIGURATION_C

    expected = {"cublas_workspace_config": ":4096:8", "deterministic_algorithms": True,
                "warn_only": True, "flash_sdp": False, "mem_efficient_sdp": False,
                "math_sdp": True, "cudnn_sdp": False, "tf32_matmul": False, "tf32_cudnn": False}
    if (any(type(policy["settings_actual"].get(key)) is not bool for key in expected if key != "cublas_workspace_config")
            or type(policy["configuration"].get("deterministic_algorithms")) is not bool
            or type(policy["configuration"].get("warn_only")) is not bool
            or policy["configuration"] != CONFIGURATION_C or policy["settings_actual"] != expected
            or policy["restored"] is not True or not isinstance(policy["warnings"], list)
            or not isinstance(policy["warned_ops"], list)):
        raise ValueError("observer proof did not use configuration C or restore its policy")
    for warning in policy["warnings"]:
        if not all(isinstance(warning[key], str) for key in ("category", "message", "filename")):
            raise ValueError("malformed observer warning")
    expected_messages = [w["message"] for w in policy["warnings"] if "determin" in w["message"].lower()]
    if ([w["message"] for w in policy["warned_ops"]] != expected_messages
            or any(not isinstance(w["operator"], str) or not w["operator"] for w in policy["warned_ops"])):
        raise ValueError("observer determinism warning operators were omitted")


def validate_spread(row, proof, *, rehearsal=False):
    registered_identity(row, rehearsal=rehearsal)
    if (row["kind"] != "spread" or row["repetition"] != 0 or row["arm"] != proof["arm"]
            or not row["frozen_unchanged"] or row["path_before"] != row["path_after"]):
        raise ValueError("shipped spread binding changed")
    for key in ("runtime", "initial", "tokens", "nonce", "input_seal_sha256", "identity_sha256",
                "base", "provenance", "path_before"):
        if row[key] != proof[key]:
            raise ValueError("shipped spread differs from proof: " + key)
    spread = row["spread"]
    if (spread["schema"] != "dq11-shipped-spread/1" or spread["train_block"] != 0
            or spread["optimizer_updates"] != 0 or spread["gate"] is not False
            or spread["tolerance"] is not None):
        raise ValueError("spread must be zero-update reporting without a tolerance or gate")
    setting = spread["settings"]
    if row["execution_settings"] != setting:
        raise ValueError("shipped spread execution-setting record is inconsistent")
    if (setting["cublas_workspace_config"] is not None or setting["deterministic_algorithms"] is not False
            or setting["tf32_matmul"] is not False or setting["tf32_cudnn"] is not False):
        raise ValueError("spread execution settings changed")
    if [p["ordinal"] for p in spread["passes"]] != [1, 2, 3, 4]:
        raise ValueError("incomplete shipped spread passes")
    for run in spread["passes"]:
        if (run["gradient_file"] != f"spread-{row['arm']}-{run['ordinal']}.safetensors"
                or type(run["gradient_file_bytes"]) is not int or run["gradient_file_bytes"] <= 0
                or not re.fullmatch("[0-9a-f]{64}", run["gradient_file_sha256"])):
            raise ValueError("invalid raw shipped spread gradient file identity")
        if (not isinstance(run["loss"], (float, int)) or isinstance(run["loss"], bool)
                or not math.isfinite(run["loss"]) or not re.fullmatch("[0-9a-f]{64}", run["loss_sha256"])
                or set(run["gradient_hashes"]) != set(proof["initial"])
                or any(not re.fullmatch("[0-9a-f]{64}", h) for h in run["gradient_hashes"].values())):
            raise ValueError("invalid shipped spread loss or gradient hashes")
    if [(p["x"], p["y"]) for p in spread["pairs"]] != [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
        raise ValueError("incomplete shipped spread pairs")
    for pair in spread["pairs"]:
        x, y = (spread["passes"][pair[k] - 1] for k in ("x", "y"))
        expected_keys = {key for key in proof["initial"] if x["gradient_hashes"][key] != y["gradient_hashes"][key]}
        if (set(pair["gradient_differences"]) != expected_keys
                or pair["loss_bitwise_equal"] != (x["loss_sha256"] == y["loss_sha256"])
                or pair["loss_abs_difference"] != abs(x["loss"] - y["loss"])):
            raise ValueError("shipped spread pair inconsistent with passes")
        for difference in pair["gradient_differences"].values():
            for key in ("max_abs_difference", "relative_to_larger_max"):
                value = difference[key]
                if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError("nonfinite/malformed shipped spread difference")
        # No comparison against a magnitude threshold: spread is reported, never a quality/headroom gate.


def report_spreads(proofs, reads, spreads, *, rehearsal=False):
    """Optional reports cannot invalidate science, including missing or malformed raw evidence."""
    reports = {}
    for proof in proofs:
        arm = proof["arm"]
        entry = spreads.get(arm)
        if entry is None:
            reports[arm] = {"status": "incomplete", "reason": "no spread receipt or process status",
                            "policy_reference": "registered shipped policy"}
            continue
        if isinstance(entry, dict) and entry.get("status") in {"spread: skipped for time", "incomplete"}:
            reports[arm] = entry
            continue
        try:
            validate_spread(entry, proof, rehearsal=rehearsal)
            for reading in reads:
                if reading["arm"] == arm and reading["execution_settings"] != entry["spread"]["settings"]:
                    raise ValueError("read policy differs from recorded spread policy")
            reports[arm] = {"status": "complete", **entry["spread"], "policy_reference": "recorded spread policy"}
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as error:
            reports[arm] = {"status": "incomplete", "reason": str(error), "policy_reference": "registered shipped policy"}
    return reports


def load_spreads(directory):
    """Retain optional process failures and verify available raw gradients without gating science."""
    rows = {}
    for arm in ("L", "U", "U0"):
        path = directory / f"spread-{arm}.json"
        try:
            status = directory / f"spread-status-{arm}.json"
            if status.exists():
                phase = json.loads(status.read_text())
                if phase.get("status") in {"spread: skipped for time", "incomplete"}:
                    rows[arm] = phase
                    continue
            if not path.exists():
                if status.exists():
                    rows[arm] = phase
                continue
            row = json.loads(path.read_text())
            for run in row["spread"]["passes"]:
                if run["gradient_file"] != f"spread-{arm}-{run['ordinal']}.safetensors":
                    raise ValueError("raw shipped-spread gradient filename changed")
                raw = directory / run["gradient_file"]
                if raw.stat().st_size != run["gradient_file_bytes"] or file_sha(raw) != run["gradient_file_sha256"]:
                    raise ValueError("raw shipped-spread gradients changed")
            rows[arm] = row
        except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
            rows[arm] = {"status": "incomplete", "reason": str(error)}
    return rows


def validate_proofs(proofs, *, rehearsal=False):
    if len(proofs) != 3 or {row["arm"] for row in proofs} != {"L", "U", "U0"}:
        raise ValueError("missing or duplicate proof")
    anchor = proofs[0]
    for row in proofs:
        registered_identity(row, rehearsal=rehearsal)
        validate_observer_policy(row["observer_policy"])
        if (row["kind"] != "proof" or row["repetition"] != 0 or not row["observer_same_arm_bitwise"]
                or not row["execution"]["observer_removed"] or not row["frozen_unchanged"]
                or row["path_before"] != row["path_after"]):
            raise ValueError("failed proof or changed binding")
        for key in ("runtime", "initial", "tokens", "nonce", "input_seal_sha256", "identity_sha256", "provenance"):
            if row[key] != anchor[key]:
                raise ValueError("proofs differ in " + key)
        if value_identity(row["base"]) != value_identity(anchor["base"]):
            raise ValueError("quantized or represented frozen values differ")
        if len(row["initial"]) != 448 or set(row["proof_gradients_sha256"]) != set(row["initial"]):
            raise ValueError("incomplete adapter gradient witness")
        if set(row["execution"]["gradients"]) != set(row["initial"]) or not row["execution"]["operations"]:
            raise ValueError("incomplete executed precision witness")
        if any(meta["dtype"] != "torch.float32" for meta in row["execution"]["gradients"].values()):
            raise ValueError("adapter gradient storage differs")
        calls = row["execution"]["calls"]
        expected = ([f"{prefix}:{layer}" for layer in range(32)
                     for prefix in ("qkv", "o", "mlp", "LoRA_QKV", "LoRA_W", "LoRA_MLP")]
                    if row["arm"] == "U" else [key.removesuffix(".A") for key in row["initial"] if key.endswith(".A")])
        if any(not isinstance(calls.get(key), int) or isinstance(calls[key], bool) or calls[key] < 1 for key in expected):
            raise ValueError("partial executed binding witness")
        # The forward rank GEMMs are the registered storage/compute distinction.
        # Backward routing and all adapter gradients are separately witnessed.
        dtype = "torch.bfloat16" if row["arm"] == "U" else "torch.float32"
        forward = expected[:]
        if row["arm"] == "U":
            forward = [key for key in expected if not key.startswith("LoRA_")]
        for route in forward:
            operations = [op for op in row["execution"]["operations"] if op["route"] == route and "mm" in op["operator"]]
            if not operations or any(any(p["dtype"] != dtype for p in op["operands"])
                                     or op["result_dtype"] != dtype for op in operations):
                raise ValueError("forward adapter GEMM precision not witnessed")


def validate_read(row, proof, *, rehearsal=False):
    schema = "dq11-rehearsal-arm/1" if rehearsal else "dq11-arm/1"
    if row.get("schema", "dq11-arm/1") != schema or row.get("science_eligible", True) is not (not rehearsal):
        raise ValueError("reading science/rehearsal mode changed")
    if row["arm"] == "L":
        from dq11_stream_witness import validate_train_prefetch_witness

        witness = row.get("rehearsal_train_prefetch" if rehearsal else "train_prefetch")
        validate_train_prefetch_witness(witness, rehearsal=rehearsal)
        if witness["source"] != row["runtime"]["experts4bit-qlora"] or witness["nonce"] != row["nonce"]:
            raise ValueError("streaming route witness source/nonce changed")
    from dq11_proof_policy import validate_shipped_policy

    validate_shipped_policy(row["execution_settings"])
    if row["kind"] != "read" or row["path_before"] != row["path_after"] or row["path_before"] != proof["path_before"]:
        raise ValueError("read path differs from proof")
    for key in ("runtime", "initial", "tokens", "nonce", "input_seal_sha256", "identity_sha256", "base", "provenance"):
        if row[key] != proof[key]:
            raise ValueError("read differs from proof: " + key)
    training = row["training"]
    correct, measured = training["correctness"], training["measured"]
    if (training["status"] != "OK" or not row["frozen_unchanged"] or not correct["all_finite"]
            or not correct["frozen_dense_bytes_unchanged"] or not correct["adapters_moved"]
            or correct["steps_without_trained_tokens"] != 0 or len(correct["losses"]) != 40
            or len(measured["step_seconds"]) != 40 or measured["timed_steps"] != "6..40"
            or set(row["final_adapters_sha256"]) != set(row["initial"])):
        raise ValueError("incomplete/mismatched training work or integrity")
    if not all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) and x > 0
               for x in measured["step_seconds"]):
        raise ValueError("nonfinite/invalid step timing")
    if not all(math.isfinite(x) for x in correct["losses"]):
        raise ValueError("nonfinite train loss")
    return statistics.median(measured["step_seconds"][5:])


def teardown_valid(teardown, instance_id):
    if not teardown or teardown.get("complete") is not True or teardown.get("instance_id") != instance_id:
        return False
    evidence = teardown.get("evidence")
    if isinstance(evidence, str):
        evidence = json.loads(evidence)
    return (isinstance(evidence, dict) and evidence.get("instance_absent") is True
            and evidence.get("destroy", {}).get("instance_id") == instance_id
            and evidence.get("destroy", {}).get("http") in (200, 204)
            and instance_id not in evidence.get("list_after", []))


def reduce(proofs, reads, *, initial_only=False, teardown=None, instance_id=None, spreads=None):
    result = {"schema": "dq11-read/1", "scope": SCOPE, "default_unsloth_position": False,
              "capacity_licensed": False, "shipping_licensed": False}
    try:
        validate_proofs(proofs)
        result["shipped_run_to_run_spread"] = report_spreads(proofs, reads, spreads or {})
        result["observer_policy"] = {p["arm"]: p["observer_policy"] for p in proofs}
        if not initial_gate(proofs):
            return {**result, "verdict": "QUALITY_FAIL", "phase": "initial", "recommendation": None}
        if initial_only:
            return {**result, "verdict": "INITIAL_GATE_PASS", "scientific_read": False}
        if len(reads) != 6 or [(row["repetition"], row["arm"]) for row in reads] != list(ORDER):
            raise ValueError("missing, duplicate or reordered reading")
        proof_by_arm = {row["arm"]: row for row in proofs}
        medians = {(row["repetition"], row["arm"]): validate_read(row, proof_by_arm[row["arm"]]) for row in reads}
        by = {(row["repetition"], row["arm"]): row for row in reads}
        quality_ok = all(quality(by[(rep, b)], by[(rep, "U")], field)
                         for rep in (1, 2) for b in ("L", "U0") for field in ("initial_quality", "final_quality"))
        pairs = [{"repetition": rep, "median_seconds": {arm: medians[(rep, arm)] for arm in ("L", "U", "U0")},
                  "fusion_headroom": (medians[(rep, "U0")] - medians[(rep, "U")]) / medians[(rep, "U0")],
                  "whole_path_ratio_L_over_U": medians[(rep, "L")] / medians[(rep, "U")],
                  "whole_path_time_saved": (medians[(rep, "L")] - medians[(rep, "U")]) / medians[(rep, "L")]}
                 for rep in (1, 2)]
        result.update(pairs=pairs, raw_reads_sha256=object_sha(reads), quality_pass=quality_ok)
        if not quality_ok:
            return {**result, "verdict": "QUALITY_FAIL", "phase": "trained", "recommendation": None}
        if teardown is None:
            return {**result, "verdict": "AWAITING_TEARDOWN", "recommendation": None}
        if not teardown_valid(teardown, instance_id):
            raise ValueError("teardown lacks verified instance absence")
        result.update(verdict="VALID_CONTROLLED_READ",
                      recommendation="BUILD_CANDIDATE" if all(p["fusion_headroom"] >= 0.05 for p in pairs) else "DEFER_FUSION",
                      controlled_position="U_FASTER_CONTROLLED" if all(p["whole_path_time_saved"] >= 0.05 for p in pairs) else "NO_STABLE_POSITION")
        return result
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return {**result, "verdict": "VOID", "reason": str(error), "recommendation": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--teardown", type=Path)
    parser.add_argument("--instance-id")
    args = parser.parse_args()
    try:
        proofs = [json.loads((args.directory / f"proof-{arm}.json").read_text()) for arm in ("L", "U", "U0")]
        reads = [] if args.initial_only else [json.loads((args.directory / f"read-{rep}-{arm}.json").read_text()) for rep, arm in ORDER]
        teardown = json.loads(args.teardown.read_text()) if args.teardown else None
        result = reduce(proofs, reads, initial_only=args.initial_only, teardown=teardown, instance_id=args.instance_id, spreads=load_spreads(args.directory))
    except (OSError, ValueError) as error:
        result = {"schema": "dq11-read/1", "scope": SCOPE, "verdict": "VOID", "reason": str(error), "recommendation": None}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["verdict"] in {"INITIAL_GATE_PASS", "AWAITING_TEARDOWN", "VALID_CONTROLLED_READ"} else 12)
