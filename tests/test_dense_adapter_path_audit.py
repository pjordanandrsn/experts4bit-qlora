"""CPU identity controls; no GPU, model download or Unsloth import."""

import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import types

import torch
from torch import nn

from experts4bit_qlora.lora import LoRALinear


SOURCE = Path(__file__).resolve().parents[1] / "bench/dq1/adapter_path_audit.py"
SPEC = importlib.util.spec_from_file_location("dense_adapter_path_audit", SOURCE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def original_forward(self, x):
    return x + 1


def replacement_forward(self, x):
    return x + 2


def fused_qkv(self, x):
    return x, x, x


def fused_o(x, scale=1):
    return x * scale


def native():
    return LoRALinear(nn.Linear(8, 8, bias=False, dtype=torch.bfloat16), r=2, dtype=torch.float32)


def test_real_native_storage_and_explicit_unknowns_do_not_run_or_mutate():
    model = nn.Sequential(native())
    snapshot = {name: value.clone() for name, value in model.state_dict().items()}
    declared = {"base_dtype": "bf16", "compute_dtype": "bf16", "autocast": False}
    first = AUDIT.audit_adapter_paths(model, declared_compute=declared)
    assert first == AUDIT.audit_adapter_paths(model, declared_compute=declared)
    assert json.loads(json.dumps(first)) == first
    assert first["declared_compute"] == declared
    assert first["per_operation_precision"] == "unknown" and first["execution_observed"] is False
    row, = first["loaded_paths"]
    assert row["adapter_storage_dtypes"] == {"lora_A": "torch.float32", "lora_B": "torch.float32"}
    assert row["forward"]["qualname"] == "LoRALinear.forward"
    assert row["forward"]["source_status"] == "known"
    assert all(value["status"] == "unknown" for value in row["attention_bindings"].values())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in snapshot.items())
    assert str(Path(__file__).parent) not in json.dumps(first)


def test_loaded_fused_bindings_and_partial_are_identified_without_execution():
    model = nn.Module()
    model.apply_qkv = types.MethodType(fused_qkv, model)
    model.apply_o = functools.partial(fused_o, scale=2)
    row, = AUDIT.audit_adapter_paths(model)["loaded_paths"]
    qkv, output = row["attention_bindings"].values()
    assert qkv["kind"] == "bound_method" and qkv["qualname"] == "fused_qkv"
    assert qkv["source_file_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert output["kind"] == "partial" and output["function"]["qualname"] == "fused_o"
    assert output["bound_keyword_names"] == ["scale"] and output["bound_values"] == "unknown"


def test_changed_forward_mutant_changes_audit_even_with_same_source_file():
    model = native()
    model.forward = types.MethodType(original_forward, model)
    before = AUDIT.audit_adapter_paths(model)
    model.forward = types.MethodType(replacement_forward, model)
    after = AUDIT.audit_adapter_paths(model)
    a, b = before["loaded_paths"][0]["forward"], after["loaded_paths"][0]["forward"]
    assert a["source_file_sha256"] == b["source_file_sha256"]
    assert a["code_sha256"] != b["code_sha256"] and before != after


def test_runtime_code_replacement_mutant_keeps_name_and_changes_audit():
    mutant = types.FunctionType(original_forward.__code__, globals(), "same_name")
    before = AUDIT.callable_identity(mutant)
    mutant.__code__ = replacement_forward.__code__
    after = AUDIT.callable_identity(mutant)
    assert before["module"] == after["module"] and before["qualname"] == after["qualname"]
    assert before["source_file_sha256"] == after["source_file_sha256"]
    assert before["code_sha256"] != after["code_sha256"]


def test_dynamic_source_missing_and_empty_census_are_explicit():
    namespace = {}
    exec(compile("def dynamic(x): return x", "<dynamic>", "exec"), namespace)
    identity = AUDIT.callable_identity(namespace["dynamic"])
    assert identity["code_sha256"] and identity["source_file_sha256"] is None
    assert identity["source_status"] == "unknown"
    assert AUDIT.callable_identity(None) == {"status": "unknown", "reason": "missing_or_not_callable"}
    empty = AUDIT.audit_adapter_paths(nn.Linear(8, 8))
    assert empty["census_status"] == "empty" and empty["loaded_paths"] == []


def test_builtin_and_callable_object_do_not_recurse_or_invent_source():
    builtin = AUDIT.callable_identity(len)
    assert builtin["kind"] == "builtin" and builtin["qualname"] == "len"
    assert builtin["source_status"] == "unknown" and builtin["code_sha256"] is None

    class Forward:
        def __call__(self, x):
            raise AssertionError("audit must not run callables")

    obj = AUDIT.callable_identity(Forward())
    assert obj["kind"] == "callable_object"
    assert obj["call"]["qualname"].endswith("Forward.__call__")
    assert obj["call"]["code_sha256"]


def test_code_identity_excludes_checkout_paths():
    first = types.FunctionType(original_forward.__code__.replace(co_filename="/checkout/one.py"), globals())
    second = types.FunctionType(original_forward.__code__.replace(co_filename="/checkout/two.py"), globals())
    a, b = AUDIT.callable_identity(first), AUDIT.callable_identity(second)
    assert a["code_sha256"] == b["code_sha256"]
    assert a["source_status"] == b["source_status"] == "unknown"
