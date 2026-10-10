"""Opt-in loaded-path census for a future registered dense matched-work comparison.

Call ``audit_adapter_paths(model, declared_compute={...})`` after loader/patch setup
and save the returned JSON beside that comparison's receipts. This helper runs no
forward, installs no hooks, imports no framework, and does not alter the model.
Loaded callable identity is not evidence of execution or per-operation precision.
Compute settings are caller declarations; tensor dtypes describe storage only.
No historical TC1 receipts are changed. Wiring requires the future lane's amendment.
"""

import functools
import hashlib
import inspect
import marshal
from pathlib import Path
import platform
import types


def _code_without_paths(code):
    # Keep executable constants, including nested code, but exclude checkout paths.
    constants = tuple(_code_without_paths(c) if isinstance(c, types.CodeType) else c for c in code.co_consts)
    return code.replace(co_filename="", co_consts=constants)


def callable_identity(fn):
    """Describe the current callable, including runtime code; unknowns stay explicit.

    Code hashes are Python-version-specific. Closure cells, bound argument values
    and mutable globals are not captured, so this is not a full semantic fingerprint.
    Source hashes alone would miss a runtime replacement of ``function.__code__``.
    No repr, absolute path, tensor value or bound argument value is written.
    """
    if not callable(fn):
        return {"status": "unknown", "reason": "missing_or_not_callable"}
    if isinstance(fn, functools.partial):
        return {"status": "known", "kind": "partial", "function": callable_identity(fn.func),
                "bound_arg_types": [type(a).__name__ for a in fn.args],
                "bound_keyword_names": sorted(fn.keywords or {}), "bound_values": "unknown"}
    if inspect.isbuiltin(fn) or inspect.ismethoddescriptor(fn):
        return {"status": "known", "kind": "builtin", "module": getattr(fn, "__module__", None),
                "qualname": getattr(fn, "__qualname__", None), "code_sha256": None,
                "source_file_sha256": None, "source_status": "unknown"}
    kind = "bound_method" if inspect.ismethod(fn) else "function" if inspect.isfunction(fn) else "callable_object"
    target = fn.__func__ if inspect.ismethod(fn) else fn
    if kind == "callable_object":
        return {"status": "known", "kind": kind,
                "module": type(fn).__module__, "qualname": type(fn).__qualname__,
                "call": callable_identity(type(fn).__call__)}
    record = {"status": "known", "kind": kind, "module": target.__module__, "qualname": target.__qualname__,
              "code_sha256": None, "source_file_sha256": None, "source_status": "unknown"}
    code = getattr(target, "__code__", None)
    if code is not None:
        record["code_sha256"] = hashlib.sha256(marshal.dumps(_code_without_paths(code))).hexdigest()
    try:
        source = inspect.getsourcefile(target)
        if source:
            record["source_file_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            record["source_status"] = "known"
    except (OSError, TypeError):
        pass
    return record


def audit_adapter_paths(model, *, declared_compute=None):
    """Return a JSON-safe census of adapter wrappers and attention fusion bindings.

    Detection uses adapter parameter structure and callable attributes, not framework
    class names. Absent apply_qkv/apply_o bindings are explicit unknowns on adapter
    rows too. Call only after the future lane has finished replacing its callables.
    An empty census is recorded as empty, never as evidence of an unfused path.
    """
    rows = []
    for name, module in model.named_modules():
        storage = {}
        for parameter_name, parameter in module.named_parameters(recurse=True):
            first = parameter_name.split(".")[0]
            if first in {"lora_A", "lora_B"} or (
                "." not in parameter_name and parameter_name.endswith(("_lora_A", "_lora_B"))
            ):
                storage[parameter_name] = str(parameter.dtype)
        bindings = {key: getattr(module, key, None) for key in ("apply_qkv", "apply_o")}
        if storage or any(value is not None for value in bindings.values()):
            rows.append({"module_path": name, "class_module": type(module).__module__,
                         "class_qualname": type(module).__qualname__, "adapter_storage_dtypes": dict(sorted(storage.items())),
                         "forward": callable_identity(module.forward),
                         "attention_bindings": {key: callable_identity(value) for key, value in bindings.items()}})
    return {"schema": 1, "python": platform.python_version(), "loaded_paths": sorted(rows, key=lambda row: row["module_path"]),
            "declared_compute": declared_compute, "execution_observed": False, "per_operation_precision": "unknown",
            "closure_and_bound_values": "unknown", "census_status": "nonempty" if rows else "empty"}
