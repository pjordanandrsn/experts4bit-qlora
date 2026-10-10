"""Repeated equality of a supplied Python function, not provenance authority.

No supplied function is invoked. Globals, dependencies, descriptors, classes,
sites and native extensions require separate reviewed bindings. Matching forged
reference and live functions can pass. Serialized cooperating callers only.
"""

import hashlib
import marshal
import math
import types

MAX_BYTES = 1024 * 1024
MAX_SCALAR_BYTES = 65536
MAX_ITEMS = 4096
MAX_DEPTH = 16
MAX_CODE_OBJECTS = 256


def _require(ok, message):
    if not ok:
        raise ValueError("Python function snapshot: " + message)


def _typed(value, budget, active, depth=0):
    _require(depth <= MAX_DEPTH, "metadata depth bound")
    budget[0] += 1
    _require(budget[0] <= MAX_ITEMS, "metadata item bound")
    kind = type(value)
    if kind is str:
        try:
            raw = value.encode("utf-8")
        except UnicodeError as error:
            raise ValueError("Python function snapshot: metadata UTF-8") from error
        _require(len(raw) <= MAX_SCALAR_BYTES, "metadata scalar byte bound")
        budget[1] += len(raw)
        result = ("str", value)
    elif kind is bytes:
        _require(len(value) <= MAX_SCALAR_BYTES, "metadata scalar byte bound")
        budget[1] += len(value)
        result = ("bytes", value)
    elif kind is int:
        _require(value.bit_length() <= 4096, "metadata integer bound")
        budget[1] += (value.bit_length() + 7) // 8
        result = ("int", value)
    elif kind is float:
        _require(math.isfinite(value), "finite metadata float")
        result = ("float", value.hex())
    elif kind is bool or value is None:
        result = (kind.__name__, value)
    elif kind in (tuple, list, dict):
        _require(id(value) not in active, "cyclic metadata")
        _require(len(value) <= MAX_ITEMS, "metadata container bound")
        active.add(id(value))
        try:
            if kind is dict:
                _require(all(type(key) is str for key in value), "metadata string keys only")
                result = (
                    "dict",
                    tuple(
                        (_typed(key, budget, active, depth + 1), _typed(value[key], budget, active, depth + 1))
                        for key in sorted(value)
                    ),
                )
            else:
                result = (kind.__name__, tuple(_typed(item, budget, active, depth + 1) for item in value))
        finally:
            active.remove(id(value))
    else:
        raise ValueError("Python function snapshot: opaque metadata needs separate binding")
    _require(budget[1] <= MAX_BYTES, "metadata total byte bound")
    return result


def _code(code):
    count = [0]

    def visit(value, depth=0):
        _require(type(value) is types.CodeType and depth <= MAX_DEPTH, "code object depth bound")
        count[0] += 1
        _require(count[0] <= MAX_CODE_OBJECTS and len(value.co_code) <= MAX_BYTES, "code object bound")
        for constant in value.co_consts:
            if type(constant) is types.CodeType:
                visit(constant, depth + 1)

    visit(code)
    try:
        raw = marshal.dumps(code)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("Python function snapshot: unsupported code serialization") from error
    _require(len(raw) <= MAX_BYTES, "complete code byte bound")
    return raw, count[0]


def _snapshot(function):
    _require(type(function) is types.FunctionType, "original Python function only")
    # Reading deferred annotations can execute arbitrary supplied annotation code.
    # Refuse before accessing __annotations__, including on Python 3.14.
    _require(getattr(function, "__annotate__", None) is None, "deferred annotations need separate binding")
    code, count = _code(function.__code__)
    try:
        closure = tuple(cell.cell_contents for cell in function.__closure__) if function.__closure__ else None
    except ValueError as error:
        raise ValueError("Python function snapshot: empty closure cell") from error
    metadata = _typed(
        {
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "doc": function.__doc__,
            "defaults": function.__defaults__,
            "kwdefaults": function.__kwdefaults__,
            "annotations": function.__annotations__,
            "attributes": function.__dict__,
            "closure": closure,
            "type_params": getattr(function, "__type_params__", ()),
        },
        [0, 0],
        set(),
    )
    try:
        raw = marshal.dumps(metadata)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("Python function snapshot: unsupported metadata serialization") from error
    _require(len(raw) <= MAX_BYTES, "complete metadata byte bound")
    return code, metadata, raw, count


class FunctionBinding:
    """Anchor one original object to a supplied reference code/metadata snapshot.

    The reference has no source authority here. This checks same-process snapshot
    equality only; it admits imports, attribute dispatch and changed global
    dependencies without authenticating them. It never calls either function.
    A failed repeated check is terminal. Private fields are not a tamper barrier.
    """

    def __init__(self, original, reference):
        self._failed = False
        self._original = original
        self._expected = _snapshot(reference)
        _require(_snapshot(reference) == self._expected, "supplied reference snapshot drift")
        self.check(original)

    def check(self, original):
        _require(not self._failed, "terminal refusal")
        try:
            _require(original is self._original, "original function identity drift")
            observed = _snapshot(original)
            _require(observed[0] == self._expected[0], "complete code drift")
            _require(observed[1] == self._expected[1], "complete typed metadata drift")
        except BaseException:
            self._failed = True
            raise
        return {
            "schema": 1,
            "status": "SUPPLIED_PYTHON_FUNCTION_SNAPSHOT_EQUAL",
            "code_sha256": hashlib.sha256(observed[0]).hexdigest(),
            "code_bytes": len(observed[0]),
            "code_objects": observed[3],
            "metadata_sha256": hashlib.sha256(observed[2]).hexdigest(),
            "metadata_bytes": len(observed[2]),
            "original_object_preserved": True,
            "source_authenticated": False,
            "runtime_authenticated": False,
            "globals_authenticated": False,
            "dependency_graph_authenticated": False,
            "class_site_authenticated": False,
            "native_extension_authenticated": False,
            "native_read_authenticated": False,
            "consumer_verified": False,
        }
