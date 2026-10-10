"""Original-function snapshot controls on isolated OWN Python fixtures only."""

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ra_python_function", ROOT / "bench/ra/ra_python_function.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

SOURCE = '''
def make(factor):
    def original(value=2, *, pad=False):
        """OWN synthetic function; never called by the binding."""
        return abs(value) * factor + pad
    return original
original = make(3)
'''


def fixture(source=SOURCE):
    namespace = {"__name__": "owned_fixture"}
    exec(compile(source, "owned-python-function-fixture.py", "exec", optimize=2), namespace)
    return namespace["original"]


def pair():
    live, reference = fixture(), fixture()
    for function in (live, reference):
        function.__annotations__ = {"value": "int", "return": "int"}
        function.details = {"tail": [1, 2], "tuple": ("café", b"bytes", -0.0)}
    return live, reference, gate.FunctionBinding(live, reference)


def test_complete_original_snapshot_and_authority_boundary():
    live, _, binding = pair()
    result = binding.check(live)
    assert result["status"] == "SUPPLIED_PYTHON_FUNCTION_SNAPSHOT_EQUAL"
    assert result["original_object_preserved"] and result["code_objects"] == 1
    assert all(
        not value for key, value in result.items() if key.endswith("authenticated") or key == "consumer_verified"
    )


def test_original_and_reference_are_never_invoked():
    live = fixture("def original(): raise RuntimeError('must never call')\n")
    ref = fixture("def original(): raise RuntimeError('must never call')\n")
    assert gate.FunctionBinding(live, ref).check(live)["original_object_preserved"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: setattr(f, "__name__", "changed"),
        lambda f: setattr(f, "__qualname__", "changed.qualname"),
        lambda f: setattr(f, "__module__", "changed_origin_label"),
        lambda f: setattr(f, "__doc__", "changed doc"),
        lambda f: setattr(f, "__defaults__", (3,)),
        lambda f: setattr(f, "__defaults__", (2.0,)),
        lambda f: setattr(f, "__defaults__", (True,)),
        lambda f: setattr(f, "__kwdefaults__", {"pad": 0}),
        lambda f: setattr(f, "__kwdefaults__", {"pad": False, "extra": 0}),
        lambda f: f.__annotations__.update(value="changed"),
        lambda f: f.__annotations__.update(extra=False),
        lambda f: f.details["tail"].__setitem__(-1, 3),
        lambda f: f.details["tail"].__setitem__(-1, 2.0),
        lambda f: f.details["tail"].__setitem__(-1, True),
        lambda f: f.details.update(extra=None),
        lambda f: f.details.update(tuple=("café", b"bytes", 0.0)),
        lambda f: f.details.update(tuple=["café", b"bytes", -0.0]),
        lambda f: f.__dict__.clear(),
        lambda f: setattr(f.__closure__[0], "cell_contents", 4),
        lambda f: setattr(f.__closure__[0], "cell_contents", 3.0),
    ],
)
def test_complete_metadata_mutants_refuse_and_are_terminal(mutate):
    live, _, binding = pair()
    mutate(live)
    with pytest.raises(ValueError, match="metadata drift"):
        binding.check(live)
    with pytest.raises(ValueError, match="terminal refusal"):
        binding.check(live)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.replace(co_filename="other-origin.py"),
        lambda c: c.replace(co_consts=c.co_consts + ("unused full tail",)),
        lambda c: c.replace(co_names=c.co_names + ("extra",)),
        lambda c: c.replace(co_firstlineno=c.co_firstlineno + 1),
        lambda c: c.replace(co_name="changed"),
    ],
)
def test_complete_code_not_only_executed_body(mutate):
    live, _, binding = pair()
    live.__code__ = mutate(live.__code__)
    with pytest.raises(ValueError, match="complete code drift"):
        binding.check(live)


@pytest.mark.parametrize("hold_alias", [False, True])
def test_qualified_name_mutation_refuses_metadata_with_or_without_code_alias(hold_alias):
    live, _, binding = pair()
    code = live.__code__
    held = [live.__qualname__] if hold_alias else []
    live.__qualname__ = "changed.qualname"
    assert live.__code__ is code
    with pytest.raises(ValueError, match="complete typed metadata drift"):
        binding.check(live)
    with pytest.raises(ValueError, match="terminal refusal"):
        binding.check(live)
    assert len(held) == int(hold_alias)


def test_external_code_and_metadata_references_do_not_change_snapshot_hashes():
    live, reference, binding = pair()
    initial = binding.check(live)
    held = [
        live.__code__,
        live.__code__.co_consts,
        live.__code__.co_filename,
        live.__name__,
        live.__qualname__,
        live.__defaults__,
        live.__annotations__["value"],
        live.details["tuple"][0],
        reference.__code__,
        reference.__qualname__,
    ]
    assert binding.check(live) == initial
    held.clear()
    assert binding.check(live) == initial


@pytest.mark.parametrize("field", ["co_qualname", "co_linetable", "co_exceptiontable"])
def test_additional_interpreter_code_fields_are_bound(field):
    live, _, binding = pair()
    if hasattr(live.__code__, field):
        value = getattr(live.__code__, field)
        changed = value + (".changed" if type(value) is str else b"\x00")
        live.__code__ = live.__code__.replace(**{field: changed})
        with pytest.raises(ValueError, match="complete code drift"):
            binding.check(live)
    else:
        # Python 3.10 has no qualified-name or exception-table code fields.
        assert binding.check(live)["original_object_preserved"]


def test_nested_code_body_is_in_complete_code():
    text = "def original():\n    def nested(): return 1\n    return nested\n"
    live, reference = fixture(text), fixture(text)
    binding = gate.FunctionBinding(live, reference)
    assert binding.check(live)["code_objects"] == 2
    constants = list(live.__code__.co_consts)
    index = next(i for i, c in enumerate(constants) if type(c) is type(live.__code__))
    constants[index] = constants[index].replace(co_consts=(None, 2))
    live.__code__ = live.__code__.replace(co_consts=tuple(constants))
    with pytest.raises(ValueError, match="complete code drift"):
        binding.check(live)


def test_equivalent_replacement_function_is_not_original():
    live, reference, binding = pair()
    with pytest.raises(ValueError, match="identity drift"):
        binding.check(reference)
    with pytest.raises(ValueError, match="terminal refusal"):
        binding.check(live)


def test_reference_later_mutation_cannot_reseal_expected_snapshot():
    live, reference, binding = pair()
    live.details["tail"][-1] = 77
    reference.details["tail"][-1] = 77
    with pytest.raises(ValueError, match="metadata drift"):
        binding.check(live)


def test_matching_forged_reference_deliberately_passes_without_authority():
    live, reference = fixture(), fixture()
    live.__module__ = reference.__module__ = "forged_matching_label"
    live.__defaults__ = reference.__defaults__ = (77,)
    result = gate.FunctionBinding(live, reference).check(live)
    assert not result["source_authenticated"] and not result["dependency_graph_authenticated"]


@pytest.mark.parametrize("value", [object(), {1: 2}, {"nan": math.nan}, {"inf": math.inf}, {"tail": set()}])
def test_opaque_and_invalid_metadata_refuse(value):
    live, reference = fixture(), fixture()
    live.extra = reference.extra = value
    with pytest.raises(ValueError):
        gate.FunctionBinding(live, reference)


def test_metadata_subclass_hooks_never_execute():
    class HostileList(list):
        def __iter__(self):
            raise AssertionError("must not iterate subclass")

    live, reference = fixture(), fixture()
    live.extra = reference.extra = HostileList([1, 2])
    with pytest.raises(ValueError, match="opaque metadata"):
        gate.FunctionBinding(live, reference)


@pytest.mark.parametrize("value", ["a" * 65537, b"a" * 65537, 1 << 4096, list(range(4097)), "\ud800"])
def test_metadata_bounds(value):
    live, reference = fixture(), fixture()
    live.extra = reference.extra = value
    with pytest.raises(ValueError):
        gate.FunctionBinding(live, reference)


def test_aggregate_item_bound():
    live, reference = fixture(), fixture()
    live.extra = reference.extra = [list(range(100)) for _ in range(100)]
    with pytest.raises(ValueError, match="item bound"):
        gate.FunctionBinding(live, reference)


def test_aggregate_byte_bound():
    live, reference = fixture(), fixture()
    live.extra = reference.extra = ["x" * 65536 for _ in range(17)]
    with pytest.raises(ValueError, match="total byte bound"):
        gate.FunctionBinding(live, reference)


def test_metadata_cycle_and_depth_refuse():
    live, reference = fixture(), fixture()
    cycle = []
    cycle.append(cycle)
    live.extra = reference.extra = cycle
    with pytest.raises(ValueError, match="cyclic"):
        gate.FunctionBinding(live, reference)
    value = 0
    for _ in range(18):
        value = [value]
    live.extra = reference.extra = value
    with pytest.raises(ValueError, match="depth bound"):
        gate.FunctionBinding(live, reference)


def test_empty_closure_cell_refuses():
    live, reference = fixture(), fixture()
    del reference.__closure__[0].cell_contents
    with pytest.raises(ValueError, match="empty closure cell"):
        gate.FunctionBinding(live, reference)


def test_large_code_payload_refuses():
    live, reference = fixture(), fixture()
    reference.__code__ = reference.__code__.replace(co_consts=reference.__code__.co_consts + ("x" * (1024 * 1024),))
    with pytest.raises(ValueError, match="code byte bound"):
        gate.FunctionBinding(live, reference)


def test_unsupported_code_constant_refuses():
    live, reference = fixture(), fixture()
    reference.__code__ = reference.__code__.replace(co_consts=reference.__code__.co_consts + (object(),))
    with pytest.raises(ValueError, match="unsupported code serialization"):
        gate.FunctionBinding(live, reference)


def test_code_object_inventory_bound():
    live, reference = fixture(), fixture()
    reference.__code__ = reference.__code__.replace(co_consts=(live.__code__,) * 256)
    with pytest.raises(ValueError, match="code object bound"):
        gate.FunctionBinding(live, reference)


def test_nested_code_depth_bound():
    live, reference = fixture(), fixture()
    code = reference.__code__
    for _ in range(18):
        code = code.replace(co_consts=(code,))
    reference.__code__ = code
    with pytest.raises(ValueError, match="code object depth bound"):
        gate.FunctionBinding(live, reference)


@pytest.mark.parametrize("value", [abs, object(), None])
def test_non_python_functions_refuse(value):
    with pytest.raises(ValueError, match="original Python function only"):
        gate.FunctionBinding(value, value)


def test_deferred_annotation_code_not_executed():
    live, reference = fixture(), fixture()
    if hasattr(reference, "__annotate__"):

        def forbidden(*args):
            raise AssertionError("annotation code must never execute")

        reference.__annotate__ = forbidden
        with pytest.raises(ValueError, match="deferred annotations"):
            gate.FunctionBinding(live, reference)
    else:
        assert gate.FunctionBinding(live, reference).check(live)["original_object_preserved"]


def test_global_dependency_drift_is_explicitly_outside_authority():
    text = "OFFSET = 1\ndef original(value): return value + OFFSET\n"
    live, reference = fixture(text), fixture(text)
    binding = gate.FunctionBinding(live, reference)
    live.__globals__["OFFSET"] = 77
    assert not binding.check(live)["globals_authenticated"]


@pytest.mark.parametrize(
    "text", ["def original():\n    import json\n    return 0\n", "def original(peer): return peer.operation()\n"]
)
def test_import_and_parameter_dispatch_deliberately_admitted_without_graph_authority(text):
    live, reference = fixture(text), fixture(text)
    result = gate.FunctionBinding(live, reference).check(live)
    assert not result["dependency_graph_authenticated"]
