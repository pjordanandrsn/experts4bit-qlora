# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""CPU guard for residency overrides (#1477): unavailable kernels never skip it."""
import ast
import importlib
import inspect
from pathlib import Path

import pytest

ENGINES = Path(__file__).resolve().parents[1] / "experts4bit_qlora" / "engines"
OLD_ROUTING_NAMES = {"router_indices", "router_scores"}


def _ast_signature(method):
    args = method.args
    positional = args.posonlyargs + args.args
    defaults = [inspect.Parameter.empty] * (len(positional) - len(args.defaults)) + [None] * len(args.defaults)
    parameters = [inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_ONLY if i < len(args.posonlyargs)
                                    else inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default)
                  for i, (arg, default) in enumerate(zip(positional, defaults))]
    if args.vararg:
        parameters.append(inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    parameters.extend(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                                       default=inspect.Parameter.empty if default is None else None)
                      for arg, default in zip(args.kwonlyargs, args.kw_defaults))
    if args.kwarg:
        parameters.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


def _catalog(directory):
    trees, classes, imports = {}, {}, {}
    for path in sorted(directory.glob("*.py")):
        module = path.stem
        trees[module] = ast.parse(path.read_text(), filename=str(path))
        imports[module] = {}
        for node in trees[module].body:
            if isinstance(node, ast.ImportFrom):
                source = node.module.rsplit(".", 1)[-1] if node.module else None
                for alias in node.names:
                    imports[module][alias.asname or alias.name] = (source, alias.name)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[module][alias.asname or alias.name] = (alias.name.rsplit(".", 1)[-1], None)
            if isinstance(node, ast.ClassDef):
                classes[module, node.name] = node
    parents = {}
    for key, node in classes.items():
        parents[key] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                parent = imports[key[0]].get(base.id, (key[0], base.id))
                if parent in classes:
                    parents[key].append(parent)
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                source, imported = imports[key[0]].get(base.value.id, (base.value.id, None))
                parent = (imported if imported in trees else source, base.attr)
                if parent in classes:
                    parents[key].append(parent)
    return trees, classes, parents


def _methods(node):
    return {n.name: n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _residencies(classes, parents):
    # Identify independent residency roots by their constructor/forward protocol,
    # then discover every descendant, including ones with inherited methods.
    found = set()
    for key, node in classes.items():
        methods = _methods(node)
        if "forward" in methods and "__init__" in methods:
            parameters = _ast_signature(methods["__init__"]).parameters
            if {"mod", "hot_ids", "device"} <= parameters.keys():
                found.add(key)
    while True:
        expanded = found | {key for key, bases in parents.items() if any(base in found for base in bases)}
        if expanded == found:
            return found
        found = expanded


def _signature(key, method, classes, loaded):
    module, name = key
    if module not in loaded:
        try:
            loaded[module] = importlib.import_module("experts4bit_qlora.engines." + module)
        except (ImportError, OSError, RuntimeError):
            # CPU installations may lack Torch/Triton/native CUDA libraries.
            # Source signatures remain mandatory; there is no importorskip.
            loaded[module] = None
    if loaded[module] is not None:
        return inspect.signature(getattr(getattr(loaded[module], name), method))
    return _ast_signature(_methods(classes[key])[method])


def _ancestors(key, parents):
    for parent in parents[key]:
        yield parent
        yield from _ancestors(parent, parents)


def _assert_accepts(base, override, label):
    values = {p.name: object() for p in base.parameters.values()
              if p.name not in ("self", "cls") and p.kind in
              (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in base.parameters.values()):
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in override.parameters.values()), label
    positional = [object() for p in base.parameters.values()
                  if p.kind == inspect.Parameter.POSITIONAL_ONLY or p.name in ("self", "cls")]
    try:
        override.bind(*positional, **values)
    except TypeError as exc:
        raise AssertionError(f"{label}: base {base}, override {override}: {exc}") from exc


def _assert_no_old_routing_keywords(trees):
    # The gpt-oss pipelined override now follows its base's keyword contract.
    # Audit all forward calls in engines for either historical spelling;
    # current callers are positional, and future base-name keywords are valid.
    for module, tree in trees.items():
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != "forward":
                continue
            label = f"{module}:{call.lineno}: obsolete or unprovable routing keyword in {ast.unparse(call)}"
            for keyword in call.keywords:
                if keyword.arg is not None:
                    assert keyword.arg not in OLD_ROUTING_NAMES, label
                else:
                    assert isinstance(keyword.value, ast.Dict), label
                    assert all(isinstance(key, ast.Constant) and isinstance(key.value, str)
                               and key.value not in OLD_ROUTING_NAMES for key in keyword.value.keys), label


def test_residency_overrides_accept_base_keywords():
    trees, classes, parents = _catalog(ENGINES)
    found = _residencies(classes, parents)
    assert ("hot_residency", "_HotResidency") in found
    assert ("hybrid", "_HybridTier") in found
    loaded, checks = {}, []
    for key in sorted(found):
        for parent in _ancestors(key, parents):
            base_methods = _methods(classes[parent])
            keyword_hooks = {call.func.attr for method in base_methods.values() for call in ast.walk(method)
                             if isinstance(call, ast.Call) and call.keywords and isinstance(call.func, ast.Attribute)
                             and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"}
            for name in sorted({"forward"} | keyword_hooks):
                if name not in _methods(classes[key]) or name not in base_methods:
                    continue
                label = f"{key[0]}.{key[1]}.{name} overrides {parent[0]}.{parent[1]}.{name}"
                _assert_accepts(_signature(parent, name, classes, loaded), _signature(key, name, classes, loaded), label)
                checks.append(label)
    assert checks, "no residency overrides were checked"
    _assert_no_old_routing_keywords(trees)


def test_ast_signatures_enforce_keyword_only_and_kwargs_contracts():
    def signature(source):
        return _ast_signature(ast.parse(source).body[0])
    base = signature("def hook(self, x, *, residual=None): pass")
    _assert_accepts(base, signature("def hook(self, x, **kwargs): pass"), "kwargs accepts residual")
    with pytest.raises(AssertionError, match="residual"):
        _assert_accepts(base, signature("def hook(self, x): pass"), "missing keyword")
    with pytest.raises(AssertionError, match="residual"):
        _assert_accepts(base, signature("def hook(self, x, residual=None, /): pass"), "positional-only keyword")


def test_callers_use_base_routing_names():
    trees, _, _ = _catalog(ENGINES)
    _assert_no_old_routing_keywords({**trees, "new_caller": ast.parse(
        "state.forward(hidden, top_k_index=idx, top_k_weights=wts)")})
    for name in OLD_ROUTING_NAMES:
        with pytest.raises(AssertionError, match=name):
            _assert_no_old_routing_keywords({**trees, "old_caller": ast.parse(f"state.forward(hidden, {name}=routed)")})
        with pytest.raises(AssertionError, match=name):
            _assert_no_old_routing_keywords({**trees, "old_caller": ast.parse(
                f"state.forward(hidden, **{{'{name}': routed}})")})
    with pytest.raises(AssertionError, match="unprovable"):
        _assert_no_old_routing_keywords({**trees, "opaque_caller": ast.parse("state.forward(hidden, **kwargs)")})


def test_full_residency_guard_uses_ast_when_cpu_imports_are_unavailable(monkeypatch):
    original = importlib.import_module

    def without_engines(name, *args, **kwargs):
        if name.startswith("experts4bit_qlora.engines."):
            raise ImportError("simulate CPU installation without CUDA/Triton dependencies")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without_engines)
    test_residency_overrides_accept_base_keywords()


def test_discovery_follows_imported_and_module_qualified_subclasses(tmp_path):
    (tmp_path / "root.py").write_text(
        "class State:\n"
        " def __init__(self, mod, hot_ids, device): pass\n"
        " def forward(self, hidden_states, top_k_index, top_k_weights): pass\n")
    (tmp_path / "child.py").write_text(
        "from .root import State as Base\n"
        "from . import root as r\n"
        "class Imported(Base): pass\n"
        "class Qualified(r.State): pass\n"
        "class Grandchild(Qualified): pass\n")
    _, classes, parents = _catalog(tmp_path)
    assert _residencies(classes, parents) == {
        ("root", "State"), ("child", "Imported"), ("child", "Qualified"), ("child", "Grandchild")}
