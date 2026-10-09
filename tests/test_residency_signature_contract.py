# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""CPU guard for residency overrides (#1477): unavailable kernels never skip it."""
import ast
import importlib
import inspect
from pathlib import Path

import pytest

ENGINES = Path(__file__).resolve().parents[1] / "experts4bit_qlora" / "engines"
ADAPTER = ("pipelined", "_GptOssPipelined", "forward")
ROUTING_ALIASES = {"top_k_index": "router_indices", "top_k_weights": "router_scores"}


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


def _assert_accepts(base, override, label, aliases=None):
    values = {p.name: object() for p in base.parameters.values()
              if p.name not in ("self", "cls") and p.kind in
              (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    if aliases:
        missing = values.keys() - override.parameters.keys()
        assert missing == aliases.keys(), f"{label}: positional adapter exception changed: {missing}"
        values = {aliases.get(name, name): value for name, value in values.items()}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in base.parameters.values()):
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in override.parameters.values()), label
    positional = [object() for p in base.parameters.values()
                  if p.kind == inspect.Parameter.POSITIONAL_ONLY or p.name in ("self", "cls")]
    try:
        override.bind(*positional, **values)
    except TypeError as exc:
        raise AssertionError(f"{label}: base {base}, override {override}: {exc}") from exc


def _assert_positional_dispatch(trees, classes):
    # ONE named exception: gpt-oss's pipelined positional adapter renames
    # top_k_index/top_k_weights to router_indices/router_scores. Conservatively
    # check every dynamic forward receiver in engines, not just today's `st`
    # alias: a new caller cannot introduce routing keywords behind the waiver.
    calls = []
    for module, tree in trees.items():
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != "forward":
                continue
            receiver = call.func.value
            if isinstance(receiver, ast.Name) and (module, receiver.id) in classes:
                continue  # explicit class implementation, not state dispatch
            if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Name) and receiver.func.id in ("super", "type"):
                continue  # statically bound implementation on self
            calls.append((module, call))
    assert calls, "positional adapter exception has no dispatch sites to check"
    assert any(module == "pipelined" for module, _ in calls), "pipelined dispatch was not audited"
    for module, call in calls:
        label = f"{module}:{call.lineno}: {ast.unparse(call)}"
        assert len(call.args) >= 3 and not any(isinstance(arg, ast.Starred) for arg in call.args[:3]), label
        assert all(kw.arg is not None and kw.arg not in {*ROUTING_ALIASES, *ROUTING_ALIASES.values()}
                   for kw in call.keywords), label


def test_residency_overrides_accept_base_keywords():
    trees, classes, parents = _catalog(ENGINES)
    found = _residencies(classes, parents)
    assert ("hot_residency", "_HotResidency") in found
    assert ("hybrid", "_HybridTier") in found
    assert (ADAPTER[0], ADAPTER[1]) in found
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
                aliases = ROUTING_ALIASES if (*key, name) == ADAPTER else None
                _assert_accepts(_signature(parent, name, classes, loaded), _signature(key, name, classes, loaded), label, aliases)
                checks.append(label)
    assert checks, "no residency overrides were checked"
    _assert_positional_dispatch(trees, classes)


def test_ast_signatures_enforce_keyword_only_and_kwargs_contracts():
    def signature(source):
        return _ast_signature(ast.parse(source).body[0])
    base = signature("def hook(self, x, *, residual=None): pass")
    _assert_accepts(base, signature("def hook(self, x, **kwargs): pass"), "kwargs accepts residual")
    with pytest.raises(AssertionError, match="residual"):
        _assert_accepts(base, signature("def hook(self, x): pass"), "missing keyword")
    with pytest.raises(AssertionError, match="residual"):
        _assert_accepts(base, signature("def hook(self, x, residual=None, /): pass"), "positional-only keyword")


def test_positional_adapter_exception_refuses_keyword_callers():
    trees, classes, _ = _catalog(ENGINES)
    tree = ast.parse("state.forward(hidden, top_k_index=idx, top_k_weights=wts)")
    with pytest.raises(AssertionError, match="top_k_index"):
        _assert_positional_dispatch({**trees, "new_caller": tree}, classes)


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
