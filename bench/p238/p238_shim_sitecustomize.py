"""P238 amendment 5: the missing ``__wrapped__`` on unsloth's ``FastModel.get_peft_model`` (staged flat as
``p238_shim_sitecustomize.py``; the runner copies it to ``$W/p238_site/sitecustomize.py`` and puts that directory first on the
op and step processes' PYTHONPATH, so it loads in the venv too: a venv runs ``sitecustomize``, never ``usercustomize``).

Why. tc1_arm's T24 passes PEFT ``target_parameters`` only when ``inspect.signature(get_peft_model)`` names it, and otherwise
refuses the arm. In the pinned unsloth 2026.10.8 (read from the wheel, sha256 12c5d8fb7ba7a9092d880f309d350a3491773e9a86bd8ecedf
9056b7676e2ac1, never imported):
- ``FastLanguageModel.get_peft_model`` (unsloth/models/loader.py:1183-1190) is decorated ``@functools.wraps(FastLlamaModel.
  get_peft_model)``, so its signature shows ``target_parameters`` (models/llama.py:4358);
- ``FastModel.get_peft_model`` (loader.py:2127-2136), P238's registered loader, is a bare ``(model, *args, **kwargs)`` dispatcher
  that forwards both unchanged to ``FastBaseModel.get_peft_model`` (models/vision.py:4808, which names ``target_parameters``),
  with no ``functools.wraps``. So T24 read ``(model, *args, **kwargs)`` and refused U0/U1/UA on p238-shake-r5.

What. After ``unsloth.models.loader`` executes, set ``FastModel.get_peft_model.__wrapped__ = FastBaseModel.get_peft_model``:
the same metadata ``functools.wraps`` gives the sibling dispatcher. No call path changes; T24 then sees the function the kwargs
reach and passes ``target_parameters`` explicitly. It REFUSES (raises at that import, so the arm fails loud) unless the
installed unsloth is the pin, ``FastModel`` subclasses ``FastBaseModel``, both methods are staticmethods, the dispatcher's
signature is exactly ``(model, *args, **kwargs)`` with no ``__wrapped__`` yet, and the target names ``target_parameters``.
Inert in any process that never imports ``unsloth.models.loader`` (the e4b arms).

On the box, unsloth's vision.py:5709-5718 rebinds ``FastBaseModel.get_peft_model`` at import as
``staticmethod(wrap_loader_for_grouped_moe(...))`` (unsloth_zoo's wrapper, ``functools.wraps`` of the original), so the target
the shim reads is that wrapper and ``inspect.signature`` follows it to the original's ``target_parameters``. A sitecustomize
this file shadows is chained (run after the hook is installed), so placing the shim changes nothing else."""
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

PIN = "2026.10.8"
TARGET = "unsloth.models.loader"


def _refuse(why):
    raise RuntimeError(f"P238 shim: {why}; refusing (amendment 5 verified unsloth {PIN} only)")


def patch(mod):
    """Set the dispatcher's ``__wrapped__``, after every check above. Returns the line it prints."""
    import importlib.metadata as md
    import inspect
    try:
        got = md.version("unsloth")
    except md.PackageNotFoundError:
        _refuse("unsloth has no installed distribution metadata")
    if got != PIN:
        _refuse(f"unsloth is {got}, not {PIN}")
    fm, fb = getattr(mod, "FastModel", None), getattr(mod, "FastBaseModel", None)
    if not (isinstance(fm, type) and isinstance(fb, type) and issubclass(fm, fb)):
        _refuse(f"{TARGET} has no FastModel subclassing FastBaseModel")
    d, t = fm.__dict__.get("get_peft_model"), fb.__dict__.get("get_peft_model")
    if not (isinstance(d, staticmethod) and isinstance(t, staticmethod)):
        _refuse("FastModel.get_peft_model or FastBaseModel.get_peft_model is not a staticmethod")
    f, tf = d.__func__, t.__func__
    if hasattr(f, "__wrapped__"):
        _refuse("FastModel.get_peft_model already carries __wrapped__")
    P = inspect.Parameter
    shape = [(p.name, p.kind) for p in inspect.signature(f, follow_wrapped=False).parameters.values()]
    if shape != [("model", P.POSITIONAL_OR_KEYWORD), ("args", P.VAR_POSITIONAL), ("kwargs", P.VAR_KEYWORD)]:
        _refuse(f"FastModel.get_peft_model's signature is {shape}, not (model, *args, **kwargs)")
    if "target_parameters" not in inspect.signature(tf).parameters:
        _refuse("FastBaseModel.get_peft_model does not name target_parameters")
    f.__wrapped__ = tf
    line = (f"P238_SHIM FastModel.get_peft_model.__wrapped__ = FastBaseModel.get_peft_model (unsloth {got}); "
            f"inspect.signature now names target_parameters")
    print(line, file=sys.stderr, flush=True)
    return line


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch(module)
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())


def chain_shadowed():
    """Run the sitecustomize this one shadows, if any (one later on sys.path would otherwise never run), as Python would
    without the shim. Returns its path, or None."""
    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.loader is None:
        return None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    print(f"P238_SHIM chained the shadowed sitecustomize {spec.origin}", file=sys.stderr, flush=True)
    return spec.origin


chain_shadowed()
