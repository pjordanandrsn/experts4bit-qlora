# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Switch transformers' Gated DeltaNet functions between the kernels they resolved and their torch path, in one process.

transformers 5.17 resolves each of the four functions once, at import, through ``use_kernel_func_from_hub_with_fallback``:
the module-level name is a ``wrapped`` closure over ``implementation`` (fla's or causal-conv1d's function when the package
imports, else the torch function), ``is_new_implementation`` and ``applicable_params`` (the kwargs it forwards). The
models' forwards look the name up in their module at call time. Writing those three cells to the torch function's values
reproduces exactly what the closure holds when the package is absent; writing the saved values back restores the kernels.

Without the ``kernels`` hub package the outer decorators are identities, so the module attribute IS ``wrapped``; the
constructor refuses anything else.
"""
import inspect

GDN_FUNCS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")
_CELLS = ("implementation", "is_new_implementation", "applicable_params")


class GdnToggle:
    def __init__(self, modules):
        self.entries = []
        for mod in modules:
            for name in GDN_FUNCS:
                fn = getattr(mod, name)
                cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
                missing = set(_CELLS + ("torch_function",)) - cells.keys()
                if missing:
                    raise RuntimeError(f"{mod.__name__}.{name} is not transformers' fallback closure (missing {sorted(missing)})")
                tf = cells["torch_function"].cell_contents
                if getattr(fn, "__wrapped__", None) is not tf:
                    raise RuntimeError(f"{mod.__name__}.{name} does not wrap its torch function directly")
                resolved = tuple(cells[k].cell_contents for k in _CELLS)
                torch_path = (tf, False, tuple(inspect.signature(tf).parameters))
                self.entries.append((f"{mod.__name__}.{name}", cells, resolved, torch_path))
        self.state = "resolved"

    def use(self, which):
        """``"torch"``: every function takes the torch path; ``"resolved"``: what transformers resolved at import."""
        if which not in ("torch", "resolved"):
            raise ValueError(which)
        for _, cells, resolved, torch_path in self.entries:
            for k, v in zip(_CELLS, resolved if which == "resolved" else torch_path):
                cells[k].cell_contents = v
        self.state = which

    def record(self):
        """The implementation each function calls right now, by module (the engagement record)."""
        return {key: cells["implementation"].cell_contents.__module__ for key, cells, _, _ in self.entries}

    def resolved_record(self):
        return {key: resolved[0].__module__ for key, _, resolved, _ in self.entries}
