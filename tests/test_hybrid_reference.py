# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``tests/hybrid_reference.py``: the CPU hybrid tests' masked import of transformers' hybrid modeling modules (#1454).

CI installs none of ``causal_conv1d``, ``fla`` and ``mamba_ssm``, so on CI an unmasked import is harmless and the
hazard never shows. Here pure-Python stand-ins shadow whatever is installed, in a fresh process: ``causal_conv1d`` and
``mamba_ssm`` export every function transformers looks up, each raising what a CUDA-only build raises on a CPU tensor,
and ``fla`` is empty. So both directions run anywhere, on a Gated DeltaNet, a short-conv and two Mamba families:

- a plain import binds the stand-ins and a CPU forward raises from them; ``reference_modeling`` then refuses that
  module, naming every bound function (read from the closures: the Mamba ones are not named after their kernels);
- ``reference_modeling`` on a family not yet imported binds the torch functions, the CPU forward runs, and the
  stand-ins are importable again afterwards (the mask is lifted after the import).

And no test module imports a modeling module transformers binds a kernel into, or one of its model classes, at
collection: that runs before any test can mask it. The families are read from the installed transformers' source.
"""
import ast
import importlib.util
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from hybrid_reference import KERNELS

pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
                    reason="needs transformers with Qwen3.5-MoE")

TESTS = Path(__file__).resolve().parent

RAISE = '''
def _cuda_only(*args, **kwargs):
    raise RuntimeError("stand-in kernel: Expected x.is_cuda() to be true")
'''

#: stand-in package -> {module path inside it: the functions transformers resolves there}
STAND_INS = {
    "causal_conv1d": {"": ("causal_conv1d_fn", "causal_conv1d_update")},
    "fla": {"": ()},
    "mamba_ssm": {"": (), "ops": (), "ops/triton": (),
                  "ops/triton/selective_state_update": ("selective_state_update",),
                  "ops/triton/ssd_combined": ("mamba_chunk_scan_combined", "mamba_split_conv1d_scan_combined"),
                  "ops/selective_scan_interface": ("mamba_inner_fn", "selective_scan_fn")},
}

SCRIPT = '''
import importlib, json, sys
sys.path[:0] = [{fake!r}, {tests!r}]
import torch
import transformers as tr
from hybrid_reference import bound_kernels, reference_modeling

def forward(model, n=8):
    try:
        with torch.no_grad():
            model.eval()(input_ids=torch.randint(0, 64, (1, n)))
        return "ran"
    except RuntimeError as e:
        return str(e)

COMMON = dict(vocab_size=64, hidden_size=64, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
GDN = dict(linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
           linear_conv_kernel_dim=4, layer_types=["linear_attention", "full_attention"], num_hidden_layers=2,
           head_dim=16)
BUILD = {{
    "qwen3_5": ("Qwen3_5ForCausalLM", tr.Qwen3_5TextConfig(intermediate_size=128, **COMMON, **GDN)),
    "qwen3_5_moe": ("Qwen3_5MoeForCausalLM", tr.Qwen3_5MoeTextConfig(
        moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
        **COMMON, **GDN)),
    "lfm2_moe": ("Lfm2MoeForCausalLM", tr.Lfm2MoeConfig(
        intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=4, num_experts=8, num_experts_per_tok=2,
        num_dense_layers=1, layer_types=["conv", "full_attention", "conv", "full_attention"], **COMMON)),
    "granitemoehybrid": ("GraniteMoeHybridForCausalLM", tr.GraniteMoeHybridConfig(
        intermediate_size=64, shared_intermediate_size=32, num_local_experts=4, num_experts_per_tok=2,
        num_hidden_layers=2, layer_types=["mamba", "attention"], mamba_n_heads=4, mamba_d_head=32, mamba_d_state=16,
        mamba_n_groups=1, mamba_chunk_size=8, **COMMON)),
    "nemotron_h": ("NemotronHForCausalLM", tr.NemotronHConfig(
        intermediate_size=64, num_hidden_layers=3, head_dim=16, layers_block_type=["mamba", "moe", "attention"],
        n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, moe_shared_expert_intermediate_size=32,
        n_groups=1, n_group=1, topk_group=1, mamba_num_heads=4, mamba_head_dim=16, ssm_state_size=16, chunk_size=8,
        **COMMON)),
}}

def build(mod, family):
    torch.manual_seed(0)
    cls, cfg = BUILD[family]
    return getattr(mod, cls)(cfg)

out = {{}}
for f in ("qwen3_5", "granitemoehybrid"):                       # plain imports: the stand-ins bind
    mod = importlib.import_module(f"transformers.models.{{f}}.modeling_{{f}}")
    out[f] = {{"bound": bound_kernels(mod), "forward": forward(build(mod, f))}}
    try:
        reference_modeling(f)
        out[f]["refused"] = None
    except RuntimeError as e:
        out[f]["refused"] = str(e)
for f in ("qwen3_5_moe", "lfm2_moe", "nemotron_h"):             # through the helper first
    mod = reference_modeling(f)
    out[f] = {{"bound": bound_kernels(mod), "forward": forward(build(mod, f))}}
out["after"] = sorted(k for k in ("causal_conv1d", "fla", "mamba_ssm")
                      if importlib.import_module(k).__file__.startswith({fake!r}))
print(json.dumps(out))
'''


def _stand_ins(root):
    for pkg, mods in STAND_INS.items():
        for sub, funcs in mods.items():
            d = root / pkg / sub if sub else root / pkg
            d.parent.mkdir(parents=True, exist_ok=True)
            body = (RAISE + "".join(f"{f} = _cuda_only\n" for f in funcs)) if funcs else ""
            if sub and not any(k.startswith(sub + "/") for k in mods):
                d.with_suffix(".py").write_text(body)              # a leaf module
            else:
                d.mkdir(exist_ok=True)
                (d / "__init__.py").write_text(body)


def test_a_plain_import_binds_the_kernels_and_reference_modeling_does_not(tmp_path):
    _stand_ins(tmp_path)
    script = textwrap.dedent(SCRIPT).format(fake=str(tmp_path), tests=str(TESTS))
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    want = {"qwen3_5": {"causal_conv1d_fn", "causal_conv1d_update"},
            "granitemoehybrid": {"causal_conv1d_fn", "causal_conv1d_update", "mamba2_split_conv1d_scan_combined",
                                 "mamba2_selective_state_update", "mamba2_chunk_scan"}}
    for f, names in want.items():
        assert set(out[f]["bound"]) == names, (f, out[f]["bound"])
        assert "stand-in kernel" in out[f]["forward"], (f, out[f]["forward"])
        assert out[f]["refused"] and f"modeling_{f} was imported" in out[f]["refused"], (f, out[f]["refused"])
        assert all(n in out[f]["refused"] for n in names), (f, out[f]["refused"])
    for f in ("qwen3_5_moe", "lfm2_moe", "nemotron_h"):
        assert out[f] == {"bound": [], "forward": "ran"}, (f, out[f])
    assert out["after"] == sorted(KERNELS)


def _kernel_modeling():
    """{modeling module: its ``__all__``} for every family whose modeling module binds a :data:`KERNELS` package at
    import, read from the installed transformers' source (nothing imported)."""
    root = Path(importlib.util.find_spec("transformers").submodule_search_locations[0]) / "models"
    binds = re.compile(r'use_kernel_func_from_hub_with_fallback\(\s*"[^"]+"\s*,\s*"(%s)"' % "|".join(KERNELS))
    out = {}
    for path in sorted(root.glob("*/modeling_*.py")):
        if path.name != f"modeling_{path.parent.name}.py":
            continue
        text = path.read_text(encoding="utf-8")
        if binds.search(text):
            names = set()
            for node in ast.parse(text).body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
                    names = set(ast.literal_eval(node.value))
            out[f"transformers.models.{path.parent.name}.{path.stem}"] = names
    return out


IMPORTERS = ("importorskip", "import_module", "__import__")


def _collection_imports(tree):
    """``(name, line)`` for what a test module's own statements reach when it is collected (outside every def and
    class): imports, ``importorskip`` / ``import_module`` / ``__import__`` calls, and attribute names."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Import):
                yield from ((a.name, n.lineno) for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                yield from ((f"{n.module}.{a.name}", n.lineno) for a in n.names)
            elif (isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant)
                  and getattr(n.func, "attr", getattr(n.func, "id", None)) in IMPORTERS):
                yield str(n.args[0].value), n.lineno
            elif isinstance(n, ast.Attribute):
                yield f"transformers.{n.attr}", n.lineno               # tr.<Model>: a lazy import of its module


def test_no_test_module_imports_a_kernel_binding_modeling_module_at_collection():
    modeling = _kernel_modeling()
    assert {"transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "transformers.models.lfm2_moe.modeling_lfm2_moe",
            "transformers.models.nemotron_h.modeling_nemotron_h"} <= modeling.keys()
    classes = {f"transformers.{c}" for names in modeling.values() for c in names}
    found = [f"{path.name}:{line} {name}" for path in sorted(TESTS.glob("*.py"))
             for name, line in _collection_imports(ast.parse(path.read_text(encoding="utf-8")))
             if name in classes or any(name == m or name.startswith(m + ".") for m in modeling)]
    assert not found, f"import these through hybrid_reference.reference_modeling, inside the test: {found}"
