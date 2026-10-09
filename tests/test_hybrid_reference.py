# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``tests/hybrid_reference.py``: the CPU hybrid tests' masked import of transformers' Gated DeltaNet modeling modules.

CI installs neither ``causal_conv1d`` nor ``fla``, so on CI an unmasked import is harmless and the hazard never shows.
Here a stand-in ``causal_conv1d`` (pure Python, raising what the CUDA-only build raises on a CPU tensor) and an empty
``fla`` shadow whatever is installed, in a fresh process, so both directions run anywhere:

- a plain import binds the stand-in and a CPU forward raises from it; ``reference_modeling`` then refuses that module,
  naming the bound functions;
- ``reference_modeling`` on a family not yet imported binds the torch functions, the CPU forward runs, and the stand-in
  is importable again afterwards (the mask is lifted after the import).

And no test module imports one of these modeling modules at module scope: that runs at collection, before any mask.
"""
import ast
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
                    reason="needs transformers with Qwen3.5-MoE")

TESTS = Path(__file__).resolve().parent

STAND_IN = '''
def _cuda_only(x, *args, **kwargs):
    raise RuntimeError("stand-in causal_conv1d: Expected x.is_cuda() to be true")

causal_conv1d_fn = causal_conv1d_update = _cuda_only
'''

SCRIPT = '''
import importlib, json, sys
sys.path[:0] = [{fake!r}, {tests!r}]
import torch
from transformers import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
import transformers.models.qwen3_5.modeling_qwen3_5 as dense          # plain import: binds the stand-in
from hybrid_reference import reference_modeling

LAYERS = dict(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
              linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
              vocab_size=64, hidden_size=64, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
              max_position_embeddings=128)

def forward(model):
    try:
        with torch.no_grad():
            model.eval()(input_ids=torch.randint(0, 64, (1, 6)))
        return "ran"
    except RuntimeError as e:
        return str(e)

out = {{"plain": forward(dense.Qwen3_5ForCausalLM(Qwen3_5TextConfig(intermediate_size=128, **LAYERS)))}}
try:
    reference_modeling("qwen3_5")
    out["refused"] = None
except RuntimeError as e:
    out["refused"] = str(e)
moe = reference_modeling("qwen3_5_moe")
out["masked"] = forward(moe.Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(
    moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2, **LAYERS)))
out["after"] = importlib.import_module("causal_conv1d").__file__.startswith({fake!r})
print(json.dumps(out))
'''


def test_a_plain_import_binds_the_kernel_and_reference_modeling_does_not(tmp_path):
    (tmp_path / "causal_conv1d").mkdir()
    (tmp_path / "causal_conv1d" / "__init__.py").write_text(STAND_IN)
    (tmp_path / "fla").mkdir()
    (tmp_path / "fla" / "__init__.py").write_text("")
    script = textwrap.dedent(SCRIPT).format(fake=str(tmp_path), tests=str(TESTS))
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert "stand-in causal_conv1d" in out["plain"]
    assert out["refused"] and "modeling_qwen3_5 " in out["refused"]
    assert "causal_conv1d_fn" in out["refused"] and "causal_conv1d_update" in out["refused"]
    assert out["masked"] == "ran"
    assert out["after"] is True


MODELING = re.compile(r"transformers\.models\.(qwen3_5_moe|qwen3_5|qwen3_next)\.modeling_")
MODEL_CLASS = re.compile(r"transformers\.(Qwen3_5|Qwen3Next)\w*(Model|For\w+)$")   # a lazy attribute that imports one
IMPORTERS = ("importorskip", "import_module", "__import__")


def _collection_imports(tree):
    """``(module, line)`` for what the module's own statements import when it is collected: import statements and
    ``importorskip`` / ``import_module`` / ``__import__`` calls outside every def and class."""
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


def test_no_test_module_imports_a_gated_deltanet_modeling_module_at_collection():
    found = [f"{path.name}:{line} {name}" for path in sorted(TESTS.glob("*.py"))
             for name, line in _collection_imports(ast.parse(path.read_text(encoding="utf-8")))
             if MODELING.match(name) or MODEL_CLASS.match(name)]
    assert not found, f"import these through hybrid_reference.reference_modeling, inside the test: {found}"
