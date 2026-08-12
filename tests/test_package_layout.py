# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The subpackage split must not break anyone's imports.

Architectures, on-disk formats and execution engines were flat siblings in one
namespace, which left no visible answer to "where does a new one go". They now live
in arch/, formats/ and engines/ (see experts4bit_qlora/README-LAYOUT.md).

The public API is `__all__` and did not move. But this is a published package, so
somebody may already import a submodule by path — and breaking that at import time
is the worst way for them to find out. These pin both halves of the promise.
"""
import importlib

import pytest

pytest.importorskip("torch")

import experts4bit_qlora as e4b  # noqa: E402

LEGACY = ["awq", "gptq", "nvfp4", "mxfp4", "compressed_int", "fp8_blocks", "dense_disk",
          "pipelined", "cold_engine", "hot_residency", "offload", "capture", "kv_cache",
          "deepseek_v4", "glm5", "axk1", "mixtral", "gptoss", "moe_conventions"]


@pytest.mark.parametrize("old", LEGACY)
def test_the_old_submodule_path_still_imports(old):
    """`import experts4bit_qlora.awq` kept working across the move."""
    mod = importlib.import_module(f"experts4bit_qlora.{old}")
    assert mod is not None
    assert mod.__name__.startswith("experts4bit_qlora")


@pytest.mark.parametrize("old", LEGACY)
def test_the_old_name_is_reachable_as_an_ATTRIBUTE_too(old):
    """`experts4bit_qlora.awq` must resolve without an explicit import of it.

    Registering in `sys.modules` satisfies `import experts4bit_qlora.awq` and nothing
    else: a real import also binds the submodule on its parent package, and a cache
    hit skips that step, so attribute access falls through to `__getattr__` and
    raises. The test above passes over this entirely -- which is how the first
    version of the alias layer shipped with attribute access broken.
    """
    import experts4bit_qlora as pkg
    assert hasattr(pkg, old), f"experts4bit_qlora.{old} is not bound on the package"
    assert getattr(pkg, old) is importlib.import_module(f"experts4bit_qlora.{old}")


def test_new_paths_and_old_paths_are_the_same_module_object():
    """Arming: two distinct module objects would each pass the test above while
    holding separate copies of any module-level state."""
    new = importlib.import_module("experts4bit_qlora.engines.pipelined")
    old = importlib.import_module("experts4bit_qlora.pipelined")
    assert new is old, (new, old)


def test_the_public_api_did_not_move():
    """`__all__` is the contract; the reorg must be invisible through it."""
    missing = [n for n in e4b.__all__ if not hasattr(e4b, n)]
    assert not missing, f"names in __all__ that no longer resolve: {missing}"


@pytest.mark.parametrize("sub", ["arch", "formats", "engines"])
def test_each_subpackage_exists_and_is_documented(sub):
    importlib.import_module(f"experts4bit_qlora.{sub}")
    layout = (__import__("pathlib").Path(e4b.__file__).parent / "README-LAYOUT.md").read_text()
    assert f"`{sub}/`" in layout, f"{sub}/ is undocumented in README-LAYOUT.md"
