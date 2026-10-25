"""bench/common/unsloth_moe_counter.py on CPU stand-ins of unsloth_zoo 2026.10.8's two grouped paths.

The stand-ins sit at the real module paths (``unsloth_zoo.temporary_patches.moe_grouped_fp16`` / ``moe_nf4_grouped``) and reproduce
what the counter depends on in the release-proven wheel (sha256 db8c283b...; the line numbers are the wheel's):
- moe_utils resolves ``mg.generic_grouped_mm`` / ``mg.generic_grouped_wgrad`` through the module at call time (602, 615, 1408);
- ``generic_grouped_mm`` / ``generic_grouped_wgrad`` bump ``GENERIC_CALLS`` after the kernel returns (750, 763);
- ``triton_grouped_available`` runs ``_self_check_generic`` through module globals, once per device, and restores ``GENERIC_CALLS``
  around it (816-834); the self-check calls both functions through module globals (788-790);
- moe_utils imports ``maybe_nf4_grouped_mm`` at call time (1188); it returns None on a decline, and ``NF4_FUSED_STATS`` counts
  ``launches`` inside the op (269; -1 at 275 on a failed launch that then decodes eagerly), ``calls`` after a fused return (546)
  and ``fallbacks`` when the spec is None (531). Shape mismatches (534-537), ``FAILED_SHAPES`` entries (538-542) and ``auto`` above
  its cap (523) return None with no counter.
No real kernel runs here; the real binding check (the expected fused launch count) is the lane's, on the box.
"""
import functools
import importlib.machinery
import importlib.util
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_counter():
    spec = importlib.util.spec_from_file_location("unsloth_moe_counter", os.path.join(ROOT, "bench", "common", "unsloth_moe_counter.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_counter()


def _fake_zoo():
    """Stand-ins for the two library modules, plus a moe_utils-shaped caller."""
    pkg = types.ModuleType("unsloth_zoo")
    pkg.__path__ = []
    tp = types.ModuleType("unsloth_zoo.temporary_patches")
    tp.__path__ = []
    mg = types.ModuleType(C.FP16)
    mg.GENERIC_CALLS = {"gemm": 0, "wgrad": 0, "fallback": 0}
    mg._SELF_CHECKED = {}

    def generic_grouped_mm(inputs, weight, offsets, config_rows=None):
        out = inputs @ weight[0]
        mg.GENERIC_CALLS["gemm"] += 1
        return out

    def generic_grouped_wgrad(inputs, grad, offsets, num_experts):
        dw = (inputs.T @ grad).unsqueeze(0).expand(num_experts, -1, -1)
        mg.GENERIC_CALLS["wgrad"] += 1
        return dw

    def _self_check_generic(device):
        x, w = torch.ones(3, 2), torch.ones(1, 2, 2)
        mg.generic_grouped_mm(x, w, None)          # through module globals, as 788-790
        mg.generic_grouped_mm(x, w, None)
        mg.generic_grouped_wgrad(x, torch.ones(3, 2), None, 1)

    def triton_grouped_available(device):
        if device in mg._SELF_CHECKED:
            return mg._SELF_CHECKED[device]
        saved = dict(mg.GENERIC_CALLS)
        try:
            mg._self_check_generic(device)
            ok = True
        except RuntimeError:
            ok = False                               # 821-830: the generic path is disabled for the process
        finally:
            mg.GENERIC_CALLS.update(saved)          # "the self-check's own launches are not engagement" (833-834)
        mg._SELF_CHECKED[device] = ok
        return ok

    mg.generic_grouped_mm, mg.generic_grouped_wgrad = generic_grouped_mm, generic_grouped_wgrad
    mg._self_check_generic, mg.triton_grouped_available = _self_check_generic, triton_grouped_available

    nf = types.ModuleType(C.NF4)
    nf.NF4_FUSED_STATS = {"calls": 0, "fallbacks": 0, "launches": 0}
    nf.MODE = {"decline": None, "fail_launch": False, "ignore_auto_cap": False}
    nf.AUTO_MAX_ROWS = 128

    def nf4_fused_mode():                            # 56-64, as the library parses the environment
        mode = os.environ.get("UNSLOTH_MOE_NF4_FUSED", "0").strip().lower()
        if mode in ("1", "true", "on", "yes"):
            return "1"
        return mode if mode in ("auto", "force") else "0"

    def nf4_auto_max_rows():                         # 72-76
        try:
            return int(os.environ.get("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", nf.AUTO_MAX_ROWS))
        except ValueError:
            return nf.AUTO_MAX_ROWS

    def nf4_auto_fused(rows, num_experts):           # 79-81
        return rows <= nf4_auto_max_rows() * num_experts

    nf.nf4_fused_mode, nf.nf4_auto_max_rows, nf.nf4_auto_fused = nf4_fused_mode, nf4_auto_max_rows, nf4_auto_fused

    def maybe_nf4_grouped_mm(inputs, offsets, source, preprocess):
        if nf4_fused_mode() == "auto" and not nf4_auto_fused(inputs.shape[0], offsets.shape[0]) and not nf.MODE["ignore_auto_cap"]:
            return None                              # 523: auto above its cap, by design
        if nf4_fused_mode() == "0":
            return None                              # the spec is None under mode 0 (458), no fallback counted (530)
        if nf.MODE["decline"] in ("shape", "failed_shapes"):
            return None                              # 534-537 / 538-542: no counter
        if nf.MODE["decline"] == "spec_none":
            nf.NF4_FUSED_STATS["fallbacks"] += 1     # 531
            return None
        nf.NF4_FUSED_STATS["launches"] += 1          # 269, inside the op
        if nf.MODE["fail_launch"]:
            nf.NF4_FUSED_STATS["launches"] -= 1      # 275: decodes eagerly, uncounted
        out = inputs @ source
        nf.NF4_FUSED_STATS["calls"] += 1             # 546
        return out

    nf.maybe_nf4_grouped_mm = maybe_nf4_grouped_mm

    def nf4_dx(dy, source):
        """The custom op's backward (dX): counted inside the op, never through maybe_nf4_grouped_mm."""
        nf.NF4_FUSED_STATS["launches"] += 1
        return dy @ source.T

    nf.nf4_dx = nf4_dx

    def moe_utils_expert_forward(x, w, lora, device="cuda:0"):
        """One expert-module forward, the way moe_utils.forward_native_grouped_mm calls the two paths."""
        from unsloth_zoo.temporary_patches.moe_nf4_grouped import maybe_nf4_grouped_mm as maybe   # resolved at call time (1188)
        if not mg.triton_grouped_available(device):                      # 599: the self-check failed, torch._grouped_mm
            gm = lambda a, b: a @ b                                      # noqa: E731 -- uncounted here, as in the library
        else:
            gm = lambda a, b: mg.generic_grouped_mm(a, b.unsqueeze(0), None)   # noqa: E731 -- 602
        base = maybe(x, OFFS, w, None)
        if base is None:
            base = gm(x, w)                                              # the dequant path's base GEMM
        return base + gm(gm(x, lora[0]), lora[1])

    return pkg, tp, mg, nf, moe_utils_expert_forward


FAKE_MOE_UTILS = """
import os
import torch
_AUTO = {"lora": -1, "many": 2048, "few": 256}            # sm_120's row (moe_utils.py:505)
_FORCE_LORA_TORCH = False


def _triton_grouped_mm_kind(K, N, E):                      # 561
    return "lora" if min(K, N) <= 32 else ("many" if E >= 64 else "few")


def _triton_grouped_mm_max_rows(index, kind="lora"):       # 521-558: the env read at first use
    mode = os.environ.get("UNSLOTH_MOE_GROUPED_TRITON", "auto")
    if mode == "0" or os.environ.get("UNSLOTH_DISABLE_MOE_TRITON") == "1":
        return 0
    return -1 if mode == "1" else _AUTO.get(kind, 0)


def _triton_grouped_mm_wanted(inputs, weight):             # 572-589
    if _triton_grouped_mm_max_rows(inputs.device.index, "lora") == 0 or weight.dim() != 3:
        return False
    E, K, N = weight.shape
    if _FORCE_LORA_TORCH and min(K, N) <= 32:
        return False                                       # e.g. a tensor subclass: False, silently
    lim = _triton_grouped_mm_max_rows(inputs.device.index, _triton_grouped_mm_kind(K, N, E))
    return lim != 0 and (lim < 0 or inputs.shape[0] <= lim * E)


def _torch_gm(inputs, weight):
    return inputs @ weight[0]


def _grouped_mm_with_backward_fix(inputs, weight, offsets):   # 366-412
    if _triton_grouped_mm_wanted(inputs, weight):
        from unsloth_zoo.temporary_patches import moe_grouped_fp16 as mg
        if not mg.triton_grouped_available(inputs.device):
            mg.GENERIC_CALLS["fallback"] += 1                 # 600
            return _torch_gm(inputs, weight)
        return mg.generic_grouped_mm(inputs, weight, offsets)
    return _torch_gm(inputs, weight)


def _lora_rank_pad_needed(inputs, first_weight):           # 2997-3011: a padding probe, not a GEMM
    return not _triton_grouped_mm_wanted(inputs, first_weight)


def forward_native(x, base, lora_a, lora_b):               # forward_native_grouped_mm, 5244-
    _lora_rank_pad_needed(x, lora_a)
    with torch.no_grad():
        y = _grouped_mm_with_backward_fix(x, base, None)     # the dequant path's recompute forward (1100): no backward node
    h = _grouped_mm_with_backward_fix(x, lora_a, None)
    return y + _grouped_mm_with_backward_fix(h, lora_b, None)


def forward_moe_backend(self, x, base, lora_a, lora_b):    # 2051-2112: the bnb 4-bit hand-off first, by ABSOLUTE import
    try:
        from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import _moe_uses_bnb4bit_expert_weights, forward_moe_backend_bnb4bit
    except ImportError:
        _moe_uses_bnb4bit_expert_weights = forward_moe_backend_bnb4bit = None
    if _moe_uses_bnb4bit_expert_weights is not None and _moe_uses_bnb4bit_expert_weights(self):
        return forward_moe_backend_bnb4bit(self, x, base, lora_a, lora_b)
    return forward_native(x, base, lora_a, lora_b)


_CACHE = None


def get_forward_moe_backend():                             # 348-356
    return _CACHE.forward_moe_backend if _CACHE is not None else forward_moe_backend
"""


def _module_from_source(name, source, path):
    path.write_text(source, encoding="utf-8")
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


class _Zoo(tuple):
    """(mg, nf, fwd), with the moe_utils stand-in and the test's tmp dir as attributes."""


def _package(name, root):
    """A stand-in package with a real source directory, so the counter's whole-package scan (find_spec) reads its files."""
    mod = types.ModuleType(name)
    mod.__path__ = [str(root)]
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    mod.__spec__.submodule_search_locations = [str(root)]
    return mod


@pytest.fixture
def zoo(monkeypatch, tmp_path):
    for k in [k for k in os.environ if k.startswith("UNSLOTH_")]:  # the arm's environment starts with no UNSLOTH_* key
        monkeypatch.delenv(k)
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "1")
    pkg, tp, mg, nf, fwd = _fake_zoo()
    zroot, uroot = tmp_path / "pkgs" / "unsloth_zoo", tmp_path / "pkgs" / "unsloth"
    tdir = zroot / "temporary_patches"
    tdir.mkdir(parents=True)
    uroot.mkdir(parents=True)
    (zroot / "__init__.py").write_text("", encoding="utf-8")
    (tdir / "moe_grouped_fp16.py").write_text('os.environ.get("UNSLOTH_GPTOSS_FP16_GEMM")\n', encoding="utf-8")
    (tdir / "moe_nf4_grouped.py").write_text('os.environ.get("UNSLOTH_MOE_NF4_FUSED", "0")\nos.environ.get("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS")\n'
                                             'os.environ.get("UNSLOTH_MOE_NF4_AUTOTUNE", "0")\n', encoding="utf-8")
    (tdir / "moe_router_triton.py").write_text('_ON = os.environ.get("UNSLOTH_MOE_FUSED_ROUTER", "1") == "1"\n', encoding="utf-8")
    (uroot / "__init__.py").write_text('if "UNSLOTH_COMPILE_DISABLE" in os.environ: pass\n', encoding="utf-8")
    mg.__file__, nf.__file__ = str(tdir / "moe_grouped_fp16.py"), str(tdir / "moe_nf4_grouped.py")
    mu = _module_from_source(C.MOE_UTILS, FAKE_MOE_UTILS, tdir / "moe_utils.py")
    pkg, tp = _package("unsloth_zoo", zroot), _package("unsloth_zoo.temporary_patches", tdir)
    for name, mod in (("unsloth_zoo", pkg), ("unsloth_zoo.temporary_patches", tp), (C.FP16, mg), (C.NF4, nf), (C.MOE_UTILS, mu),
                      ("unsloth", _package("unsloth", uroot))):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "unsloth_zoo.temporary_patches.moe_utils_bnb4bit", raising=False)
    monkeypatch.delitem(sys.modules, "unsloth_zoo.temporary_patches.moe_utils_fp8", raising=False)
    z = _Zoo((mg, nf, fwd))
    z.mu, z.tmp, z.mp = mu, tdir, monkeypatch
    return z


X, W, LORA = torch.ones(4, 2), torch.ones(2, 3), (torch.ones(2, 1), torch.ones(1, 3))
OFFS = torch.tensor([1, 2, 3, 4], dtype=torch.int32)            # 4 experts: auto admits up to 128 x 4 = 512 rows
X_BIG = torch.ones(600, 2)                                       # 600 rows > 512: auto declines, by design


def test_targets_are_the_release_module_paths():
    assert C.FP16 == "unsloth_zoo.temporary_patches.moe_grouped_fp16"
    assert C.NF4 == "unsloth_zoo.temporary_patches.moe_nf4_grouped"
    assert C.TARGETS == {"generic_gemm": (C.FP16, "generic_grouped_mm"), "generic_wgrad": (C.FP16, "generic_grouped_wgrad"),
                         "nf4_maybe": (C.NF4, "maybe_nf4_grouped_mm")}
    assert C.SELF_CHECK == (C.FP16, "_self_check_generic")


def test_unfused_arm_counts_generic_and_keeps_the_self_check_apart(zoo, monkeypatch):
    mg, nf, fwd = zoo
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "0")
    c = C.UnslothMoeCounter().install()
    for _ in range(3):
        fwd(X, W, LORA)
    s = c.snapshot()
    assert s["counts"]["generic_gemm"] == 9                     # 3 forwards x (base + LoRA A + LoRA B)
    assert (s["counts"]["selfcheck_gemm"], s["counts"]["selfcheck_wgrad"]) == (2, 1)
    assert C.check_library(s) == (True, "")
    assert C.check_unfused(s) == (True, "")
    assert C.check_fused(s, 6)[0] is False
    c.uninstall()
    assert mg.generic_grouped_mm.__name__ == "generic_grouped_mm" and not hasattr(mg.generic_grouped_mm, "__wrapped__")


def test_fused_arm_binds_on_launches_and_the_fused_path_bypasses_the_generic_counter(zoo):
    mg, nf, fwd = zoo
    c = C.UnslothMoeCounter().install()
    for _ in range(3):
        fwd(X, W, LORA)
        nf.nf4_dx(torch.ones(4, 3), W)                            # the base GEMM's dX, in the op's backward
    s = c.snapshot()
    # the fused base GEMM BYPASSES the generic counter: only the 2 LoRA GEMMs per forward reach it
    assert s["counts"]["generic_gemm"] == 6 and s["counts"]["nf4_fused_calls"] == 3
    assert s["library_delta"]["nf4"]["launches"] == 6             # 3 forward + 3 dX
    assert C.check_fused(s, 6) == (True, "")
    assert C.check_library(s) == (True, "")
    assert C.check_unfused(s)[0] is False
    # MUTANT: a generic-only bar would read the fused arm as 3 base GEMMs short
    assert s["counts"]["generic_gemm"] != 9


@pytest.mark.parametrize("decline", ["shape", "failed_shapes"])
def test_each_silent_decline_voids_the_fused_arm(zoo, decline):
    mg, nf, fwd = zoo
    nf.MODE["decline"] = decline
    c = C.UnslothMoeCounter().install()
    for _ in range(3):
        fwd(X, W, LORA)
    s = c.snapshot()
    assert s["library_delta"]["nf4"] == {"calls": 0, "fallbacks": 0, "launches": 0}   # silent: no library counter moved
    ok, why = C.check_fused(s, 6)
    assert not ok and "launches 0 != expected 6" in why


def test_a_failed_launch_that_decodes_eagerly_voids_the_fused_arm(zoo):
    mg, nf, fwd = zoo
    nf.MODE["fail_launch"] = True
    c = C.UnslothMoeCounter().install()
    for _ in range(3):
        fwd(X, W, LORA)
        nf.nf4_dx(torch.ones(4, 3), W)
    s = c.snapshot()
    assert s["counts"]["nf4_fused_calls"] == 3                    # calls alone would pass: it is reported, never binding
    assert not C.check_fused(s, 6)[0]


def test_a_local_alias_bound_before_install_is_missed_and_the_library_cross_check_catches_it(zoo, monkeypatch):
    mg, nf, fwd = zoo
    alias = mg.generic_grouped_mm                                 # bound BEFORE install: bypasses the wrapper
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "0")
    c = C.UnslothMoeCounter().install()
    alias(X, W.unsqueeze(0), None)
    s = c.snapshot()
    assert s["counts"]["generic_gemm"] == 0                       # missed
    ok, why = C.check_library(s)
    assert not ok and "generic_gemm: wrapper 0 != library GENERIC_CALLS 1" in why


def test_a_captured_block_undercounts_the_wrapper_and_check_equal_catches_it(zoo):
    mg, nf, fwd = zoo
    c = C.UnslothMoeCounter().install()
    lib = torch.library.Library("umc_test", "FRAGMENT")
    lib.define("launch(Tensor x) -> Tensor")
    in_op = {"n": 0}

    def _impl(x):
        in_op["n"] += 1                                           # an in-op counter, like NF4_FUSED_STATS.launches
        return x * 2

    lib.impl("launch", _impl, "CPU")
    lib.impl("launch", lambda x: torch.empty_like(x), "Meta")

    class Block(torch.nn.Module):
        def forward(self, x):
            return torch.ops.umc_test.launch(mg.generic_grouped_mm(x, W.unsqueeze(0), None))

    # a captured program: the Python wrapper runs during capture, never when the program runs (as under CUDA-graph replay)
    captured = torch.export.export(Block(), (X,), strict=False).module()
    before_wrapper, before_op = c.counts["generic_gemm"], in_op["n"]
    for _ in range(4):
        captured(X)
    ran_wrapper, ran_op = c.counts["generic_gemm"] - before_wrapper, in_op["n"] - before_op
    assert ran_op == 4 and ran_wrapper == 0                       # the custom op counted every call; the wrapper none
    ok, why = C.check_equal("generic_gemm", ran_wrapper, ran_op, "in-op launches")
    assert not ok and "traced or captured" in why
    assert C.check_equal("generic_gemm", 4, 4, "in-op launches") == (True, "")


def test_an_absent_target_fails_every_check(monkeypatch):
    pkg, tp, mg, nf, fwd = _fake_zoo()
    del mg.generic_grouped_wgrad
    for name, mod in (("unsloth_zoo", pkg), ("unsloth_zoo.temporary_patches", tp), (C.FP16, mg), (C.NF4, nf)):
        monkeypatch.setitem(sys.modules, name, mod)
    c = C.UnslothMoeCounter().install()
    s = c.snapshot()
    assert any("generic_grouped_wgrad: absent" in a for a in s["absent"])
    assert not C.check_library(s)[0]


def test_no_library_at_all_reads_unread_never_pass(monkeypatch):
    for name in ("unsloth_zoo", "unsloth_zoo.temporary_patches", C.FP16, C.NF4):
        monkeypatch.setitem(sys.modules, name, None)              # import fails
    c = C.UnslothMoeCounter().install()
    s = c.snapshot()
    assert s["library_delta"] == {"generic": None, "nf4": None}
    assert not C.check_library(s)[0] and not C.check_fused(s, 6)[0] and not C.check_unfused(s)[0]


def test_a_failed_self_check_leaves_the_generic_counts_at_zero_so_a_lane_bar_fails_closed(zoo, monkeypatch):
    mg, nf, fwd = zoo
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "0")

    def failing_self_check(device):
        raise RuntimeError("generic grouped GEMM self-check bfloat16 forward: 1.0")

    mg._self_check_generic = failing_self_check                   # the library then uses torch._grouped_mm (821-830)
    c = C.UnslothMoeCounter().install()
    for _ in range(3):
        fwd(X, W, LORA)
    s = c.snapshot()
    assert s["counts"]["generic_gemm"] == 0 and s["counts"]["selfcheck_gemm"] == 0
    assert C.check_library(s) == (True, "")                       # consistent: nothing reached the generic path
    ok, why = C.check_equal("generic_gemm", s["counts"]["generic_gemm"], 9, "the lane's expected 3E")
    assert not ok                                                 # the lane's bar fails CLOSED, never passes on zero


def _run(fwd, nf, x, n=3):
    for _ in range(n):
        fwd(x, W, LORA)


def test_auto_at_an_eligible_shape_fuses_every_call(zoo, monkeypatch):
    mg, nf, fwd = zoo
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "auto")
    c = C.UnslothMoeCounter().install()
    _run(fwd, nf, X)
    s = c.snapshot()
    assert (s["counts"]["nf4_auto_eligible"], s["counts"]["nf4_auto_ineligible"]) == (3, 0)
    assert C.fused_calls_expected(s, "auto") == 3 == s["counts"]["nf4_fused_calls"]
    assert C.check_auto(s) == (True, "")


def test_auto_at_an_ineligible_shape_declines_by_design_and_reads_like_the_unfused_route(zoo, monkeypatch):
    mg, nf, fwd = zoo
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "auto")
    c = C.UnslothMoeCounter().install()
    _run(fwd, nf, X_BIG)
    s = c.snapshot()
    assert (s["counts"]["nf4_auto_eligible"], s["counts"]["nf4_auto_ineligible"]) == (0, 3)
    assert C.fused_calls_expected(s, "auto") == 0
    assert C.check_auto(s) == (True, "")
    assert C.check_fused(s, 0) == (True, "")                      # 0 launches expected above the cap
    assert s["counts"]["generic_gemm"] == 9                       # the dequant route, as the =0 arm


def test_auto_mutant_a_forced_decline_at_an_eligible_shape_voids(zoo, monkeypatch):
    mg, nf, fwd = zoo
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "auto")
    nf.MODE["decline"] = "shape"
    c = C.UnslothMoeCounter().install()
    _run(fwd, nf, X)
    s = c.snapshot()
    ok, why = C.check_auto(s)
    assert not ok and "0 fused calls != 3" in why
    assert not C.check_fused(s, 2 * C.fused_calls_expected(s, "auto"))[0]


def test_auto_mutant_b_a_launch_at_an_ineligible_shape_voids(zoo, monkeypatch):
    mg, nf, fwd = zoo
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "auto")
    nf.MODE["ignore_auto_cap"] = True
    c = C.UnslothMoeCounter().install()
    _run(fwd, nf, X_BIG)
    s = c.snapshot()
    ok, why = C.check_auto(s)
    assert not ok and "3 fused calls != 0" in why
    assert not C.check_fused(s, 0)[0]


def test_auto_mutant_c_an_ambient_auto_max_rows_voids_by_the_env_check(zoo, monkeypatch):
    mg, nf, fwd = zoo
    pins = {"UNSLOTH_MOE_NF4_FUSED": "auto", "UNSLOTH_MOE_NF4_AUTO_MAX_ROWS": "128", "UNSLOTH_MOE_NF4_AUTOTUNE": "0"}
    for k, v in pins.items():
        monkeypatch.setenv(k, v)
    snap = C.UnslothMoeCounter().install().snapshot()
    assert C.check_env(snap, dict(os.environ), pins) == (True, "")
    monkeypatch.setenv("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", "4096")   # ambient: moves auto's cap
    ok, why = C.check_env(snap, dict(os.environ), pins)
    assert not ok and "UNSLOTH_MOE_NF4_AUTO_MAX_ROWS='4096' (registered '128')" in why
    monkeypatch.setenv("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", "128")
    monkeypatch.delenv("UNSLOTH_MOE_NF4_AUTOTUNE")                # a registered pin missing
    assert not C.check_env(snap, dict(os.environ), pins)[0]


def test_the_auto_count_uses_the_librarys_own_predicate_never_a_retyped_formula(zoo, monkeypatch):
    mg, nf, fwd = zoo
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "auto")
    nf.AUTO_MAX_ROWS = 1000                                       # the library's constant moves: 600 rows become eligible
    c = C.UnslothMoeCounter().install()
    _run(fwd, nf, X_BIG)
    s = c.snapshot()
    assert s["counts"]["nf4_auto_eligible"] == 3 and C.check_auto(s) == (True, "")


def test_an_unreadable_auto_predicate_reads_unread_and_voids(zoo, monkeypatch):
    mg, nf, fwd = zoo
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "auto")
    del nf.nf4_auto_fused                                         # the module attribute the counter reads at call time
    c = C.UnslothMoeCounter().install()
    _run(fwd, nf, X)
    s = c.snapshot()
    assert s["counts"]["nf4_mode_unread"] == 3
    assert C.fused_calls_expected(s, "auto") is None
    assert not C.check_auto(s)[0]


# -- the per-call dispatch: moe_utils' own predicate, attributed to the expert-module pass it ran in -------------------------------
E_BASE, R = 64, 16
CKPT = ("none", "nonreentrant", "reentrant", "unsloth")       # the four checkpoint shapes (forwards / under grad / dX per step):
#                                                              none 1/1/1, HF non-reentrant 2/2/1, reentrant 2/1/1, Unsloth's 2/1/1


def _stacks(base_rows=8):
    x = torch.ones(base_rows, 48, requires_grad=True)
    base = torch.ones(E_BASE, 48, 40)                             # "many" (E >= 64, min(K, N) > 32), frozen
    lora_a = torch.ones(E_BASE, 48, R, requires_grad=True)        # "lora" (min(K, N) <= 32)
    lora_b = torch.ones(E_BASE, R, 40, requires_grad=True)
    return x, base, lora_a, lora_b


def _backward_as_the_library_would(mg, n_gx, n_dw):
    """The Triton autograd's backward on CPU: gx through generic_grouped_mm (634-636), LoRA dW through generic_grouped_wgrad."""
    for _ in range(n_gx):
        mg.generic_grouped_mm(torch.ones(2, 3), torch.ones(1, 3, 2), None)
    for _ in range(n_dw):
        mg.generic_grouped_wgrad(torch.ones(2, 3), torch.ones(2, 2), None, 1)


def _experts(forward, name="Qwen3_5MoeExperts", n=1, nf4=False):
    """n expert modules whose class forward is ``forward`` (as patch_function binds it, qwen3_5_moe.py:53-60); ``nf4`` marks their
    stacks bnb 4-bit for the hand-off predicate stand-in."""
    cls = type(name, (torch.nn.Module,), {"forward": forward})
    mods = [cls() for _ in range(n)]
    for m in mods:
        m.nf4 = nf4
    return mods


def _copy(zoo, name):
    return _module_from_source("unsloth_cached_moe_utils", FAKE_MOE_UTILS, zoo.tmp / f"{name}.py")


def _bnb4bit(zoo):
    """moe_utils_bnb4bit: the library's own predicate, and a hand-off whose RELATIVE import reaches the PACKAGE moe_utils
    (moe_utils_bnb4bit.py:367-376), never the compile-cache copy."""
    b = types.ModuleType("unsloth_zoo.temporary_patches.moe_utils_bnb4bit")
    b.__package__ = "unsloth_zoo.temporary_patches"

    def _moe_uses_bnb4bit_expert_weights(self):
        return bool(getattr(self, "nf4", False))

    def forward_moe_backend_bnb4bit(self, x, base, lora_a, lora_b):
        return sys.modules[f"{b.__package__}.moe_utils"].forward_native(x, base, lora_a, lora_b)

    b._moe_uses_bnb4bit_expert_weights, b.forward_moe_backend_bnb4bit = _moe_uses_bnb4bit_expert_weights, forward_moe_backend_bnb4bit
    zoo.mp.setitem(sys.modules, b.__name__, b)
    return b


def _step(c, mg, module, i, ckpt="unsloth", args=None, mb=1):
    """One optimizer step through ``module`` under a checkpoint shape, then the backward the library would run: per micro-batch, the
    Triton gx of the two LoRA GEMMs (inputs that require grad) and their two dW. The base GEMM runs under no_grad (1100)."""
    args = args or _stacks()
    c.mark_step(i)
    for _ in range(mb):
        if ckpt == "none":
            module(*args)
        elif ckpt == "nonreentrant":                              # both passes under grad; only the original is backwarded
            module(*args)
            module(*args)
        else:                                                     # reentrant / Unsloth: no-grad first, then the replay under grad
            with torch.no_grad():
                module(*args)
            module(*args)
        _backward_as_the_library_would(mg, 2, 2)


def test_dispatch_on_the_package_module_and_the_structural_counts(zoo):
    mg, nf, fwd = zoo
    mu = zoo.mu
    ex = _experts(mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    _step(c, mg, ex, 0, "none")
    c.mark_step(1)
    s, st = c.snapshot(), c.structural(1, 1)
    assert s["served"][0]["is_package"] and s["dispatch_served"] == {"package": {"gemm": 3, "probe": 1}}
    assert s["dispatch"] == {"probe:lora:triton": 1, "gemm:many:triton": 1, "gemm:lora:triton": 2}
    assert st["faults"] == [] and st["expected"] == {"dx_nf4": 0, "gx": 2, "wgrad": 2}
    assert C.implied_generic(s, st) == {"generic_gemm": 5, "generic_wgrad": 2}            # 3 forwards + 2 gx; 2 LoRA dW
    assert C.check_dispatch(s, st, base_must_be_triton=True) == (True, "")
    assert any(k.endswith(":many") and v == 2048 for k, v in s["limits"].items())


@pytest.mark.parametrize("ckpt", CKPT)
def test_the_structural_count_reads_right_under_every_checkpoint_shape(zoo, ckpt):
    mg, nf, fwd = zoo
    ex = _experts(zoo.mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    for i in range(3):
        _step(c, mg, ex, i, ckpt)
    c.mark_step(3)
    s, st = c.snapshot(), c.structural(1, 3)
    assert st["faults"] == [] and st["expected"]["gx"] == 6 and st["expected"]["wgrad"] == 6
    assert C.check_dispatch(s, st) == (True, "")
    # the under-grad split agrees only where each under-grad pass is backwarded once
    assert st["split"]["agrees"] is (ckpt != "nonreentrant")


def test_mutant_the_under_grad_rule_double_counts_under_hf_non_reentrant(zoo):
    """The old rule (forwards + the backward gx of every Triton call made under grad) expects 2x the real gx there."""
    mg, nf, fwd = zoo
    ex = _experts(zoo.mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    _step(c, mg, ex, 0, "nonreentrant")
    c.mark_step(1)
    s = c.snapshot()
    fwd_triton = sum(v for k, v in s["dispatch"].items() if k.startswith("gemm:") and k.endswith(":triton"))
    old = fwd_triton + s["dispatch_grad"]["triton_grad_input"]
    assert old != s["counts"]["generic_gemm"] == C.implied_generic(s, c.structural(1, 1))["generic_gemm"]


def test_micro_batches_multiply_the_backward(zoo):
    mg, nf, fwd = zoo
    ex = _experts(zoo.mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    _step(c, mg, ex, 0, "unsloth", mb=4)
    c.mark_step(1)
    st = c.structural(4, 1)
    assert st["expected"]["gx"] == 8 and C.check_dispatch(c.snapshot(), st) == (True, "")
    assert C.check_dispatch(c.snapshot(), c.structural(1, 1))[0] is False     # a lane that understates its micro-batches VOIDs


def test_passes_that_decide_differently_inside_one_step_void(zoo):
    mg, nf, fwd = zoo
    ex = _experts(zoo.mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    c.mark_step(0)
    with torch.no_grad():
        ex(*_stacks(base_rows=8))
    ex(*_stacks(base_rows=2048 * E_BASE + 1))                     # the replay at another shape: the base GEMM decides torch
    c.mark_step(1)
    st = c.structural(1, 1)
    assert st["faults"] and "decided differently" in st["faults"][0]
    assert C.check_dispatch(c.snapshot(), st)[0] is False


def test_an_eval_pass_is_not_a_training_pass(zoo):
    mg, nf, fwd = zoo
    ex = _experts(zoo.mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    _step(c, mg, ex, 0, "unsloth")
    ex.eval()
    ex(*_stacks(base_rows=4))                                     # tc1_arm's eval_loss: eval mode, grad on, another shape
    ex.train()
    c.mark_step(1)
    s, st = c.snapshot(), c.structural(1, 1)
    assert st["faults"] == [] and C.check_dispatch(s, st) == (True, "")


def test_a_step_with_no_training_pass_and_a_pass_outside_every_step_void(zoo):
    mg, nf, fwd = zoo
    ex = _experts(zoo.mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    ex(*_stacks())                                                # before the first mark
    c.mark_step(0)
    c.mark_step(1)
    st = c.structural(1, 1)
    assert any("outside a marked step" in f for f in st["faults"]) and any("ran no training pass" in f for f in st["faults"])


def test_no_structure_fails_closed(zoo):
    mg, nf, fwd = zoo
    s = C.UnslothMoeCounter().install().snapshot()
    assert C.implied_generic(s, None) is None
    ok, why = C.check_dispatch(s, None)
    assert not ok and "no structural count" in why
    assert C.UnslothMoeCounter().install().structural(1, 1)["faults"]          # nothing tracked


def test_a_cache_copy_serving_the_forward_is_counted(zoo):
    mg, nf, fwd = zoo
    copy = _copy(zoo, "cached_moe_utils")
    ex = _experts(copy.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    _step(c, mg, ex, 0)
    c.mark_step(1)
    s = c.snapshot()
    assert s["served"][0]["is_package"] is False and s["served"][0]["module"] == "unsloth_cached_moe_utils"
    assert s["dispatch_served"]["copy1"]["gemm"] == 6
    assert C.check_dispatch(s, c.structural(1, 1)) == (True, "")


def test_mutant_a_cache_copy_appearing_after_install_without_patch_loaded_voids(zoo):
    mg, nf, fwd = zoo
    mu = zoo.mu
    c = C.UnslothMoeCounter().install()                           # the package served at install; nothing tracked
    mu._CACHE = _copy(zoo, "late_cache")
    ex = _experts(mu.get_forward_moe_backend())[0]
    ex(*_stacks())
    ok, why = C.check_dispatch(c.snapshot(), c.structural(1, 1))
    assert not ok and "no decision counted through the serving module" in why


# -- bnb 4-bit experts are served by the PACKAGE moe_utils, whatever object their class forward came from ----------------------------
@pytest.mark.parametrize("name", ["Qwen3_5MoeExperts", "Qwen3MoeExperts"])
def test_bnb4bit_experts_on_the_copys_forward_are_served_by_the_package(zoo, name):
    mg, nf, fwd = zoo
    _bnb4bit(zoo)
    copy = _copy(zoo, "cache_b")
    ex = _experts(copy.forward_moe_backend, name=name, n=2, nf4=True)
    c = C.UnslothMoeCounter().install().patch_loaded(ex)
    _step(c, mg, ex[0], 0)
    _step(c, mg, ex[1], 0)
    c.mark_step(1)
    s = c.snapshot()
    assert [(x["tag"], x["source"]) for x in s["served"]] == [("package", "bnb4bit")]
    assert s["expert_forwards"][f"{__name__}.{name}"][0]["rule"] == "bnb4bit"
    assert set(s["dispatch_served"]) == {"package"}
    assert C.check_dispatch(s, c.structural(1, 1)) == (True, "")


@pytest.mark.parametrize("name", ["Qwen3_5MoeExperts", "Qwen3MoeExperts"])
def test_mutant_without_the_bnb4bit_rule_the_nf4_experts_void(zoo, name):
    mg, nf, fwd = zoo
    _bnb4bit(zoo)
    zoo.mp.setattr(C, "HANDOFF", ())                              # the rule dropped: serve by the class forward only
    copy = _copy(zoo, "cache_m")
    ex = _experts(copy.forward_moe_backend, name=name, nf4=True)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    _step(c, mg, ex, 0)
    c.mark_step(1)
    ok, why = C.check_dispatch(c.snapshot(), c.structural(1, 1))
    assert not ok and "no decision counted through the serving module (copy1" in why and "serves no forward" in why


def test_a_16bit_stack_on_the_copy_still_requires_the_copys_decisions(zoo):
    mg, nf, fwd = zoo
    _bnb4bit(zoo)
    copy = _copy(zoo, "cache_16")
    ex = _experts(copy.forward_moe_backend, nf4=False)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    _step(c, mg, ex, 0)
    c.mark_step(1)
    s = c.snapshot()
    assert [(x["tag"], x["source"]) for x in s["served"]] == [("copy1", "loaded")]
    assert C.check_dispatch(s, c.structural(1, 1)) == (True, "")


def test_a_mixed_model_one_bnb4bit_type_and_one_16bit_type(zoo):
    mg, nf, fwd = zoo
    _bnb4bit(zoo)
    copy = _copy(zoo, "cache_mix")
    a = _experts(copy.forward_moe_backend, name="ExpertsNF4", nf4=True)[0]
    b = _experts(copy.forward_moe_backend, name="ExpertsBF16", nf4=False)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([a, b])
    _step(c, mg, a, 0)
    ok, why = C.check_dispatch(c.snapshot(), c.structural(1, 1))
    assert not ok                                                 # b never ran: no decision through the copy, no pass for b
    _step(c, mg, b, 0)
    c.mark_step(1)
    s = c.snapshot()
    assert {(x["tag"], x["source"]) for x in s["served"]} == {("package", "bnb4bit"), ("copy1", "loaded")}
    assert {e["rule"] for v in s["expert_forwards"].values() for e in v} == {"bnb4bit", "dispatcher"}
    assert C.check_dispatch(s, c.structural(1, 1)) == (True, "")


def test_mutant_experts_left_on_the_original_forward_void(zoo):
    mg, nf, fwd = zoo

    def original_forward(self, hidden_states):
        return hidden_states

    c = C.UnslothMoeCounter().install().patch_loaded(_experts(original_forward, n=3))
    s = c.snapshot()
    assert s["expert_forwards"][f"{__name__}.Qwen3_5MoeExperts"][0]["globals_module"] is None
    ok, why = C.check_dispatch(s, c.structural(1, 1))
    assert not ok and "no _triton_grouped_mm_wanted in its globals (3 modules)" in why


def test_a_wrapped_forward_unwraps_to_the_predicates_globals(zoo):
    mg, nf, fwd = zoo
    inner = zoo.mu.forward_moe_backend

    @functools.wraps(inner)
    def decorated(*args, **kwargs):                               # a decorator or compile carrier defined elsewhere
        return inner(*args, **kwargs)

    assert C.DISPATCH not in decorated.__globals__
    c = C.UnslothMoeCounter().install().patch_loaded(_experts(decorated))
    s = c.snapshot()
    assert s["expert_forwards"][f"{__name__}.Qwen3_5MoeExperts"][0]["unwrapped"] == 1
    assert [x["tag"] for x in s["served"]] == ["package"]


def test_the_wrappers_and_hooks_observe_and_never_change_anything(zoo):
    mg, nf, fwd = zoo
    mu = zoo.mu
    copy = _copy(zoo, "cache_eq")
    ex = _experts(copy.forward_moe_backend)[0]
    args = _stacks()
    want = ex(*args)                                              # before install: the plain call
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    assert torch.equal(ex(*args), want)                           # the hooks pass inputs and output through
    x, base, la, lb = args
    huge = torch.ones(2048 * E_BASE + 1, 48)
    for g in (vars(mu), vars(copy)):
        wrapped = g[C.DISPATCH]
        assert wrapped is not wrapped.__wrapped__
        for a in ((x, base), (x, la), (huge, base), (x, torch.ones(4, 4))):
            assert wrapped(*a) == wrapped.__wrapped__(*a)
    c.uninstall()
    assert not hasattr(vars(mu)[C.DISPATCH], "__wrapped__") and not hasattr(vars(copy)[C.DISPATCH], "__wrapped__")
    n = len(c.passes)
    ex(*args)
    assert len(c.passes) == n                                     # uninstall removed the hooks


def test_mutant_a_lora_call_deciding_torch_voids(zoo):
    mg, nf, fwd = zoo
    zoo.mu._FORCE_LORA_TORCH = True
    ex = _experts(zoo.mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    _step(c, mg, ex, 0)
    c.mark_step(1)
    ok, why = C.check_dispatch(c.snapshot(), c.structural(1, 1))
    assert not ok and "LoRA-class GEMMs decided torch" in why


def test_mutant_a_triton_decided_call_that_falls_back_eager_voids(zoo):
    mg, nf, fwd = zoo

    def failing_self_check(device):
        raise RuntimeError("self-check")

    mg._self_check_generic = failing_self_check                   # triton_grouped_available is now False (821-830)
    ex = _experts(zoo.mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    c.mark_step(0)
    ex(*_stacks())
    c.mark_step(1)
    ok, why = C.check_dispatch(c.snapshot(), c.structural(1, 1))
    assert not ok and "fallback moved by 3" in why


def test_mutant_backward_off_the_triton_branch_voids_by_the_structural_count(zoo):
    mg, nf, fwd = zoo
    ex = _experts(zoo.mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    c.mark_step(0)
    with torch.no_grad():
        ex(*_stacks())
    ex(*_stacks())
    _backward_as_the_library_would(mg, 0, 2)                      # create_graph: gx went eager (642-645), not generic
    c.mark_step(1)
    ok, why = C.check_dispatch(c.snapshot(), c.structural(1, 1))
    assert not ok and "generic_gemm: wrapper 6 != 8 the decisions imply" in why


def test_the_sm120_guard_voids_a_base_gemm_that_decided_torch(zoo):
    mg, nf, fwd = zoo
    ex = _experts(zoo.mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    _step(c, mg, ex, 0, args=_stacks(base_rows=2048 * E_BASE + 1))   # one row past "many"'s 2048 x E
    c.mark_step(1)
    s, st = c.snapshot(), c.structural(1, 1)
    assert s["dispatch"]["gemm:many:torch"] == 2
    assert C.check_dispatch(s, st) == (True, "")                 # a mixed lane (P237's sm_89) registers this
    ok, why = C.check_dispatch(s, st, base_must_be_triton=True)
    assert not ok and "2 base GEMMs decided torch" in why


# -- the NF4 dX, by structure --------------------------------------------------------------------------------------------------------
def _nf4_expert(fwd):
    """An expert module whose forward is the NF4 caller (moe_utils_expert_forward): one maybe_nf4_grouped_mm per pass."""
    return _experts(lambda self, x: fwd(x, W, LORA))[0]


@pytest.mark.parametrize("ckpt", CKPT)
def test_expected_launches_are_fused_calls_plus_the_structural_dx(zoo, ckpt):
    mg, nf, fwd = zoo
    ex = _nf4_expert(fwd)
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    xg = X.clone().requires_grad_(True)
    for i in range(2):
        c.mark_step(i)
        if ckpt == "none":
            ex(xg)
        elif ckpt == "nonreentrant":
            ex(xg)
            ex(xg)
        else:
            with torch.no_grad():
                ex(xg)
            ex(xg)
        nf.nf4_dx(torch.ones(4, 3), W)                            # the one backward: one dX launch inside the op
    ex.eval()
    ex(xg)                                                        # an eval forward: a fused launch, no dX
    ex.train()
    c.mark_step(2)
    s, st = c.snapshot(), c.structural(1, 2)
    assert st["faults"] == [] and st["expected"]["dx_nf4"] == 2
    assert C.expected_launches(s, st) == s["library_delta"]["nf4"]["launches"]
    assert C.check_fused(s, C.expected_launches(s, st)) == (True, "")
    # MUTANT: the under-grad rule (fused calls + fused calls under grad) counts a dX for the grad-on eval forward (tc1_arm's
    # eval_loss runs with grad on), and twice per step under HF non-reentrant; the structural rule reads right in all four
    under = s["counts"]["nf4_fused_calls"] + s["counts"]["nf4_fused_calls_grad"]
    assert under != s["library_delta"]["nf4"]["launches"]


def test_mutant_a_dx_that_ran_outside_the_op_voids_the_launch_equality(zoo):
    mg, nf, fwd = zoo
    ex = _nf4_expert(fwd)
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    c.mark_step(0)
    ex(X.clone().requires_grad_(True))                            # one training pass: the structure expects one dX launch ...
    c.mark_step(1)
    s = c.snapshot()                                              # ... that never ran in the op (e.g. an eager dX)
    ok, why = C.check_fused(s, C.expected_launches(s, c.structural(1, 1)))
    assert not ok and "launches 1 != expected 2" in why


def test_expected_launches_read_unread_without_structure_or_with_an_unread_mode(zoo):
    mg, nf, fwd = zoo
    c = C.UnslothMoeCounter().install()
    assert C.expected_launches(c.snapshot(), None) is None
    assert C.expected_launches(c.snapshot(), {"faults": ["x"], "expected": {"dx_nf4": 0}}) is None
    c.counts["nf4_mode_unread"] = 1
    assert C.expected_launches(c.snapshot(), {"faults": [], "expected": {"dx_nf4": 0}}) is None


# -- the environment: every module of the installed packages, and every SET key registered ---------------------------------------
def test_check_env_scans_every_module_of_both_packages(zoo, monkeypatch):
    mg, nf, fwd = zoo
    pins = {"UNSLOTH_MOE_NF4_FUSED": "1", "UNSLOTH_MOE_GROUPED_TRITON": "auto", "UNSLOTH_MOE_FUSED_ROUTER": "1"}
    for k, v in pins.items():
        monkeypatch.setenv(k, v)
    s = C.UnslothMoeCounter().install().snapshot()
    assert {"UNSLOTH_MOE_FUSED_ROUTER", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_DISABLE_MOE_TRITON", "UNSLOTH_GPTOSS_FP16_GEMM"} <= set(s["env_keys"])
    assert s["env_counts"]["unsloth_zoo/temporary_patches/moe_router_triton.py"] == 1 and s["env_counts"]["unsloth/__init__.py"] == 1
    assert C.check_env(s, dict(os.environ), pins) == (True, "")
    monkeypatch.setenv("UNSLOTH_MOE_FUSED_ROUTER", "0")           # a router switch outside the old three-module list
    ok, why = C.check_env(s, dict(os.environ), pins)
    assert not ok and "UNSLOTH_MOE_FUSED_ROUTER='0' (registered '1')" in why
    monkeypatch.setenv("UNSLOTH_MOE_FUSED_ROUTER", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_MOE_TRITON", "1")         # ambient, scanned
    assert "ambient UNSLOTH_DISABLE_MOE_TRITON='1'" in C.check_env(s, dict(os.environ), pins)[1]
    monkeypatch.delenv("UNSLOTH_DISABLE_MOE_TRITON")
    monkeypatch.setenv("UNSLOTH_NOT_IN_ANY_FILE", "1")            # ambient and read by nothing we scanned: still VOID
    assert "ambient UNSLOTH_NOT_IN_ANY_FILE='1'" in C.check_env(s, dict(os.environ), pins)[1]
    monkeypatch.delenv("UNSLOTH_NOT_IN_ANY_FILE")
    ok, why = C.check_env(s, dict(os.environ), {**pins, "UNSLOTH_MOE_TYPO": "1"})
    assert not ok and "registered UNSLOTH_MOE_TYPO is not a key the library reads" in why


def test_an_unscanned_package_fails_check_env(zoo, monkeypatch):
    mg, nf, fwd = zoo
    monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))   # no spec: the package cannot be located
    s = C.UnslothMoeCounter().install().snapshot()
    assert s["env_unscanned"] and not C.check_env(s, dict(os.environ), {})[0]
