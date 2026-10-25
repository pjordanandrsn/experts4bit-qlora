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


def forward_moe_backend(x, base, lora_a, lora_b):
    _lora_rank_pad_needed(x, lora_a)
    with torch.no_grad():
        y = _grouped_mm_with_backward_fix(x, base, None)     # the dequant path's recompute forward (1100): no backward node
    h = _grouped_mm_with_backward_fix(x, lora_a, None)
    return y + _grouped_mm_with_backward_fix(h, lora_b, None)


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


@pytest.fixture
def zoo(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "1")
    for k in ("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", "UNSLOTH_MOE_GROUPED_TRITON", "UNSLOTH_DISABLE_MOE_TRITON", "UNSLOTH_MOE_NF4_AUTOTUNE"):
        monkeypatch.delenv(k, raising=False)
    pkg, tp, mg, nf, fwd = _fake_zoo()
    (tmp_path / "fp16.py").write_text('os.environ.get("UNSLOTH_GPTOSS_FP16_GEMM")\n', encoding="utf-8")
    (tmp_path / "nf4.py").write_text('os.environ.get("UNSLOTH_MOE_NF4_FUSED", "0")\nos.environ.get("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS")\n'
                                     'os.environ.get("UNSLOTH_MOE_NF4_AUTOTUNE", "0")\n', encoding="utf-8")
    mg.__file__, nf.__file__ = str(tmp_path / "fp16.py"), str(tmp_path / "nf4.py")
    mu = _module_from_source(C.MOE_UTILS, FAKE_MOE_UTILS, tmp_path / "moe_utils.py")
    for name, mod in (("unsloth_zoo", pkg), ("unsloth_zoo.temporary_patches", tp), (C.FP16, mg), (C.NF4, nf), (C.MOE_UTILS, mu)):
        monkeypatch.setitem(sys.modules, name, mod)
    z = _Zoo((mg, nf, fwd))
    z.mu, z.tmp = mu, tmp_path
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


# -- the per-call dispatch: moe_utils' own predicate, in every module object that can serve the forward -------
E_BASE, R = 64, 16


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


def test_dispatch_on_the_package_module_and_the_implied_counts(zoo):
    mg, nf, fwd = zoo
    mu = zoo.mu
    c = C.UnslothMoeCounter().install()
    mu.get_forward_moe_backend()(*_stacks())
    _backward_as_the_library_would(mg, 2, 2)
    s = c.snapshot()
    assert s["served"][0]["is_package"] and s["dispatch_served"] == {"package": {"gemm": 3, "probe": 1}}
    assert s["dispatch"] == {"probe:lora:triton": 1, "gemm:many:triton": 1, "gemm:lora:triton": 2}
    assert C.implied_generic(s) == {"generic_gemm": 5, "generic_wgrad": 2}           # 3 forwards + 2 gx; 2 LoRA dW
    assert C.check_dispatch(s, base_must_be_triton=True) == (True, "")
    assert any(k.endswith(":many") and v == 2048 for k, v in s["limits"].items())    # the resolved limit is recorded


def test_a_cache_copy_serving_the_forward_is_counted(zoo):
    mg, nf, fwd = zoo
    mu = zoo.mu
    mu._CACHE = _module_from_source("unsloth_cached_moe_utils", FAKE_MOE_UTILS, zoo.tmp / "cached_moe_utils.py")
    c = C.UnslothMoeCounter().install()
    mu.get_forward_moe_backend()(*_stacks())
    _backward_as_the_library_would(mg, 2, 2)
    s = c.snapshot()
    assert s["served"][0]["is_package"] is False and s["served"][0]["module"] == "unsloth_cached_moe_utils"
    assert s["dispatch_served"] == {"copy1": {"gemm": 3, "probe": 1}}
    assert C.check_dispatch(s) == (True, "")


def test_mutant_a_cache_copy_appearing_after_install_is_missed_and_voids(zoo):
    mg, nf, fwd = zoo
    mu = zoo.mu
    c = C.UnslothMoeCounter().install()                           # the package served at install
    mu._CACHE = _module_from_source("unsloth_cached_moe_utils", FAKE_MOE_UTILS, zoo.tmp / "late_cache.py")
    mu.get_forward_moe_backend()(*_stacks())                      # now served by an unpatched copy
    s = c.snapshot()
    ok, why = C.check_dispatch(s)
    assert not ok and "no decision counted through the serving module" in why


# -- patch_loaded: the forward the LOADED experts run, read off the model after its load --------------------------
def _experts(forward, name="Qwen3_5MoeExperts", n=3):
    """n expert modules whose class forward is ``forward``, as patch_function binds the zoo's answer (qwen3_5_moe.py:53-60)."""
    cls = type(name, (torch.nn.Module,), {"forward": forward})
    return [cls() for _ in range(n)]


def _copy(zoo, name):
    return _module_from_source("unsloth_cached_moe_utils", FAKE_MOE_UTILS, zoo.tmp / f"{name}.py")


def test_patch_loaded_serves_the_cache_copy_that_appeared_during_the_load(zoo):
    mg, nf, fwd = zoo
    mu = zoo.mu
    c = C.UnslothMoeCounter().install()                           # post_import: the package serves
    mu._CACHE = _copy(zoo, "load_time_cache")                     # the load creates the cache copy ...
    experts = _experts(mu.get_forward_moe_backend())              # ... and the zoo binds its forward into the class
    c.patch_loaded(experts)                                       # post_load
    type(experts[0]).forward(*_stacks())
    _backward_as_the_library_would(mg, 2, 2)
    s = c.snapshot()
    assert [(x["tag"], x["source"], x["is_package"]) for x in s["served"]] == [("copy1", "loaded", False)]
    assert s["expert_forwards"] == {f"{__name__}.Qwen3_5MoeExperts": [
        {"forward": "unsloth_cached_moe_utils.forward_moe_backend", "globals_module": "unsloth_cached_moe_utils", "unwrapped": 0,
         "modules": 3}]}
    assert C.check_dispatch(s, base_must_be_triton=True) == (True, "")


@pytest.mark.parametrize("passed", [True, False])
def test_mutant_experts_on_a_forward_other_than_the_fresh_answer_count_only_when_passed(zoo, passed):
    """The model's experts hold a forward from a cache copy that is no longer get_forward_moe_backend()'s answer (a later reload
    replaced it). Read off the model, it is counted; left to the fresh call, the decisions run unwrapped and the arm VOIDs."""
    mg, nf, fwd = zoo
    mu = zoo.mu
    stale = _copy(zoo, "stale_cache")
    experts = _experts(stale.forward_moe_backend)
    mu._CACHE = _copy(zoo, "current_cache")                       # the fresh answer is now a DIFFERENT module object
    c = C.UnslothMoeCounter().install()
    if passed:
        c.patch_loaded(experts)
    type(experts[0]).forward(*_stacks())
    _backward_as_the_library_would(mg, 2, 2)
    s = c.snapshot()
    assert s["served_fresh"]["file"].endswith("current_cache.py")
    ok, why = C.check_dispatch(s)
    if passed:
        assert ok, why
        assert s["served"][0]["file"].endswith("stale_cache.py")
    else:
        assert not ok and "no decision counted through the serving module" in why


def test_a_mixed_model_records_each_forward_and_every_served_object_must_decide(zoo):
    mg, nf, fwd = zoo
    mu = zoo.mu
    copy = _copy(zoo, "cache_a")
    a, b = _experts(copy.forward_moe_backend, "ExpertsA", 2), _experts(mu.forward_moe_backend, "ExpertsB", 1)
    c = C.UnslothMoeCounter().install().patch_loaded(a + b)
    type(a[0]).forward(*_stacks())
    s = c.snapshot()
    assert {k.rsplit(".", 1)[1]: [x["modules"] for x in v] for k, v in s["expert_forwards"].items()} == {"ExpertsA": [2], "ExpertsB": [1]}
    ok, why = C.check_dispatch(s)                                 # B's forward never ran
    assert not ok and "no decision counted through the serving module (package" in why
    type(b[0]).forward(*_stacks())
    _backward_as_the_library_would(mg, 4, 4)
    assert C.check_dispatch(c.snapshot()) == (True, "")


def test_mutant_experts_left_on_the_original_forward_void(zoo):
    """patch_function refused (can_safely_patch) and the transformers forward runs: its globals hold no predicate."""
    mg, nf, fwd = zoo

    def original_forward(self, hidden_states):
        return hidden_states

    c = C.UnslothMoeCounter().install().patch_loaded(_experts(original_forward))
    s = c.snapshot()
    assert s["expert_forwards"][f"{__name__}.Qwen3_5MoeExperts"][0]["globals_module"] is None
    ok, why = C.check_dispatch(s)
    assert not ok and "no _triton_grouped_mm_wanted in its globals (3 modules)" in why


def test_a_wrapped_forward_unwraps_to_the_predicates_globals(zoo):
    mg, nf, fwd = zoo
    mu = zoo.mu
    inner = mu.forward_moe_backend

    @functools.wraps(inner)
    def decorated(*args, **kwargs):                               # a decorator or compile carrier defined elsewhere
        return inner(*args, **kwargs)

    assert C.DISPATCH not in decorated.__globals__                 # the carrier's own globals hold no predicate
    c = C.UnslothMoeCounter().install().patch_loaded(_experts(decorated, n=1))
    s = c.snapshot()
    assert s["expert_forwards"][f"{__name__}.Qwen3_5MoeExperts"][0]["unwrapped"] == 1
    assert [x["tag"] for x in s["served"]] == ["package"]


def test_the_dispatch_wrapper_observes_and_never_changes_the_decision(zoo):
    mg, nf, fwd = zoo
    mu = zoo.mu
    copy = _copy(zoo, "cache_eq")
    c = C.UnslothMoeCounter().install().patch_loaded(_experts(copy.forward_moe_backend))
    x, base, la, lb = _stacks()
    huge = torch.ones(2048 * E_BASE + 1, 48)
    for g in (vars(mu), vars(copy)):
        wrapped = g[C.DISPATCH]
        assert wrapped is not wrapped.__wrapped__
        for args in ((x, base), (x, la), (huge, base), (x, torch.ones(4, 4))):
            assert wrapped(*args) == wrapped.__wrapped__(*args)
    c.uninstall()
    assert not hasattr(vars(mu)[C.DISPATCH], "__wrapped__") and not hasattr(vars(copy)[C.DISPATCH], "__wrapped__")


def test_mutant_a_lora_call_deciding_torch_voids(zoo):
    mg, nf, fwd = zoo
    zoo.mu._FORCE_LORA_TORCH = True
    c = C.UnslothMoeCounter().install()
    zoo.mu.get_forward_moe_backend()(*_stacks())
    ok, why = C.check_dispatch(c.snapshot())
    assert not ok and "LoRA-class GEMMs decided torch" in why


def test_mutant_a_triton_decided_call_that_falls_back_eager_voids(zoo):
    mg, nf, fwd = zoo

    def failing_self_check(device):
        raise RuntimeError("self-check")

    mg._self_check_generic = failing_self_check                   # triton_grouped_available is now False (821-830)
    c = C.UnslothMoeCounter().install()
    zoo.mu.get_forward_moe_backend()(*_stacks())
    ok, why = C.check_dispatch(c.snapshot())
    assert not ok and "fallback moved by 3" in why


def test_mutant_backward_off_the_triton_branch_voids_by_the_implied_count(zoo):
    mg, nf, fwd = zoo
    c = C.UnslothMoeCounter().install()
    zoo.mu.get_forward_moe_backend()(*_stacks())
    _backward_as_the_library_would(mg, 0, 2)                      # create_graph: gx went eager (642-645), not generic
    ok, why = C.check_dispatch(c.snapshot())
    assert not ok and "generic_gemm: wrapper 3 != 5 the decisions imply" in why


def test_the_sm120_guard_voids_a_base_gemm_that_decided_torch(zoo):
    mg, nf, fwd = zoo
    c = C.UnslothMoeCounter().install()
    x, base, la, lb = _stacks(base_rows=2048 * E_BASE + 1)        # one row past "many"'s 2048 x E
    zoo.mu.get_forward_moe_backend()(x, base, la, lb)
    _backward_as_the_library_would(mg, 2, 2)
    s = c.snapshot()
    assert s["dispatch"]["gemm:many:torch"] == 1
    assert C.check_dispatch(s) == (True, "")                      # a mixed lane (P237's sm_89) registers this
    ok, why = C.check_dispatch(s, base_must_be_triton=True)
    assert not ok and "1 base GEMMs decided torch" in why


def test_a_checkpointed_double_forward_is_counted_twice(zoo):
    mg, nf, fwd = zoo
    c = C.UnslothMoeCounter().install()
    for _ in range(2):                                            # the forward, then its recompute under checkpointing
        zoo.mu.get_forward_moe_backend()(*_stacks())
    _backward_as_the_library_would(mg, 4, 4)
    s = c.snapshot()
    assert s["dispatch"]["gemm:lora:triton"] == 4 and C.check_dispatch(s) == (True, "")


def test_check_env_scans_the_library_source_not_a_prefix(zoo, monkeypatch):
    mg, nf, fwd = zoo
    pins = {"UNSLOTH_MOE_NF4_FUSED": "1", "UNSLOTH_MOE_GROUPED_TRITON": "auto"}
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "auto")
    s = C.UnslothMoeCounter().install().snapshot()
    assert {"UNSLOTH_DISABLE_MOE_TRITON", "UNSLOTH_MOE_GROUPED_TRITON", "UNSLOTH_GPTOSS_FP16_GEMM"} <= set(s["env_keys"])
    assert C.check_env(s, dict(os.environ), pins) == (True, "")
    monkeypatch.setenv("UNSLOTH_DISABLE_MOE_TRITON", "1")          # ambient, and not UNSLOTH_MOE_*: a prefix test misses it
    ok, why = C.check_env(s, dict(os.environ), pins)
    assert not ok and "ambient UNSLOTH_DISABLE_MOE_TRITON='1'" in why
    monkeypatch.delenv("UNSLOTH_DISABLE_MOE_TRITON")
    ok, why = C.check_env(s, dict(os.environ), {**pins, "UNSLOTH_MOE_TYPO": "1"})
    assert not ok and "registered UNSLOTH_MOE_TYPO is not a key the library reads" in why


def test_an_unscanned_library_fails_check_env(zoo, monkeypatch):
    mg, nf, fwd = zoo
    zoo.mu.__file__ = str(zoo.tmp / "missing.py")
    s = C.UnslothMoeCounter().install().snapshot()
    assert s["env_unscanned"] and not C.check_env(s, dict(os.environ), {})[0]
