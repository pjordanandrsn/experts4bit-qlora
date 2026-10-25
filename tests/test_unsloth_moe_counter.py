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
    nf.MODE = {"mode": "1", "decline": None, "fail_launch": False}

    def maybe_nf4_grouped_mm(inputs, offsets, source, preprocess):
        if nf.MODE["mode"] == "0":
            return None
        if nf.MODE["decline"] in ("shape", "failed_shapes", "auto_cap"):
            return None                              # 523 / 534-537 / 538-542: no counter
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
        base = maybe(x, None, w, None)
        if base is None:
            base = gm(x, w)                                              # the dequant path's base GEMM
        return base + gm(gm(x, lora[0]), lora[1])

    return pkg, tp, mg, nf, moe_utils_expert_forward


@pytest.fixture
def zoo(monkeypatch):
    pkg, tp, mg, nf, fwd = _fake_zoo()
    for name, mod in (("unsloth_zoo", pkg), ("unsloth_zoo.temporary_patches", tp), (C.FP16, mg), (C.NF4, nf)):
        monkeypatch.setitem(sys.modules, name, mod)
    return mg, nf, fwd


X, W, LORA = torch.ones(4, 2), torch.ones(2, 3), (torch.ones(2, 1), torch.ones(1, 3))


def test_targets_are_the_release_module_paths():
    assert C.FP16 == "unsloth_zoo.temporary_patches.moe_grouped_fp16"
    assert C.NF4 == "unsloth_zoo.temporary_patches.moe_nf4_grouped"
    assert C.TARGETS == {"generic_gemm": (C.FP16, "generic_grouped_mm"), "generic_wgrad": (C.FP16, "generic_grouped_wgrad"),
                         "nf4_maybe": (C.NF4, "maybe_nf4_grouped_mm")}
    assert C.SELF_CHECK == (C.FP16, "_self_check_generic")


def test_unfused_arm_counts_generic_and_keeps_the_self_check_apart(zoo):
    mg, nf, fwd = zoo
    nf.MODE["mode"] = "0"
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


@pytest.mark.parametrize("decline", ["shape", "failed_shapes", "auto_cap"])
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


def test_a_local_alias_bound_before_install_is_missed_and_the_library_cross_check_catches_it(zoo):
    mg, nf, fwd = zoo
    alias = mg.generic_grouped_mm                                 # bound BEFORE install: bypasses the wrapper
    nf.MODE["mode"] = "0"
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


def test_a_failed_self_check_leaves_the_generic_counts_at_zero_so_a_lane_bar_fails_closed(zoo):
    mg, nf, fwd = zoo
    nf.MODE["mode"] = "0"

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
