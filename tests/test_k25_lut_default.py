# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_K25_LUT``'s per-family default (lane P158, FLIP; bench/p158/RESULTS-p158.md): an unset knob resolves at engine
build to ``prmt`` on qwen3_moe where the installed grouped-nf4-gemm carries K39's decode on an NVIDIA device, and to
``tree`` everywhere else with the reason recorded. A default never refuses a build: an older kernel package or a ROCm
device resolves it to the tree (the same values). An explicit ``prmt`` keeps its refusals. On CPU with the route
tests' kernel stubs."""
import pytest

pytest.importorskip("nf4_grouped")

from test_nf4_grouped_smallm_route import _install_stubs, _run, _set_mode  # noqa: E402

from experts4bit_qlora.engines import hot_residency as hr  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Each test starts from an unresolved process (the default is the tree until a build resolves it)."""
    monkeypatch.setattr(hr, "_K25_LUT_DEFAULT", ["tree"])
    monkeypatch.setattr(hr, "K25_LUT_SEEN", {})
    monkeypatch.delenv("E4B_K25_LUT", raising=False)
    _set_mode(monkeypatch, None)


def test_only_qwen3_moe_is_licensed():
    assert hr.K25_LUT_DEFAULT_FAMILIES == {"qwen3_moe": "prmt"}


def test_qwen3_moe_unset_takes_prmt_where_the_kernel_carries_it_and_the_forward_follows(monkeypatch):
    calls = _install_stubs(monkeypatch, prmt=True)
    r = hr.resolve_k25_lut("qwen3_moe")
    assert r == {"env": None, "lut": "prmt", "why": None, "source": "default-allowlisted"}
    assert hr._K25_LUT_DEFAULT == ["prmt"] and hr._k25_lut_env() == "prmt"
    _run(24)
    assert [p["lut"] for p in calls["k25_plan"]] == ["prmt", "prmt"] and hr.K25_LUT_SEEN == {"prmt|le256": 1}


def test_qwen3_moe_unset_on_an_older_kernel_package_resolves_the_tree_never_a_refusal(monkeypatch):
    calls = _install_stubs(monkeypatch, tree=True)            # 0.49-era: the select tree, no K39
    r = hr.resolve_k25_lut("qwen3_moe")
    assert (r["lut"], r["why"]) == ("tree", None) and r["source"].startswith("default-off: the installed grouped-nf4-gemm")
    assert hr._K25_LUT_DEFAULT == ["tree"]
    _run(24)
    assert [p["lut"] for p in calls["k25_plan"]] == ["tree", "tree"]          # gate_up and down


def test_qwen3_moe_unset_on_rocm_resolves_the_tree(monkeypatch):
    import torch
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising=False)
    _install_stubs(monkeypatch, prmt=True)
    r = hr.resolve_k25_lut("qwen3_moe")
    assert r["lut"] == "tree" and "ROCm" in r["source"] and hr._K25_LUT_DEFAULT == ["tree"]


@pytest.mark.parametrize("mt", ["mixtral", "olmoe", "qwen2_moe", None])
def test_other_families_stay_on_the_tree_with_the_reason(monkeypatch, mt):
    _install_stubs(monkeypatch, prmt=True)
    r = hr.resolve_k25_lut(mt)
    assert r == {"env": None, "lut": "tree", "why": None, "source": f"default-off: no read licenses model_type {mt!r}"}


def test_an_explicit_tree_opts_qwen3_moe_out(monkeypatch):
    monkeypatch.setenv("E4B_K25_LUT", "tree")
    calls = _install_stubs(monkeypatch, prmt=True)
    assert hr.resolve_k25_lut("qwen3_moe") == {"env": "tree", "lut": "tree", "why": None, "source": "explicit"}
    _run(24)
    assert [p["lut"] for p in calls["k25_plan"]] == ["tree", "tree"]          # gate_up and down


@pytest.mark.parametrize("kernel,hip,match", [({"tree": True}, None, "needs grouped-nf4-gemm with K39"),
                                              ({"prmt": True}, "6.4.0", "on ROCm keep E4B_K25_LUT=tree")],
                         ids=["kernel-without-k39", "rocm"])
def test_an_explicit_prmt_keeps_its_refusals_on_qwen3_moe(monkeypatch, kernel, hip, match):
    import torch
    monkeypatch.setenv("E4B_K25_LUT", "prmt")
    if hip:
        monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    _install_stubs(monkeypatch, **kernel)
    with pytest.raises(RuntimeError, match=match):
        hr.resolve_k25_lut("qwen3_moe")


def test_k25_off_resolves_no_decode_for_qwen3_moe_without_a_refusal(monkeypatch):
    _set_mode(monkeypatch, "0")
    _install_stubs(monkeypatch, prmt=True)
    r = hr.resolve_k25_lut("qwen3_moe")
    assert (r["lut"], r["why"], hr._K25_LUT_DEFAULT) == (None, "E4B_NF4_GROUPED_SMALLM=0", ["tree"])


def test_a_kernel_package_without_k25_resolves_no_decode_for_qwen3_moe_without_a_refusal(monkeypatch):
    _install_stubs(monkeypatch, with_k25=False)
    r = hr.resolve_k25_lut("qwen3_moe")
    assert (r["lut"], r["why"]) == (None, "the kernel package predates K25") and hr._K25_LUT_DEFAULT == ["tree"]


def test_each_resolution_starts_from_the_tree(monkeypatch):
    """A later build for another family must not inherit an earlier qwen3_moe default."""
    _install_stubs(monkeypatch, prmt=True)
    assert hr.resolve_k25_lut("qwen3_moe")["lut"] == "prmt"
    assert hr.resolve_k25_lut("mixtral")["lut"] == "tree" and hr._K25_LUT_DEFAULT == ["tree"]


def test_a_default_prmt_the_package_can_no_longer_serve_takes_the_tree_in_the_forward(monkeypatch):
    """The forward's safety branch: a resolved default whose kernel later lacks prmt takes the tree, never a refusal
    (only an explicit prmt refuses there)."""
    monkeypatch.setattr(hr, "_K25_LUT_DEFAULT", ["prmt"])
    calls = _install_stubs(monkeypatch, tree=True)
    _run(24)
    assert [p["lut"] for p in calls["k25_plan"]] == ["tree", "tree"]          # gate_up and down


def test_build_engine_resolves_the_family_default_from_the_config_before_any_weight_loads():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "experts4bit_qlora" / "serve_paged.py").read_text(encoding="utf-8")
    body = src[src.index("def build_engine(cfg: PagedServeConfig)"):]
    early = body.index("    if _k25_lut_raw() or _k25_plan_raw():")
    early_call = body.index("        resolve_k25_plan(resolve_k25_lut(), family_known=False)")
    config = body.index("model_config = AutoConfig.from_pretrained(")
    resolved = body.index('    k25_lut = resolve_k25_lut(getattr(model_config, "model_type", None))')
    plan = body.index("    k25_plan = resolve_k25_plan(k25_lut)")          # E4B_K25_PLAN, on the family's decode
    load = body.index("load_moe_4bit_streaming(")
    assert early < early_call < config < resolved < plan < load
    assert 'info["k25_lut"] = k25_lut' in body and 'info["k25_plan"] = k25_plan' in body
