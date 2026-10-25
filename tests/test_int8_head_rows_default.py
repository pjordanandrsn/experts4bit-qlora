# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_INT8_HEAD_ROWS``'s per-family default (lane P232, FLIP_ROWS_8; bench/p232/RESULTS-p232.md, after P229's
microbench, bench/p229/RESULTS-p229.md): an unset knob resolves to ``8`` on qwen3_moe at the head shape P232 read
(151936 x 2048) only where the int8 head is on, K37 imports and speculative decode is off; to ``1`` everywhere else,
every other model_type included, with the reason recorded and never a crash or a refusal. Explicit values keep their
behaviour and every refusal. On CPU with stub kernels; the route itself (the routing, the refusals under an explicit
value) is tests/test_int8_head_bw.py's, and the head swap's own default is tests/test_int8_head_default.py's."""
import inspect
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from experts4bit_qlora import loader  # noqa: E402
from experts4bit_qlora import serve_paged as sp  # noqa: E402
from experts4bit_qlora.arch import moe_conventions  # noqa: E402
from experts4bit_qlora.engines import int8_head_bw as ih  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
UNSET = sp.FUSION_UNSET
QWEN3_HEAD = (151936, 2048)
TOY_HEAD = (40, 16)


def _gemv(x, q, s):
    return (s * (q.float() @ x.float().reshape(-1))).to(torch.bfloat16).reshape(1, -1)


def _quantize(w):
    """A per-row int8 grid (absmax / 127): the swap's own grid is tests/test_int8_head_bw.py's; here only the shape
    matters."""
    s = (w.float().abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    return torch.round(w.float() / s[:, None]).clamp(-127, 127).to(torch.int8), s


class _Linear:
    """Stands where K37's ``linear_i8`` would."""

    def __call__(self, x, q, s):
        return ((x.float() @ q.float().T) * s).to(torch.bfloat16)


def _plan(n, k, m):
    return (32, 256, 4)


class _Toy(nn.Module):
    """A model with a family and an output head (vocab x hidden), as build_engine sees one after its swaps."""

    def __init__(self, model_type="qwen3_moe", vocab=TOY_HEAD[0], hidden=TOY_HEAD[1], tied=False):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.embed = nn.Embedding(vocab, hidden, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(hidden, vocab, bias=False, dtype=torch.bfloat16)
        if tied:
            self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, m):
        self.lm_head = m


@pytest.fixture
def toy_family(monkeypatch):
    """Both default tables with the toy's head shape (40 x 16) in place of Qwen3's 151936 x 2048 (0.6 GB in bf16)."""
    monkeypatch.setattr(sp, "INT8_HEAD_BW_DEFAULT_FAMILIES", {"qwen3_moe": (TOY_HEAD,)})
    monkeypatch.setattr(sp, "INT8_HEAD_ROWS_DEFAULT_FAMILIES", {"qwen3_moe": {TOY_HEAD: 8}})


@pytest.fixture
def k37(monkeypatch):
    """A grouped-nf4-gemm with K37: ``dense_i8`` importable, with stub ``linear_i8`` and ``plan``."""
    fake = types.ModuleType("dense_i8")
    fake.linear_i8, fake.plan = _Linear(), _plan
    monkeypatch.setitem(sys.modules, "dense_i8", fake)
    return fake


@pytest.fixture
def no_k37(monkeypatch):
    """A grouped-nf4-gemm without K37 (older than 0.49.0, e.g. ``[fast]``'s floor): ``import dense_i8`` fails."""
    monkeypatch.setitem(sys.modules, "dense_i8", None)


def _cfg(monkeypatch, rows=None, head_bw=None):
    monkeypatch.setattr(sp, "_capability", lambda device: None)
    for name, val in (("E4B_INT8_HEAD_ROWS", rows), ("E4B_INT8_HEAD_BW", head_bw)):
        if val is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, val)
    for name in ("E4B_PAGED_SPEC", "E4B_PAGED_SPEC_K", "E4B_PAGED_SPEC_HEAD"):
        monkeypatch.delenv(name, raising=False)
    return sp.PagedServeConfig.from_env()


def _build(cfg, model):
    """build_engine's two steps, in its order: the head swap (its own default), then the route over it."""
    head = sp.apply_int8_head_bw_resolved(model, cfg, gemv=_gemv, quantize=_quantize)
    return head, sp.apply_int8_head_rows_resolved(model, cfg, head)


def _known_model_types() -> set:
    """Every model_type the codebase names: the loader's admitted architectures, the MoE conventions' families, and
    every per-family default table's keys."""
    known = set(loader.SUPPORTED_ARCHITECTURES)
    for conv in moe_conventions.CONVENTIONS:
        known |= set(conv.model_types)
    for table in (sp.FUSION_DEFAULT_FAMILIES, sp.FUSION_UNLICENSED, sp.PREFILL_GLUE_DEFAULT_FAMILIES,
                  sp.ATTN_SPLIT_CAP_DEFAULT_FAMILIES, sp.INT8_HEAD_BW_DEFAULT_FAMILIES):
        known |= set(table)
    return known


OTHER_FAMILIES = sorted(_known_model_types() - {"qwen3_moe"})


# ------------------------------------------------------------------------------------------ the table, the resolver --

def test_qwen3_moe_alone_takes_8_rows_by_default_at_p232s_shape():
    """P232 (p232-5090-1) read FLIP_ROWS_8 for qwen3_moe at 151936 x 2048; nothing else is read."""
    assert sp.INT8_HEAD_ROWS_DEFAULT_FAMILIES == {"qwen3_moe": {QWEN3_HEAD: 8}}
    assert len(sp.INT8_HEAD_ROWS_DEFAULT_FAMILIES) == 1
    assert sp.resolve_int8_head_rows(UNSET, "qwen3_moe", QWEN3_HEAD) == (8, "default-allowlisted")
    assert 8 <= ih.MAX_ROWS


def test_the_family_list_is_the_whole_codebase():
    """The parametrized families below are every model_type the codebase knows, not a hand-picked few."""
    assert {"olmoe", "granitemoe", "gpt_oss", "gemma4_text", "mixtral", "qwen3_5_moe_text", "qwen3_5_moe",
            "granitemoehybrid", "granitemoeshared", "nemotron_h", "jamba", "lfm2_moe", "deepseek_v4",
            "kimi_k3"} <= set(OTHER_FAMILIES)
    assert "qwen3_moe" not in OTHER_FAMILIES and len(OTHER_FAMILIES) >= 40


@pytest.mark.parametrize("family", OTHER_FAMILIES + [None])
def test_every_other_model_type_keeps_1(family):
    """The maintainer's requirement: every model_type but qwen3_moe keeps today's default of 1, even at Qwen3's head
    shape with the head on, K37 present and spec off."""
    rows, source = sp.resolve_int8_head_rows(UNSET, family, QWEN3_HEAD)
    assert (rows, source) == (1, f"default-off: no read licenses model_type {family!r}"), family


@pytest.mark.parametrize("family", OTHER_FAMILIES)
def test_every_other_model_type_keeps_1_through_the_build_and_never_imports_k37(monkeypatch, toy_family, no_k37,
                                                                                  family):
    """End to end with the head swapped on (an explicit E4B_INT8_HEAD_BW=1, the only way another family has one):
    unset rows stays 1 with the family's reason, and K37 is never asked (no_k37 would name it)."""
    m = _Toy(model_type=family)
    head, rows = _build(_cfg(monkeypatch, head_bw="1"), m)
    assert head["mode"] == "1" and isinstance(m.lm_head, ih.Int8HeadLinear)
    assert rows == {"rows_max": 1, "source": f"default-off: no read licenses model_type {family!r}"}
    assert m.lm_head.rows_max == 1 and m.lm_head._linear is None


def test_another_qwen3_head_shape_keeps_1():
    """Qwen3-235B-A22B's head (151936 x 4096) was not read."""
    rows, source = sp.resolve_int8_head_rows(UNSET, "qwen3_moe", (151936, 4096))
    assert rows == 1 and source == ("default-off: head shape [151936, 4096] is not one read on qwen3_moe "
                                    "([151936, 2048])")
    assert sp.resolve_int8_head_rows(UNSET, "qwen3_moe", None)[0] == 1


@pytest.mark.parametrize("kw,why", [
    (dict(head_off="the int8 head is off (E4B_INT8_HEAD_BW resolved 0: explicit)"),
     "default-off: the int8 head is off (E4B_INT8_HEAD_BW resolved 0: explicit)"),
    (dict(spec=True), "default-off: E4B_PAGED_SPEC is on: its gate and draft were read with the head on F.linear"),
    (dict(kernel_error="grouped-nf4-gemm without dense_i8.linear_i8 (K37, 0.49.0 or later): x"),
     "default-off: grouped-nf4-gemm without dense_i8.linear_i8 (K37, 0.49.0 or later): x"),
])
def test_each_unmet_condition_turns_the_default_to_1_with_its_reason(kw, why):
    assert sp.resolve_int8_head_rows(UNSET, "qwen3_moe", QWEN3_HEAD, **kw) == (1, why)


@pytest.mark.parametrize("value", [1, 2, 4, 8])
def test_an_explicit_value_passes_through_whatever_the_conditions(value):
    """An explicit value is not resolved: the route (head off, no K37) and the config (spec) refuse it, loudly."""
    assert sp.resolve_int8_head_rows(value, "qwen3_moe", QWEN3_HEAD) == (value, "explicit")
    assert sp.resolve_int8_head_rows(value, "mixtral", (32000, 4096), head_off="off", spec=True,
                                     kernel_error="no K37") == (value, "explicit")


# ------------------------------------------------------------------------------------------ the env and the config --

@pytest.mark.parametrize("value,want", [("", UNSET), ("  ", UNSET), (None, UNSET), ("1", 1), (" 1 ", 1), ("2", 2),
                                        ("4", 4), ("8", 8)])
def test_the_env_reader_keeps_unset_as_the_sentinel(value, want):
    assert sp._int8_head_rows_env(value) == want


@pytest.mark.parametrize("value", ["0", "9", "16", "auto", "2.0", "-2", "on"])
def test_the_env_reader_still_refuses_anything_else(value):
    with pytest.raises(ValueError, match="E4B_INT8_HEAD_ROWS=.*expected 1 to 8"):
        sp._int8_head_rows_env(value)


def test_the_server_config_leaves_an_unset_knob_to_the_build(monkeypatch):
    """from_env no longer turns unset into 1 eagerly: the sentinel survives to the build, which resolves it."""
    assert sp.PagedServeConfig.__dataclass_fields__["int8_head_rows"].default == UNSET
    assert _cfg(monkeypatch).int8_head_rows == UNSET
    assert _cfg(monkeypatch, rows="").int8_head_rows == UNSET
    assert _cfg(monkeypatch, rows="1").int8_head_rows == 1
    assert _cfg(monkeypatch, rows="8").int8_head_rows == 8
    with pytest.raises(ValueError, match="expected 1 to 8"):
        _cfg(monkeypatch, rows="9")


def _spec_cfg(**kw):
    base = dict(graphs=True, buckets=(1, 2, 4, 8, 16), max_seqs=16, spec="eagle3", spec_k=2, spec_head="/heads/e3")
    base.update(kw)
    return sp.PagedServeConfig(**base)


def test_spec_with_the_unset_knob_validates_and_an_explicit_bound_above_1_still_refuses():
    _spec_cfg().validate()                                   # the dataclass default is the sentinel
    _spec_cfg(int8_head_rows=UNSET).validate()               # unset: resolves to 1 at the build, never refused
    _spec_cfg(int8_head_rows=1).validate()
    for v in (2, 8):
        with pytest.raises(ValueError, match=f"E4B_PAGED_SPEC and E4B_INT8_HEAD_ROWS={v} do not combine"):
            _spec_cfg(int8_head_rows=v).validate()
    sp.PagedServeConfig(graphs=True, buckets=(1, 2, 4, 8, 16), max_seqs=16, int8_head_rows=8).validate()   # no spec


# ------------------------------------------------------------------------- end to end through the build's helpers --

def test_the_default_widens_the_route_to_8_on_qwen3_moe(monkeypatch, toy_family, k37):
    """Both knobs unset on qwen3_moe at the read shape, K37 present, spec off: the head swaps (P153's default) and its
    2..8 rows go to K37 (P232's), recorded as /health's levers.int8_head_bw.rows."""
    m = _Toy()
    head, rows = _build(_cfg(monkeypatch), m)
    assert head["mode"] == "1" and head["source"] == "default-allowlisted"
    assert rows == {"rows_max": 8, "kernel": "dense_i8.linear_i8 (K37)",
                    "plan": {str(r): [32, 256, 4] for r in range(2, 9)}, "source": "default-allowlisted"}
    assert m.lm_head.rows_max == 8 and m.lm_head._linear is k37.linear_i8


@pytest.mark.parametrize("head_bw,tied,why", [
    ("0", False, "default-off: the int8 head is off (E4B_INT8_HEAD_BW resolved 0: explicit)"),
    (None, True, "default-off: the int8 head is off (E4B_INT8_HEAD_BW resolved 0: default-off: the output head is "
                 "tied to the input embeddings"),
])
def test_the_head_off_keeps_1_with_the_heads_reason(monkeypatch, toy_family, k37, head_bw, tied, why):
    m = _Toy(tied=tied)
    head, rows = _build(_cfg(monkeypatch, head_bw=head_bw), m)
    assert head["mode"] == "0" and type(m.lm_head) is nn.Linear
    assert rows["rows_max"] == 1 and rows["source"].startswith(why)


def test_the_head_off_by_its_own_family_default_keeps_1(monkeypatch, toy_family, k37):
    """The head's default is off at an unread shape, so the rows' default is off too (the shape is named first)."""
    m = _Toy(vocab=48)
    head, rows = _build(_cfg(monkeypatch), m)
    assert head["mode"] == "0" and rows["rows_max"] == 1
    assert rows["source"] == "default-off: head shape [48, 16] is not one read on qwen3_moe ([40, 16])"
    # with the head forced on at that shape, the rows' default still reads the shape
    m = _Toy(vocab=48)
    head, rows = _build(_cfg(monkeypatch, head_bw="1"), m)
    assert head["mode"] == "1" and rows == {"rows_max": 1, "source": "default-off: head shape [48, 16] is not one "
                                                                       "read on qwen3_moe ([40, 16])"}


def test_without_k37_the_default_keeps_1_and_an_explicit_8_refuses(monkeypatch, toy_family, no_k37):
    m = _Toy()
    head, rows = _build(_cfg(monkeypatch), m)
    assert head["mode"] == "1"
    assert rows["rows_max"] == 1 and rows["source"].startswith(
        "default-off: grouped-nf4-gemm without dense_i8.linear_i8 (K37, 0.49.0 or later)")
    assert m.lm_head.rows_max == 1 and m.lm_head._linear is None
    with pytest.raises(RuntimeError, match=r"needs grouped-nf4-gemm with dense_i8.linear_i8 \(K37"):
        _build(_cfg(monkeypatch, rows="8"), _Toy())


def test_spec_on_keeps_1_never_raises_and_never_imports_k37(monkeypatch, toy_family, no_k37):
    """Speculative decode's gate and draft were read with the head on F.linear: unset resolves to 1 and the reason is
    spec's (no_k37 would have named K37 had it been probed)."""
    cfg = _cfg(monkeypatch)
    cfg.spec = "eagle3"
    m = _Toy()
    head, rows = _build(cfg, m)
    assert head["mode"] == "1"
    assert rows == {"rows_max": 1, "source": "default-off: E4B_PAGED_SPEC is on: its gate and draft were read with "
                                             "the head on F.linear"}
    assert m.lm_head.rows_max == 1


def _probe_raises(monkeypatch):
    """K37's probe replaced by one that fails the test if it is ever called."""
    def probed():
        raise AssertionError("rows_kernel_missing was called: the unset knob probed K37 where the default cannot apply")
    monkeypatch.setattr(ih, "rows_kernel_missing", probed)


@pytest.mark.parametrize("case", ["unlisted family", "qwen3 head off", "qwen3 spec on"])
def test_an_unset_knob_never_probes_k37_where_the_default_cannot_apply(monkeypatch, toy_family, k37, case):
    """The probe (and so ``import dense_i8``) runs only for a listed family with the head on and spec off: elsewhere an
    unset knob resolves to 1 without the call (a mutant that probes every family fails here)."""
    _probe_raises(monkeypatch)
    if case == "unlisted family":
        cfg, m = _cfg(monkeypatch, head_bw="1"), _Toy(model_type="olmoe")
    elif case == "qwen3 head off":
        cfg, m = _cfg(monkeypatch, head_bw="0"), _Toy()
    else:
        cfg, m = _cfg(monkeypatch), _Toy()
        cfg.spec = "eagle3"
    head, rows = _build(cfg, m)
    assert rows["rows_max"] == 1 and rows["source"].startswith("default-off: "), (case, rows)


def test_the_probe_runs_where_the_default_can_apply(monkeypatch, toy_family, k37):
    """The positive control for the test above: on the listed family, head on, spec off, the unset knob does call the
    probe, so the raising stub is the one the build reaches."""
    _probe_raises(monkeypatch)
    with pytest.raises(AssertionError, match="rows_kernel_missing was called"):
        _build(_cfg(monkeypatch), _Toy())


def test_an_explicit_1_opts_out_on_qwen3_moe(monkeypatch, toy_family, k37):
    m = _Toy()
    head, rows = _build(_cfg(monkeypatch, rows="1"), m)
    assert head["mode"] == "1" and rows == {"rows_max": 1, "source": "explicit"}
    assert m.lm_head.rows_max == 1 and m.lm_head._linear is None


@pytest.mark.parametrize("value", ["2", "4", "8"])
def test_an_explicit_value_routes_anywhere_the_head_is_on(monkeypatch, toy_family, k37, value):
    m = _Toy(model_type="mixtral")
    head, rows = _build(_cfg(monkeypatch, rows=value, head_bw="1"), m)
    assert rows["rows_max"] == int(value) and rows["source"] == "explicit" and m.lm_head.rows_max == int(value)


def test_an_explicit_value_above_1_still_refuses_a_head_that_is_off(monkeypatch, toy_family, k37):
    with pytest.raises(RuntimeError, match="E4B_INT8_HEAD_ROWS=8: the output head is Linear, not the int8 head"):
        _build(_cfg(monkeypatch, rows="8", head_bw="0"), _Toy())


def test_build_engine_resolves_the_route_after_the_swap_and_before_any_capture():
    src = inspect.getsource(sp.build_engine)
    at = [src.index(s) for s in ("int8_head = apply_int8_head_bw_resolved(model, cfg)",
                                 "int8_head_rows = apply_int8_head_rows_resolved(model, cfg, int8_head)",
                                 "runner.enable_decode_graphs(",
                                 'info["int8_head_bw"] = {**int8_head, "rows": int8_head_rows, "engagement"')]
    assert at == sorted(at), at
    assert "apply_int8_head_rows(model," not in src
