# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_INT8_HEAD_BW``'s per-family default (lane P153, FLIP; bench/p153/RESULTS-p153.md): an unset knob resolves to
``1`` on qwen3_moe at the head shape P153 read (151936 x 2048), to ``0`` everywhere else with the reason recorded, and
an ineligible head or a grouped-nf4-gemm without K36 resolves the default off rather than crashing a build. The copy's
bytes enter ``E4B_PAGED_MAX_SEQS=auto``'s footprint whenever the knob resolves on. On CPU with a stub kernel; the swap
itself (the grid, the routing, the refusals under an explicit ``1``) is tests/test_int8_head_bw.py's."""
import ast
import hashlib
import inspect
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from experts4bit_qlora import serve_paged as sp  # noqa: E402
from experts4bit_qlora.engines import int8_head_bw as ih  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
UNSET = sp.FUSION_UNSET
QWEN3_HEAD = (151936, 2048)
QUANTIZER_SHA256 = "8ea6dd9ee1b52594419f40339272059e413956aaed2a2f02d382450f854e8c6c"   # K36's pin of P148's bytes


def _load_quantize():
    """P148's ``quantize_rows_int8`` executed from its own bytes after its sha256 check (as tests/test_int8_head_bw.py)."""
    src = (REPO / "bench" / "p148" / "p148_kernels.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "quantize_rows_int8")
    seg = ast.get_source_segment(src, fn)
    assert hashlib.sha256(seg.encode("utf-8")).hexdigest() == QUANTIZER_SHA256
    ns = {"torch": torch}
    exec(compile(seg, "p148_kernels.quantize_rows_int8", "exec"), ns)
    return ns["quantize_rows_int8"]


_quantize = _load_quantize()


def _stub(x, q, s):
    return (s * (q.float() @ x.float().reshape(-1))).to(torch.bfloat16).reshape(1, -1)


class _Int4Like(nn.Linear):
    """A head subclass (as an int4 head is an Int4Linear): never swapped."""


class _Toy(nn.Module):
    """A model with a family and an output head (vocab x hidden), as build_engine sees one after its swaps."""

    def __init__(self, model_type="qwen3_moe", vocab=40, hidden=16, tied=False, bias=False, dtype=torch.bfloat16,
                 head_cls=nn.Linear):
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.embed = nn.Embedding(vocab, hidden, dtype=dtype)
        self.lm_head = head_cls(hidden, vocab, bias=bias, dtype=dtype)
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
    """The default table with the toy's head shape (40 x 16) in place of Qwen3's 151936 x 2048, which is 0.6 GB in bf16."""
    monkeypatch.setattr(sp, "INT8_HEAD_BW_DEFAULT_FAMILIES", {"qwen3_moe": ((40, 16),)})


def _cfg(monkeypatch, env):
    monkeypatch.setattr(sp, "_capability", lambda device: None)
    if env is None:
        monkeypatch.delenv("E4B_INT8_HEAD_BW", raising=False)
    else:
        monkeypatch.setenv("E4B_INT8_HEAD_BW", env)
    return sp.PagedServeConfig.from_env()


# ------------------------------------------------------------------------------------------ the table, the resolver --

def test_qwen3_moe_alone_takes_the_head_by_default_at_p153s_shape():
    """P153 (p153-5090-1) read FLIP for qwen3_moe at 151936 x 2048, the one shape K36 has a tuned plan for; nothing else
    is read. Qwen3-235B-A22B's head (151936 x 4096) takes K36's untuned plan and waits for a read of its own."""
    assert sp.INT8_HEAD_BW_DEFAULT_FAMILIES == {"qwen3_moe": (QWEN3_HEAD,)}
    assert sp.resolve_int8_head_bw(UNSET, "qwen3_moe", QWEN3_HEAD) == ("1", "default-allowlisted")
    mode, why = sp.resolve_int8_head_bw(UNSET, "qwen3_moe", (151936, 4096))
    assert mode == "0" and why.startswith("default-off: head shape [151936, 4096] is not one read on qwen3_moe")
    for family in ("mixtral", "qwen3_5_moe_text", "gpt_oss", "granitemoe", "olmoe", None):
        mode, why = sp.resolve_int8_head_bw(UNSET, family, QWEN3_HEAD)
        assert (mode, why) == ("0", f"default-off: no read licenses model_type {family!r}"), family
    assert sp.resolve_int8_head_bw("0", "qwen3_moe", QWEN3_HEAD) == ("0", "explicit")       # the way out
    assert sp.resolve_int8_head_bw("1", "mixtral", (32000, 4096)) == ("1", "explicit")      # explicit works anywhere


def test_a_refusal_or_a_missing_kernel_turns_the_default_off_and_never_an_explicit_1():
    assert sp.resolve_int8_head_bw(UNSET, "qwen3_moe", QWEN3_HEAD, refusal="tied") == ("0", "default-off: tied")
    assert sp.resolve_int8_head_bw(UNSET, "qwen3_moe", QWEN3_HEAD, kernel_error="no K36") == ("0", "default-off: no K36")
    # an explicit 1 is not resolved: the swap itself refuses an ineligible head or a missing kernel, loudly
    assert sp.resolve_int8_head_bw("1", "qwen3_moe", QWEN3_HEAD, refusal="tied", kernel_error="no K36") == ("1", "explicit")


@pytest.mark.parametrize("value,want", [("", UNSET), ("  ", UNSET), ("0", "0"), ("1", "1"), (" 1 ", "1")])
def test_the_env_reader(value, want):
    assert sp._int8_head_bw_env(value) == want


@pytest.mark.parametrize("value", ["2", "auto", "true", "on"])
def test_the_env_reader_refuses_anything_else(value):
    with pytest.raises(ValueError, match="E4B_INT8_HEAD_BW"):
        sp._int8_head_bw_env(value)


def test_the_server_config_leaves_an_unset_knob_to_the_family(monkeypatch):
    assert sp.PagedServeConfig.__dataclass_fields__["int8_head_bw"].default == UNSET
    assert _cfg(monkeypatch, None).int8_head_bw == UNSET
    assert _cfg(monkeypatch, "0").int8_head_bw == "0"


# ------------------------------------------------------------------------- end to end through the build's helper --

@pytest.mark.parametrize("env,model_type,vocab,swapped,source", [
    (None, "qwen3_moe", 40, True, "default-allowlisted"),                  # the flip: knob unset, the family's default
    (None, "mixtral", 40, False, "default-off: no read licenses model_type 'mixtral'"),
    (None, "qwen3_moe", 48, False, "default-off: head shape [48, 16] is not one read on qwen3_moe ([40, 16])"),
    ("0", "qwen3_moe", 40, False, "explicit"),                             # the way out
    ("1", "mixtral", 40, True, "explicit"),                                # explicit still works anywhere
])
def test_the_builds_helper_swaps_the_head_from_the_default(monkeypatch, toy_family, env, model_type, vocab, swapped,
                                                           source):
    """The server config read from the environment -> the per-family, per-shape resolution -> the swap and its record,
    through build_engine's own helper (apply_int8_head_bw_resolved). Only the kernel is a stub."""
    cfg = _cfg(monkeypatch, env)
    m = _Toy(model_type=model_type, vocab=vocab)
    rec = sp.apply_int8_head_bw_resolved(m, cfg, gemv=_stub, quantize=_quantize)
    assert rec["source"] == source and rec["mode"] == ("1" if swapped else "0")
    assert isinstance(m.get_output_embeddings(), ih.Int8HeadLinear) is swapped
    assert rec["modules"] == int(swapped)


@pytest.mark.parametrize("kw,why", [
    (dict(tied=True), "tied to the input embeddings"),
    (dict(bias=True), "has a bias"),
    (dict(dtype=torch.float32), "not a bf16"),
    (dict(head_cls=_Int4Like), "_Int4Like, not nn.Linear"),
])
def test_an_ineligible_head_turns_the_default_off_with_its_reason_never_a_crash(monkeypatch, toy_family, kw, why):
    cfg = _cfg(monkeypatch, None)
    m = _Toy(**kw)
    w = m.lm_head.weight.detach().clone()
    rec = sp.apply_int8_head_bw_resolved(m, cfg, gemv=_stub, quantize=_quantize)
    assert rec["mode"] == "0" and rec["source"].startswith("default-off: ") and why in rec["source"]
    assert type(m.lm_head) is not ih.Int8HeadLinear and torch.equal(m.lm_head.weight, w)       # untouched
    with pytest.raises(RuntimeError, match=why):                                                # an explicit 1 refuses
        sp.apply_int8_head_bw_resolved(_Toy(**kw), _cfg(monkeypatch, "1"), gemv=_stub, quantize=_quantize)


def test_a_grouped_nf4_gemm_without_k36_turns_the_default_off_and_an_explicit_1_refuses(monkeypatch, toy_family):
    """The floor (>= 0.48.0) protects normal installs; this protects an odd one (``--no-deps`` over an older release)."""
    monkeypatch.setitem(sys.modules, "dense_gemv", None)
    m = _Toy()
    rec = sp.apply_int8_head_bw_resolved(m, _cfg(monkeypatch, None))
    assert rec["mode"] == "0" and "K36, 0.48.0 or later" in rec["source"] and type(m.lm_head) is nn.Linear
    with pytest.raises(RuntimeError, match="K36"):
        sp.apply_int8_head_bw_resolved(_Toy(), _cfg(monkeypatch, "1"))


def test_build_engine_sets_the_head_through_apply_int8_head_bw_resolved():
    src = inspect.getsource(sp.build_engine)
    assert "apply_int8_head_bw_resolved(model, cfg)" in src and "apply_int8_head_bw(model," not in src


# ----------------------------------------------------------------------------------- memory: E4B_PAGED_MAX_SEQS=auto --

def _qwen3_config(**kw):
    tr = pytest.importorskip("transformers")
    args = dict(hidden_size=2048, intermediate_size=6144, moe_intermediate_size=768, num_experts=128,
                num_experts_per_tok=8, num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4,
                head_dim=128, vocab_size=151936, max_position_embeddings=40960, decoder_sparse_step=1,
                tie_word_embeddings=False)
    args.update(kw)
    return tr.Qwen3MoeConfig(**args)


@pytest.fixture
def kernel_present(monkeypatch):
    monkeypatch.setattr(ih, "kernel_missing", lambda: None)


def test_the_copys_bytes_are_priced_whenever_the_knob_resolves_on(monkeypatch, kernel_present):
    """311,772,672 bytes at Qwen3-30B-A3B's head: the codes and scales, the rise in memory_allocated P153 read."""
    assert ih.copy_bytes(*QWEN3_HEAD) == 311_772_672
    monkeypatch.delenv("E4B_SERVE_LMHEAD_INT4_CALIB", raising=False)
    unset, off, on = (_cfg(monkeypatch, v) for v in (None, "0", "1"))
    assert sp.int8_head_copy_bytes(unset, _qwen3_config()) == 311_772_672             # the default, on
    assert sp.int8_head_copy_bytes(off, _qwen3_config()) == 0                          # opted out
    assert sp.int8_head_copy_bytes(unset, _qwen3_config(hidden_size=4096)) == 0       # an unread shape: default-off
    assert sp.int8_head_copy_bytes(on, _qwen3_config(hidden_size=4096)) == ih.copy_bytes(151936, 4096)   # explicit
    assert sp.int8_head_copy_bytes(unset, _qwen3_config(tie_word_embeddings=True)) == 0   # the swap would refuse
    assert sp.int8_head_copy_bytes(unset, None) == 0 and sp.int8_head_copy_bytes(unset, object()) == 0
    monkeypatch.setenv("E4B_SERVE_LMHEAD_INT4_CALIB", "1")                            # an int4 head: never swapped
    assert sp.int8_head_copy_bytes(unset, _qwen3_config()) == 0


def test_without_k36_the_default_prices_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "dense_gemv", None)
    monkeypatch.delenv("E4B_SERVE_LMHEAD_INT4_CALIB", raising=False)
    assert sp.int8_head_copy_bytes(_cfg(monkeypatch, None), _qwen3_config()) == 0


def test_the_auto_width_drops_by_the_copys_share_at_the_margin():
    """At free memory exactly the 64-slot need, 64 fits with nothing extra; with the int8 head's bytes it does not, and
    the next width is taken. The candidates carry the term, for /health."""
    from experts4bit_qlora.arch.topology import describe_moe
    from experts4bit_qlora.serve_recipe import ServeSetup, choose_max_seqs
    topo = describe_moe(_qwen3_config())
    st = ServeSetup(max_tokens_per_seq=2048, exp_int4=True, attn_int4=True)
    extra = ih.copy_bytes(*QWEN3_HEAD)
    need64 = choose_max_seqs(topo, st, 1 << 40)["candidates"][0]
    assert need64["max_seqs"] == 64 and "extra_bytes" not in need64
    free = need64["need_bytes"]
    assert choose_max_seqs(topo, st, free)["max_seqs"] == 64
    r = choose_max_seqs(topo, st, free, extra_bytes=extra)
    assert r["max_seqs"] == 16 and r["extra_bytes"] == extra
    assert r["candidates"][0]["need_bytes"] == need64["need_bytes"] + extra and not r["candidates"][0]["fits"]
    assert choose_max_seqs(topo, st, free + extra, extra_bytes=extra)["max_seqs"] == 64


def test_resolve_max_seqs_hands_the_copys_bytes_to_the_chooser(monkeypatch, kernel_present):
    from experts4bit_qlora import serve_recipe
    monkeypatch.setenv("E4B_PAGED_MAX_SEQS", "auto")
    monkeypatch.delenv("E4B_SERVE_LMHEAD_INT4_CALIB", raising=False)
    seen = {}

    def fake_choose(topo, setup, free, *, widths, extra_bytes=0):
        seen["extra"] = extra_bytes
        return {"max_seqs": 16, "free_bytes": free, "candidates": [], "why": "test"}
    monkeypatch.setattr(serve_recipe, "choose_max_seqs", fake_choose)
    sp.resolve_max_seqs(_cfg(monkeypatch, None), _qwen3_config(), 1 << 40)
    assert seen["extra"] == 311_772_672
    sp.resolve_max_seqs(_cfg(monkeypatch, "0"), _qwen3_config(), 1 << 40)
    assert seen["extra"] == 0
