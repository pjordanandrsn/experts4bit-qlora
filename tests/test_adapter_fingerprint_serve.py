# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The served base's qfp-v1 against a backend-adapter/2 manifest's (docs/ADAPTER_PORTABILITY.md; refusals 11 and 12).

The arena here is written by grouped-nf4-gemm's own bake (``nvme_bake_nf4.bake_nf4`` on a tiny MoE), never hand-built
bytes, so the resident-vs-arena parity is checked on the writer that serves. Training's quantizer
(``Experts4bit.from_float``) and the bake both call ``bitsandbytes.functional.quantize_4bit(w, blocksize=64, nf4)`` per
expert on ``cat([gate, up])``, so an fp32-absmax-trained adapter verifies; a double-quantized one does not (F1)."""
import importlib.util
import json
import os
import pathlib
import sys
import types

import pytest

#: under CI (GitHub Actions sets CI=true) every dependency is a HARD import, so a green check proves these tests ran:
#: CI's grouped-nf4-gemm ships nvme_bake_nf4 and nvme_residency, and ColdTier(pinned=False) needs no CUDA
IN_CI = os.environ.get("CI") == "true"


def _need(name):
    return importlib.import_module(name) if IN_CI else pytest.importorskip(name)


torch = _need("torch")
_need("bitsandbytes")

ROOT = pathlib.Path(__file__).resolve().parents[1]
L, E, INTER, H = 2, 3, 64, 128
REV = "c" * 40


def _weights():
    g = torch.Generator().manual_seed(0)
    return {(li, e, proj): torch.randn(*shape, generator=g).to(torch.bfloat16)
            for li in range(L) for e in range(E)
            for proj, shape in (("gate_proj", (INTER, H)), ("up_proj", (INTER, H)), ("down_proj", (H, INTER)))}


def _bake(tmp_path, W):
    """A real arena: the tiny MoE in an HF-cache-shaped snapshot, quantized by the bake's own quantizer on the CPU."""
    nb = _need("nvme_bake_nf4")
    st = _need("safetensors.torch")
    snap = tmp_path / "models--org--tiny-moe" / "snapshots" / REV
    snap.mkdir(parents=True)
    tensors = {f"model.layers.{li}.mlp.experts.{e}.{proj}.weight": t for (li, e, proj), t in W.items()}
    st.save_file(tensors, str(snap / "model-00001-of-00001.safetensors"))
    wm = {k: "model-00001-of-00001.safetensors" for k in tensors}
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}), encoding="utf-8")
    arena = str(tmp_path / "nf4.arena")
    nb.bake_nf4(str(snap), arena, quantize_fn=nb.default_quantize_expert(dev="cpu"), log=lambda *a, **k: None)
    return arena


def _model(W, wrap=False):
    """A served-shaped tree: each layer's attention projections (bf16) and its expert stack, quantized as training does
    (``wrap``: inside training's ExpertsLoRA, the form whose absmax training double-quantizes by default)."""
    from torch import nn

    from experts4bit_qlora._vendor.experts import Experts4bit
    from experts4bit_qlora.lora import ExpertsLoRA
    g = torch.Generator().manual_seed(1)
    layers = nn.ModuleList()
    for li in range(L):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = nn.Linear(H, H, bias=False).to(torch.bfloat16)
            lin.weight.data = torch.randn(H, H, generator=g).to(torch.bfloat16)
            setattr(layer.self_attn, proj, lin)
        layer.mlp = nn.Module()
        gu = torch.stack([torch.cat([W[(li, e, "gate_proj")], W[(li, e, "up_proj")]], 0) for e in range(E)])
        dn = torch.stack([W[(li, e, "down_proj")] for e in range(E)])
        stack = Experts4bit.from_float(gu, dn, quant_type="nf4", blocksize=64)
        layer.mlp.experts = ExpertsLoRA(stack, r=2, alpha=4) if wrap else stack
        layers.append(layer)
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = layers
    return root


def _ad(fingerprints, form="fp32", schema="backend-adapter/2"):
    return types.SimpleNamespace(manifest={"schema": schema, "fingerprints": fingerprints,
                                           "expert_absmax": {"form": form}}, attention={"q": None})


def _fps(model, arena=None):
    from experts4bit_qlora import adapter_schema as S
    return {"attention": S.quant_fingerprint(model, "attention"),
            "experts": S.quant_fingerprint(model, "experts", source="arena", arena=arena) if arena
            else S.quant_fingerprint(model, "experts")}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    W = _weights()
    arena = _bake(tmp_path_factory.mktemp("bake"), W)
    return W, arena, _model(W)


def test_the_arena_fingerprint_equals_the_resident_one_on_a_real_bake(served):
    from experts4bit_qlora import adapter_schema as S
    _W, arena, model = served
    resident = S.quant_fingerprint(model, "experts")
    from_arena = S.quant_fingerprint(model, "experts", source="arena", arena=arena)
    assert from_arena["sha256"] == resident["sha256"] and from_arena["modules"] == resident["modules"] == 2 * L
    assert from_arena["source"] == "arena" and from_arena["bytes_hashed"] == resident["bytes_hashed"]


def test_a_matching_manifest_verifies_both_scopes(served):
    from experts4bit_qlora import adapter_dir as AD
    _W, arena, model = served
    refusals, report = AD.fingerprint_check(_ad(_fps(model)), model, arena)
    assert refusals == [] and {r["status"] for r in report.values()} == {"verified"}
    bm = AD.base_match(_ad(_fps(model)), fingerprints=report)
    assert bm["experts"].startswith("verified (qfp-v1 ") and "base verified" in bm["attention"]


def _dq(W):
    """A dq-trained base as training builds it: the stacks in ExpertsLoRA, the absmax compressed (training's default)."""
    from experts4bit_qlora.absmax_dq import compress_expert_absmax_
    dq_model = _model(W, wrap=True)
    assert compress_expert_absmax_(dq_model) == L                 # training's default absmax form (F1)
    return dq_model


def _without_source(fps):
    """A manifest written before the fp32 source record (#2014's form)."""
    return dict(fps, experts={k: v for k, v in fps["experts"].items() if k != "fp32_source"})


def test_a_dq_trained_adapter_verifies_by_its_fp32_source_without_an_override(served):
    from experts4bit_qlora import adapter_dir as AD
    W, arena, model = served
    trained = _fps(_dq(W))                                         # what a dq-trained /2 manifest records
    src = trained["experts"]["fp32_source"]
    assert trained["experts"]["sha256"] != src["sha256"]           # the trained form is the compressed one
    refusals, report = AD.fingerprint_check(_ad(trained, form="double-quantized"), model, arena)
    assert refusals == [] and report["experts"]["status"] == "verified (source)"
    assert report["experts"]["fp32_source"] == src["sha256"] == report["experts"]["served"]
    assert report["experts"]["trained"] == trained["experts"]["sha256"] and report["attention"]["status"] == "verified"
    bm = AD.base_match(_ad(trained, form="double-quantized"), fingerprints=report)["experts"]
    # the BASE is proven, never "served == trained"
    assert bm.startswith("verified (source, qfp-v1 ") and bm.endswith(
        "trained on double-quantized absmax, served on fp32 absmax")


def test_the_fp32_source_is_the_full_fp32_absmax_hashed_before_compression(served):
    """Condition (c): the recorded hash covers each stack's FULL fp32 absmax, taken before compression, keyed per
    stack: it equals the same base's fingerprint while fp32, stack by stack, and the arena's (condition (b) on CPU:
    training's quantize and the bake's reach the same fp32 absmax)."""
    from experts4bit_qlora import adapter_schema as S
    W, arena, model = served
    src = S.quant_fingerprint(_dq(W), "experts")["fp32_source"]
    fp32 = S.quant_fingerprint(model, "experts")
    served_fp = S.quant_fingerprint(model, "experts", source="arena", arena=arena)
    assert src["sha256"] == fp32["sha256"] == served_fp["sha256"]
    assert src["absmax_by_stack"] == fp32["absmax_by_stack"] == served_fp["absmax_by_stack"]
    assert sorted(src["absmax_by_stack"]) == sorted(f"model.layers.{li}.mlp.experts.{w}" for li in range(L)
                                                    for w in ("gate_up", "down"))
    stack = model.model.layers[0].mlp.experts
    full = S.absmax_sha256(stack.gate_up_absmax)                    # the whole buffer's bytes
    assert src["absmax_by_stack"]["model.layers.0.mlp.experts.gate_up"] == full
    assert "fp32_source" not in fp32                               # an fp32 base records no second fingerprint


def test_a_dq_adapter_without_the_source_record_is_refusal_12_naming_the_override_and_the_dq_case(served):
    from experts4bit_qlora import adapter_dir as AD
    W, arena, model = served
    trained = _without_source(_fps(_dq(W)))                        # a dq-trained /2 manifest from before the record
    refusals, report = AD.fingerprint_check(_ad(trained, form="double-quantized"), model, arena)
    assert [c for c, _ in refusals] == [12] and report["experts"]["status"] == "mismatch"
    msg = refusals[0][1]
    assert "E4B_PAGED_ADAPTER_ACCEPT_BASE_MISMATCH=experts" in msg            # the exact override, named
    assert "dq-vs-fp32 absmax case" in msg and "records the fp32 source fingerprint" in msg
    assert report["attention"]["status"] == "verified"
    ok, rep = AD.fingerprint_check(_ad(trained, form="double-quantized"), model, arena,
                                   accept_base_mismatch=frozenset({"experts"}))
    assert ok == [] and rep["experts"]["status"] == "mismatch (served under E4B_PAGED_ADAPTER_ACCEPT_BASE_MISMATCH=experts)"
    assert AD.base_match(_ad(trained), fingerprints=rep)["experts"].startswith("mismatch (served under")


def test_a_source_fingerprint_the_arena_does_not_match_is_refusal_12_naming_the_stack(served):
    from experts4bit_qlora import adapter_dir as AD
    W, arena, model = served
    trained = _fps(_dq(W))
    src = trained["experts"]["fp32_source"]
    key = "model.layers.1.mlp.experts.down"
    trained["experts"]["fp32_source"] = {"sha256": "2" * 64,
                                         "absmax_by_stack": dict(src["absmax_by_stack"], **{key: "3" * 64})}
    refusals, report = AD.fingerprint_check(_ad(trained, form="double-quantized"), model, arena)
    assert [c for c, _ in refusals] == [12] and report["experts"]["status"] == "mismatch"
    msg = refusals[0][1]
    assert "matches neither" in msg and f"the first stack whose absmax differs: {key}" in msg
    assert "dq-vs-fp32" not in msg                                 # a recorded source rules the dq case out
    ok, rep = AD.fingerprint_check(_ad(trained, form="double-quantized"), model, arena,
                                   accept_base_mismatch=frozenset({"experts"}))
    assert ok == [] and rep["experts"]["status"].startswith("mismatch (served under")


def test_a_malformed_fp32_source_is_refused_by_the_verifier(served):
    from experts4bit_qlora import adapter_dir as AD
    W, _arena, _model_ = served
    fp = _fps(_dq(W))["experts"]
    AD._verify_fp32_source(fp, "experts", ValueError)              # well formed
    src = fp["fp32_source"]
    for bad, scope in ((dict(src, sha256="x"), "experts"),
                       (dict(src, absmax_by_stack=dict(list(src["absmax_by_stack"].items())[1:])), "experts"),
                       (dict(src, absmax_by_stack={k: "nope" for k in src["absmax_by_stack"]}), "experts"),
                       ("ab" * 32, "experts"), (src, "attention")):
        with pytest.raises(ValueError, match="fp32_source"):
            AD._verify_fp32_source(dict(fp, fp32_source=bad), scope, ValueError)


def test_an_fp32_manifest_mismatch_does_not_claim_the_dq_case(served):
    from experts4bit_qlora import adapter_dir as AD
    _W, arena, model = served
    fps = _fps(model)
    fps["experts"] = dict(fps["experts"], sha256="0" * 64)
    refusals, _ = AD.fingerprint_check(_ad(fps, form="fp32"), model, arena)
    assert [c for c, _ in refusals] == [12] and "dq-vs-fp32" not in refusals[0][1]
    assert "not baked from the base this adapter was trained on" in refusals[0][1]


def test_an_attention_mismatch_is_refusal_11_with_no_override(served):
    from experts4bit_qlora import adapter_dir as AD
    _W, arena, model = served
    fps = _fps(model)
    fps["attention"] = dict(fps["attention"], sha256="1" * 64)
    for accept in (frozenset(), frozenset({"experts"})):
        refusals, report = AD.fingerprint_check(_ad(fps), model, arena, accept_base_mismatch=accept)
        assert [c for c, _ in refusals] == [11] and "no override" in refusals[0][1]
        assert report["attention"]["status"] == "mismatch"


def test_a_v1_manifest_or_an_unknown_scheme_is_not_compared(served):
    from experts4bit_qlora import adapter_dir as AD
    _W, arena, model = served
    assert AD.fingerprint_check(_ad(_fps(model), schema="backend-adapter/1"), model, arena) == ([], {})
    odd = {k: dict(v, scheme_id="qfp-v9") for k, v in _fps(model).items()}
    assert AD.fingerprint_check(_ad(odd), model, arena) == ([], {})


def test_the_mismatch_override_names_its_scope_and_never_attention():
    from experts4bit_qlora import adapter_dir as AD
    assert AD.parse_accept_base_mismatch("experts") == frozenset({"experts"})
    assert AD.parse_accept_base_mismatch("") == frozenset()
    for bad in ("attention", "1", "true", "experts,attention"):
        with pytest.raises(ValueError, match="E4B_PAGED_ADAPTER_ACCEPT_BASE_MISMATCH"):
            AD.parse_accept_base_mismatch(bad)


def test_the_fingerprint_is_taken_after_the_load_before_the_merge_and_every_lever():
    src = (ROOT / "experts4bit_qlora" / "serve_paged.py").read_text(encoding="utf-8")
    body = src[src.index("def build_engine("):]
    order = [body.index(x) for x in ("load_moe_4bit_streaming(", "_raise_adapter_refusals(_adapter_refusals(adapter, cfg, "
                                     "arena_index, served_revision=served))", "fingerprint_check(adapter, model, cfg.arena",
                                     "_raise_adapter_refusals(fp_refusals)", "report = merge_attention_lora(model, adapter)",
                                     "enable_hybrid_tier(", "levers = _apply_levers(", "fusions = _apply_fusions(")]
    assert order == sorted(order)
    assert '"fingerprints": fingerprints' in body and "base_match(adapter, fingerprints=fingerprints)" in body


@pytest.mark.skipif(sys.platform != "linux", reason="nvme_residency's ColdTier reads with os.preadv (Linux); the "
                                                     "reader comparison runs in Linux CI, never skipped there")
def test_the_hashed_bytes_are_the_bytes_the_server_reads(served):
    """A second reader could hash bytes the server never reads: PA-1's seek/read must return exactly what
    nvme_residency.segment_tensor (the serving tier's reader) returns for the same rows and segments."""
    nr = _need("nvme_residency")
    from experts4bit_qlora import adapter_schema as S
    from experts4bit_qlora.engines.nvme_train import arena_offload_view
    _W, arena, _model_ = served
    index = json.loads(open(arena + ".index.json", encoding="utf-8").read())
    view, segs = arena_offload_view(index)
    offsets = {(int(r[0]), int(r[1])): int(r[2]) for r in view["rows"]}
    tier = nr.ColdTier(arena, hot_rows=E, pinned=False)
    try:
        with open(arena, "rb") as fh:
            for layer in range(L):
                for key in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax"):
                    mine = S._arena_segment(fh, view, offsets, layer, E, segs[key])
                    served_bytes = nr.segment_tensor(tier, view, layer, range(E), segs[key])
                    assert torch.equal(mine.view(torch.uint8), served_bytes.contiguous().view(torch.uint8)), (layer, key)
    finally:
        tier.close()


MUTANTS = [
    ('        if scope == "attention":\n            refusals.append((11,',
     '        if False:\n            refusals.append((11,'),                                            # refusal 11
    ('        if "experts" in accept_base_mismatch:\n', '        if True:\n'),                        # 12 always waived
    ('        match = got["sha256"] == want.get("sha256")', '        match = True or got["sha256"] == want.get("sha256")'),
    ('            elif _dq_trained(ad):\n', '            elif False:\n'),                         # the dq explanation
    ('        if src is not None and src.get("sha256") == got["sha256"] and', '        if src is not None and'),
    ('        if src is not None and src.get("sha256") == got["sha256"] and', '        if False and'),
]


@pytest.mark.parametrize("old,new", MUTANTS)
def test_each_mutant_is_caught(served, tmp_path, monkeypatch, old, new):
    src = (ROOT / "experts4bit_qlora" / "adapter_dir.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    path = tmp_path / "adapter_dir_fp_mutant.py"
    path.write_bytes(src.replace(old, new).encode("utf-8"))
    spec = importlib.util.spec_from_file_location("experts4bit_qlora.adapter_dir_fp_mutant", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    import experts4bit_qlora as pkg
    monkeypatch.setattr(pkg, "adapter_dir", mod, raising=False)
    monkeypatch.setitem(sys.modules, "experts4bit_qlora.adapter_dir", mod)
    caught = 0
    for test in (test_a_dq_adapter_without_the_source_record_is_refusal_12_naming_the_override_and_the_dq_case,
                 test_a_dq_trained_adapter_verifies_by_its_fp32_source_without_an_override,
                 test_a_source_fingerprint_the_arena_does_not_match_is_refusal_12_naming_the_stack,
                 test_an_attention_mismatch_is_refusal_11_with_no_override,
                 test_an_fp32_manifest_mismatch_does_not_claim_the_dq_case):
        try:
            test(served)
        except AssertionError:
            caught += 1
    assert caught >= 1, f"mutant survived: {new!r}"


SOURCE_MUTANTS = [
    ("        source[which] = absmax_sha256(absmax)\n",
     "        source[which] = absmax_sha256(absmax.reshape(-1)[:64])\n"),                 # a subset, not the full absmax
    ("        source[which] = absmax_sha256(absmax)\n", ""),                              # never recorded
]


@pytest.mark.parametrize("old,new", SOURCE_MUTANTS)
def test_each_training_side_mutant_is_caught(served, tmp_path, monkeypatch, old, new):
    """Condition (c) on the writer: the hash must be the FULL fp32 absmax, recorded before compression."""
    src = (ROOT / "experts4bit_qlora" / "absmax_dq.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    path = tmp_path / "absmax_dq_mutant.py"
    path.write_bytes(src.replace(old, new).encode("utf-8"))
    spec = importlib.util.spec_from_file_location("experts4bit_qlora.absmax_dq", path,
                                                  submodule_search_locations=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "experts4bit_qlora"
    monkeypatch.setitem(sys.modules, "experts4bit_qlora.absmax_dq", mod)
    spec.loader.exec_module(mod)
    with pytest.raises((AssertionError, KeyError, NameError)):
        test_the_fp32_source_is_the_full_fp32_absmax_hashed_before_compression(served)
