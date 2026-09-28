# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The calibrated int4 attention as a hash-pinned artifact (#674).

Before this, the attention half of the licensed Qwen3 stack was re-derived from a
Hessian pass on every load, so a pack fingerprint identified the experts only. These
tests pin the new contract on CPU (reference kernels stubbed as in
``test_int4_attn_calib.py``): a dump names its bytes; a licensed load installs
exactly those bytes and never calibrates; every drift refuses.
"""
import json
import types

import pytest
import torch
from torch import nn

from test_int4_attn_calib import HEADS, D, H, ToyModel, _stubs

REV = "0123456789abcdef0123456789abcdef01234567"


def _model(seed=0, rev=REV, cls=ToyModel):
    torch.manual_seed(seed)
    m = cls()
    m.config = types.SimpleNamespace(_commit_hash=rev, _name_or_path="toy/attn")
    return m


def _calibrated(monkeypatch, seed=0, cls=ToyModel):
    _stubs(monkeypatch, [])
    from experts4bit_qlora.engines.int4_attn_calib import (
        calibrate_attention_hessians, enable_serve_attn_int4_calib)
    m = _model(seed, cls=cls)
    torch.manual_seed(100 + seed)
    hs = calibrate_attention_hessians(m, [torch.randint(0, 50, (2, 8))], device="cpu")
    assert enable_serve_attn_int4_calib(m, hs) == 2
    return m


def _stores(m):
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    return {n: (mod.packed.clone(), mod.scales.clone(), None if mod.bias is None else mod.bias.clone())
            for n, mod in m.named_modules() if isinstance(mod, Int4Linear)}


def test_a_licensed_load_installs_exactly_the_dumped_bytes_and_never_calibrates(monkeypatch, tmp_path):
    from experts4bit_qlora.engines import int4_attn_calib as ac
    src = _calibrated(monkeypatch, seed=0)
    man = ac.dump_attn_int4_artifact(src, str(tmp_path / "pack"))
    assert man["layout"] == "int4_b32.attn.v1"
    assert [r["name"] for r in man["layers"]] == ["attn.o_proj", "attn.qkv_proj"]
    assert all(r["calibrated"] and r["group"] == "attention" for r in man["layers"])

    # a DIFFERENT model's weights: whatever it serves after the load came from the artifact
    dst = _model(seed=7)
    monkeypatch.setattr(ac, "calibrate_attention_hessians",
                        lambda *a, **k: pytest.fail("a licensed load must not calibrate"))
    n = ac.enable_serve_attn_int4_from_artifact(dst, str(tmp_path / "pack"),
                                               expected_fingerprint=man["pack_fingerprint"])
    assert n == 2
    a, b = _stores(src), _stores(dst)
    assert set(a) == set(b)
    for name in a:
        assert torch.equal(a[name][0], b[name][0]), name
        assert torch.equal(a[name][1], b[name][1]), name
        assert getattr(dst.get_submodule(name), "_e4b_calibrated") is True


def test_the_fingerprint_names_the_bytes_not_the_directory(monkeypatch, tmp_path):
    from experts4bit_qlora.engines import int4_attn_calib as ac
    m0 = _calibrated(monkeypatch, seed=0)
    fp_a = ac.dump_attn_int4_artifact(m0, str(tmp_path / "a"))["pack_fingerprint"]
    fp_b = ac.dump_attn_int4_artifact(m0, str(tmp_path / "b"))["pack_fingerprint"]
    assert fp_a == fp_b
    m1 = _calibrated(monkeypatch, seed=1)
    assert ac.dump_attn_int4_artifact(m1, str(tmp_path / "c"))["pack_fingerprint"] != fp_a


def test_the_receipt_names_the_attention_pack(monkeypatch, tmp_path):
    from experts4bit_qlora.engines import int4_attn_calib as ac
    from experts4bit_qlora.engines.pack_manifest import merge_provenance_into_receipt
    m = _calibrated(monkeypatch)
    fp = ac.dump_attn_int4_artifact(m, str(tmp_path / "p"))["pack_fingerprint"]
    rep = merge_provenance_into_receipt({}, m)
    assert rep["attn_pack_fingerprint"] == fp and rep["attn_pack_source"] == "calibrated-and-dumped"
    dst = _model(seed=3)
    ac.enable_serve_attn_int4_from_artifact(dst, str(tmp_path / "p"), expected_fingerprint=fp)
    rep = merge_provenance_into_receipt({}, dst)
    assert rep["attn_pack_fingerprint"] == fp and rep["attn_pack_source"] == "artifact"


def _flip_a_payload_byte(root):
    p = next((root / "payloads" / "attn").glob("*.packed.bin"))
    b = bytearray(p.read_bytes())
    b[-1] ^= 0x01
    p.write_bytes(bytes(b))


def _edit_manifest_layers(root):
    mp = root / "manifest.json"
    man = json.loads(mp.read_text())
    man["layers"][0]["N"] += 32
    mp.write_text(json.dumps(man))


@pytest.mark.parametrize("drift,match", [
    ("payload", "does not match the manifest"),
    ("fingerprint", "!= expected"),
    ("revision", "model_revision"),
    ("no_live_revision", "_commit_hash"),
    ("manifest_layers", "hashed identity payload"),
])
def test_a_licensed_load_refuses_every_drift(monkeypatch, tmp_path, drift, match):
    from experts4bit_qlora.engines import int4_attn_calib as ac
    from experts4bit_qlora.engines.pack_manifest import PackManifestError
    m = _calibrated(monkeypatch)
    root = tmp_path / "p"
    fp = ac.dump_attn_int4_artifact(m, str(root))["pack_fingerprint"]
    dst = _model(seed=5)
    if drift == "payload":
        _flip_a_payload_byte(root)
    elif drift == "fingerprint":
        fp = "sha256:" + "0" * 64
    elif drift == "revision":
        dst.config._commit_hash = "f" * 40
    elif drift == "no_live_revision":
        dst.config._commit_hash = None
    elif drift == "manifest_layers":
        _edit_manifest_layers(root)
    before = _stores(dst)
    with pytest.raises(PackManifestError, match=match):
        ac.enable_serve_attn_int4_from_artifact(dst, str(root), expected_fingerprint=fp)
    assert _stores(dst) == before == {}, "a refused load must not have swapped anything"


class _WideAttention(nn.Module):
    """ToyModel's attention plus one more projection: a different target set."""

    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(H, 3 * HEADS * D, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(HEADS * D, H, bias=False, dtype=torch.bfloat16)
        self.extra_proj = nn.Linear(H, H, bias=False, dtype=torch.bfloat16)


def test_a_licensed_load_refuses_a_different_projection_set(monkeypatch, tmp_path):
    from experts4bit_qlora.engines import int4_attn_calib as ac
    from experts4bit_qlora.engines.pack_manifest import PackManifestError
    m = _calibrated(monkeypatch)
    fp = ac.dump_attn_int4_artifact(m, str(tmp_path / "p"))["pack_fingerprint"]
    dst = _model(seed=2)
    dst.attn = _WideAttention()
    with pytest.raises(PackManifestError, match="partial licensed load"):
        ac.enable_serve_attn_int4_from_artifact(dst, str(tmp_path / "p"), expected_fingerprint=fp)


def test_the_dump_refuses_fused_modules(monkeypatch, tmp_path):
    from experts4bit_qlora.engines import int4_attn_calib as ac
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    m = _calibrated(monkeypatch)
    m.attn.qkv_proj = Int4Linear.fuse([m.attn.qkv_proj, m.attn.o_proj])
    with pytest.raises(RuntimeError, match="before qkv_fuse"):
        ac.dump_attn_int4_artifact(m, str(tmp_path / "p"))
    assert not (tmp_path / "p").exists(), "a refused dump must not leave a partial artifact"


def test_the_dump_refuses_an_unknown_revision_unless_asked(monkeypatch, tmp_path):
    from experts4bit_qlora.engines import int4_attn_calib as ac
    m = _calibrated(monkeypatch)
    m.config._commit_hash = None
    with pytest.raises(RuntimeError, match="revision"):
        ac.dump_attn_int4_artifact(m, str(tmp_path / "p"))
    man = ac.dump_attn_int4_artifact(m, str(tmp_path / "q"), allow_unknown_revision=True)
    assert man["model_revision_missing"] is True


class _BiasedModel(ToyModel):
    def __init__(self):
        super().__init__()
        self.attn.o_proj = nn.Linear(HEADS * D, H, bias=True, dtype=torch.bfloat16)


def test_a_projection_bias_rides_in_the_pack(monkeypatch, tmp_path):
    from experts4bit_qlora.engines import int4_attn_calib as ac
    from experts4bit_qlora.engines.pack_manifest import PackManifestError
    src = _calibrated(monkeypatch, cls=_BiasedModel)
    man = ac.dump_attn_int4_artifact(src, str(tmp_path / "p"))
    assert {r["name"]: r["bias"] for r in man["layers"]} == {"attn.o_proj": True, "attn.qkv_proj": False}
    dst = _model(seed=9, cls=_BiasedModel)
    ac.enable_serve_attn_int4_from_artifact(dst, str(tmp_path / "p"),
                                           expected_fingerprint=man["pack_fingerprint"])
    assert torch.equal(dst.attn.o_proj.bias, src.attn.o_proj.bias)
    # and a live projection WITHOUT the bias the pack carries is refused, not silently biased
    with pytest.raises(PackManifestError, match="bias"):
        ac.enable_serve_attn_int4_from_artifact(_model(seed=9), str(tmp_path / "p"),
                                               expected_fingerprint=man["pack_fingerprint"])


def test_enable_from_env_dumps_loads_and_refuses_by_name(monkeypatch, tmp_path):
    from experts4bit_qlora.engines import int4_attn_calib as ac
    _stubs(monkeypatch, [])
    ids = [torch.randint(0, 50, (2, 8))]
    for k in ("E4B_SERVE_ATTN_INT4_CALIB", "E4B_SERVE_LMHEAD_INT4_CALIB", "E4B_SERVE_DENSE_INT4_CALIB",
              "E4B_SERVE_ATTN_INT4_ARTIFACT", "E4B_SERVE_ATTN_INT4_FINGERPRINT", "E4B_SERVE_ATTN_INT4_DUMP"):
        monkeypatch.delenv(k, raising=False)

    # calibrate + dump
    monkeypatch.setenv("E4B_SERVE_ATTN_INT4_CALIB", "1")
    monkeypatch.setenv("E4B_SERVE_ATTN_INT4_DUMP", str(tmp_path / "p"))
    src = _model(seed=0)
    assert ac.enable_from_env(src, ids) == 2
    man = json.loads((tmp_path / "p" / "manifest.json").read_text())
    assert man["calibration_token_stream_sha"].startswith("sha256:")
    monkeypatch.delenv("E4B_SERVE_ATTN_INT4_DUMP")

    # an artifact without its fingerprint refuses
    monkeypatch.setenv("E4B_SERVE_ATTN_INT4_ARTIFACT", str(tmp_path / "p"))
    with pytest.raises(RuntimeError, match="FINGERPRINT"):
        ac.enable_from_env(_model(seed=1), ids)

    # with it: installs, and never calibrates
    monkeypatch.setenv("E4B_SERVE_ATTN_INT4_FINGERPRINT", man["pack_fingerprint"])
    monkeypatch.setattr(ac, "calibrate_attention_hessians",
                        lambda *a, **k: pytest.fail("the artifact path must not calibrate"))
    dst = _model(seed=1)
    assert ac.enable_from_env(dst, ids) == 2
    assert all(torch.equal(x[0], y[0]) for x, y in zip(_stores(src).values(), _stores(dst).values()))

    # an artifact with no flag naming its groups refuses rather than doing nothing
    monkeypatch.delenv("E4B_SERVE_ATTN_INT4_CALIB")
    with pytest.raises(RuntimeError, match="no E4B_SERVE"):
        ac.enable_from_env(_model(seed=1), ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_kernels_a_loaded_pack_serves_the_calibrated_function_bitwise(tmp_path):
    """No stubs: grouped-nf4-gemm's own GPTQ packer and int4 kernels. A model loaded
    from the pack must serve exactly what the calibrated model served, at decode
    (1 row, the GEMV) and above it (several rows)."""
    pytest.importorskip("gptq_pack", reason="needs grouped-nf4-gemm with gptq_pack")
    pytest.importorskip("int4_b32", reason="needs grouped-nf4-gemm with int4_b32")
    from experts4bit_qlora.engines import int4_attn_calib as ac

    src = _model(seed=0).cuda()
    torch.manual_seed(11)
    calib = [torch.randint(0, 50, (2, 16), device="cuda") for _ in range(4)]
    hs = ac.calibrate_attention_hessians(src, calib)
    assert ac.enable_serve_attn_int4_calib(src, hs) == 2
    man = ac.dump_attn_int4_artifact(src, str(tmp_path / "p"))

    dst = _model(seed=7).cuda()          # different bf16 weights underneath
    ac.enable_serve_attn_int4_from_artifact(dst, str(tmp_path / "p"),
                                           expected_fingerprint=man["pack_fingerprint"])
    for n_rows in (1, 4, 24):
        ids = torch.randint(0, 50, (1, n_rows), device="cuda")
        # the embedding is not part of the pack; feed both the source's embedding
        x = src.emb(ids).to(torch.bfloat16)
        with torch.no_grad():
            want, got = src.attn(x), dst.attn(x)
        assert torch.equal(want, got), f"{n_rows} row(s): the loaded pack serves a different function"
    # and it is not trivially equal: RTN of the same weights gives different bytes
    rtn = _model(seed=0).cuda()
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    rtn_q = Int4Linear(rtn.attn.qkv_proj)
    assert not torch.equal(rtn_q.packed, src.attn.qkv_proj.packed), \
        "calibrated and RTN bytes coincide; this fixture cannot tell the two apart"
