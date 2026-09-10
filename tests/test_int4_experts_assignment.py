# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""A recorded gptq/rtn assignment is honoured, persisted, and round-tripped (#530).

The gptq/rtn decision is ``rows >= min_rows`` on routed-row counts, which
sit at the router-flip noise floor -- so it flips between boxes (P37: 10
of 12,288). These tests pin the repair: the decision travels with the
pack as a HASHED payload, a re-pack honours it instead of re-deriving it,
disagreement with local routing is counted not applied, and every hole
is a refusal rather than a silent fallback to the recipe.

Hermetic: no GPU, no network. The ``stubs`` GPTQ packer returns the RTN
grid of ``w * 0.5`` so gptq-vs-rtn is visible in the bytes.
"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("safetensors")
pytest.importorskip("int4_b32")

from experts4bit_qlora.engines import int4_experts as ie  # noqa: E402
from experts4bit_qlora.engines.int4_experts import (  # noqa: E402
    dump_calibrated_artifact, enable_serve_experts_int4, enable_serve_experts_int4_calibrated,
    enable_serve_experts_int4_from_artifact)
from experts4bit_qlora.engines.pack_manifest import (  # noqa: E402
    ASSIGNMENT_PATH, PackManifestError, assignment_index, load_payload_tensors, method_map_hash,
    provenance_from_model, read_assignment, verify_artifact, write_artifact)
from int4_pack_ref import dequant_int4_ref, pack_int4_b32  # noqa: E402

from test_int4_experts import E, K1, N1, _family_case, _live_model, _PlanTree, _write_ckpt  # noqa: E402
from test_int4_experts_calib import _Acc, _fake_gemm  # noqa: E402


@pytest.fixture
def stubs(monkeypatch):
    """Same stand-ins as test_int4_experts_calib.stubs (its helpers are imported, not
    copied): a GPTQ packer that returns the RTN grid of ``w * 0.5`` -- so gptq-vs-rtn
    is visible in the bytes -- and a mock grouped GEMM. Defined here rather than
    imported because pytest binds a fixture by name and ruff reads the re-use as a
    redefinition of the import."""
    import types
    gp = types.ModuleType("gptq_pack")
    gp.HessianAccumulator = _Acc
    calls = {"gptq": 0}

    def gptq_pack_int4_b32(w, hessian, **kw):
        calls["gptq"] += 1
        assert hessian.shape == (w.shape[1], w.shape[1]), (hessian.shape, w.shape)
        return pack_int4_b32(w * 0.5)
    gp.gptq_pack_int4_b32 = gptq_pack_int4_b32
    monkeypatch.setitem(sys.modules, "gptq_pack", gp)
    nf = sys.modules.get("nf4_grouped") or types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "nf4_grouped", nf)
    monkeypatch.setattr(nf, "gemm_4bit_grouped", _fake_gemm, raising=False)
    return calls


def _hess(rows_by_expert):
    Hg, Hd = torch.eye(K1) * 4.0, torch.eye(N1) * 4.0
    return {0: {e: (Hg, Hd, r) for e, r in rows_by_expert.items()}}


def _record(**methods):
    """method_map rows for layer 0; ``methods`` maps 'e<idx>' -> ('gu', 'dn') methods."""
    rows = []
    for name, (gu, dn) in methods.items():
        e = int(name[1:])
        rows.append({"layer": 0, "expert": e, "role": "gu", "method": gu})
        rows.append({"layer": 0, "expert": e, "role": "dn", "method": dn})
    return rows


def _stacks(src, names):
    from experts4bit_qlora.arch.moe_load import make_plan_reader, read_fused_expert_layer
    from experts4bit_qlora.arch.moe_plan import plan_moe_checkpoint
    from experts4bit_qlora.engines.int4_experts import safetensors_reader
    keys, read_tensor = safetensors_reader(src)
    plan = plan_moe_checkpoint(keys, _PlanTree(names), "qwen3_moe")
    read = make_plan_reader(plan, read_tensor, torch.float32)
    return read_fused_expert_layer(plan, 0, read, device="cpu", dtype=torch.float32)


def _assert_packed_as(stores, stack, role, e, method):
    w = stack[e] * (0.5 if method == "gptq" else 1.0)
    pk, sc = pack_int4_b32(w)
    n, k = stores[role]["N"], stores[role]["K"]
    got = dequant_int4_ref(stores[role]["packed"][e].cpu(), stores[role]["scales"][e].cpu(), n, k)
    assert torch.equal(got, dequant_int4_ref(pk, sc, n, k)), (role, e, method)


# ------------------------------------------------------------------ honouring --

def test_assignment_overrides_min_rows_in_both_directions(tmp_path, stubs, monkeypatch):
    """rows=3 recorded gptq -> gptq; rows=500 recorded rtn -> rtn. The rule would
    have picked the opposite for both, so both are counted as disagreements."""
    monkeypatch.delenv("E4B_INT4_KEEP_NF4", raising=False)
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, st = _live_model(wrap, "qwen3_moe")
    rec = _record(e0=("rtn", "rtn"), e1=("gptq", "gptq"))
    n = enable_serve_experts_int4(live, src, model_type="qwen3_moe", plan_model=_PlanTree(names),
                                  expert_hessians=_hess({0: 500, 1: 3}), assignment=rec)
    assert n == 1
    first, down = _stacks(src, names)
    stores = st._int4_stores
    _assert_packed_as(stores, first, "gu", 0, "rtn")
    _assert_packed_as(stores, down, "dn", 0, "rtn")
    _assert_packed_as(stores, first, "gu", 1, "gptq")
    _assert_packed_as(stores, down, "dn", 1, "gptq")
    assert stores["calibrated"] == (2, 2) and stubs["gptq"] == 2
    prov = provenance_from_model(live)
    assert prov["assignment_honoured"] == {"method_map_hash": method_map_hash(rec),
                                           "disagreements": 4}
    # the decision that was USED is what the provenance carries, not the rule's pick
    assert method_map_hash(prov["method_map"]) == method_map_hash(rec)


def test_assignment_agreeing_with_the_rule_reports_zero_disagreements(tmp_path, stubs, monkeypatch):
    monkeypatch.delenv("E4B_INT4_KEEP_NF4", raising=False)
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, _st = _live_model(wrap, "qwen3_moe")
    rec = _record(e0=("gptq", "gptq"), e1=("rtn", "rtn"))
    enable_serve_experts_int4(live, src, model_type="qwen3_moe", plan_model=_PlanTree(names),
                              expert_hessians=_hess({0: 500, 1: 3}), assignment=rec)
    assert provenance_from_model(live)["assignment_honoured"]["disagreements"] == 0


def test_gptq_named_for_an_expert_this_box_never_routed_to_refuses(tmp_path, stubs):
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, _st = _live_model(wrap, "qwen3_moe")
    rec = _record(e0=("gptq", "gptq"), e1=("gptq", "gptq"))
    with pytest.raises(RuntimeError, match="never routed to it"):
        enable_serve_experts_int4(live, src, model_type="qwen3_moe", plan_model=_PlanTree(names),
                                  expert_hessians=_hess({0: 500}), assignment=rec)   # expert 1: no Hessian


def test_an_expert_the_record_does_not_name_refuses(tmp_path, stubs):
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, _st = _live_model(wrap, "qwen3_moe")
    rec = _record(e0=("gptq", "gptq"))   # nothing for expert 1
    with pytest.raises(RuntimeError, match="not named by the assignment"):
        enable_serve_experts_int4(live, src, model_type="qwen3_moe", plan_model=_PlanTree(names),
                                  expert_hessians=_hess({0: 500, 1: 500}), assignment=rec)


def test_assignment_without_hessians_refuses(tmp_path, stubs):
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, _st = _live_model(wrap, "qwen3_moe")
    with pytest.raises(RuntimeError, match="without expert_hessians"):
        enable_serve_experts_int4(live, src, model_type="qwen3_moe", plan_model=_PlanTree(names),
                                  assignment=_record(e0=("rtn", "rtn"), e1=("rtn", "rtn")))


def test_no_assignment_is_the_recipe_unchanged(tmp_path, stubs, monkeypatch):
    """The existing calibrated path, byte for byte, with no assignment fields in provenance."""
    monkeypatch.delenv("E4B_INT4_KEEP_NF4", raising=False)
    ck, names, wrap = _family_case("qwen3_moe")
    src = _write_ckpt(tmp_path, ck)
    live, st = _live_model(wrap, "qwen3_moe")
    enable_serve_experts_int4(live, src, model_type="qwen3_moe", plan_model=_PlanTree(names),
                              expert_hessians=_hess({0: 500, 1: 3}))
    first, down = _stacks(src, names)
    _assert_packed_as(st._int4_stores, first, "gu", 0, "gptq")
    _assert_packed_as(st._int4_stores, down, "dn", 1, "rtn")
    prov = provenance_from_model(live)
    assert "assignment_honoured" not in prov
    # the recipe's own decision is recorded as lists so a dump can persist it
    assert len(prov["method_map"]) == E * 2 and prov["row_counts"] == [   # E experts x (gu, dn)
        {"layer": 0, "expert": 0, "rows": 500}, {"layer": 0, "expert": 1, "rows": 3}]


# ------------------------------------------------------------------ manifest --

def _tensors():
    return {
        (0, "gu", "packed"): torch.randint(0, 255, (E, N1, K1 // 2), dtype=torch.uint8),
        (0, "gu", "scales"): torch.rand(E, N1, K1 // 32, dtype=torch.float16),
        (0, "dn", "packed"): torch.randint(0, 255, (E, K1, N1 // 2), dtype=torch.uint8),
        (0, "dn", "scales"): torch.rand(E, K1, N1 // 32, dtype=torch.float16),
    }


def _meta():
    return {"model_id": "Qwen/Qwen3-30B-A3B", "model_revision": "ad44e777" + "0" * 32,
            "min_rows": 32, "damping": 0.01,
            "layers": [{"index": 0, "Ngu": N1, "Kgu": K1, "Ndn": K1, "Kdn": N1, "calibrated": [2, 2]}]}


def test_assignment_is_a_hashed_payload_and_the_manifest_hash_is_derived(tmp_path):
    rec = _record(e0=("gptq", "gptq"), e1=("rtn", "gptq"))
    rows = [{"layer": 0, "expert": 0, "rows": 500}, {"layer": 0, "expert": 1, "rows": 3}]
    man = write_artifact(tmp_path, tensors=_tensors(), meta=_meta(),
                         assignment={"method_map": rec, "row_counts": rows})
    assert any(p["path"] == ASSIGNMENT_PATH for p in man["payloads"])
    assert man["method_map_hash"] == method_map_hash(rec)
    got = read_assignment(tmp_path)
    assert got["method_map_hash"] == method_map_hash(rec)
    assert got["min_rows"] == 32 and got["row_counts"] == rows
    assert assignment_index(got["method_map"])[(0, 1, "gu")] == "rtn"
    verify_artifact(tmp_path, expected_fingerprint=man["pack_fingerprint"])
    # the tensor loader ignores it, like the identity payload
    assert set(load_payload_tensors(tmp_path, man)) == set(_tensors())


def test_editing_the_manifest_method_map_hash_refuses(tmp_path):
    rec = _record(e0=("gptq", "gptq"), e1=("rtn", "rtn"))
    man = write_artifact(tmp_path, tensors=_tensors(), meta=_meta(), assignment={"method_map": rec})
    mp = tmp_path / "manifest.json"
    edited = json.loads(mp.read_text())
    edited["method_map_hash"] = "sha256:" + "0" * 64
    mp.write_text(json.dumps(edited, indent=2, sort_keys=True))
    with pytest.raises(PackManifestError, match="hashed assignment payload"):
        verify_artifact(tmp_path, expected_fingerprint=man["pack_fingerprint"])


def test_a_changed_decision_is_a_different_fingerprint(tmp_path):
    a = write_artifact(tmp_path / "a", tensors=_tensors(), meta=_meta(),
                       assignment={"method_map": _record(e0=("gptq", "gptq"), e1=("rtn", "rtn"))})
    b = write_artifact(tmp_path / "b", tensors=_tensors(), meta=_meta(),
                       assignment={"method_map": _record(e0=("gptq", "gptq"), e1=("gptq", "rtn"))})
    assert a["pack_fingerprint"] != b["pack_fingerprint"]
    # and tampering with the payload itself is caught by the ordinary payload hash
    (tmp_path / "a" / ASSIGNMENT_PATH).write_bytes(b"{}")
    with pytest.raises(PackManifestError, match="does not match the manifest"):
        verify_artifact(tmp_path / "a")


def test_artifacts_without_an_assignment_still_verify(tmp_path):
    """#405 artifacts predate the payload; they must keep loading."""
    man = write_artifact(tmp_path, tensors=_tensors(), meta=_meta())
    assert not any(p["path"] == ASSIGNMENT_PATH for p in man["payloads"])
    verify_artifact(tmp_path, expected_fingerprint=man["pack_fingerprint"])
    with pytest.raises(PackManifestError, match="no assignment payload"):
        read_assignment(tmp_path)


def test_malformed_records_refuse(tmp_path):
    with pytest.raises(PackManifestError, match="twice with different methods"):
        assignment_index([{"layer": 0, "expert": 0, "role": "gu", "method": "gptq"},
                          {"layer": 0, "expert": 0, "role": "gu", "method": "rtn"}])
    with pytest.raises(PackManifestError, match="role="):
        method_map_hash([{"layer": 0, "expert": 0, "role": "up", "method": "gptq"}])
    with pytest.raises(PackManifestError, match="empty"):
        write_artifact(tmp_path / "never", tensors=_tensors(), meta=_meta(), assignment={"method_map": []})
    # the refusal came before any write: no half-built artifact directory is left behind
    assert not (tmp_path / "never").exists()


# ------------------------------------------------------------------ round trip --

def test_dump_persists_the_decision_and_a_licensed_load_carries_it_back(tmp_path, stubs, monkeypatch):
    """recipe enable -> dump -> artifact carries the payload -> from_artifact load puts the
    decision back on the model -> a second dump reproduces the same assignment bytes."""
    monkeypatch.delenv("E4B_INT4_KEEP_NF4", raising=False)
    monkeypatch.setenv("E4B_INT4_KEEP_NF4", "1")   # the loader needs the wrapper's stacks around
    ck, names, wrap = _family_case("qwen3_moe")
    (tmp_path / "ck").mkdir()
    src = _write_ckpt(tmp_path / "ck", ck)
    live, st = _live_model(wrap, "qwen3_moe")
    live.config._commit_hash = "ad44e777" + "0" * 32
    live.config._name_or_path = "Qwen/Qwen3-30B-A3B"
    monkeypatch.setattr(ie, "_meta_twin", lambda m: _PlanTree(names))
    enable_serve_experts_int4(live, src, model_type="qwen3_moe", plan_model=_PlanTree(names),
                              expert_hessians=_hess({0: 500, 1: 3}))
    recipe_mm = provenance_from_model(live)["method_map"]
    art = tmp_path / "art"
    man = dump_calibrated_artifact(live, src, str(art), model_type="qwen3_moe")
    assert any(p["path"] == ASSIGNMENT_PATH for p in man["payloads"])
    assert read_assignment(art)["method_map_hash"] == method_map_hash(recipe_mm)

    live2, _st2 = _live_model(wrap, "qwen3_moe")
    live2.config._commit_hash = live.config._commit_hash
    n = enable_serve_experts_int4_from_artifact(live2, src, str(art), expected_fingerprint=man["pack_fingerprint"],
                                                model_type="qwen3_moe")
    assert n == 1
    prov2 = provenance_from_model(live2)
    assert prov2["loaded_from_artifact"] and method_map_hash(prov2["method_map"]) == method_map_hash(recipe_mm)
    man2 = dump_calibrated_artifact(live2, src, str(tmp_path / "art2"), model_type="qwen3_moe")
    assert man2["pack_fingerprint"] == man["pack_fingerprint"], "round trip changed the licensed bytes"


def test_calibrated_entry_resolves_the_assignment_from_env(tmp_path, monkeypatch):
    """A lane hook pins the licensed split with E4B_INT4_ASSIGNMENT (file or artifact dir);
    the streamed enable hands the READ record to every chunk's pack."""
    rec = _record(e0=("gptq", "rtn"), e1=("rtn", "rtn"))
    write_artifact(tmp_path / "art", tensors=_tensors(), meta=_meta(), assignment={"method_map": rec})
    seen = {}

    def fake_calibrate(model, src, batches, **kw):
        return {0: {}}

    def fake_enable(model, src, **kw):
        seen["assignment"] = kw.get("assignment")
        return 1
    monkeypatch.setattr(ie, "calibrate_expert_hessians", fake_calibrate)
    monkeypatch.setattr(ie, "enable_serve_experts_int4", fake_enable)
    monkeypatch.setattr(ie, "_expert_layers", lambda *a, **k: (None, [(0, object())]))
    monkeypatch.setenv("E4B_INT4_ASSIGNMENT", str(tmp_path / "art"))

    class _M:
        config = type("C", (), {"hidden_size": 8, "moe_intermediate_size": 8, "num_experts": 2})()
    enable_serve_experts_int4_calibrated(_M(), str(tmp_path), batches=[torch.zeros(1, 4, dtype=torch.long)])
    assert seen["assignment"]["method_map_hash"] == method_map_hash(rec)
    # explicit argument wins over the env
    other = _record(e0=("rtn", "rtn"), e1=("rtn", "rtn"))
    enable_serve_experts_int4_calibrated(_M(), str(tmp_path), batches=[torch.zeros(1, 4, dtype=torch.long)],
                                         assignment={"method_map": other})
    assert method_map_hash(seen["assignment"]["method_map"]) == method_map_hash(other)
    # a bad path is a refusal, never a silent recipe build
    monkeypatch.setenv("E4B_INT4_ASSIGNMENT", str(tmp_path / "nope"))
    with pytest.raises(PackManifestError, match="no assignment payload"):
        enable_serve_experts_int4_calibrated(_M(), str(tmp_path), batches=[torch.zeros(1, 4, dtype=torch.long)])
