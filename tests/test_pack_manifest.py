"""Hash-pinned pack artifacts: fingerprint stability, round-trip, refusals (#405)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experts4bit_qlora.engines.pack_manifest import (
    IDENTITY_PATH, LAYOUT, PackManifestError, check_manifest_dims, compute_pack_fingerprint,
    fingerprint_mismatch_reason, int4_store_dims, load_payload_tensors, method_map_hash,
    require_artifact_for_licensed_load, require_model_revision, row_count_vector_hash,
    tensor_from_payload_bytes, tensor_payload_bytes, verify_artifact, write_artifact,
)
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated


def _dummy_tensors():
    return {
        (0, "gu", "packed"): torch.arange(24, dtype=torch.int32).reshape(2, 3, 4),
        (0, "gu", "scales"): torch.linspace(0.1, 0.8, 6, dtype=torch.float32).reshape(2, 3),
        (0, "dn", "packed"): torch.arange(100, 124, dtype=torch.int32).reshape(2, 3, 4),
        (0, "dn", "scales"): torch.linspace(1.1, 1.8, 6, dtype=torch.float32).reshape(2, 3),
    }


def _meta():
    return {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "model_revision": "ad44e777" + "0" * 32,
        "min_rows": 32,
        "damping": 0.01,
        "layers": [{"index": 0, "Ngu": 8, "Kgu": 8, "Ndn": 8, "Kdn": 8,
                    "calibrated": [2, 2]}],
    }


def test_fingerprint_is_stable_across_rewrites(tmp_path: Path):
    t = _dummy_tensors()
    a = write_artifact(tmp_path / "a", tensors=t, meta=_meta())
    b = write_artifact(tmp_path / "b", tensors=t, meta=_meta())
    assert a["pack_fingerprint"] == b["pack_fingerprint"]
    assert a["pack_fingerprint"].startswith("sha256:")
    assert len(a["pack_fingerprint"]) == 71
    man = verify_artifact(tmp_path / "a", expected_fingerprint=a["pack_fingerprint"])
    assert man["pack_fingerprint"] == a["pack_fingerprint"]


def test_byte_round_trip(tmp_path: Path):
    t = _dummy_tensors()
    man = write_artifact(tmp_path, tensors=t, meta=_meta())
    loaded = load_payload_tensors(tmp_path, man)
    for key, src in t.items():
        got = loaded[key]
        assert got.dtype == src.dtype
        assert got.shape == src.shape
        assert torch.equal(got, src)
    raw = tensor_payload_bytes(t[(0, "gu", "packed")])
    assert torch.equal(tensor_from_payload_bytes(raw), t[(0, "gu", "packed")])


def test_corruption_and_missing_file_refuse(tmp_path: Path):
    man = write_artifact(tmp_path, tensors=_dummy_tensors(), meta=_meta())
    payload = tmp_path / man["payloads"][0]["path"]
    payload.write_bytes(payload.read_bytes() + b"\x00")
    with pytest.raises(PackManifestError, match="does not match the manifest"):
        verify_artifact(tmp_path, expected_fingerprint=man["pack_fingerprint"])
    payload.unlink()
    with pytest.raises(PackManifestError, match="missing payload"):
        verify_artifact(tmp_path)


def test_wrong_model_revision_and_layout_refuse(tmp_path: Path):
    man = write_artifact(tmp_path, tensors=_dummy_tensors(), meta=_meta())
    with pytest.raises(PackManifestError, match="model_revision"):
        verify_artifact(tmp_path, expected_model_revision="deadbeef" * 5)
    with pytest.raises(PackManifestError, match="layout"):
        verify_artifact(tmp_path, expected_layout="not-this-layout")
    with pytest.raises(PackManifestError, match="!= expected"):
        verify_artifact(tmp_path, expected_fingerprint="sha256:" + "0" * 64)
    assert man["layout"] == LAYOUT


def test_licensed_path_never_falls_back_to_the_recipe():
    fp = "sha256:" + "d" * 64
    with pytest.raises(PackManifestError, match="refusing to rebuild"):
        require_artifact_for_licensed_load(None, fp)
    with pytest.raises(PackManifestError, match="refusing to rebuild"):
        require_artifact_for_licensed_load("", fp)
    require_artifact_for_licensed_load("/tmp/artifact", fp)
    require_artifact_for_licensed_load(None, None)


def test_calibrated_enable_with_fingerprint_does_not_enter_the_recipe(tmp_path: Path):
    fp = "sha256:" + "e" * 64
    with pytest.raises(PackManifestError, match="refusing to rebuild"):
        enable_serve_experts_int4_calibrated(
            object(), str(tmp_path), batches=[], expected_fingerprint=fp)


def test_method_and_row_hashes_are_order_insensitive():
    a = [{"layer": 1, "expert": 2, "role": "dn", "method": "rtn"},
         {"layer": 1, "expert": 1, "role": "gu", "method": "gptq"}]
    b = list(reversed(a))
    assert method_map_hash(a) == method_map_hash(b)
    rows_a = [{"layer": 0, "expert": 3, "rows": 9}, {"layer": 0, "expert": 1, "rows": 40}]
    assert row_count_vector_hash(rows_a) == row_count_vector_hash(list(reversed(rows_a)))


def test_reducer_fingerprint_mismatch_reason():
    fp = "sha256:" + "f" * 64
    assert fingerprint_mismatch_reason(None, None) is None
    assert fingerprint_mismatch_reason(fp, None) is None
    assert "no pack_fingerprint" in fingerprint_mismatch_reason(None, fp)
    assert "!=" in fingerprint_mismatch_reason("sha256:" + "0" * 64, fp)
    assert fingerprint_mismatch_reason(fp, fp) is None


def test_payload_order_does_not_change_the_root_hash():
    p1 = [{"path": "payloads/b.bin", "size": 2, "sha256": "aa"},
          {"path": "payloads/a.bin", "size": 1, "sha256": "bb"}]
    p2 = list(reversed(p1))
    assert compute_pack_fingerprint(p1) == compute_pack_fingerprint(p2)


# --- review fixes on PR #439 (identity in the hash; N/K from bytes; revision at dump time) ---



def _b32_tensors(E=2, N=8, K=64):
    """Real int4_b32 shapes: packed [E, N, K//2] uint8, scales [E, N, K//32] fp16."""
    return {
        (0, "gu", "packed"): torch.randint(0, 255, (E, N, K // 2), dtype=torch.uint8),
        (0, "gu", "scales"): torch.rand(E, N, K // 32, dtype=torch.float16),
        (0, "dn", "packed"): torch.randint(0, 255, (E, N, K // 2), dtype=torch.uint8),
        (0, "dn", "scales"): torch.rand(E, N, K // 32, dtype=torch.float16),
    }


def test_identity_is_a_hashed_payload_and_the_manifest_copy_is_cross_checked(tmp_path: Path):
    man = write_artifact(tmp_path, tensors=_dummy_tensors(), meta=_meta())
    assert any(p["path"] == IDENTITY_PATH for p in man["payloads"])
    ident = json.loads((tmp_path / IDENTITY_PATH).read_text())
    assert ident["model_revision"] == _meta()["model_revision"]
    assert ident["layers"] == _meta()["layers"]
    assert ident["layout"] == LAYOUT
    verify_artifact(tmp_path, expected_fingerprint=man["pack_fingerprint"])
    # editing the manifest's revision leaves the payload hashes intact -- and is now refused
    mp = tmp_path / "manifest.json"
    edited = json.loads(mp.read_text())
    edited["model_revision"] = "deadbeef" * 5
    mp.write_text(json.dumps(edited, indent=2, sort_keys=True))
    with pytest.raises(PackManifestError, match="differs from the hashed identity payload"):
        verify_artifact(tmp_path, expected_fingerprint=man["pack_fingerprint"])
    # so is editing layers[] (the N/K cross-check source)
    edited["model_revision"] = _meta()["model_revision"]
    edited["layers"][0]["Ngu"] = 4096
    mp.write_text(json.dumps(edited, indent=2, sort_keys=True))
    with pytest.raises(PackManifestError, match="'layers'"):
        verify_artifact(tmp_path, expected_fingerprint=man["pack_fingerprint"])


def test_identity_changes_the_fingerprint(tmp_path: Path):
    a = write_artifact(tmp_path / "a", tensors=_dummy_tensors(), meta=_meta())
    other = dict(_meta(), model_revision="cafe" * 10)
    b = write_artifact(tmp_path / "b", tensors=_dummy_tensors(), meta=other)
    assert a["pack_fingerprint"] != b["pack_fingerprint"], "same bytes, different checkpoint: different licence"


def test_missing_identity_payload_refuses(tmp_path: Path):
    man = write_artifact(tmp_path, tensors=_dummy_tensors(), meta=_meta())
    (tmp_path / IDENTITY_PATH).unlink()
    with pytest.raises(PackManifestError, match="missing payload"):
        verify_artifact(tmp_path)
    # an older-style manifest that never listed an identity payload is refused too
    man = json.loads((tmp_path / "manifest.json").read_text())
    man["payloads"] = [p for p in man["payloads"] if p["path"] != IDENTITY_PATH]
    man["pack_fingerprint"] = compute_pack_fingerprint(man["payloads"])
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(PackManifestError, match="no hashed identity payload"):
        verify_artifact(tmp_path)


def test_tensor_loader_skips_the_identity_payload(tmp_path: Path):
    man = write_artifact(tmp_path, tensors=_dummy_tensors(), meta=_meta())
    loaded = load_payload_tensors(tmp_path, man)
    assert set(loaded) == set(_dummy_tensors())


def test_n_k_come_from_the_bytes_and_layers_only_cross_checks():
    t = _b32_tensors(E=2, N=8, K=64)
    assert int4_store_dims(t[(0, "gu", "packed")], t[(0, "gu", "scales")]) == (8, 64)
    meta = {"Ngu": 8, "Kgu": 64, "Ndn": 8, "Kdn": 64}
    check_manifest_dims(0, "gu", meta, 8, 64)
    with pytest.raises(PackManifestError, match="manifest layers\\[\\] says N=4096"):
        check_manifest_dims(0, "gu", {"Ngu": 4096, "Kgu": 64}, 8, 64)
    with pytest.raises(PackManifestError, match="packed and scales disagree"):
        int4_store_dims(t[(0, "gu", "packed")], torch.rand(2, 8, 3, dtype=torch.float16))
    with pytest.raises(PackManifestError, match="lacks N/K"):
        check_manifest_dims(0, "dn", {}, 8, 64)


def test_unknown_model_revision_is_refused_at_dump_time_unless_marked():
    assert require_model_revision("ad44e777" + "0" * 32) == ("ad44e777" + "0" * 32, False)
    with pytest.raises(PackManifestError, match="never become a licensed load"):
        require_model_revision(None)
    with pytest.raises(PackManifestError):
        require_model_revision("")
    assert require_model_revision(None, allow_unknown=True) == (None, True)
