"""Hash-pinned pack artifacts: fingerprint stability, round-trip, refusals (#405)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experts4bit_qlora.engines.pack_manifest import (
    LAYOUT, PackManifestError, compute_pack_fingerprint, fingerprint_mismatch_reason,
    load_payload_tensors, method_map_hash, require_artifact_for_licensed_load,
    row_count_vector_hash, tensor_from_payload_bytes, tensor_payload_bytes,
    verify_artifact, write_artifact,
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
