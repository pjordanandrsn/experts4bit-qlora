# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Source selection must reach the real int4 reader without a Hub download."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("huggingface_hub")

from safetensors.torch import save_file  # noqa: E402
from experts4bit_qlora import serve_paged  # noqa: E402
from experts4bit_qlora.engines import int4_experts  # noqa: E402


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "local checkpoint"
    path.mkdir()
    save_file({"expert.weight": torch.arange(8).reshape(2, 4)}, str(path / "model.safetensors"))
    return path


@pytest.fixture
def model():
    return SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"))


@pytest.mark.parametrize("calibrated", (False, True))
def test_local_source_reaches_real_reader_without_hub(source, model, monkeypatch, calibrated):
    import huggingface_hub

    monkeypatch.setenv("E4B_SERVE_EXP_INT4", "1")
    monkeypatch.setenv("E4B_SERVE_EXP_INT4_CALIB", "1" if calibrated else "0")
    for knob in ("E4B_SERVE_ATTN_INT4", "E4B_SERVE_ATTN_INT4_CALIB", "E4B_SERVE_LMHEAD_INT4_CALIB",
                 "E4B_SERVE_DENSE_INT4_CALIB"):
        monkeypatch.delenv(knob, raising=False)
    monkeypatch.setenv("E4B_INT4_ARTIFACT_DIR", "licensed-artifact")
    monkeypatch.setenv("E4B_INT4_EXPECTED_FINGERPRINT", "registered-fingerprint")
    monkeypatch.setenv("E4B_CALIB_LAYERS_PER_PASS", "3")

    def no_hub(*args, **kwargs):
        pytest.fail("a local checkpoint must not invoke snapshot_download")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_hub)
    batches = [object()]
    monkeypatch.setattr(serve_paged, "_calib_batches", lambda tok: batches)
    calls = []

    def enable(actual_model, directory, *args, **kwargs):
        assert actual_model is model
        keys, reader = int4_experts.safetensors_reader(directory)
        assert keys == ["expert.weight"]
        assert torch.equal(reader(keys[0]), torch.arange(8).reshape(2, 4))
        calls.append((directory, args, kwargs))
        return 2

    name = "enable_serve_experts_int4_calibrated" if calibrated else "enable_serve_experts_int4"
    monkeypatch.setattr(int4_experts, name, enable)
    result = serve_paged._apply_levers(model, serve_paged.PagedServeConfig(model=str(source)), object())
    assert result["exp_int4_layers_enabled"] == 2
    assert calls[0][0] == str(source)
    if calibrated:
        assert calls[0][1] == (batches,)
        assert calls[0][2]["artifact_dir"] == "licensed-artifact"
        assert calls[0][2]["expected_fingerprint"] == "registered-fingerprint"
        assert calls[0][2]["layers_per_pass"] == 3
    else:
        assert calls[0][1:] == ((), {})


def test_hub_source_preserves_revision_and_allow_patterns(source, model, monkeypatch):
    import huggingface_hub

    monkeypatch.setenv("E4B_SERVE_EXP_INT4", "1")
    monkeypatch.setenv("E4B_SERVE_EXP_INT4_CALIB", "0")
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(source)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(int4_experts, "enable_serve_experts_int4", lambda m, src: 2 if src == str(source) else 0)
    serve_paged._apply_levers(model, serve_paged.PagedServeConfig(model="org/model", revision="frozen-revision"), None)
    assert calls == [("org/model", {"allow_patterns": ["*.json", "*.safetensors"], "revision": "frozen-revision"})]


def test_missing_local_source_keeps_hub_validation_refusal(tmp_path, model, monkeypatch):
    from huggingface_hub.errors import HFValidationError

    monkeypatch.setenv("E4B_SERVE_EXP_INT4", "1")
    with pytest.raises(HFValidationError):
        serve_paged._apply_levers(model, serve_paged.PagedServeConfig(model=str(tmp_path / "missing")), None)


@pytest.mark.parametrize("calibrated", (False, True))
def test_zero_patched_layers_still_refuses(source, model, monkeypatch, calibrated):
    monkeypatch.setenv("E4B_SERVE_EXP_INT4", "1")
    monkeypatch.setenv("E4B_SERVE_EXP_INT4_CALIB", "1" if calibrated else "0")
    monkeypatch.setattr(serve_paged, "_calib_batches", lambda tok: [])
    name = "enable_serve_experts_int4_calibrated" if calibrated else "enable_serve_experts_int4"
    monkeypatch.setattr(int4_experts, name, lambda *a, **k: 0)
    with pytest.raises(RuntimeError, match="patched 0 layers"):
        serve_paged._apply_levers(model, serve_paged.PagedServeConfig(model=str(source)), None)
