"""A pinned, complete local cache loads offline (no Hub call); a partial one is refused by name.

Offline, some huggingface_hub releases resolve snapshot_download(..., revision=<sha>) from the trees/<sha>.json
listing a download writes; a cache with the complete snapshot and no listing (an older release's, or one assembled by
hand) makes them list the repo tree online: observed on 1.26.0, 1.27.0, 1.28.0, 1.29.0, 1.30.0, 1.31.0, 1.32.0,
1.33.0, 2.0.0, 2.1.0 and 2.1.1; 1.25.0 and 2.2.0 resolve such a cache locally. The fixtures here write no listing,
the layout that fails (test_loader_architectures covers the listed layout through the real hub).
These tests patch snapshot_download to RAISE, so they hold under any hub version: offline with a pinned sha the loader
must not call it at all (the previous inline call always did, so these tests fail on it)."""
import json

import pytest

loader = pytest.importorskip("experts4bit_qlora.loader")

SHA = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
MODEL = "Qwen/Qwen3-30B-A3B"


def _cache(tmp_path, files=None, index_shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
           with_index=True):
    repo = tmp_path / ("models--" + MODEL.replace("/", "--"))
    snap = repo / "snapshots" / SHA
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(SHA)
    if with_index:
        (snap / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(index_shards)}}))
    for f in (files if files is not None else ["config.json", "tokenizer.json", *index_shards]):
        (snap / f).write_text("x")
    return snap


@pytest.fixture
def offline(monkeypatch, tmp_path):
    from huggingface_hub import constants
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    calls = []

    def no_hub(*a, **kw):
        calls.append((a, kw))
        raise AssertionError("snapshot_download called: an offline pinned snapshot must resolve with no Hub call")

    monkeypatch.setattr(loader, "snapshot_download", no_hub)
    return calls


def test_complete_pinned_snapshot_resolves_offline_without_the_hub(offline, tmp_path):
    snap = _cache(tmp_path)
    assert loader._resolve_snapshot(MODEL, SHA) == str(snap)
    assert offline == []


def test_single_file_checkpoint_resolves_offline(offline, tmp_path):
    snap = _cache(tmp_path, files=["config.json", "model.safetensors"], with_index=False)
    assert loader._resolve_snapshot(MODEL, SHA) == str(snap)


def test_a_missing_shard_is_refused_by_name(offline, tmp_path):
    _cache(tmp_path, files=["config.json", "tokenizer.json", "model-00001-of-00002.safetensors"])
    with pytest.raises(FileNotFoundError, match="model-00002-of-00002.safetensors"):
        loader._resolve_snapshot(MODEL, SHA)


def test_a_missing_config_is_refused(offline, tmp_path):
    _cache(tmp_path, files=["tokenizer.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    with pytest.raises(FileNotFoundError, match="config.json"):
        loader._resolve_snapshot(MODEL, SHA)


def test_no_tokenizer_is_required(offline, tmp_path):
    """The streaming loader reads config and weights only; a weights-only snapshot is complete."""
    snap = _cache(tmp_path, files=["config.json", "model-00001-of-00002.safetensors",
                                   "model-00002-of-00002.safetensors"])
    assert loader._resolve_snapshot(MODEL, SHA) == str(snap)
    assert offline == []


def test_a_pinned_sha_not_in_the_cache_is_refused_offline(offline, tmp_path):
    with pytest.raises(FileNotFoundError, match="not in the cache"):
        loader._resolve_snapshot(MODEL, SHA)


def test_a_branch_revision_offline_keeps_snapshot_downloads_behaviour(monkeypatch, tmp_path):
    """Documented: a branch name is never guessed into a ref here; snapshot_download resolves refs/<branch>."""
    from huggingface_hub import constants
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    seen = []
    monkeypatch.setattr(loader, "snapshot_download", lambda *a, **kw: seen.append(kw) or "from-hub-resolver")
    _cache(tmp_path)
    assert loader._resolve_snapshot(MODEL, "main") == "from-hub-resolver"
    assert seen == [{"allow_patterns": loader._SNAPSHOT_PATTERNS, "revision": "main"}]


def test_online_behaviour_is_unchanged(monkeypatch, tmp_path):
    from huggingface_hub import constants
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    seen = []
    monkeypatch.setattr(loader, "snapshot_download", lambda *a, **kw: seen.append((a, kw)) or "online")
    _cache(tmp_path)
    assert loader._resolve_snapshot(MODEL, SHA) == "online"
    assert seen == [((MODEL,), {"allow_patterns": loader._SNAPSHOT_PATTERNS, "revision": SHA})]


def test_a_local_directory_is_used_as_given(offline, tmp_path):
    d = tmp_path / "local-model"
    d.mkdir()
    assert loader._resolve_snapshot(str(d), SHA) == str(d)
    assert offline == []
