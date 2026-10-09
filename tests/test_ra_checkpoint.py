"""Synthetic repository/index controls; no actual checkpoint or GPU execution."""

import copy
import hashlib
import importlib.util
import json
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ra_checkpoint", ROOT / "bench/ra/ra_checkpoint.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
inputs = gate.load_inputs()


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


@pytest.fixture
def sample(tmp_path):
    root = tmp_path.resolve() / "checkpoint"
    root.mkdir()
    pins = inputs.read_json(ROOT / "bench/ra/source-pins.json")
    model = inputs.MODELS["proof"][1]
    files = {
        "config.json": b"{}",
        "tokenizer_config.json": b"{}",
        "tokenizer.json": b"{}",
        "README.md": b"synthetic documentation",
        "model-1.safetensors": b"synthetic one",
        "model-2.safetensors": b"synthetic two",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}
        ).encode(),
    }
    siblings, tree = [], []
    for name, data in files.items():
        (root / name).write_bytes(data)
        if name.endswith(".safetensors"):
            sha = hashlib.sha256(data).hexdigest()
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {len(data)}\n".encode()
            oid = blob(pointer)
            lf = {"oid": sha, "size": len(data), "pointerSize": len(pointer)}
            sibling = {
                "rfilename": name,
                "size": len(data),
                "blobId": oid,
                "lfs": {"sha256": sha, "size": len(data), "pointerSize": len(pointer)},
            }
            row = {"type": "file", "path": name, "size": len(data), "oid": oid, "lfs": lf}
        else:
            oid = blob(data)
            sibling = {"rfilename": name, "size": len(data), "blobId": oid}
            row = {"type": "file", "path": name, "size": len(data), "oid": oid}
        siblings.append(sibling)
        tree.append(row)
    info = {
        "id": model,
        "modelId": model,
        "sha": pins["models"][model],
        "private": False,
        "gated": False,
        "disabled": False,
        "siblings": siblings,
    }
    return root, info, tree


def authority(sample):
    _, info, tree = sample
    responses = iter((copy.deepcopy(info), copy.deepcopy(tree)))
    pins = inputs.read_json(ROOT / "bench/ra/source-pins.json")
    model = inputs.MODELS["proof"][1]
    return gate.observe(model, pins["models"][model], lambda _: next(responses))


def test_complete_materialization_and_explicit_limits(sample):
    root, info, tree = sample
    expected = inputs.inventory(root)
    bound = authority(sample)
    got = gate.bind(root, bound, expected)
    assert got["file_count"] == len(info["siblings"]) == len(tree) == 7
    assert got["shard_count"] == got["weight_names"] == 2
    assert got["files"] == expected
    assert bound["revision"] == info["sha"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "foreign/model"),
        ("modelId", "foreign/model"),
        ("sha", "0" * 40),
        ("private", True),
        ("gated", "auto"),
        ("disabled", True),
    ],
)
def test_foreign_or_restricted_revision_refuses(sample, field, value):
    sample[1][field] = value
    with pytest.raises(ValueError):
        authority(sample)


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_info",
        "duplicate_tree",
        "missing_info",
        "missing_tree",
        "extra_dir",
        "traversal",
        "absolute",
        "backslash",
        "empty_component",
        "bad_oid",
        "bad_size",
        "unknown_type",
        "lfs_sha",
        "lfs_size",
        "lfs_pointer_size",
        "pointer_oid",
        "info_lfs_sha",
        "info_lfs_size",
        "info_pointer_size",
        "surplus_lfs",
        "info_blob",
        "info_size",
        "info_bool_size",
        "empty",
        "info_limit",
    ],
)
def test_index_mutants_refuse(sample, kind):
    _, info, tree = sample
    lfs = next(row for row in tree if "lfs" in row)
    slfs = next(row for row in info["siblings"] if "lfs" in row)
    if kind == "duplicate_info":
        info["siblings"].append(copy.deepcopy(info["siblings"][0]))
    elif kind == "duplicate_tree":
        tree.append(copy.deepcopy(tree[0]))
    elif kind == "missing_info":
        info["siblings"].pop()
    elif kind == "missing_tree":
        tree.pop()
    elif kind == "extra_dir":
        tree.append({"path": "extra", "type": "directory", "oid": "a" * 40})
    elif kind in ("traversal", "absolute", "backslash", "empty_component"):
        tree[0]["path"] = {"traversal": "../x", "absolute": "/x", "backslash": "x\\y", "empty_component": "a//b"}[kind]
    elif kind == "bad_oid":
        tree[0]["oid"] = "0" * 39
    elif kind == "bad_size":
        tree[0]["size"] = True
    elif kind == "unknown_type":
        tree[0]["type"] = "symlink"
    elif kind == "lfs_sha":
        lfs["lfs"]["oid"] = "g" * 64
    elif kind == "lfs_size":
        lfs["lfs"]["size"] += 1
    elif kind == "lfs_pointer_size":
        lfs["lfs"]["pointerSize"] += 1
    elif kind == "pointer_oid":
        lfs["oid"] = "a" * 40
    elif kind == "info_lfs_sha":
        slfs["lfs"]["sha256"] = "b" * 64
    elif kind == "info_lfs_size":
        slfs["lfs"]["size"] += 1
    elif kind == "info_pointer_size":
        slfs["lfs"]["pointerSize"] += 1
    elif kind == "surplus_lfs":
        info["siblings"][0]["lfs"] = slfs["lfs"]
    elif kind == "info_blob":
        info["siblings"][0]["blobId"] = "a" * 40
    elif kind == "info_size":
        info["siblings"][0]["size"] += 1
    elif kind == "info_bool_size":
        info["siblings"][0]["size"] = True
    elif kind == "empty":
        tree.clear()
    elif kind == "info_limit":
        info["siblings"] *= 150
    with pytest.raises(ValueError):
        authority(sample)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "symlink",
        "git_mutant",
        "lfs_mutant",
        "pointer_file",
        "resealed_git",
        "resealed_lfs",
        "missing_index_shard",
        "surplus_index_shard",
        "wrong_config",
    ],
)
def test_materialized_mutants_refuse(sample, kind):
    root, _, _ = sample
    bound = authority(sample)
    expected = inputs.inventory(root)
    if kind == "missing":
        (root / "README.md").unlink()
    elif kind == "extra":
        (root / "cache.bin").write_bytes(b"extra")
    elif kind == "symlink":
        (root / "README.md").unlink()
        (root / "README.md").symlink_to(root / "config.json")
    elif kind in ("git_mutant", "resealed_git"):
        (root / "README.md").write_bytes(b"resealed bytes")
    elif kind in ("lfs_mutant", "resealed_lfs"):
        (root / "model-1.safetensors").write_bytes(b"mutated one!!")
    elif kind == "pointer_file":
        (root / "model-1.safetensors").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    elif kind in ("missing_index_shard", "surplus_index_shard"):
        index = {"weight_map": {"a": "model-1.safetensors"}}
        if kind == "surplus_index_shard":
            index["weight_map"]["b"] = "foreign.safetensors"
        data = json.dumps(index).encode()
        (root / "model.safetensors.index.json").write_bytes(data)
        bound["files"]["model.safetensors.index.json"].update(size=len(data), git_blob_oid=blob(data))
    elif kind == "wrong_config":
        (root / "config.json").write_bytes(b"[]")
        bound["files"]["config.json"]["git_blob_oid"] = blob(b"[]")
    if kind.startswith("resealed") or kind.endswith("index_shard") or kind == "wrong_config":
        expected = inputs.inventory(root)
    with pytest.raises(ValueError):
        gate.bind(root, bound, expected)


def test_directory_inventory_and_info_tree_match(sample):
    root, info, tree = sample
    old = root / "README.md"
    new = root / "docs" / "README.md"
    new.parent.mkdir()
    old.rename(new)
    info["siblings"][3]["rfilename"] = "docs/README.md"
    tree[3]["path"] = "docs/README.md"
    tree.append({"type": "directory", "path": "docs", "oid": "a" * 40})
    assert gate.bind(root, authority(sample), inputs.inventory(root))["file_count"] == 7


def setup_verify(sample, monkeypatch):
    root, info, _ = sample
    lock = {"model": info["id"], "revision": info["sha"], "trees": {"checkpoint": inputs.inventory(root)}}
    spec = {"trees": {"checkpoint": str(root)}}
    manifest = {"input_spec": str(root / "config.json"), "input_lock": str(root / "tokenizer_config.json")}
    monkeypatch.setattr(gate, "check", lambda _: (spec, lock, {"common_identity": "synthetic"}))
    monkeypatch.setattr(gate.sys, "flags", SimpleNamespace(isolated=1, no_site=1))
    monkeypatch.setattr(gate.sys, "dont_write_bytecode", True)
    rows = iter(
        [copy.deepcopy(sample[1]), copy.deepcopy(sample[2]), copy.deepcopy(sample[1]), copy.deepcopy(sample[2])]
    )
    return manifest, rows


def test_verify_brackets_inventory_and_keeps_limits_false(sample, monkeypatch):
    manifest, rows = setup_verify(sample, monkeypatch)
    got = gate.verify(manifest, lambda _: next(rows))
    assert got["proves_hub_index_checkpoint_equality"] is True
    for key in (
        "proves_dataset_authority",
        "proves_calibration_authority",
        "proves_tokenizer_execution",
        "proves_runtime_consumption",
        "proves_publisher_signature",
        "proves_gpu_engagement",
        "launch_authority",
    ):
        assert got[key] is False
    with pytest.raises(StopIteration):
        next(rows)


def test_authority_changed_after_binding_refuses(sample, monkeypatch):
    manifest, rows = setup_verify(sample, monkeypatch)
    records = list(rows)
    records[2]["sha"] = "f" * 40
    rows = iter(records)
    with pytest.raises(ValueError):
        gate.verify(manifest, lambda _: next(rows))


def test_input_mutation_between_authority_reads_refuses(sample, monkeypatch):
    manifest, rows = setup_verify(sample, monkeypatch)
    count = 0

    def fetch(_):
        nonlocal count
        count += 1
        if count == 3:
            (sample[0] / "config.json").write_bytes(b'{"changed":true}')
        return next(rows)

    with pytest.raises(ValueError):
        gate.verify(manifest, fetch)


@pytest.mark.parametrize("flag", ["isolated", "no_site", "dont_write_bytecode"])
def test_wrong_interpreter_flags_refuse(sample, monkeypatch, flag):
    manifest, rows = setup_verify(sample, monkeypatch)
    if flag == "dont_write_bytecode":
        monkeypatch.setattr(gate.sys, flag, False)
    else:
        monkeypatch.setattr(gate.sys, "flags", SimpleNamespace(isolated=flag != "isolated", no_site=flag != "no_site"))
    with pytest.raises(ValueError):
        gate.verify(manifest, lambda _: next(rows))


@pytest.mark.parametrize(
    "kind",
    [
        "positive",
        "redirect",
        "pagination",
        "encoding",
        "content_type",
        "status",
        "duplicate",
        "nonfinite",
        "too_large",
        "empty",
        "foreign_url",
    ],
)
def test_transport_refusals_retain_bounded_response(tmp_path, monkeypatch, kind):
    url = gate.urls("registered/model", "a" * 40)[0]
    headers = Message()
    headers["Content-Type"] = "application/json"
    if kind == "pagination":
        headers["Link"] = '<https://example.invalid/page>; rel="next"'
    if kind == "encoding":
        headers["Content-Encoding"] = "gzip"
    if kind == "content_type":
        headers.replace_header("Content-Type", "text/plain")
    data = {
        "duplicate": b'{"x":1,"x":2}',
        "nonfinite": b'{"x":NaN}',
        "too_large": b"x" * (gate.LIMIT + 1),
        "empty": b"",
    }.get(kind, b'{"x":1}')

    class Response:
        status = 403 if kind == "status" else 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def geturl(self):
            return url + "&changed=true" if kind == "redirect" else url

        def read(self, size):
            assert size == gate.LIMIT + 1
            return data[:size]

    Response.headers = headers

    def opener(*handlers):
        assert isinstance(handlers[0], gate.urllib.request.ProxyHandler) and handlers[0].proxies == {}
        assert isinstance(handlers[1], gate.NoRedirect)
        return SimpleNamespace(open=lambda *a, **kw: Response())

    monkeypatch.setattr(gate.urllib.request, "build_opener", opener)
    collector = gate.Collector(tmp_path, [url])
    if kind == "positive":
        assert collector(url) == {"x": 1}
        assert collector.records[0]["sha256"] == hashlib.sha256(data).hexdigest()
    else:
        with pytest.raises(ValueError):
            collector(url + "&foreign=true" if kind == "foreign_url" else url)
        if kind in ("duplicate", "nonfinite", "too_large", "empty"):
            assert len(collector.records) == 1 and (tmp_path / "response-00.json").read_bytes() == data


def test_redirect_handler_never_follows():
    with pytest.raises(ValueError):
        gate.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid")


def input_fixture_module():
    spec = importlib.util.spec_from_file_location("checkpoint_input_fixture", ROOT / "tests/test_ra_inputs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fixture_module = input_fixture_module()
stage = fixture_module.stage


def test_composed_reviewed_input_lock_and_checkpoint_binding(sample, tmp_path, stage, monkeypatch):
    own = fixture_module.fixture(tmp_path.resolve() / "composed")
    root = Path(own["trees"]["checkpoint"])
    (root / "bytes.bin").unlink()  # this test's tiny synthetic fixture only
    for p in sample[0].iterdir():
        (root / p.name).write_bytes(p.read_bytes())
    spec_path = tmp_path.resolve() / "input-spec.json"
    lock_path = tmp_path.resolve() / "input-lock.json"
    fixture_module.write(spec_path, own)
    fixture_module.write(lock_path, fixture_module.inputs.inspect(own, stage))
    manifest = {
        "schema": 1,
        "input_spec": str(spec_path),
        "stage": str(stage),
        "input_lock": str(lock_path),
        "input_lock_sha256": fixture_module.inputs.file_record(lock_path)["sha256"],
    }
    monkeypatch.setattr(gate, "load_inputs", lambda: fixture_module.inputs)
    monkeypatch.setattr(gate.sys, "flags", SimpleNamespace(isolated=1, no_site=1))
    monkeypatch.setattr(gate.sys, "dont_write_bytecode", True)
    records = iter(
        [copy.deepcopy(sample[1]), copy.deepcopy(sample[2]), copy.deepcopy(sample[1]), copy.deepcopy(sample[2])]
    )
    result = gate.verify(manifest, lambda _: next(records))
    assert result["checkpoint"]["file_count"] == 7
    assert result["common_inputs"]["status"] == "INPUT_BYTES_AND_PROJECTIONS_MATCH"
    changed = copy.deepcopy(manifest)
    changed["input_lock_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        gate.check(changed)


@pytest.mark.parametrize("change", ["schema", "extra", "bad_digest"])
def test_manifest_schema_refuses_without_network(sample, change):
    manifest = {
        "schema": 1,
        "input_spec": "/unused",
        "stage": "/unused",
        "input_lock": "/unused",
        "input_lock_sha256": "a" * 64,
    }
    if change == "schema":
        manifest["schema"] = True
    elif change == "extra":
        manifest["model"] = sample[1]["id"]
    else:
        manifest["input_lock_sha256"] = "a" * 63
    with pytest.raises(ValueError):
        gate.check(manifest)
