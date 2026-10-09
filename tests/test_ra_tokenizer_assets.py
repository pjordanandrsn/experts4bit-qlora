"""Synthetic indexed/config/receipt controls; no native tokenizer or model execution."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load("ra_tokenizer_assets", ROOT / "bench/ra/ra_tokenizer_assets.py")
inputs, cp = gate.peers()
fixture_module = load("tokenizer_assets_inputs_fixture", ROOT / "tests/test_ra_inputs.py")
stage = fixture_module.stage
inputs = fixture_module.inputs


def encoded(value):
    return json.dumps(value, separators=(",", ":")).encode()


def payloads(model):
    family, cls, names = gate.POLICIES[model]
    token = {
        "content": "<synthetic>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    }
    data = {name: b"synthetic raw bytes" for name in names}
    data.update(
        {
            "config.json": encoded({"model_type": family, "eos_token_id": 7, "pad_token_id": 7}),
            "tokenizer_config.json": encoded(
                {
                    "tokenizer_class": cls,
                    "eos_token": "<synthetic>",
                    "pad_token": "<synthetic>",
                    "added_tokens_decoder": {"7": token},
                    "add_bos_token": False,
                }
            ),
            "tokenizer.json": encoded({"model": {"type": "BPE"}, "added_tokens": [{"id": 7, **token}]}),
        }
    )
    return data


@pytest.fixture(params=list(gate.POLICIES))
def sample(tmp_path, request):
    model = request.param
    root = tmp_path.resolve() / "checkpoint"
    root.mkdir()
    data = payloads(model)
    data["README.md"] = b"synthetic docs"
    siblings, tree = [], []
    for name, content in data.items():
        (root / name).write_bytes(content)
        row = {"path": name, "type": "file", "oid": cp.git_blob(content), "size": len(content)}
        sibling = {"rfilename": name, "blobId": row["oid"], "size": row["size"]}
        if name == "tokenizer.json":
            digest = hashlib.sha256(content).hexdigest()
            pointer = (
                f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(content)}\n".encode()
            )
            row.update(
                oid=cp.git_blob(pointer), lfs={"oid": digest, "size": len(content), "pointerSize": len(pointer)}
            )
            sibling.update(
                blobId=row["oid"], lfs={"sha256": digest, "size": len(content), "pointerSize": len(pointer)}
            )
        tree.append(row)
        siblings.append(sibling)
    pins = inputs.read_json(ROOT / "bench/ra/source-pins.json")
    info = {
        "id": model,
        "modelId": model,
        "sha": pins["models"][model],
        "private": False,
        "gated": False,
        "disabled": False,
        "siblings": siblings,
    }
    return root, data, info, tree


def observed(sample):
    _, _, info, tree = sample
    rows = iter((copy.deepcopy(info), copy.deepcopy(tree)))
    return cp.observe(info["id"], info["sha"], lambda _: next(rows))


def test_complete_asset_bytes_and_declared_configuration(sample):
    root, data, info, _ = sample
    result = gate.bind(root, observed(sample), inputs.inventory(root))
    assert set(result["files"]) == gate.POLICIES[info["id"]][2]
    assert result["configuration"]["special_tokens"]["eos"]["declared_added_token_id"] == 7
    assert result["configuration"]["runtime_class_observed"] is False
    assert result["configuration"]["native_asset_read_observed"] is False
    assert result["file_count"] == len(data) - 1


@pytest.mark.parametrize(
    "kind",
    [
        "family",
        "class",
        "auto_map_config",
        "auto_map_tokenizer",
        "asset_override",
        "missing_eos",
        "missing_pad",
        "bool_id",
        "float_id",
        "wrong_id",
        "duplicate_content",
        "bool_mode",
        "backend_type",
        "missing_added",
        "extra_added",
        "bool_added_id",
        "float_added_id",
        "wrong_body",
        "extra_body",
        "negative_decoder",
        "leading_zero_decoder",
        "extra_decoder_body",
        "nonboolean_body",
    ],
)
def test_configuration_mutants_refuse(sample, kind):
    _, data, info, _ = sample
    selected = {n: data[n] for n in gate.POLICIES[info["id"]][2]}
    config = json.loads(selected["config.json"])
    tc = json.loads(selected["tokenizer_config.json"])
    backend = json.loads(selected["tokenizer.json"])
    if kind == "family":
        config["model_type"] = "foreign"
    elif kind == "class":
        tc["tokenizer_class"] = "foreign"
    elif kind == "auto_map_config":
        config["auto_map"] = {}
    elif kind == "auto_map_tokenizer":
        tc["auto_map"] = {}
    elif kind == "asset_override":
        tc["tokenizer_file"] = "/foreign"
    elif kind == "missing_eos":
        tc.pop("eos_token")
    elif kind == "missing_pad":
        tc.pop("pad_token")
    elif kind in ("bool_id", "float_id", "wrong_id"):
        config["eos_token_id"] = {"bool_id": True, "float_id": 7.0, "wrong_id": 8}[kind]
    elif kind == "duplicate_content":
        tc["added_tokens_decoder"]["8"] = copy.deepcopy(tc["added_tokens_decoder"]["7"])
    elif kind == "bool_mode":
        tc["add_bos_token"] = 0
    elif kind == "backend_type":
        backend["model"]["type"] = "Unigram"
    elif kind == "missing_added":
        backend["added_tokens"] = []
    elif kind == "extra_added":
        backend["added_tokens"] += copy.deepcopy(backend["added_tokens"])
    elif kind == "bool_added_id":
        backend["added_tokens"][0]["id"] = True
    elif kind == "float_added_id":
        backend["added_tokens"][0]["id"] = 7.0
    elif kind == "wrong_body":
        backend["added_tokens"][0]["content"] += "changed"
    elif kind == "extra_body":
        backend["added_tokens"][0]["extra"] = 1
    elif kind in ("negative_decoder", "leading_zero_decoder"):
        tc["added_tokens_decoder"]["-7" if kind == "negative_decoder" else "07"] = tc["added_tokens_decoder"].pop("7")
    elif kind == "extra_decoder_body":
        tc["added_tokens_decoder"]["7"]["extra"] = 1
    elif kind == "nonboolean_body":
        tc["added_tokens_decoder"]["7"]["special"] = 1
    selected.update(
        {"config.json": encoded(config), "tokenizer_config.json": encoded(tc), "tokenizer.json": encoded(backend)}
    )
    with pytest.raises(ValueError):
        gate.configuration(info["id"], selected)


@pytest.mark.parametrize("kind", ["git_bytes", "lfs_bytes", "size", "missing_asset", "extra_local", "symlink"])
def test_index_or_local_mutants_refuse(sample, kind):
    root, _, _, _ = sample
    authority = observed(sample)
    expected = inputs.inventory(root)
    if kind == "git_bytes":
        authority["files"]["config.json"]["git_blob_oid"] = "a" * 40
    elif kind == "lfs_bytes":
        authority["files"]["tokenizer.json"]["lfs_sha256"] = "b" * 64
    elif kind == "size":
        authority["files"]["config.json"]["size"] = True
    elif kind == "missing_asset":
        authority["files"].pop("vocab.json")
    elif kind == "extra_local":
        (root / "extra").write_bytes(b"x")
    else:
        (root / "config.json").unlink()
        (root / "config.json").symlink_to(root / "tokenizer.json")
    with pytest.raises(ValueError):
        gate.bind(root, authority, expected)


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}', b'{"x":NaN}', b"", b"not JSON"])
def test_bounded_strict_json_refuses(data):
    with pytest.raises(ValueError):
        gate.parse(data)


@pytest.mark.parametrize("kind", ["extra", "bool_schema", "bool_parent", "zero_parent"])
def test_manifest_overrides_refuse(kind):
    manifest = {k: "/unused" for k in gate.KEYS}
    manifest.update(schema=1, expected_parent=1)
    if kind == "extra":
        manifest["model"] = "foreign"
    elif kind == "bool_schema":
        manifest["schema"] = True
    elif kind == "bool_parent":
        manifest["expected_parent"] = True
    else:
        manifest["expected_parent"] = 0
    with pytest.raises(ValueError):
        gate.manifest_fields(manifest)


def setup_audit(sample, monkeypatch):
    # Collection can reload same-path helpers; bind all synthetic audit peers.
    monkeypatch.setattr(gate, "peers", lambda: (inputs, cp))
    monkeypatch.setattr(cp, "load_inputs", lambda: inputs)
    root, _, info, _ = sample
    lock = {"model": info["id"], "revision": info["sha"], "trees": {"checkpoint": inputs.inventory(root)}}
    spec = {"trees": {"checkpoint": str(root)}}
    manifest = {
        "schema": 1,
        "input_spec": str(root / "README.md"),
        "stage": str(root),
        "input_lock": str(root / "README.md"),
        "input_lock_sha256": "a" * 64,
        "expected_parent": 22,
    }
    monkeypatch.setattr(cp, "check", lambda _: (spec, lock, {"synthetic_common_inputs": True}))
    return manifest


def responses(sample):
    return iter(copy.deepcopy([sample[2], sample[3], sample[2], sample[3]]))


def test_repeated_index_and_explicit_scope(sample, monkeypatch):
    manifest = setup_audit(sample, monkeypatch)
    rows = responses(sample)
    result = gate.audit(manifest, lambda _: next(rows))
    assert result["proves_indexed_tokenizer_asset_and_config_equality"] is True
    assert all(
        v is False
        for k, v in result.items()
        if k.startswith("proves_") and k != "proves_indexed_tokenizer_asset_and_config_equality"
    )
    assert result["launch_authority"] is False
    with pytest.raises(StopIteration):
        next(rows)


@pytest.mark.parametrize("kind", ["authority", "local", "helper"])
@pytest.mark.parametrize("peer_reload", [False, True])
def test_post_binding_drift_refuses(sample, monkeypatch, kind, peer_reload):
    if peer_reload:
        peer = load("ra_inputs", ROOT / "bench/ra/ra_inputs.py")
        assert peer is not inputs
        monkeypatch.setitem(sys.modules, "ra_inputs", peer)
    manifest = setup_audit(sample, monkeypatch)
    rows = responses(sample)
    count = 0
    original = inputs.file_record

    def fetch(_):
        nonlocal count
        count += 1
        if count == 3:
            if kind == "local":
                (sample[0] / "tokenizer.json").write_bytes(b"changed")
            elif kind == "helper":

                def record(p):
                    v = original(p)
                    return {**v, "sha256": "f" * 64} if Path(p).name == "ra_tokenizer_assets.py" else v

                monkeypatch.setattr(inputs, "file_record", record)
            else:
                value = next(rows)
                value["sha"] = "f" * 40
                return value
        return next(rows)

    with pytest.raises(ValueError):
        gate.audit(manifest, fetch)


@pytest.mark.parametrize(
    "kind",
    [
        "positive",
        "summary",
        "bool_schema",
        "float_schema",
        "extra_summary",
        "extra_file",
        "raw",
        "raw_resealed",
        "url",
        "clock",
        "cleanup",
        "null_return",
        "bool_pid",
        "guard",
    ],
)
def test_parent_receipt_mutants_refuse(sample, monkeypatch, tmp_path, kind):
    manifest = setup_audit(sample, monkeypatch)
    rows = responses(sample)
    expected = gate.audit(manifest, lambda _: next(rows))
    out = tmp_path.resolve() / "receipts"
    out.mkdir()
    observations = []
    for index, value in enumerate(copy.deepcopy([sample[2], sample[3], sample[2], sample[3]])):
        name = f"response-{index:02d}.json"
        content = encoded(value)
        (out / name).write_bytes(content)
        observations.append(
            {
                "url": (cp.urls(sample[2]["id"], sample[2]["sha"]) * 2)[index],
                "file": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "started_at": f"2026-10-09T00:00:0{index}+00:00",
                "finished_at": f"2026-10-09T00:00:0{index}+00:00",
            }
        )
    guard = {"pid": 88, "expected_parent": 22, "observed_parent": 22, "signal": 9, "platform": "linux"}
    result = {
        **expected,
        "parent_guard": guard,
        "status": "PASS",
        "pid": 88,
        "manifest_sha256": "b" * 64,
        "observations": observations,
    }
    process = {
        "status": "OK",
        "pid": 88,
        "returncode": 0,
        "cleanup_complete": True,
        "parent_death_guard": {"required": True, "verified": True, "expected_parent": 22},
    }
    if kind == "summary":
        result["assets"]["configuration"]["special_tokens"]["eos"]["declared_added_token_id"] = 8
    elif kind in ("bool_schema", "float_schema"):
        result["schema"] = True if kind == "bool_schema" else 1.0
    elif kind == "extra_summary":
        result["cleared"] = True
    elif kind == "extra_file":
        (out / "unknown").write_bytes(b"x")
    elif kind in ("raw", "raw_resealed"):
        content = encoded({**sample[2], "sha": "f" * 40})
        (out / observations[0]["file"]).write_bytes(content)
        if kind == "raw_resealed":
            observations[0].update(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    elif kind == "url":
        observations[0]["url"] += "&unknown=true"
    elif kind == "clock":
        observations[1]["started_at"] = "2026-10-08T00:00:00+00:00"
    elif kind == "cleanup":
        process["cleanup_complete"] = False
    elif kind == "null_return":
        process["returncode"] = None
    elif kind == "bool_pid":
        process["pid"] = True
    elif kind == "guard":
        process["parent_death_guard"]["verified"] = False
    (out / "result.json").write_bytes(encoded(result))
    if kind == "positive":
        assert gate.retained(manifest, out, process, "b" * 64) == expected
    else:
        with pytest.raises(ValueError):
            gate.retained(manifest, out, process, "b" * 64)


def test_verify_with_explicit_guard_double(sample, monkeypatch):
    manifest = setup_audit(sample, monkeypatch)
    rows = responses(sample)
    expected = gate.audit(manifest, lambda _: next(rows))
    guard = {"pid": 88, "expected_parent": 22, "observed_parent": 22, "signal": 9, "platform": "linux"}
    monkeypatch.setattr(gate, "parent_guard", lambda _: guard)
    rows = responses(sample)
    assert gate.verify(manifest, lambda _: next(rows)) == {**expected, "parent_guard": guard}


def test_composed_common_input_lock(sample, tmp_path, stage, monkeypatch):
    battery = "proof" if sample[2]["id"].startswith("ibm-granite/") else "reading"
    spec = fixture_module.fixture(tmp_path.resolve() / "composed", battery)
    root = Path(spec["trees"]["checkpoint"])
    for name, data in sample[1].items():
        (root / name).write_bytes(data)
    # Preserve this fixture's non-tokenizer file in the complete reviewed tree.
    spec_path, lock_path = tmp_path.resolve() / "spec.json", tmp_path.resolve() / "lock.json"
    fixture_module.write(spec_path, spec)
    fixture_module.write(lock_path, fixture_module.inputs.inspect(spec, stage))
    manifest = {
        "schema": 1,
        "input_spec": str(spec_path),
        "stage": str(stage),
        "input_lock": str(lock_path),
        "input_lock_sha256": fixture_module.inputs.file_record(lock_path)["sha256"],
        "expected_parent": 22,
    }
    monkeypatch.setattr(gate, "peers", lambda: (fixture_module.inputs, cp))
    monkeypatch.setattr(cp, "load_inputs", lambda: fixture_module.inputs)
    rows = responses(sample)
    result = gate.audit(manifest, lambda _: next(rows))
    assert result["common_inputs"]["status"] == "INPUT_BYTES_AND_PROJECTIONS_MATCH"
    assert result["assets"]["file_count"] == len(gate.POLICIES[sample[2]["id"]][2])
    wrong = {**manifest, "input_lock_sha256": "f" * 64}
    with pytest.raises(ValueError):
        gate.audit(wrong, lambda _: pytest.fail("network before bad lock refusal"))


@pytest.fixture
def cli_fixture(sample, tmp_path, stage):
    root = tmp_path.resolve() / "cli"
    root.mkdir()
    spec = fixture_module.fixture(
        root / "common", "proof" if sample[2]["id"].startswith("ibm-granite/") else "reading"
    )
    checkpoint = Path(spec["trees"]["checkpoint"])
    for name, data in sample[1].items():
        (checkpoint / name).write_bytes(data)
    fixture_module.write(root / "spec.json", spec)
    fixture_module.write(root / "lock.json", inputs.inspect(spec, stage))
    manifest = {
        "schema": 1,
        "input_spec": str(root / "spec.json"),
        "input_lock": str(root / "lock.json"),
        "input_lock_sha256": inputs.file_record(root / "lock.json")["sha256"],
        "stage": str(stage),
        "expected_parent": 22,
    }
    fixture_module.write(root / "manifest.json", manifest)
    return root, manifest


@pytest.mark.parametrize(
    "kind", ["positive", "manifest", "helper", "raw", "extra_receipt", "guard", "host_unprotected"]
)
def test_cli_final_and_failed_prefix_controls(sample, cli_fixture, monkeypatch, kind):
    import os
    import sys

    root, manifest = cli_fixture
    out = root / "receipt"
    monkeypatch.setattr(gate, "peers", lambda: (inputs, cp))
    monkeypatch.setattr(cp, "load_inputs", lambda: inputs)
    guard = {"pid": os.getpid(), "expected_parent": 22, "observed_parent": 22, "signal": 9, "platform": "linux"}
    if kind != "host_unprotected":
        monkeypatch.setattr(gate, "parent_guard", lambda _: guard)
    original_record = inputs.file_record

    class Replay(cp.Collector):
        def __call__(self, url):
            index = len(self.records)
            assert url == (cp.urls(sample[2]["id"], sample[2]["sha"]) * 2)[index]
            data = encoded(copy.deepcopy([sample[2], sample[3], sample[2], sample[3]][index]))
            name = f"response-{index:02d}.json"
            path = self.out / name
            path.write_bytes(data)
            self.records.append(
                {
                    "url": url,
                    "file": name,
                    "sha256": original_record(path)["sha256"],
                    "bytes": len(data),
                    "started_at": cp.clock(),
                    "finished_at": cp.clock(),
                }
            )
            return json.loads(data)

    monkeypatch.setattr(cp, "Collector", Replay)
    if kind not in ("positive", "host_unprotected"):
        original_verify = gate.verify

        def verify(m, fetch):
            result = original_verify(m, fetch)
            if kind == "manifest":
                (root / "manifest.json").write_bytes((root / "manifest.json").read_bytes() + b" ")
            elif kind == "helper":

                def record(p):
                    row = original_record(p)
                    return {**row, "sha256": "f" * 64} if Path(p).name == "ra_tokenizer_assets.py" else row

                monkeypatch.setattr(inputs, "file_record", record)
            elif kind == "raw":
                (out / "response-00.json").write_bytes(b"{}")
            elif kind == "extra_receipt":
                (out / "extra").write_bytes(b"x")
            else:
                monkeypatch.setattr(gate, "parent_guard", lambda _: {**guard, "observed_parent": 23})
            return result

        monkeypatch.setattr(gate, "verify", verify)
    monkeypatch.setattr(sys, "argv", ["gate", "--manifest", str(root / "manifest.json"), "--out", str(out)])
    if kind == "positive":
        gate.main()
        result = inputs.read_json(out / "result.json")
        assert result["status"] == "PASS" and len(result["observations"]) == 4
    else:
        with pytest.raises(ValueError):
            gate.main()
        result = inputs.read_json(out / "result.json")
        assert result["status"] == "FAILED"
        assert len(result["observations"]) == (0 if kind == "host_unprotected" else 4)
        if kind == "host_unprotected":
            isolated_linux = (
                sys.platform == "linux"
                and sys.flags.isolated
                and sys.flags.no_site
                and sys.dont_write_bytecode
            )
            expected = "expected live parent" if isolated_linux else "selected Linux"
            assert expected in result["error"]


def test_local_oversized_asset_refuses_before_read(sample):
    root, _, _, _ = sample
    authority = observed(sample)
    expected = inputs.inventory(root)
    authority["files"]["tokenizer.json"]["size"] = gate.LIMIT + 1
    with pytest.raises(ValueError, match="bounded indexed asset size"):
        gate.bind(root, authority, expected)


def test_overlong_decoder_id_refuses(sample):
    data = sample[1].copy()
    tc = json.loads(data["tokenizer_config.json"])
    tc["added_tokens_decoder"]["1" * 10000] = tc["added_tokens_decoder"].pop("7")
    data["tokenizer_config.json"] = encoded(tc)
    with pytest.raises(ValueError, match="typed complete added-token declaration"):
        gate.configuration(sample[2]["id"], {n: data[n] for n in gate.POLICIES[sample[2]["id"]][2]})
