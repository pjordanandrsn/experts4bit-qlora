"""Synthetic checkpoint/process joins; no live Hub, runtime or GPU proof."""

import copy
import datetime
from pathlib import Path

import pytest
import test_ra_abba_publication as c

abba = c.abba
plan = c.plan
abba_stage = c.abba_stage
verified = c.verified
published = c.published
checkpoint = c.c.c.controls.load("ra_checkpoint")


@pytest.fixture
def checkpointed(published, monkeypatch):
    spec = abba.pinned(published["inputs"]["spec"])
    root = Path(spec["trees"]["checkpoint"])
    for name, value in {
        "config.json": {},
        "tokenizer.json": {},
        "tokenizer_config.json": {},
        "model.safetensors.index.json": {"weight_map": {"synthetic.weight": "model.safetensors"}},
    }.items():
        c.c.write(root / name, value)
    (root / "model.safetensors").write_bytes(b"tiny synthetic shard; not a model")
    lock = abba.ra_inputs.inspect(spec, published["stage"]["path"])
    published["inputs"]["lock"] = c.c.write(Path(published["inputs"]["lock"]["path"]), lock)
    pins = abba.ra_inputs.read_json(abba.HERE / "source-pins.json")
    pins["models"] = {lock["model"]: lock["revision"]}
    c.c.write(abba.HERE / "source-pins.json", pins)
    # Own exact peer bytes and owner substitution; no production pin change.
    for name in ("ra_checkpoint.py", "ra_inputs.py", "ra_stage.py"):
        (abba.HERE / name).write_bytes((c.c.c.controls.ROOT / "bench/ra" / name).read_bytes())
    for name in ("ra_checkpoint.py", "ra_inputs.py", "ra_stage.py", "source-pins.json"):
        published["startup"]["tools"][name] = c.c.c.pin(abba.HERE / name)["sha256"]
    monkeypatch.setattr(checkpoint, "__file__", str(abba.HERE / "ra_checkpoint.py"))
    monkeypatch.setattr(checkpoint, "load_inputs", lambda: abba.ra_inputs)
    published["schema"] = 4
    published["checkpoint_authority"] = {"timeout_s": 10}
    return published


def raw_indexes(plan):
    lock = abba.pinned(plan["inputs"]["lock"])
    spec = abba.pinned(plan["inputs"]["spec"])
    siblings, tree = [], []
    for name in lock["trees"]["checkpoint"]:
        data = (Path(spec["trees"]["checkpoint"]) / name).read_bytes()
        row = {"type": "file", "path": name, "size": len(data), "oid": checkpoint.git_blob(data)}
        sibling = {"rfilename": name, "size": len(data), "blobId": row["oid"]}
        if name.endswith(".safetensors"):
            sha = c.c.c.pin(Path(spec["trees"]["checkpoint"]) / name)["sha256"]
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {len(data)}\n".encode()
            row.update(
                oid=checkpoint.git_blob(pointer), lfs={"oid": sha, "size": len(data), "pointerSize": len(pointer)}
            )
            sibling.update(blobId=row["oid"], lfs={"sha256": sha, "size": len(data), "pointerSize": len(pointer)})
        tree.append(row)
        siblings.append(sibling)
    info = {
        "id": lock["model"],
        "modelId": lock["model"],
        "sha": lock["revision"],
        "private": False,
        "gated": False,
        "disabled": False,
        "siblings": siblings,
    }
    return [info, tree, copy.deepcopy(info), copy.deepcopy(tree)]


def gate_result(plan, output, manifest, pid):
    output.mkdir()
    lock = abba.pinned(plan["inputs"]["lock"])
    spec = abba.pinned(plan["inputs"]["spec"])
    raw = raw_indexes(plan)
    values = iter(raw)
    authority = checkpoint.observe(lock["model"], lock["revision"], lambda url: next(values))
    materialized = checkpoint.bind(Path(spec["trees"]["checkpoint"]), authority, lock["trees"]["checkpoint"])
    rows = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for i, (url, value) in enumerate(zip(checkpoint.urls(lock["model"], lock["revision"]) * 2, raw)):
        pin = c.c.write(output / f"response-{i:02d}.json", value)
        rows.append(
            {
                "url": url,
                "file": Path(pin["path"]).name,
                "sha256": pin["sha256"],
                "bytes": Path(pin["path"]).stat().st_size,
                "started_at": now,
                "finished_at": now,
            }
        )
    result = {
        "schema": 1,
        "status": "PASS",
        "pid": pid,
        "manifest_sha256": c.c.c.pin(manifest)["sha256"],
        "authority": authority,
        "checkpoint": materialized,
        "observations": rows,
        "common_inputs": abba.ra_inputs.verify(
            spec, plan["stage"]["path"], plan["inputs"]["lock"]["path"], plan["inputs"]["lock"]["sha256"]
        ),
        "proves_hub_index_checkpoint_equality": True,
    }
    for key in (
        "proves_dataset_authority",
        "proves_calibration_authority",
        "proves_tokenizer_execution",
        "proves_runtime_consumption",
        "proves_publisher_signature",
        "proves_gpu_engagement",
        "launch_authority",
    ):
        result[key] = False
    c.c.write(output / "result.json", result)
    return result


@pytest.mark.parametrize("mutation", [None, "fields", "timeout", "model", "helper", "pins", "schema"])
def test_checkpoint_fixed_plan(checkpointed, mutation):
    if mutation == "fields":
        checkpointed["checkpoint_authority"]["argv"] = []
    elif mutation == "timeout":
        checkpointed["checkpoint_authority"]["timeout_s"] = True
    elif mutation == "model":
        c.c.write(abba.HERE / "source-pins.json", {"models": {}})
        checkpointed["startup"]["tools"]["source-pins.json"] = c.c.c.pin(abba.HERE / "source-pins.json")["sha256"]
    elif mutation == "helper":
        (abba.HERE / "ra_checkpoint.py").write_bytes(b"resealed helper")
    elif mutation == "pins":
        checkpointed["startup"]["tools"]["source-pins.json"] = "0" * 64
    elif mutation == "schema":
        checkpointed["schema"] = 3
    if mutation:
        with pytest.raises(ValueError):
            abba.check(checkpointed)
    else:
        abba.check(checkpointed)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "cleanup",
        "guard",
        "returncode",
        "status",
        "pid",
        "pid_bool",
        "manifest",
        "scope",
        "common",
        "authority",
        "count",
        "raw",
        "resealed_raw",
        "clock",
        "url",
        "missing",
        "extra",
        "symlink",
        "duplicate",
        "limit",
        "during",
        "manifest_drift",
    ],
)
def test_checkpoint_receipt_raw_local_and_process_joins(checkpointed, tmp_path, monkeypatch, mutation):
    out = tmp_path / "gate/before"

    def run(argv, **kw):
        assert argv[:5] == [
            checkpointed["versions"]["old"]["python"]["path"],
            "-I",
            "-S",
            "-B",
            str(abba.HERE / "ra_checkpoint.py"),
        ]
        assert argv[5:] == ["--manifest", str(out.with_suffix(".manifest.json")), "--out", str(out)]
        assert abba.ra_inputs.read_json(argv[6]) == abba.checkpoint_manifest(checkpointed)
        process = {
            "status": "OK",
            "returncode": 0,
            "pid": 321,
            "cleanup_complete": True,
            "parent_death_guard": {"verified": True},
        }
        result = gate_result(checkpointed, out, Path(argv[6]), 321)
        if mutation == "cleanup":
            process["cleanup_complete"] = False
        elif mutation == "guard":
            process["parent_death_guard"]["verified"] = False
        elif mutation == "returncode":
            process["returncode"] = 1
        elif mutation == "status":
            result["status"] = "FAILED"
        elif mutation in ("pid", "pid_bool"):
            result["pid"] = 322 if mutation == "pid" else True
        elif mutation == "manifest":
            result["manifest_sha256"] = "0" * 64
        elif mutation == "scope":
            result["proves_runtime_consumption"] = True
        elif mutation == "common":
            result["common_inputs"]["identity_sha256"] = "0" * 64
        elif mutation == "authority":
            result["authority"]["revision"] = "0" * 40
        elif mutation == "count":
            result["checkpoint"]["shard_count"] += 1
        elif mutation in ("raw", "resealed_raw"):
            p = out / "response-00.json"
            value = abba.ra_inputs.read_json(p)
            value["siblings"][0]["blobId"] = "0" * 40
            pin = c.c.write(p, value)
            if mutation == "resealed_raw":
                result["observations"][0].update(sha256=pin["sha256"], bytes=p.stat().st_size)
        elif mutation == "clock":
            result["observations"][0]["finished_at"] = "2000-01-01T00:00:00+00:00"
        elif mutation == "url":
            result["observations"][0]["url"] = "https://example.invalid/foreign"
        elif mutation == "missing":
            (out / "response-00.json").unlink()
        elif mutation == "extra":
            (out / "extra.json").write_text("{}")
        elif mutation == "symlink":
            p = out / "response-00.json"
            p.rename(out / "hidden")
            p.symlink_to(out / "hidden")
        elif mutation == "duplicate":
            p = out / "response-00.json"
            p.write_text('{"id": 1, "id": 2}')
            result["observations"][0].update(sha256=c.c.c.pin(p)["sha256"], bytes=p.stat().st_size)
        elif mutation == "limit":
            result["observations"][0]["bytes"] = checkpoint.LIMIT + 1
        elif mutation == "during":
            spec = abba.pinned(checkpointed["inputs"]["spec"])
            (Path(spec["trees"]["checkpoint"]) / "bytes.bin").write_bytes(b"mutant during probe")
        elif mutation == "manifest_drift":
            Path(argv[6]).write_text("{}")
        c.c.write(out / "result.json", result)
        return process

    monkeypatch.setattr(abba.ra_process, "run", run)
    if mutation:
        with pytest.raises((ValueError, FileNotFoundError)):
            abba.checkpoint_gate(checkpointed, "old", out)
        assert (out / "result.json").exists()
    else:
        result = abba.checkpoint_gate(checkpointed, "old", out)
        assert result["identity"]["checkpoint"]["shard_count"] == 1
        assert result["release_cleared"] is False


@pytest.mark.parametrize("mutation,prefix", [(None, 24), ("before", 0), ("after", 6), ("repeat", 12)])
def test_schema4_all_positions_same_checkpoint_and_failure_prefix(
    checkpointed, tmp_path, monkeypatch, mutation, prefix
):
    checked = abba.check(checkpointed)
    seen, count = [], 0

    def publication(plan, slot, out):
        return {
            "identity": {"slot": slot},
            "result_sha256": "synthetic",
            "release_archives": checked["source_publication"][slot],
        }

    def run(argv, **kw):
        nonlocal count
        process = {
            "status": "OK",
            "returncode": 0,
            "pid": 321,
            "cleanup_complete": True,
            "parent_death_guard": {"verified": True},
        }
        if Path(argv[4]).name == "ra_checkpoint.py":
            out = Path(argv[-1])
            gate_result(checkpointed, out, Path(argv[6]), 321)
            seen.append((out.parent.name, "checkpoint_" + out.name))
            if (
                (mutation == "before" and count == 0)
                or (mutation == "after" and count == 1)
                or (mutation == "repeat" and count == 4)
            ):
                process["cleanup_complete"] = False
            count += 1
        else:
            dest = kw["cwd"]
            binding = {
                "worker_spec": c.c.c.pin(dest / "worker-spec.json"),
                "manifest": c.c.c.pin(dest / "handoff-spec.json"),
                "handoff_receipt": str(dest / "handoff.json"),
                "worker_receipt": str(dest / "worker.json"),
            }
            h, r = c.c.receipts(binding, process, checked["startup"]["common"])
            for name, identity in checked["source_publication"]["old"].items():
                h["payload"]["distributions"][name] = {**identity, "tree_sha256": "f" * 64}
            c.c.write(Path(binding["handoff_receipt"]), h)
            r.update(handoff_evidence=h, handoff_sha256=c.c.c.pin(Path(binding["handoff_receipt"]))["sha256"])
            c.c.write(Path(binding["worker_receipt"]), r)
            c.c.native(abba.pinned(checkpointed["inputs"]["lock"]), dest.name, dest)
            seen.append((dest.parent.name, dest.name))
        return process

    monkeypatch.setattr(abba, "publication_gate", publication)
    monkeypatch.setattr(abba.ra_process, "run", run)
    out = tmp_path / "run"
    if mutation:
        with pytest.raises(ValueError):
            abba.supervise(checkpointed, out, tmp_path / "ra.lock")
        result = abba.ra_inputs.read_json(out / "supervisor.json")
        assert result["status"] == "FAILED" and result["checkpoint_authority_verified"] is False
    else:
        result = abba.supervise(checkpointed, out, tmp_path / "ra.lock")
        assert len(result["checkpoint_gates"]) == len(result["publication_gates"]) == 8
        assert result["checkpoint_authority_verified"] is result["source_publication_verified"] is True
        assert (
            result["runtime_consumption_verified"]
            is result["proves_gpu_engagement"]
            is result["release_cleared"]
            is False
        )
        assert seen == [
            (tag, phase)
            for tag, _ in abba.POSITIONS
            for phase in ("checkpoint_before", *abba.PHASES, "checkpoint_after")
        ]
    assert len(result["completed"]) == prefix


def test_checkpoint_deadline_budget_and_protected_inputs(checkpointed, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(abba.ra_process, "window", lambda deadline, timeout: calls.append(timeout))
    abba.check(checkpointed)
    native = sum(2 * j["timeout_s"] for v in checkpointed["versions"].values() for j in v["jobs"].values())
    assert (
        native
        + 8 * checkpointed["source_publication"]["timeout_s"]
        + 8 * checkpointed["checkpoint_authority"]["timeout_s"]
        in calls
    )
    spec = abba.pinned(checkpointed["inputs"]["spec"])
    with pytest.raises(ValueError, match="protected output"):
        abba.checkpoint_gate(checkpointed, "old", Path(spec["trees"]["checkpoint"]) / "child")
    with pytest.raises(ValueError, match="overlaps"):
        abba.supervise(checkpointed, tmp_path / "run", Path(spec["trees"]["checkpoint"]) / "lock")


@pytest.mark.parametrize("mutation", [None, "missing", "resealed", "extra"])
def test_fixed_worker_expands_only_to_checkpoint(tmp_path, mutation):
    import test_ra_verified_worker as worker_fixture

    assert len(worker_fixture.worker.TOOLS) == 35
    assert "ra_checkpoint.py" in worker_fixture.worker.TOOLS
    manifest, job, python, tool, _ = worker_fixture.setup(tmp_path)
    if mutation == "missing":
        del job["tools"]["ra_checkpoint.py"]
    elif mutation == "resealed":
        (tool / "ra_checkpoint.py").write_text('raise RuntimeError("must never import")\n')
    elif mutation == "extra":
        (tool / "ra_checkpoint_unknown.py").write_text('raise RuntimeError("must never import")\n')
    result = worker_fixture.run(tmp_path, manifest, job, python, tool)
    assert (result.returncode == 0) is (mutation is None), result.stderr
