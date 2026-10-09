"""Synthetic registered source/process composition; no consumption/GPU proof."""

import shutil
from pathlib import Path

import pytest
import test_ra_abba_checkpoint as c
import test_ra_alpaca as source_fixture

abba = c.abba
plan = c.plan
abba_stage = c.abba_stage
verified = c.verified
published = c.published
checkpointed = c.checkpointed
template = source_fixture.template
alpaca = c.c.c.c.controls.load("ra_alpaca")


@pytest.fixture
def sourced(checkpointed, template, tmp_path, monkeypatch):
    """Own full-size synthetic source and narrowed stage; no production edit."""
    p = checkpointed
    spec = abba.pinned(p["inputs"]["spec"])
    source = Path(spec["trees"]["datasets"]) / alpaca.SOURCE_NAME
    source.write_bytes((template / "inputs/datasets" / alpaca.SOURCE_NAME).read_bytes())
    stage = tmp_path / "alpaca-stage"
    shutil.copytree(p["stage"]["path"], stage)
    builder = stage / "tp4_alpaca.py"
    builder.write_bytes((template / "stage/tp4_alpaca.py").read_bytes())
    pins = abba.ra_inputs.read_json(stage / "source-pins.json")
    name = "bench/tp4/tp4_alpaca.py"
    pins["instruments"][name] = source_fixture.sha(builder)
    c.c.c.write(stage / "source-pins.json", pins)
    staged = abba.ra_inputs.read_json(stage / "stage-manifest.json")
    staged["source_pins_sha256"] = source_fixture.sha(stage / "source-pins.json")
    staged["files"][builder.name] = {"source_path": name, "registered": True, "sha256": source_fixture.sha(builder)}
    c.c.c.write(stage / "stage-manifest.json", staged)
    (stage / "SHA256SUMS").write_text(
        "".join(
            source_fixture.sha(f) + "  " + f.name + "\n" for f in sorted(stage.iterdir()) if f.name != "SHA256SUMS"
        )
    )
    monkeypatch.setattr(abba.ra_inputs, "REGISTERED_PINS", stage / "source-pins.json")
    p["stage"] = c.c.c.c.pin(stage / "stage-manifest.json")
    p["stage"]["path"] = str(stage)
    source_fixture.literal_builder(builder)["build"](str(source), spec["files"]["train_data"])
    tokens = abba.ra_inputs.read_json(spec["files"]["train_tokens"])
    tokens["dataset_sha256"] = source_fixture.sha(Path(spec["files"]["train_data"]))
    c.c.c.write(Path(spec["files"]["train_tokens"]), tokens)
    lock = abba.ra_inputs.inspect(spec, stage)
    p["inputs"]["lock"] = c.c.c.write(Path(p["inputs"]["lock"]["path"]), lock)
    for v in p["versions"].values():
        for phase in ("training", "training_profile"):
            native = abba.pinned(v["jobs"][phase]["spec"])
            for key, role in (("data", "train_data"), ("tokens", "train_tokens")):
                native[key] = c.c.c.c.pin(Path(spec["files"][role]))
            v["jobs"][phase]["spec"] = c.c.c.write(Path(v["jobs"][phase]["spec"]["path"]), native)
    (abba.HERE / "ra_alpaca.py").write_bytes((source_fixture.ROOT / "bench/ra/ra_alpaca.py").read_bytes())
    p["startup"]["tools"]["ra_alpaca.py"] = source_fixture.sha(abba.HERE / "ra_alpaca.py")
    monkeypatch.setattr(alpaca, "__file__", str(abba.HERE / "ra_alpaca.py"))
    monkeypatch.setattr(alpaca, "load_inputs", lambda: abba.ra_inputs)
    p.update(schema=5, alpaca_source={"timeout_s": 10})
    return p


def result_for(plan, output, manifest, pid):
    output.mkdir()
    result = alpaca.audit(abba.checkpoint_manifest(plan))
    result.update(status="PASS", pid=pid, manifest_sha256=source_fixture.sha(manifest))
    c.c.c.write(output / "result.json", result)
    return result


@pytest.mark.parametrize("mutation", [None, "fields", "timeout", "schema", "helper", "recipe"])
def test_fixed_registered_plan(sourced, mutation):
    if mutation == "fields":
        sourced["alpaca_source"]["command"] = []
    elif mutation == "timeout":
        sourced["alpaca_source"]["timeout_s"] = True
    elif mutation == "schema":
        sourced["schema"] = 4
    elif mutation == "helper":
        (abba.HERE / "ra_alpaca.py").write_bytes(b"mutant helper")
    elif mutation == "recipe":
        source = Path(sourced["stage"]["path"]) / "tp4_alpaca.py"
        source.write_bytes(source.read_bytes().replace(b"3407", b"3408"))
    if mutation:
        with pytest.raises(ValueError):
            abba.check(sourced)
    else:
        assert abba.check(sourced)["alpaca_source"]["recipe"]["N_TRAIN"] == 1200


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "guard",
        "cleanup",
        "exit",
        "status",
        "pid",
        "bool_pid",
        "manifest",
        "scope",
        "source",
        "prepared",
        "recipe",
        "rows",
        "common",
        "extra_field",
        "extra_file",
        "symlink",
        "manifest_drift",
        "raw_drift",
        "prepared_drift",
        "helper_drift",
    ],
)
def test_parent_rechecks_source_projection_and_receipt(sourced, tmp_path, monkeypatch, mutation):
    out = tmp_path / "probe/before"

    def run(argv, **kw):
        assert argv == [
            sourced["versions"]["old"]["python"]["path"],
            "-I",
            "-S",
            "-B",
            str(abba.HERE / "ra_alpaca.py"),
            "--manifest",
            str(out.with_suffix(".manifest.json")),
            "--out",
            str(out),
        ]
        assert "E4B_NO_LIVE" not in kw["env"] and "E4B_RENT_LIVE" not in kw["env"]
        assert abba.ra_inputs.read_json(argv[6]) == abba.checkpoint_manifest(sourced)
        process = {
            "status": "OK",
            "returncode": 0,
            "pid": 123,
            "cleanup_complete": True,
            "parent_death_guard": {"verified": True},
        }
        result = result_for(sourced, out, Path(argv[6]), 123)
        if mutation in ("guard", "cleanup", "exit", "status"):
            if mutation == "guard":
                process["parent_death_guard"]["verified"] = False
            elif mutation == "cleanup":
                process["cleanup_complete"] = False
            elif mutation == "exit":
                process["returncode"] = 9
            else:
                process["status"] = "FAILED"
        elif mutation in ("pid", "bool_pid"):
            result["pid"] = True if mutation == "bool_pid" else 456
        elif mutation == "manifest":
            result["manifest_sha256"] = "0" * 64
        elif mutation == "scope":
            result["proves_runtime_consumption"] = True
        elif mutation in ("source", "prepared"):
            result["source" if mutation == "source" else "prepared_data"]["sha256"] = "0" * 64
        elif mutation == "recipe":
            result["recipe"]["SEED"] = 3408
        elif mutation == "rows":
            result["source_rows"] = 51759
        elif mutation == "common":
            result["common_inputs"] = {}
        elif mutation == "extra_field":
            result["clearance"] = True
        elif mutation == "extra_file":
            (out / "unexpected").write_text("extra")
        elif mutation == "manifest_drift":
            Path(argv[6]).write_text("{}")
        elif mutation == "helper_drift":
            (abba.HERE / "ra_alpaca.py").write_bytes(b"changed")
        elif mutation in ("raw_drift", "prepared_drift"):
            spec = abba.pinned(sourced["inputs"]["spec"])
            path = (
                Path(spec["trees"]["datasets"]) / alpaca.SOURCE_NAME
                if mutation == "raw_drift"
                else Path(spec["files"]["train_data"])
            )
            path.write_bytes(path.read_bytes() + b" ")
        c.c.c.write(out / "result.json", result)
        if mutation == "symlink":
            (out / "result.json").rename(out / "moved")
            (out / "result.json").symlink_to(out / "moved")
        return process

    monkeypatch.setattr(abba.ra_process, "run", run)
    if mutation:
        with pytest.raises(ValueError):
            abba.alpaca_gate(sourced, "old", out)
    else:
        result = abba.alpaca_gate(sourced, "old", out)
        assert result["identity"]["source_rows"] == 51760
        assert result["proves_gpu_engagement"] is result["release_cleared"] is False


@pytest.mark.parametrize("mutation,prefix", [(None, 24), ("before", 0), ("after", 6), ("repeat", 12)])
def test_eight_fixed_probes_and_failed_prefix(sourced, tmp_path, monkeypatch, mutation, prefix):
    checked = abba.check(sourced)
    seen, count = [], 0

    def publication(plan, slot, out):
        return {"identity": {"slot": slot}, "release_archives": checked["source_publication"][slot], "result_sha256": "synthetic"}

    def checkpoint(plan, slot, out):
        return {"identity": {"common": "synthetic"}}

    def run(argv, **kw):
        nonlocal count
        process = {
            "status": "OK",
            "returncode": 0,
            "pid": 321,
            "cleanup_complete": True,
            "parent_death_guard": {"verified": True},
        }
        if Path(argv[4]).name == "ra_alpaca.py":
            out = Path(argv[-1])
            result_for(sourced, out, Path(argv[6]), 321)
            seen.append((out.parent.name, "alpaca_" + out.name))
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
                "worker_spec": c.c.c.c.pin(dest / "worker-spec.json"),
                "manifest": c.c.c.c.pin(dest / "handoff-spec.json"),
                "handoff_receipt": str(dest / "handoff.json"),
                "worker_receipt": str(dest / "worker.json"),
            }
            h, r = c.c.c.receipts(binding, process, checked["startup"]["common"])
            for name, identity in checked["source_publication"]["old"].items():
                h["payload"]["distributions"][name] = {**identity, "tree_sha256": "f" * 64}
            c.c.c.write(Path(binding["handoff_receipt"]), h)
            r.update(handoff_evidence=h, handoff_sha256=c.c.c.c.pin(Path(binding["handoff_receipt"]))["sha256"])
            c.c.c.write(Path(binding["worker_receipt"]), r)
            c.c.c.native(abba.pinned(sourced["inputs"]["lock"]), dest.name, dest)
            seen.append((dest.parent.name, dest.name))
        return process

    monkeypatch.setattr(abba, "publication_gate", publication)
    monkeypatch.setattr(abba, "checkpoint_gate", checkpoint)
    monkeypatch.setattr(abba.ra_process, "run", run)
    out = tmp_path / "run"
    if mutation:
        with pytest.raises(ValueError):
            abba.supervise(sourced, out, tmp_path / "lock")
        result = abba.ra_inputs.read_json(out / "supervisor.json")
        assert result["status"] == "FAILED" and result["registered_alpaca_source_verified"] is False
    else:
        result = abba.supervise(sourced, out, tmp_path / "lock")
        assert len(result["alpaca_gates"]) == 8 and result["registered_alpaca_source_verified"] is True
        assert (
            result["runtime_consumption_verified"]
            is result["proves_gpu_engagement"]
            is result["release_cleared"]
            is False
        )
        assert seen == [
            (tag, phase) for tag, _ in abba.POSITIONS for phase in ("alpaca_before", *abba.PHASES, "alpaca_after")
        ]
    assert len(result["completed"]) == prefix


def test_complete_probe_budget_and_input_protection(sourced, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(abba.ra_process, "window", lambda deadline, timeout: calls.append(timeout))
    abba.check(sourced)
    total = sum(2 * j["timeout_s"] for v in sourced["versions"].values() for j in v["jobs"].values())
    total += 8 * sum(
        sourced[key]["timeout_s"] for key in ("source_publication", "checkpoint_authority", "alpaca_source")
    )
    assert total in calls
    spec = abba.pinned(sourced["inputs"]["spec"])
    with pytest.raises(ValueError, match="protected output"):
        abba.alpaca_gate(sourced, "old", Path(spec["trees"]["datasets"]) / "out")
    with pytest.raises(ValueError, match="overlaps"):
        abba.supervise(sourced, tmp_path / "run", Path(spec["files"]["train_data"]) / "lock")


@pytest.mark.parametrize("mutation", [None, "missing", "resealed", "extra"])
def test_explicit_fixed_inventory_adds_only_alpaca(tmp_path, mutation):
    import test_ra_verified_worker as f

    assert len(f.worker.TOOLS) == 36 and "ra_alpaca.py" in f.worker.TOOLS
    manifest, job, python, tool, _ = f.setup(tmp_path)
    if mutation == "missing":
        del job["tools"]["ra_alpaca.py"]
    elif mutation == "resealed":
        (tool / "ra_alpaca.py").write_text("raise RuntimeError('mutant')\n")
    elif mutation == "extra":
        (tool / "ra_alpaca_unknown.py").write_text("raise RuntimeError('extra')\n")
    p = f.run(tmp_path, manifest, job, python, tool)
    assert (p.returncode == 0) is (mutation is None), p.stderr
