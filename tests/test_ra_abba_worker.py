"""Synthetic ABBA startup/projection joins; no release/GPU/launch evidence."""

import copy
import hashlib
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

import test_ra_abba as c

abba = c.abba
plan = c.plan
abba_stage = c.abba_stage


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return c.pin(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture
def verified(plan, tmp_path, monkeypatch):
    """Own narrowed three-wheel/tool lock, never change production lock/pins."""
    own = tmp_path / "synthetic-tools"
    own.mkdir()
    image = write(own / "proof-image.json", {"synthetic": True})
    tools = {n: "0" * 64 for n in ("startup-execution-proposal.json", "startup-audit-adapters.json")}
    tools["proof-image.json"] = image["sha256"]
    wheels, lock = [], []
    for name in ("experts4bit-qlora", "grouped-nf4-gemm", "synthetic-common"):
        p = tmp_path / (name + "-1-py3-none-any.whl")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr(name.replace("-", "_") + "-1.dist-info/METADATA", "Name: " + name + "\nVersion: 1\n")
        wheels.append(c.pin(p))
        lock.append(
            {"name": name, "version": "1", "filename": p.name, "sha256": c.pin(p)["sha256"], "size": p.stat().st_size}
        )
    write(own / "proof-wheel-lock.json", {"wheels": lock})
    # check_tools itself is independently exercised by worker controls. This
    # composition fixture has no execution and explicitly substitutes that gate.
    monkeypatch.setitem(sys.modules, "ra_verified_worker", types.SimpleNamespace(check_tools=lambda pins: None))
    monkeypatch.setattr(abba, "HERE", own)
    # Ordinary plan check still independently binds registration bytes.
    (own / "PREREG-ra.md").write_bytes((c.controls.ROOT / "bench/ra/PREREG-ra.md").read_bytes())
    plan["schema"] = 2
    plan["startup"] = {"tools": tools, "manifests": {}}
    for slot, v in plan["versions"].items():
        m = {
            "schema": 1,
            "imports": {"e4b": ["synthetic_e4b"], "gnf4": ["synthetic_gnf4"]},
            "execution_policy_sha256": tools["startup-execution-proposal.json"],
            "payload": {
                "schema": 1,
                "venv": v["venv"],
                "image": image["path"],
                "image_sha256": image["sha256"],
                "startup_registry_sha256": tools["startup-audit-adapters.json"],
                "wheels": wheels,
            },
        }
        plan["startup"]["manifests"][slot] = write(tmp_path / (slot + "-manifest.json"), m)
    return plan


def receipts(binding, process, common):
    job = abba.pinned(binding["worker_spec"])
    manifest = abba.pinned(binding["manifest"])
    measured = {n: {**p, "tree_sha256": "1" * 64} for n, p in common.items()}
    h = {
        "status": "PASSED",
        "phase": "COMPLETE",
        "manifest_sha256": binding["manifest"]["sha256"],
        "proves_requested_release_imports": True,
        "payload": {
            "venv": manifest["payload"]["venv"],
            "proves_installed_payload": True,
            "distributions": measured,
            "wheel_lock_sha256": digest(measured),
        },
    }
    write(Path(binding["handoff_receipt"]), h)
    r = {
        "status": "WORKER_RETURNED_PENDING_GATES",
        "phase": "COMPLETE",
        "worker_phase": job["phase"],
        "pid": process["pid"],
        "spec_sha256": binding["worker_spec"]["sha256"],
        "handoff_sha256": c.pin(Path(binding["handoff_receipt"]))["sha256"],
        "handoff_evidence": h,
        "tools_sha256": digest(job["tools"]),
    }
    write(Path(binding["worker_receipt"]), r)
    return h, r


def native(lock, phase, out):
    identity = {"model": lock["model"], "revision": lock["revision"]}
    projection = lock["projection"]
    if phase.startswith("training"):
        path = out / "component/frozen/native" / ("granite_e4b_" + phase + ".json")
        write(
            path,
            {
                **identity,
                "tokens": {"sha256": projection["training_payload_sha256"], "pack": False, "eval_rows_used": 8},
            },
        )
        write(
            out / "component/component.json",
            {
                "status": "NATIVE_RECORDED_PENDING_ENGAGEMENT",
                "native_path": str(path),
                "native_sha256": c.pin(path)["sha256"],
                "verified_child_startup": {"status": "VERIFIED_TC1_CHILD_STARTUP_PENDING_ENGAGEMENT"},
            },
        )
    elif phase == "capacity":
        for point, filename in (
            ("warm", "capacity_warm.json"),
            ("burst", "capacity_burst.json"),
            ("end", "capacity.json"),
        ):
            write(
                out / "component" / filename,
                {
                    "model": lock["model"],
                    "plan": projection["capacity_plans"][point],
                    "prompts_sha256": projection["decode_prompts_sha256"],
                },
            )
        write(out / "component/capacity_evidence_end.json", {"config": identity})
        write(
            out / "component/server.json",
            {
                "status": "NATIVE_RECORDED_PENDING_ENGAGEMENT",
                "server_startup_verified": True,
                "cleanup_complete": True,
                "returncode": 0,
                "verified_driver_startups": [{}, {}, {}],
                "verified_server_startup": {"status": "VERIFIED_SERVER_STARTUP_RETURN_PENDING_GATES"},
            },
        )
    else:
        fields = (
            {"windows_sha256": {"wikitext": projection["quality_windows_sha256"]}}
            if phase.startswith("quality")
            else {"prompts_sha256": projection["decode_prompts_sha256"]}
        )
        write(out / "native.json", {**identity, **fields})


@pytest.mark.parametrize("mutation", [None, "common_tree", "missing_handoff", "native_projection"])
def test_schema2_orders_all_24_fixed_handoff_workers(verified, tmp_path, monkeypatch, mutation):
    common = abba.check(verified)["startup"]["common"]
    lock = abba.pinned(verified["inputs"]["lock"])
    seen = []

    def run(argv, **kw):
        dest = kw["cwd"]
        phase = dest.name
        assert argv[1:4] == ["-I", "-S", "-B"] and Path(argv[4]).name == "ra_handoff.py"
        wp = Path(argv[argv.index("--worker-spec") + 1])
        job = json.loads(wp.read_bytes())
        assert c.pin(wp)["sha256"] == argv[-1]
        assert job["phase"] == ("quality" if phase.startswith("quality") else phase)
        process = {
            "status": "OK",
            "pid": 100 + len(seen),
            "returncode": 0,
            "cleanup_complete": True,
            "parent_death_guard": {"verified": True},
        }
        binding = {
            "worker_spec": c.pin(wp),
            "manifest": c.pin(dest / "handoff-spec.json"),
            "handoff_receipt": str(dest / "handoff.json"),
            "worker_receipt": str(dest / "worker.json"),
        }
        h, r = receipts(binding, process, common)
        native(lock, phase, dest)
        if len(seen) == 1 and mutation:
            if mutation == "common_tree":
                h["payload"]["distributions"]["synthetic-common"]["tree_sha256"] = "2" * 64
                h["payload"]["wheel_lock_sha256"] = digest(h["payload"]["distributions"])
                write(Path(binding["handoff_receipt"]), h)
                r["handoff_evidence"] = h
                r["handoff_sha256"] = c.pin(Path(binding["handoff_receipt"]))["sha256"]
                write(Path(binding["worker_receipt"]), r)
            elif mutation == "missing_handoff":
                Path(binding["handoff_receipt"]).unlink()
            else:
                path = dest / "component/frozen/native/granite_e4b_training_profile.json"
                n = json.loads(path.read_bytes())
                n["revision"] = "mutant"
                write(path, n)
                summary = json.loads((dest / "component/component.json").read_bytes())
                summary["native_sha256"] = c.pin(path)["sha256"]
                write(dest / "component/component.json", summary)
        kw["receipt"].write_text(json.dumps(process))
        kw["log"].write_text("CPU substituted worker")
        seen.append((dest.parent.name, phase))
        return process

    monkeypatch.setattr(abba.ra_process, "run", run)
    if mutation:
        with pytest.raises((ValueError, OSError)):
            abba.supervise(verified, tmp_path / "run", tmp_path / "box.lock")
        journal = json.loads((tmp_path / "run/supervisor.json").read_bytes())
        assert journal["status"] == "FAILED" and len(journal["completed"]) == 1 and len(seen) == 2
        return
    result = abba.supervise(verified, tmp_path / "run", tmp_path / "box.lock")
    assert seen == [(tag, phase) for tag, _ in abba.POSITIONS for phase in abba.PHASES]
    assert result["verified_wrapper_startup"] is True and len(result["completed"]) == 24
    assert result["status"] == "ORDERED_COMPONENTS_RECORDED_PENDING_GATES"
    for key in (
        "proves_gpu_engagement",
        "release_cleared",
        "source_publication_verified",
        "runtime_consumption_verified",
        "native_context_absence_verified",
    ):
        assert result[key] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "slot",
        "venv",
        "image",
        "policy",
        "registry",
        "wheel_hash",
        "missing_common",
        "duplicate",
        "common_version",
    ],
)
def test_resealed_startup_mutants_refuse(verified, mutation):
    pin = verified["startup"]["manifests"]["new"]
    path = Path(pin["path"])
    m = json.loads(path.read_bytes())
    if mutation == "extra":
        m["command"] = "foreign"
    elif mutation == "slot":
        del verified["startup"]["manifests"]["old"]
    elif mutation == "venv":
        m["payload"]["venv"] = verified["versions"]["old"]["venv"]
    elif mutation == "image":
        m["payload"]["image_sha256"] = "f" * 64
    elif mutation == "policy":
        m["execution_policy_sha256"] = "f" * 64
    elif mutation == "registry":
        m["payload"]["startup_registry_sha256"] = "f" * 64
    elif mutation == "wheel_hash":
        m["payload"]["wheels"][-1]["sha256"] = "f" * 64
    elif mutation == "missing_common":
        m["payload"]["wheels"].pop()
    elif mutation == "duplicate":
        m["payload"]["wheels"].append(m["payload"]["wheels"][0])
    else:
        archive = Path(m["payload"]["wheels"][-1]["path"])
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("synthetic_common-1.dist-info/METADATA", "Name: synthetic-common\nVersion: 2\n")
        m["payload"]["wheels"][-1] = c.pin(archive)
    verified["startup"]["manifests"]["new"] = write(path, m)
    with pytest.raises((ValueError, KeyError)):
        abba.check(verified)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "pid",
        "phase",
        "spec",
        "handoff",
        "tools",
        "status",
        "cleanup",
        "guard",
        "returncode",
        "payload",
        "import",
        "common",
        "common_hash",
        "evidence",
        "missing_output",
        "missing_receipt",
    ],
)
def test_worker_receipt_mutants_refuse(verified, tmp_path, mutation):
    common = abba.check(verified)["startup"]["common"]
    out = tmp_path / "worker"
    out.mkdir()
    spec = out / "spec.json"
    write(spec, {})
    _, binding = abba.verified_command("decode", "old", spec, verified, out)
    process = {
        "status": "OK",
        "returncode": 0,
        "cleanup_complete": True,
        "pid": 123,
        "parent_death_guard": {"verified": True},
    }
    write(out / "native.json", {})
    h, r = receipts(binding, process, common)
    if mutation == "pid":
        r["pid"] += 1
    elif mutation == "phase":
        r["worker_phase"] = "quality"
    elif mutation == "spec":
        r["spec_sha256"] = "f" * 64
    elif mutation == "handoff":
        h["status"] = "FAILED"
    elif mutation == "tools":
        r["tools_sha256"] = "f" * 64
    elif mutation == "status":
        r["status"] = "FAILED"
    elif mutation == "cleanup":
        process["cleanup_complete"] = False
    elif mutation == "guard":
        process["parent_death_guard"]["verified"] = False
    elif mutation == "returncode":
        process["returncode"] = -9
    elif mutation == "payload":
        h["payload"]["proves_installed_payload"] = False
    elif mutation == "import":
        h["proves_requested_release_imports"] = False
    elif mutation == "common":
        h["payload"]["distributions"]["synthetic-common"]["version"] = "2"
    elif mutation == "common_hash":
        h["payload"]["wheel_lock_sha256"] = "f" * 64
    elif mutation == "evidence":
        r["handoff_evidence"] = {}
    write(Path(binding["handoff_receipt"]), h)
    if mutation != "evidence":
        r["handoff_evidence"] = copy.deepcopy(h)
    r["handoff_sha256"] = c.pin(Path(binding["handoff_receipt"]))["sha256"]
    write(Path(binding["worker_receipt"]), r)
    if mutation == "missing_output":
        (out / "native.json").unlink()
    elif mutation == "missing_receipt":
        Path(binding["worker_receipt"]).unlink()
    if mutation:
        with pytest.raises((ValueError, OSError)):
            abba.join_worker(binding, process, common)
    else:
        result = abba.join_worker(binding, process, common)
        assert result["pid"] == 123 and result["proves_gpu_engagement"] is False


@pytest.mark.parametrize("phase", c.abba.PHASES)
def test_native_projection_and_mutant(verified, tmp_path, phase):
    lock = abba.pinned(verified["inputs"]["lock"])
    out = tmp_path / "native"
    native(lock, phase, out)
    assert len(abba.native_joins(lock, phase, out)) == (3 if phase == "capacity" else 1)
    lock["revision"] = "mutated"
    with pytest.raises(ValueError, match="revision"):
        abba.native_joins(lock, phase, out)


def test_real_selected_test_host_handoff_joins_substituted_worker(tmp_path):
    import test_ra_verified_worker as worker

    manifest, job, python, tool, _ = worker.setup(tmp_path, copies=True)
    p = worker.run(tmp_path, manifest, job, python, tool)
    assert p.returncode == 0, p.stderr
    h = json.loads((tmp_path / "handoff.json").read_bytes())
    r = json.loads((tmp_path / "worker.json").read_bytes())
    common = {
        n: {k: v[k] for k in ("version", "wheel_sha256")}
        for n, v in h["payload"]["distributions"].items()
        if n not in ("experts4bit-qlora", "grouped-nf4-gemm")
    }
    binding = {
        "manifest": c.pin(tmp_path / "handoff-spec.json"),
        "worker_spec": c.pin(tmp_path / "worker-spec.json"),
        "handoff_receipt": str(tmp_path / "handoff.json"),
        "worker_receipt": str(tmp_path / "worker.json"),
    }
    # Native guard is not established on this non-Linux host. Explicitly
    # substitute it for receipt composition; Linux guard controls are separate.
    process = {
        "status": "OK",
        "returncode": 0,
        "cleanup_complete": True,
        "pid": r["pid"],
        "parent_death_guard": {"verified": True},
    }
    assert abba.join_worker(binding, process, common)["pid"] == r["pid"]


@pytest.mark.parametrize("role", ["manifest", "input_lock", "native_spec"])
def test_advisory_lock_cannot_overlap_verified_input(verified, tmp_path, role):
    pins = {
        "manifest": verified["startup"]["manifests"]["old"],
        "input_lock": verified["inputs"]["lock"],
        "native_spec": verified["versions"]["old"]["jobs"]["training"]["spec"],
    }
    path = Path(pins[role]["path"])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="overlaps protected"):
        abba.supervise(verified, tmp_path / "run", path)
    assert path.read_bytes() == before and not (tmp_path / "run").exists()
