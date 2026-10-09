"""Synthetic publication/process composition; no live authority or GPU claim."""

import datetime
import json
from pathlib import Path

import pytest
import test_ra_abba_worker as c

abba = c.abba
plan = c.plan
abba_stage = c.abba_stage
verified = c.verified


@pytest.fixture
def published(verified, tmp_path):
    own = abba.HERE
    startup = abba.pinned(verified["startup"]["manifests"]["old"])
    wheels = startup["payload"]["wheels"]
    # Narrowed synthetic metadata belongs only to this CPU fixture.
    import zipfile

    for pin in wheels[:2]:
        path = Path(pin["path"])
        with zipfile.ZipFile(path, "w") as archive:
            name = path.name.split("-1-")[0]
            archive.writestr(name.replace("-", "_") + "-1.dist-info/METADATA", "Name: " + name + "\nVersion: 1.0.0\n")
        pin["sha256"] = c.c.pin(path)["sha256"]
    parser = tmp_path / "packaging-1-py3-none-any.whl"
    parser.write_bytes(b"SYNTHETIC parser pin; no execution")
    lock = json.loads((own / "proof-wheel-lock.json").read_bytes())
    for row, pin in zip(lock["wheels"], wheels):
        row.update(sha256=pin["sha256"], size=Path(pin["path"]).stat().st_size)
        if row["name"] != "synthetic-common":
            row["version"] = "1.0.0"
    lock["wheels"].append(
        {
            "name": "packaging",
            "version": "1",
            "filename": parser.name,
            "sha256": c.c.pin(parser)["sha256"],
            "size": parser.stat().st_size,
        }
    )
    # common_wheels needs a real bounded packaging METADATA entry.
    with zipfile.ZipFile(parser, "w") as archive:
        archive.writestr("packaging-1.dist-info/METADATA", "Name: packaging\nVersion: 1\n")
    parser_pin = c.c.pin(parser)
    lock["wheels"][-1].update(sha256=parser_pin["sha256"], size=parser.stat().st_size)
    c.write(own / "proof-wheel-lock.json", lock)
    verified["startup"]["tools"]["proof-wheel-lock.json"] = c.c.pin(own / "proof-wheel-lock.json")["sha256"]
    for slot in ("old", "new"):
        m = abba.pinned(verified["startup"]["manifests"][slot])
        m["payload"]["wheels"] = wheels + [parser_pin]
        verified["startup"]["manifests"][slot] = c.write(tmp_path / (slot + "-startup.json"), m)
    source = []
    for pin in wheels[:2]:
        name = Path(pin["path"]).name.split("-1-")[0]
        repo = tmp_path / ("source-" + name)
        repo.mkdir()
        source.append({"repo": str(repo), "commit": "a" * 40, "version": "1.0.0", "wheel": pin})
    manifest = {
        "schema": 1,
        "releases": source,
        "parser": parser_pin,
        "parser_lock_sha256": verified["startup"]["tools"]["proof-wheel-lock.json"],
    }
    source_pins = c.write(
        own / "source-pins.json",
        {"baseline": {alias: {"version": "1.0.0", "commit": "a" * 40} for alias in ("e4b", "gnf4")}},
    )
    verified["startup"]["tools"]["source-pins.json"] = source_pins["sha256"]
    verified["schema"] = 3
    verified["source_publication"] = {
        "timeout_s": 10,
        "manifests": {slot: c.write(tmp_path / (slot + "-publication.json"), manifest) for slot in ("old", "new")},
    }
    return verified


def gate_result(published, out, *, drift=False):
    manifest = abba.pinned(published["source_publication"]["manifests"]["old"])
    expected = abba.check_publication(published)["old"]
    publications, sources, metadata, urls = {}, [], {}, []
    for name, identity in expected.items():
        publications[name] = {
            **identity,
            "project": name,
            "tag_oid": "b" * 40,
            "github_reports_verified_tag": True,
            "github_release_id": 2 if drift else 1,
        }
        sources.append({**identity, "name": name})
        metadata[name] = {**identity}
        api = "https://api.github.com/repos/pjordanandrsn/" + name
        urls += [
            api + "/releases/tags/v1.0.0",
            api + "/git/ref/tags/v1.0.0",
            api + "/git/tags/" + "b" * 40,
            "https://pypi.org/pypi/" + name + "/1.0.0/json",
        ]
    out.mkdir()
    records = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for i, url in enumerate(urls * 2):
        path = out / f"response-{i:02d}.json"
        path.write_text("{}")
        records.append(
            {
                "url": url,
                "file": path.name,
                "sha256": c.c.pin(path)["sha256"],
                "bytes": 2,
                "started_at": now,
                "finished_at": now,
            }
        )
    result = {
        "schema": 1,
        "status": "PASS",
        "manifest_sha256": published["source_publication"]["manifests"]["old"]["sha256"],
        "proves_published_release_index_binding": True,
        "publications": publications,
        "source_binding": {
            "releases": metadata,
            "runtime_source_binding": {"releases": sources, "proves_runtime_source_binding": True},
            "proves_dependency_metadata_source_binding": True,
            "parser_wheel_sha256": manifest["parser"]["sha256"],
            "parser_lock_sha256": manifest["parser_lock_sha256"],
        },
        "observations": records,
    }
    for key in (
        "proves_local_signature_verification",
        "proves_reproducible_build",
        "proves_release_authorization",
        "proves_installation",
        "startup_activated",
        "proves_release_imports",
        "proves_gpu_engagement",
    ):
        result[key] = False
    c.write(out / "result.json", result)
    return result


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "fields",
        "schema",
        "lock",
        "parser",
        "repo",
        "commit",
        "version",
        "wheel",
        "duplicate",
        "baseline_commit",
        "proof_baseline",
    ],
)
def test_publication_plan_binding_refuses(published, tmp_path, mutation):
    if mutation == "fields":
        published["source_publication"]["command"] = "foreign"
    elif mutation == "schema":
        published["schema"] = 2
    elif mutation:
        pin = published["source_publication"]["manifests"]["old"]
        m = abba.pinned(pin)
        if mutation == "lock":
            m["parser_lock_sha256"] = "0" * 64
        elif mutation == "parser":
            m["parser"]["sha256"] = "0" * 64
        elif mutation == "repo":
            m["releases"][0]["repo"] += "/missing"
        elif mutation == "commit":
            m["releases"][0]["commit"] = "short"
        elif mutation == "version":
            m["releases"][0]["version"] = "2.0.0"
        elif mutation == "wheel":
            m["releases"][0]["wheel"] = m["parser"]
        elif mutation == "baseline_commit":
            m["releases"][0]["commit"] = "b" * 40
        elif mutation == "duplicate":
            m["releases"][1] = m["releases"][0]
        else:
            lock = json.loads((abba.HERE / "proof-wheel-lock.json").read_bytes())
            lock["wheels"][0]["version"] = "2.0.0"
            c.write(abba.HERE / "proof-wheel-lock.json", lock)
            published["startup"]["tools"]["proof-wheel-lock.json"] = c.c.pin(abba.HERE / "proof-wheel-lock.json")[
                "sha256"
            ]
            for slot in ("old", "new"):
                n = abba.pinned(published["source_publication"]["manifests"][slot])
                n["parser_lock_sha256"] = published["startup"]["tools"]["proof-wheel-lock.json"]
                published["source_publication"]["manifests"][slot] = c.write(tmp_path / (slot + "-mutant.json"), n)
        if mutation != "proof_baseline":
            published["source_publication"]["manifests"]["old"] = c.write(tmp_path / "mutant.json", m)
    if mutation:
        with pytest.raises((ValueError, FileNotFoundError)):
            abba.check(published)
    else:
        assert len(abba.check(published)["source_publication"]) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "cleanup",
        "guard",
        "status",
        "manifest",
        "scope",
        "source",
        "metadata",
        "parser",
        "coverage",
        "raw",
        "size",
        "clock",
        "extra",
        "symlink",
    ],
)
def test_gate_receipts_and_raw_response_joins(published, tmp_path, monkeypatch, mutation):
    out = tmp_path / "publication/gate"

    def run(argv, **kw):
        assert argv[1:4] == ["-I", "-S", "-B"] and Path(argv[4]).name == "ra_publication.py"
        assert argv[5:] == [
            "--manifest",
            published["source_publication"]["manifests"]["old"]["path"],
            "--out",
            str(out),
        ]
        result = gate_result(published, out)
        process = {"status": "OK", "returncode": 0, "cleanup_complete": True, "parent_death_guard": {"verified": True}}
        if mutation in ("cleanup", "guard", "status"):
            if mutation == "cleanup":
                process["cleanup_complete"] = False
            elif mutation == "guard":
                process["parent_death_guard"]["verified"] = False
            else:
                result["status"] = "FAILED"
        elif mutation == "manifest":
            result["manifest_sha256"] = "0" * 64
        elif mutation == "scope":
            result["proves_gpu_engagement"] = True
        elif mutation == "source":
            result["source_binding"]["runtime_source_binding"]["releases"][0]["wheel_sha256"] = "0" * 64
        elif mutation == "metadata":
            result["source_binding"]["releases"]["experts4bit-qlora"]["commit"] = "0" * 40
        elif mutation == "parser":
            result["source_binding"]["parser_wheel_sha256"] = "0" * 64
        elif mutation == "coverage":
            result["observations"][0]["url"] = "https://foreign.invalid/"
        elif mutation == "raw":
            (out / "response-00.json").write_text('{"mutant":true}')
        elif mutation == "size":
            result["observations"][0]["bytes"] = 3
        elif mutation == "clock":
            result["observations"][0]["started_at"] = "2999-01-01T00:00:00+00:00"
        elif mutation == "extra":
            (out / "extra").write_text("foreign")
        elif mutation == "symlink":
            (out / "response-16.json").symlink_to(out / "response-00.json")
        c.write(out / "result.json", result)
        return process

    monkeypatch.setattr(abba.ra_process, "run", run)
    if mutation:
        with pytest.raises(ValueError):
            abba.publication_gate(published, "old", out)
        assert (out / "result.json").exists()
    else:
        assert len(abba.publication_gate(published, "old", out)["release_archives"]) == 2


@pytest.mark.parametrize("mutation", [None, "before", "after", "repeat", "installed"])
def test_schema3_brackets_positions_and_stops_with_prefix(published, tmp_path, monkeypatch, mutation):
    checked = abba.check(published)
    common = checked["startup"]["common"]
    lock = abba.pinned(published["inputs"]["lock"])
    seen, count = [], 0

    def run(argv, **kw):
        nonlocal count
        dest = kw["cwd"]
        process = {
            "status": "OK",
            "pid": 100 + len(seen),
            "returncode": 0,
            "cleanup_complete": True,
            "parent_death_guard": {"verified": True},
        }
        if Path(argv[4]).name == "ra_publication.py":
            out = Path(argv[-1])
            result = gate_result(
                published, out, drift=(mutation == "after" and count == 1) or (mutation == "repeat" and count == 4)
            )
            # Per-slot pin paths differ but these manifests have identical content.
            seen.append((out.parent.name, out.name))
            if mutation == "before" and count == 0:
                process["cleanup_complete"] = False
            count += 1
            c.write(out / "result.json", result)
        else:
            phase = dest.name
            binding = {
                "worker_spec": c.c.pin(dest / "worker-spec.json"),
                "manifest": c.c.pin(dest / "handoff-spec.json"),
                "handoff_receipt": str(dest / "handoff.json"),
                "worker_receipt": str(dest / "worker.json"),
            }
            h, r = c.receipts(binding, process, common)
            for name, identity in checked["source_publication"]["old"].items():
                h["payload"]["distributions"][name] = {
                    "version": identity["version"],
                    "wheel_sha256": identity["wheel_sha256"],
                    "tree_sha256": "f" * 64,
                }
            if mutation == "installed":
                h["payload"]["distributions"]["experts4bit-qlora"]["wheel_sha256"] = "0" * 64
            c.write(Path(binding["handoff_receipt"]), h)
            r.update(handoff_evidence=h, handoff_sha256=c.c.pin(Path(binding["handoff_receipt"]))["sha256"])
            c.write(Path(binding["worker_receipt"]), r)
            c.native(lock, phase, dest)
            seen.append((dest.parent.name, phase))
        return process

    monkeypatch.setattr(abba.ra_process, "run", run)
    out = tmp_path / "run"
    if mutation:
        with pytest.raises(ValueError):
            abba.supervise(published, out, tmp_path / "box.lock")
        result = json.loads((out / "supervisor.json").read_bytes())
        assert result["status"] == "FAILED" and result["source_publication_verified"] is False
        assert len(result["completed"]) == {"before": 0, "after": 6, "repeat": 12, "installed": 0}[mutation]
    else:
        result = abba.supervise(published, out, tmp_path / "box.lock")
        assert len(result["publication_gates"]) == 8 and len(result["completed"]) == 24
        assert seen == [(tag, phase) for tag, _ in abba.POSITIONS for phase in ("before", *abba.PHASES, "after")]
        assert result["source_publication_verified"] is True and result["release_cleared"] is False
        assert result["runtime_consumption_verified"] is False and result["proves_gpu_engagement"] is False
        assert all(x["joins"]["publication"] for x in result["completed"])


def test_gate_timeouts_are_included_in_whole_run_budget(published, monkeypatch):
    calls = []
    monkeypatch.setattr(abba.ra_process, "window", lambda deadline, timeout: calls.append(timeout))
    abba.check(published)
    native = sum(2 * j["timeout_s"] for v in published["versions"].values() for j in v["jobs"].values())
    assert native + 8 * published["source_publication"]["timeout_s"] in calls


@pytest.mark.parametrize("target", ["output", "lock"])
def test_publication_sources_are_protected_before_output_creation(published, tmp_path, target):
    source = Path(abba.pinned(published["source_publication"]["manifests"]["old"])["releases"][0]["repo"])
    output, lock = tmp_path / "run", tmp_path / "box.lock"
    if target == "output":
        output = source / "run"
    else:
        lock = source / "box.lock"
    with pytest.raises(ValueError, match="publication output/lock"):
        abba.supervise(published, output, lock)
    assert not output.exists() and not lock.exists()
