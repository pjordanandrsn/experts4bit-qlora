"""Synthetic registered source/process composition; no consumption/GPU proof."""

from pathlib import Path

import pytest
import test_ra_abba_checkpoint as c
import test_ra_alpaca as source_fixture
import test_ra_abba_alpaca as a
import test_ra_wikitext as raw_fixture

abba = c.abba
plan = c.plan
abba_stage = c.abba_stage
verified = c.verified
published = c.published
checkpointed = c.checkpointed
template = source_fixture.template
wiki = c.c.c.c.controls.load("ra_wikitext")
sourced = a.sourced


@pytest.fixture
def wiki_bound(sourced, monkeypatch):
    p = sourced
    spec = abba.pinned(p['inputs']['spec'])
    root = Path(spec['trees']['datasets'])
    raw = b'PAR1synthetic archive only' + (8).to_bytes(4, 'little') + b'PAR1'
    archive = root / wiki.FILENAME
    archive.parent.mkdir()
    archive.write_bytes(raw)
    c.c.c.write(root / wiki.AUTHORITY, {'schema': 1, 'repo': wiki.REPO, 'config': wiki.CONFIG,
                                     'split': 'test', 'filename': wiki.FILENAME, 'revision': raw_fixture.REVISION})
    p['inputs']['lock'] = c.c.c.write(Path(p['inputs']['lock']['path']),
                                    abba.ra_inputs.inspect(spec, p['stage']['path']))
    (abba.HERE / 'ra_wikitext.py').write_bytes((source_fixture.ROOT / 'bench/ra/ra_wikitext.py').read_bytes())
    p['startup']['tools']['ra_wikitext.py'] = source_fixture.sha(abba.HERE / 'ra_wikitext.py')
    monkeypatch.setattr(wiki, '__file__', str(abba.HERE / 'ra_wikitext.py'))
    monkeypatch.setattr(wiki, 'peers', lambda: (abba.ra_inputs, c.checkpoint))
    p.update(schema=6, wikitext_source={'timeout_s': 10})
    return p


def result_for(plan, output, manifest, pid):
    output.mkdir()
    spec = abba.pinned(plan['inputs']['spec'])
    raws = raw_fixture.records((Path(spec['trees']['datasets']) / wiki.FILENAME).read_bytes())
    values = iter(raws)
    result = wiki.audit(abba.checkpoint_manifest(plan), lambda url: next(values))
    result.update(status='PASS', pid=pid, manifest_sha256=source_fixture.sha(manifest))
    urls = wiki.urls(raw_fixture.REVISION) * 2
    observations = []
    for index, raw in enumerate(raws):
        path = output / f'response-{index:02d}.json'
        c.c.c.write(path, raw)
        observations.append({'url': urls[index], 'file': path.name, 'sha256': source_fixture.sha(path),
                             'bytes': path.stat().st_size, 'started_at': '2026-01-01T00:00:00+00:00',
                             'finished_at': '2026-01-01T00:00:01+00:00'})
    result['observations'] = observations
    c.c.c.write(output / 'result.json', result)
    return result


@pytest.mark.parametrize("mutation", [None, "fields", "timeout", "schema", "helper", "recipe"])
def test_fixed_registered_plan(wiki_bound, mutation):
    if mutation == "fields":
        wiki_bound["wikitext_source"]["command"] = []
    elif mutation == "timeout":
        wiki_bound["wikitext_source"]["timeout_s"] = True
    elif mutation == "schema":
        wiki_bound["schema"] = 5
    elif mutation == "helper":
        (abba.HERE / "ra_wikitext.py").write_bytes(b"mutant helper")
    elif mutation == "recipe":
        source = Path(wiki_bound["stage"]["path"]) / "p109_box.py"
        source.write_bytes(source.read_bytes().replace(b"Salesforce/wikitext", b"Salesforce/mutant"))
    if mutation:
        with pytest.raises(ValueError):
            abba.check(wiki_bound)
    else:
        assert abba.check(wiki_bound)["wikitext_source"]["authority"]["revision"] == raw_fixture.REVISION


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
        "raw_hash", "raw_resealed", "missing_record", "url", "bytes_bool", "bytes_limit",
        "record_name", "clock", "raw_symlink",
    ],
)
def test_parent_rechecks_source_projection_and_receipt(wiki_bound, tmp_path, monkeypatch, mutation):
    out = tmp_path / "probe/before"

    def run(argv, **kw):
        assert argv == [
            wiki_bound["versions"]["old"]["python"]["path"],
            "-I",
            "-S",
            "-B",
            str(abba.HERE / "ra_wikitext.py"),
            "--manifest",
            str(out.with_suffix(".manifest.json")),
            "--out",
            str(out),
        ]
        assert "E4B_NO_LIVE" not in kw["env"] and "E4B_RENT_LIVE" not in kw["env"]
        assert abba.ra_inputs.read_json(argv[6]) == abba.checkpoint_manifest(wiki_bound)
        process = {
            "status": "OK",
            "returncode": 0,
            "pid": 123,
            "cleanup_complete": True,
            "parent_death_guard": {"verified": True},
        }
        result = result_for(wiki_bound, out, Path(argv[6]), 123)
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
            if mutation == 'source':
                result['archive']['sha256'] = '0' * 64
            else:
                result['authority']['files'][wiki.FILENAME]['lfs_sha256'] = '0' * 64
        elif mutation == "recipe":
            result["recipe"]["split"] = "train"
        elif mutation == "rows":
            result["proves_parquet_decoding"] = True
        elif mutation == "common":
            result["common_inputs"] = {}
        elif mutation == "extra_field":
            result["clearance"] = True
        elif mutation == "extra_file":
            (out / "unexpected").write_text("extra")
        elif mutation == "manifest_drift":
            Path(argv[6]).write_text("{}")
        elif mutation == "helper_drift":
            (abba.HERE / "ra_wikitext.py").write_bytes(b"changed")
        elif mutation in ("raw_drift", "prepared_drift"):
            spec = abba.pinned(wiki_bound["inputs"]["spec"])
            path = (
                Path(spec["trees"]["datasets"]) / wiki.FILENAME
                if mutation == "raw_drift"
                else Path(spec["files"]["train_data"])
            )
            path.write_bytes(path.read_bytes() + b" ")
        elif mutation in ('raw_hash', 'raw_resealed', 'missing_record', 'url', 'bytes_bool', 'bytes_limit',
                          'record_name', 'clock', 'raw_symlink'):
            row = result['observations'][0]
            raw_path = out / row['file']
            if mutation == 'raw_hash':
                raw_path.write_text('{}')
            elif mutation == 'raw_resealed':
                raw = abba.ra_inputs.read_json(raw_path)
                raw['sha'] = 'b' * 40
                c.c.c.write(raw_path, raw)
                row.update(sha256=source_fixture.sha(raw_path), bytes=raw_path.stat().st_size)
            elif mutation == 'missing_record':
                result['observations'].pop()
            elif mutation == 'url':
                row['url'] += '&caller=override'
            elif mutation == 'bytes_bool':
                row['bytes'] = True
            elif mutation == 'bytes_limit':
                row['bytes'] = c.checkpoint.LIMIT + 1
            elif mutation == 'record_name':
                row['file'] = '../foreign.json'
            elif mutation == 'clock':
                row['finished_at'] = '2026-01-01T00:00:00-05:00'
            else:
                raw_path.rename(out / 'moved-raw')
                raw_path.symlink_to(out / 'moved-raw')
        c.c.c.write(out / "result.json", result)
        if mutation == "symlink":
            (out / "result.json").rename(out / "moved")
            (out / "result.json").symlink_to(out / "moved")
        return process

    monkeypatch.setattr(abba.ra_process, "run", run)
    if mutation:
        with pytest.raises(ValueError):
            abba.wikitext_gate(wiki_bound, "old", out)
    else:
        result = abba.wikitext_gate(wiki_bound, "old", out)
        assert result["identity"]["proves_hub_index_raw_wikitext_equality"] is True
        assert result["proves_gpu_engagement"] is result["release_cleared"] is False


@pytest.mark.parametrize("mutation,prefix", [(None, 24), ("before", 0), ("after", 6), ("repeat", 12)])
def test_eight_fixed_probes_and_failed_prefix(wiki_bound, tmp_path, monkeypatch, mutation, prefix):
    checked = abba.check(wiki_bound)
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
        if Path(argv[4]).name == "ra_wikitext.py":
            out = Path(argv[-1])
            result_for(wiki_bound, out, Path(argv[6]), 321)
            seen.append((out.parent.name, "wikitext_" + out.name))
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
            c.c.c.native(abba.pinned(wiki_bound["inputs"]["lock"]), dest.name, dest)
            seen.append((dest.parent.name, dest.name))
        return process

    monkeypatch.setattr(abba, "publication_gate", publication)
    monkeypatch.setattr(abba, "checkpoint_gate", checkpoint)
    monkeypatch.setattr(abba, 'alpaca_gate', lambda *args: {'identity': {'synthetic_prior_gate': True}, 'result_sha256': 'a' * 64})
    monkeypatch.setattr(abba.ra_process, "run", run)
    out = tmp_path / "run"
    if mutation:
        with pytest.raises(ValueError):
            abba.supervise(wiki_bound, out, tmp_path / "lock")
        result = abba.ra_inputs.read_json(out / "supervisor.json")
        assert result["status"] == "FAILED" and result["raw_wikitext_source_verified"] is False
    else:
        result = abba.supervise(wiki_bound, out, tmp_path / "lock")
        assert len(result["wikitext_gates"]) == 8 and result["raw_wikitext_source_verified"] is True
        assert (
            result["runtime_consumption_verified"]
            is result["proves_gpu_engagement"]
            is result["release_cleared"]
            is False
        )
        assert seen == [
            (tag, phase) for tag, _ in abba.POSITIONS for phase in ("wikitext_before", *abba.PHASES, "wikitext_after")
        ]
    assert len(result["completed"]) == prefix


def test_complete_probe_budget_and_input_protection(wiki_bound, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(abba.ra_process, "window", lambda deadline, timeout: calls.append(timeout))
    abba.check(wiki_bound)
    total = sum(2 * j["timeout_s"] for v in wiki_bound["versions"].values() for j in v["jobs"].values())
    total += 8 * sum(
        wiki_bound[key]["timeout_s"] for key in ("source_publication", "checkpoint_authority", "alpaca_source", "wikitext_source")
    )
    assert total in calls
    spec = abba.pinned(wiki_bound["inputs"]["spec"])
    with pytest.raises(ValueError, match="protected output"):
        abba.wikitext_gate(wiki_bound, "old", Path(spec["trees"]["datasets"]) / "out")
    with pytest.raises(ValueError, match="overlaps"):
        abba.supervise(wiki_bound, tmp_path / "run", Path(spec["files"]["train_data"]) / "lock")


@pytest.mark.parametrize("mutation", [None, "missing", "resealed", "extra"])
def test_explicit_fixed_inventory_adds_only_wikitext(tmp_path, mutation):
    import test_ra_verified_worker as f

    assert len(f.worker.TOOLS) == 36 and "ra_wikitext.py" in f.worker.TOOLS
    manifest, job, python, tool, _ = f.setup(tmp_path)
    if mutation == "missing":
        del job["tools"]["ra_wikitext.py"]
    elif mutation == "resealed":
        (tool / "ra_wikitext.py").write_text("raise RuntimeError('mutant')\n")
    elif mutation == "extra":
        (tool / "ra_wikitext_unknown.py").write_text("raise RuntimeError('extra')\n")
    p = f.run(tmp_path, manifest, job, python, tool)
    assert (p.returncode == 0) is (mutation is None), p.stderr
