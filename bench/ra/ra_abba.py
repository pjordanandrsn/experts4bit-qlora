"""Fixed serial ABBA process composition; no rental or GPU-clearance authority.

The controller must supply reviewed installation, premise and launch gates.
This library has no launch CLI until that handoff is implemented and reviewed.
"""
from __future__ import annotations

import copy
import email.parser
import hashlib
import zipfile
import fcntl
import json
import os
from pathlib import Path

import ra_env
import ra_inputs
import ra_process
import ra_stage

POSITIONS = (("old_a", "old"), ("new_a", "new"), ("new_b", "new"), ("old_b", "old"))
PHASES = ("training", "training_profile", "decode", "capacity", "quality_1", "quality_12")
HERE = Path(__file__).resolve().parent


def pinned(pin):
    return ra_inputs.read_json(ra_process.check_input(pin))


def check(plan, *, all_timeouts=True):
    if set(plan) != {"schema", "battery", "stage", "inputs", "versions", "deadline_epoch_s"} | (
            {"startup"} if plan.get("schema") == 2 else set()) or \
            type(plan["schema"]) is not int or plan["schema"] not in (1, 2) or plan["battery"] not in ("proof", "reading"):
        raise ValueError("ABBA plan fields/battery")
    if set(plan["stage"]) != {"path", "sha256"} or set(plan["inputs"]) != {"spec", "lock"}:
        raise ValueError("stage/input bindings")
    stage = ra_inputs.absolute(plan["stage"]["path"])
    if ra_process.file_digest(stage / "stage-manifest.json") != plan["stage"]["sha256"]:
        raise ValueError("reviewed stage manifest differs")
    ra_stage.verify(stage)
    spec = pinned(plan["inputs"]["spec"])
    lock = pinned(plan["inputs"]["lock"])
    if spec["battery"] != plan["battery"] or lock["battery"] != plan["battery"]:
        raise ValueError("input battery differs")
    evidence = ra_inputs.verify(spec, stage, plan["inputs"]["lock"]["path"], plan["inputs"]["lock"]["sha256"])
    if set(plan["versions"]) != {"old", "new"}:
        raise ValueError("two side-by-side version slots required")
    common = []
    total = 0
    for version in plan["versions"].values():
        if set(version) != {"venv", "python", "cache", "threads", "allocator", "jobs"} or \
                set(version["jobs"]) != set(PHASES):
            raise ValueError("version slots/fixed battery")
        venv, cache = (ra_inputs.absolute(version[k]) for k in ("venv", "cache"))
        if ra_process.check_input(version["python"]) != venv / "bin/python":
            raise ValueError("selected copied venv interpreter required")
        common.append((version["threads"], version["allocator"]))
        for phase, job in version["jobs"].items():
            if set(job) != {"spec", "fixture", "timeout_s"}:
                raise ValueError("job fields")
            ra_process.window(plan["deadline_epoch_s"], job["timeout_s"])
            total += 2 * job["timeout_s"]
            native = pinned(job["spec"])
            component = "quality" if phase.startswith("quality_") else phase
            fields = {"training": {"battery", "venv", "cache", "threads", "allocator", "data", "tokens", "prereg",
                                   "expect_trainable", "deadline_epoch_s", "timeout_s"},
                      "capacity": {"battery", "venv", "cache", "threads", "allocator", "fixture", "prompts",
                                   "deadline_epoch_s", "startup_s", "driver_s"},
                      "decode": {"kind", "short", "long", "reps", "prompts"},
                      "quality": {"kind", "group", "cont", "windows", "ref_dir"}}
            if set(native) != fields["training" if component == "training_profile" else component]:
                raise ValueError("native spec fields differ")
            ra_env.clean({}, component=component, fixture=job["fixture"], venv=venv, cache=cache,
                         threads=version["threads"], allocator=version["allocator"])
            if component in ("training", "training_profile", "capacity"):
                for key in ("venv", "cache", "threads", "allocator"):
                    if native[key] != version[key]:
                        raise ValueError("native/common execution identity differs")
                if native["battery"] != plan["battery"] or native["deadline_epoch_s"] != plan["deadline_epoch_s"]:
                    raise ValueError("native battery/deadline differs")
                if component == "capacity" and (native["fixture"] != job["fixture"] or
                        native["startup_s"] + 3 * native["driver_s"] + 10 > job["timeout_s"]):
                    raise ValueError("capacity fixture/outer cleanup window")
                if component.startswith("training"):
                    if ra_process.file_digest(ra_process.check_input(native["prereg"])) != \
                            ra_process.file_digest(HERE / "PREREG-ra.md"):
                        raise ValueError("TC1 registration bytes differ")
                    if native["timeout_s"] + 10 > job["timeout_s"]:
                        raise ValueError("training outer cleanup window")
                    for key, role in (("data", "train_data"), ("tokens", "train_tokens")):
                        if native[key] != {"path": spec["files"][role], "sha256": lock["files"][role]["sha256"]}:
                            raise ValueError("TC1 common input differs")
            elif native["kind"] != plan["battery"].upper():
                raise ValueError("serving battery differs")
            if component == "decode" and (native["prompts"] != spec["files"]["decode_prompts"] or
                    (native["short"], native["long"], native["reps"]) !=
                    ((8, 24, 1) if plan["battery"] == "proof" else (32, 160, 3))):
                raise ValueError("decode common input/battery differs")
            if component == "quality" and (native["windows"] != spec["files"]["quality_windows"] or
                    native["group"] != int(phase.split("_")[1]) or
                    native["cont"] != (32 if plan["battery"] == "proof" else 128)):
                raise ValueError("quality common input/group differs")
            if component == "capacity" and native["prompts"] != {
                    "path": spec["files"]["decode_prompts"], "sha256": lock["files"]["decode_prompts"]["sha256"]}:
                raise ValueError("capacity common prompts differ")
            if component in ("decode", "quality", "capacity"):
                if (job["fixture"].get("E4B_PAGED_MODEL"), job["fixture"].get("E4B_PAGED_REVISION")) != \
                        (lock["model"], lock["revision"]) or any(k in job["fixture"] for k in
                        ("E4B_PAGED_TRACE", "E4B_PAGED_STEP_TRACE")):
                    raise ValueError("model/trace fixture differs")
        if pinned(version["jobs"]["training"]["spec"]) != pinned(version["jobs"]["training_profile"]["spec"]):
            raise ValueError("profile must use the same TC1 fixture")
    if common[0] != common[1]:
        raise ValueError("old/new common execution identity differs")
    roots = [Path(v["venv"]) for v in plan["versions"].values()]
    if roots[0] == roots[1] or roots[0] in roots[1].parents or roots[1] in roots[0].parents:
        raise ValueError("isolated old/new environments required")
    caches = [Path(v["cache"]) for v in plan["versions"].values()]
    if caches[0] == caches[1] or caches[0] in caches[1].parents or caches[1] in caches[0].parents:
        raise ValueError("separate release output caches required")
    if all_timeouts:
        ra_process.window(plan["deadline_epoch_s"], total)
    if plan["schema"] == 2:
        evidence["startup"] = check_startup(plan)
    return evidence


def command(phase, version, spec_path, stage, output):
    python = version["python"]["path"]
    if phase in ("training", "training_profile", "capacity"):
        script = "ra_capacity.py" if phase == "capacity" else "ra_training.py"
        argv = [python, "-B", str(HERE / script), "--spec", str(spec_path), "--instruments", str(stage),
                "--out", str(output / "component")]
        return argv + (["--profile"] if phase == "training_profile" else [])
    mode = "decode" if phase == "decode" else "quality"
    return [python, "-B", str(HERE / "ra_serving.py"), "--mode", mode, "--spec", str(spec_path),
            "--instruments", str(stage), "--out", str(output / "native.json")]



def require(ok, message):
    if not ok:
        raise ValueError(message)


def common_wheels(manifest, lock):
    """Inspect metadata without imports; archive payload verification is in-child."""
    expected = {p['name']: p for p in lock['wheels'] if p['name'] not in
                ('experts4bit-qlora', 'grouped-nf4-gemm')}
    require(len(expected) == len(lock['wheels']) - 2, 'ABBA wheel lock inventory')
    found, releases = {}, set()
    for pin in manifest['payload']['wheels']:
        require(set(pin) == {'path', 'sha256'}, 'ABBA retained archive pin')
        path = ra_inputs.absolute(pin['path'])
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
                'ABBA regular retained archive')
        with zipfile.ZipFile(path) as archive:
            entries = [i for i in archive.infolist() if i.filename.count('/') == 1 and
                       i.filename.endswith('.dist-info/METADATA')]
            require(len(entries) == 1 and entries[0].file_size <= 1024 * 1024,
                    'ABBA bounded top-level wheel metadata')
            meta = email.parser.BytesParser().parsebytes(archive.read(entries[0]))
        require(len(meta.get_all('Name', [])) == len(meta.get_all('Version', [])) == 1,
                'ABBA archive identity metadata')
        name = meta['Name'].lower().replace('_', '-').replace('.', '-')
        if name in ('experts4bit-qlora', 'grouped-nf4-gemm'):
            require(name not in releases, 'ABBA duplicate release archive')
            releases.add(name)
            continue
        require(name in expected and name not in found, 'ABBA unknown/duplicate common archive')
        e = expected[name]
        require(meta['Version'] == e['version'] and path.name == e['filename'] and
                pin['sha256'] == e['sha256'] and path.stat().st_size == e['size'],
                'ABBA common archive differs from frozen lock')
        found[name] = {'version': e['version'], 'wheel_sha256': e['sha256']}
    require(set(found) == set(expected) and len(releases) == 2, 'ABBA complete retained wheel set')
    return found


def check_startup(plan):
    import ra_verified_worker as worker
    startup = plan['startup']
    require(set(startup) == {'tools', 'manifests'} and set(startup['manifests']) == {'old', 'new'},
            'ABBA startup fields/slots')
    worker.check_tools(startup['tools'])
    lock = json.loads((HERE / 'proof-wheel-lock.json').read_bytes())
    common = {}
    for slot in ('old', 'new'):
        manifest = pinned(startup['manifests'][slot])
        require(set(manifest) == {'schema', 'payload', 'imports', 'execution_policy_sha256'} and
                type(manifest['schema']) is int and manifest['schema'] == 1 and
                set(manifest['imports']) == {'e4b', 'gnf4'} and
                manifest['execution_policy_sha256'] == startup['tools']['startup-execution-proposal.json'],
                'ABBA startup manifest shape/policy')
        payload = manifest['payload']
        require(set(payload) == {'schema', 'venv', 'image', 'image_sha256', 'startup_registry_sha256', 'wheels'} and
                type(payload['schema']) is int and payload['schema'] == 1 and
                payload['venv'] == plan['versions'][slot]['venv'] and
                payload['image_sha256'] == startup['tools']['proof-image.json'] and
                ra_process.file_digest(ra_inputs.absolute(payload['image'])) == payload['image_sha256'] and
                payload['startup_registry_sha256'] == startup['tools']['startup-audit-adapters.json'],
                'ABBA startup interpreter/image/registry')
        common[slot] = common_wheels(manifest, lock)
    require(common['old'] == common['new'], 'ABBA common dependencies differ')
    return {'common': common['old'], 'archive_payloads_verified': False,
            'source_publication_verified': False, 'launch_authority': False}


def verified_command(phase, version, spec_path, plan, output):
    """Derive a fixed schema-1 worker, never accept caller argv."""
    manifest = pinned(plan['startup']['manifests'][version])
    mp, wp, hp, rp = [output / n for n in ('handoff-spec.json', 'worker-spec.json', 'handoff.json', 'worker.json')]
    job = {'schema': 1, 'phase': 'quality' if phase.startswith('quality_') else phase,
           'tools': plan['startup']['tools'],
           'native_spec': {'path': str(spec_path), 'sha256': ra_process.file_digest(spec_path)},
           'stage': plan['stage'], 'native_out': str(output / ('component' if phase in
               ('training', 'training_profile', 'capacity') else 'native.json')), 'receipt': str(rp)}
    for path, value in ((mp, manifest), (wp, job)):
        with path.open('x') as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    binding = {'manifest': {'path': str(mp), 'sha256': ra_process.file_digest(mp)},
               'worker_spec': {'path': str(wp), 'sha256': ra_process.file_digest(wp)},
               'handoff_receipt': str(hp), 'worker_receipt': str(rp)}
    argv = [plan['versions'][version]['python']['path'], '-I', '-S', '-B', str(HERE / 'ra_handoff.py'),
            '--manifest', str(mp), '--out', str(hp), '--worker-spec', str(wp),
            '--worker-sha256', binding['worker_spec']['sha256']]
    return argv, binding


def join_worker(binding, process, expected_common):
    job = pinned(binding['worker_spec'])
    manifest = pinned(binding['manifest'])
    hpath, rpath = (ra_inputs.absolute(binding[k]) for k in ('handoff_receipt', 'worker_receipt'))
    h, r = (ra_inputs.read_json(p) for p in (hpath, rpath))
    require(process.get('status') == 'OK' and process.get('returncode') == 0 and
            process.get('cleanup_complete') is True and type(process.get('pid')) is int and
            h.get('status') == 'PASSED' and h.get('phase') == 'COMPLETE' and
            h.get('manifest_sha256') == binding['manifest']['sha256'] and
            h.get('proves_requested_release_imports') is True and
            h['payload'].get('proves_installed_payload') is True and
            h['payload']['venv'] == manifest['payload']['venv'] and
            r.get('status') == 'WORKER_RETURNED_PENDING_GATES' and r.get('phase') == 'COMPLETE' and
            r.get('worker_phase') == job['phase'] and r.get('pid') == process['pid'] and
            r.get('spec_sha256') == binding['worker_spec']['sha256'] and
            r.get('handoff_sha256') == ra_process.file_digest(hpath) and r.get('handoff_evidence') == h and
            r.get('tools_sha256') == hashlib.sha256(json.dumps(job['tools'], sort_keys=True,
                separators=(',', ':')).encode()).hexdigest(), 'ABBA verified worker receipt join')
    require(process.get('parent_death_guard', {}).get('verified') is True,
            'ABBA Linux parent-death evidence required')
    actual = {n: {'version': p['version'], 'wheel_sha256': p['wheel_sha256']} for n, p in
              h['payload']['distributions'].items() if n not in ('experts4bit-qlora', 'grouped-nf4-gemm')}
    require(actual == expected_common, 'ABBA installed common dependency identity differs')
    measured = {n: {k: p[k] for k in ('version', 'wheel_sha256', 'tree_sha256')} for n, p in
                h['payload']['distributions'].items() if n in expected_common}
    common_hash = hashlib.sha256(json.dumps(measured, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(h['payload']['wheel_lock_sha256'] == common_hash, 'ABBA common dependency receipt hash')
    require(Path(job['native_out']).exists(), 'ABBA native output missing')
    return {'worker_sha256': ra_process.file_digest(rpath), 'handoff_sha256': ra_process.file_digest(hpath),
            'pid': process['pid'], 'common_dependency_sha256': common_hash,
            'proves_gpu_engagement': False, 'release_cleared': False}


def native_joins(lock, phase, output):
    """Bind retained native projections; this does not prove runtime consumption."""
    if phase in ('training', 'training_profile'):
        fam = 'granite' if lock['battery'] == 'proof' else 'qwen3'
        path = output / 'component/frozen/native' / (fam + '_e4b_' + phase + '.json')
        summary = ra_inputs.read_json(output / 'component/component.json')
        require(summary['status'] == 'NATIVE_RECORDED_PENDING_ENGAGEMENT' and
                summary['native_path'] == str(path) and summary['native_sha256'] == ra_process.file_digest(path) and
                summary['verified_child_startup']['status'] == 'VERIFIED_TC1_CHILD_STARTUP_PENDING_ENGAGEMENT',
                'ABBA TC1 verified child/native join')
        paths = [(phase, path, None)]
    elif phase == 'capacity':
        server = ra_inputs.read_json(output / 'component/server.json')
        require(server['status'] == 'NATIVE_RECORDED_PENDING_ENGAGEMENT' and
                server['server_startup_verified'] is True and server['cleanup_complete'] is True and
                server['returncode'] == 0 and len(server['verified_driver_startups']) == 3 and
                server['verified_server_startup']['status'] == 'VERIFIED_SERVER_STARTUP_RETURN_PENDING_GATES',
                'ABBA capacity verified nested joins/cleanup')
        evidence = ra_inputs.read_json(output / 'component/capacity_evidence_end.json')
        paths = [(point, output / 'component' / filename, evidence) for point, filename in
                 (('warm', 'capacity_warm.json'), ('burst', 'capacity_burst.json'), ('end', 'capacity.json'))]
    else:
        paths = [('quality' if phase.startswith('quality_') else phase, output / 'native.json', None)]
    result = []
    for component, path, evidence in paths:
        ra_inputs.bind_native(lock, ra_inputs.read_json(path), component, server_evidence=evidence)
        result.append({'component': component, 'native_sha256': ra_process.file_digest(path)})
    return result


def supervise(plan, output, lock_path):
    """Invoke the six existing wrappers in each ABBA position, with no retries.

    A shared advisory lock excludes cooperating RA supervisors only. External
    CUDA contexts, installation/import gates and launch authority remain the
    controller's obligations. Completion never clears a release.
    """
    plan = copy.deepcopy(plan)
    checked = check(plan)
    verified = plan["schema"] == 2
    output, lock_path = ra_inputs.absolute(output), ra_inputs.absolute(lock_path)
    spec = pinned(plan["inputs"]["spec"])
    protected = [Path(plan["stage"]["path"]), *map(Path, spec["trees"].values()),
                 *map(Path, spec["files"].values())]
    for v in plan["versions"].values():
        protected += [Path(v["venv"]), Path(v["cache"])]
    if any(a == b or a in b.parents or b in a.parents for a in (output, lock_path) for b in protected) or \
            output == lock_path or output in lock_path.parents:
        raise ValueError("output/lock overlaps protected inputs or outputs")
    if verified:
        protected += [HERE, *[Path(p['path']) for p in plan['startup']['manifests'].values()],
                      *[Path(p['path']) for p in plan['inputs'].values()],
                      *[Path(job['spec']['path']) for v in plan['versions'].values() for job in v['jobs'].values()]]
        for pin in plan['startup']['manifests'].values():
            manifest = pinned(pin)
            protected += [Path(manifest['payload']['image']), *[Path(p['path']) for p in manifest['payload']['wheels']]]
        require(not any(a == b or a in b.parents or b in a.parents for a in (output, lock_path) for b in protected),
                'ABBA verified output/lock overlaps protected inputs')
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        (output / "plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        record = {"status": "RUNNING", "started_at": ra_process.clock(), "attempts": 1,
                  "completed": [], "proves_gpu_engagement": False, "release_cleared": False,
                  "verified_wrapper_startup": False, "source_publication_verified": False,
                  "runtime_consumption_verified": False, "native_context_absence_verified": False}
        journal = output / "supervisor.json"
        try:
            for tag, slot in POSITIONS:
                version = plan["versions"][slot]
                for phase in PHASES:
                    record["active"] = {"tag": tag, "phase": phase}
                    journal.write_text(json.dumps(record, indent=2) + "\n")
                    before = check(plan, all_timeouts=False)
                    destination = output / "raw" / tag / phase
                    destination.mkdir(parents=True, exist_ok=False)
                    job = version["jobs"][phase]
                    native = pinned(job["spec"])
                    if phase.startswith("quality_"):
                        native["ref_dir"] = str(destination / "reference")
                    spec_path = destination / "spec.json"
                    spec_path.write_text(json.dumps(native, indent=2, allow_nan=False) + "\n")
                    component = "quality" if phase.startswith("quality_") else phase
                    env, removed = ra_env.clean(os.environ, component=component, fixture=job["fixture"],
                                               venv=Path(version["venv"]), cache=Path(version["cache"]),
                                               threads=version["threads"], allocator=version["allocator"])
                    if verified:
                        argv, binding = verified_command(phase, slot, spec_path, plan, destination)
                    else:
                        argv = command(phase, version, spec_path, Path(plan["stage"]["path"]), destination)
                    result = ra_process.run(argv,
                                            env=env, cwd=destination, log=destination / "process.log",
                                            receipt=destination / "process.json", deadline=plan["deadline_epoch_s"],
                                            timeout=job["timeout_s"])
                    if result.get("status") != "OK" or result.get("cleanup_complete") is not True:
                        raise RuntimeError("component process/cleanup incomplete")
                    if before != check(plan, all_timeouts=False):
                        raise ValueError("common binding changed during component")
                    joins = {}
                    if verified:
                        joins['startup'] = join_worker(binding, result, checked['startup']['common'])
                        common_hash = joins['startup']['common_dependency_sha256']
                        require(record.get('common_dependency_sha256', common_hash) == common_hash,
                                'ABBA installed common payload changed across positions')
                        record['common_dependency_sha256'] = common_hash
                        joins['native'] = native_joins(pinned(plan['inputs']['lock']), phase, destination)
                    record["completed"].append({"tag": tag, "phase": phase, "process": result,
                                                "removed_environment_keys": removed, "joins": joins})
            record["status"] = "ORDERED_COMPONENTS_RECORDED_PENDING_GATES"
            record["verified_wrapper_startup"] = verified
        except BaseException as exc:
            record.update(status="FAILED", error_type=type(exc).__name__)
            raise
        finally:
            record["finished_at"] = ra_process.clock()
            journal.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        return record
    finally:
        os.close(fd)
