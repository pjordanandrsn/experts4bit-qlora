"""Fixed serial ABBA process composition; no rental or GPU-clearance authority.

The controller must supply reviewed installation, premise and launch gates.
This library has no launch CLI until that handoff is implemented and reviewed.
"""
from __future__ import annotations

import copy
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
    if set(plan) != {"schema", "battery", "stage", "inputs", "versions", "deadline_epoch_s"} or \
            type(plan["schema"]) is not int or plan["schema"] != 1 or plan["battery"] not in ("proof", "reading"):
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


def supervise(plan, output, lock_path):
    """Invoke the six existing wrappers in each ABBA position, with no retries.

    A shared advisory lock excludes cooperating RA supervisors only. External
    CUDA contexts, installation/import gates and launch authority remain the
    controller's obligations. Completion never clears a release.
    """
    plan = copy.deepcopy(plan)
    check(plan)
    output, lock_path = ra_inputs.absolute(output), ra_inputs.absolute(lock_path)
    spec = pinned(plan["inputs"]["spec"])
    protected = [Path(plan["stage"]["path"]), *map(Path, spec["trees"].values()),
                 *map(Path, spec["files"].values())]
    for v in plan["versions"].values():
        protected += [Path(v["venv"]), Path(v["cache"])]
    if any(a == b or a in b.parents or b in a.parents for a in (output, lock_path) for b in protected) or \
            output == lock_path or output in lock_path.parents:
        raise ValueError("output/lock overlaps protected inputs or outputs")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        (output / "plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        record = {"status": "RUNNING", "started_at": ra_process.clock(), "attempts": 1,
                  "completed": [], "proves_gpu_engagement": False, "release_cleared": False}
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
                    result = ra_process.run(command(phase, version, spec_path, Path(plan["stage"]["path"]), destination),
                                            env=env, cwd=destination, log=destination / "process.log",
                                            receipt=destination / "process.json", deadline=plan["deadline_epoch_s"],
                                            timeout=job["timeout_s"])
                    if result.get("status") != "OK" or result.get("cleanup_complete") is not True:
                        raise RuntimeError("component process/cleanup incomplete")
                    if before != check(plan, all_timeouts=False):
                        raise ValueError("common binding changed during component")
                    record["completed"].append({"tag": tag, "phase": phase, "process": result,
                                                "removed_environment_keys": removed})
            record["status"] = "ORDERED_COMPONENTS_RECORDED_PENDING_GATES"
        except BaseException as exc:
            record.update(status="FAILED", error_type=type(exc).__name__)
            raise
        finally:
            record["finished_at"] = ra_process.clock()
            journal.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        return record
    finally:
        os.close(fd)
