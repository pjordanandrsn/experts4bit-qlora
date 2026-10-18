# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""R2-Mixtral's box harness (bench/fam235/, Amendment 2's decision): R2's lock, closure and tiny model; Q235's Python tools run
unchanged through thin wrappers; and R2's copies of Q235's bash runner and drive, held to Q235's line for line.

- **Drift.** Each copy differs from its Q235 source at the cited commit only by the declared substitutions (STRICT). A Q235 change
  after that commit is reported as a warning, never a failure, so an open Q235 amendment cannot turn R2 red.
- **The Python twin.** Every file a wrapped Q235 module reads through its ``HERE`` is classified (R2's, Q235's, or read only
  by a function the wrapper replaces); a new one fails. The rehearsal's depth is R2's own line.
- **The science mode refuses** until the amendment that registers the box pins its class, floors and caps.
- **The wrappers** read R2's files where R2 has them and Q235's everywhere else.

No network and no GPU: the runner is driven to its first refusal with a stand-in ``nvidia-smi``."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "bench" / "fam235"
Q = ROOT / "bench" / "q235"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_copies_differ_from_q235_s_only_by_their_declared_substitutions():
    drift = _mod("r2_drift", F / "r2_drift.py")
    assert drift.check(drift.CITED) == {copy: [] for copy in drift.PAIRS}


def test_a_q235_change_after_the_cited_commit_is_reported_not_failed():
    drift = _mod("r2_drift", F / "r2_drift.py")
    moved = {copy: diffs for copy, diffs in drift.check(None).items() if diffs}
    if moved:
        warnings.warn(f"Q235's working tree has moved since {drift.CITED[:8]} for {sorted(moved)}: port the change into R2's copy "
                      "and re-cite (bench/fam235/r2_drift.py)", stacklevel=1)


def test_the_drift_check_names_an_undeclared_difference():
    drift = _mod("r2_drift", F / "r2_drift.py")
    rule = drift.PAIRS["bench/fam235/r2_run.sh"]
    source = drift.source_at(rule["source"], drift.CITED)
    copy = (F / "r2_run.sh").read_text()
    assert drift.undeclared(source, copy, rule) == []
    for before, after in (("MIN_RAM_GB=64;", "MIN_RAM_GB=32;"),                # a floor on a line no rule names
                          ("finish 18; }", "finish 0; }"),                     # a refusal code
                          ("q235_box.py", "r2_box.py")):                       # a tool no rename names
        assert drift.undeclared(source, copy.replace(before, after, 1), rule), before
    science = next(ln for ln in copy.splitlines() if ln.startswith("  science)"))
    assert drift.undeclared(source, copy.replace(science, "  science) CARD_RE='^NVIDIA'; MIN_RAM_GB=1 ;;"), rule)


def test_the_committed_manifest_is_the_closure():
    """R2's own files from the working tree, Q235's at the cited commit: a later Q235 change does not move R2's manifest."""
    stage = _mod("r2_stage", F / "r2_stage.py")
    q235, sealed = stage.q235_stage(), stage.sealing_tree()
    want = dict(reversed(line.split()) for line in (F / "r2.sha256").read_text().splitlines())
    for p in stage.CLOSURE:
        assert want[Path(p).name] == q235.sha(sealed / p), f"{p} changed: rewrite r2.sha256 (r2_stage.py --write) after review"
    lock = json.loads((Q / "locked_inputs.json").read_text())["assets"]
    assert set(want) == {Path(p).name for p in stage.CLOSURE} | set(lock)


def test_the_lock_is_the_registered_revision():
    lock = json.loads((F / "model_files.json").read_text())
    assert (lock["repo"], lock["revision"]) == ("mistralai/Mixtral-8x22B-Instruct-v0.1", "cc88a6cc19fbd17d9f1c0ee0b0d70a748dce698d")
    shards = [r for r in lock["files"] if r["file"].endswith(".safetensors")]
    assert lock["n_files"] == len(lock["files"]) == 69 and len(shards) == 59
    assert lock["safetensors_bytes"] == sum(r["size"] for r in shards) == 281_260_367_720
    assert all(len(r["lfs_sha256"] or "") == 64 for r in shards) and all(len(r["blob_id"]) == 40 for r in lock["files"])


@pytest.fixture
def box(tmp_path):
    """A box workspace with R2's sealed closure staged flat, a stand-in nvidia-smi, and the TC1 lifecycle's variables."""
    stage = _mod("r2_stage", F / "r2_stage.py")
    sealed = stage.sealing_tree()                                          # Q235's files at the cited commit, R2's from the tree
    w = tmp_path / "w"
    w.mkdir()
    for p in stage.CLOSURE:
        shutil.copy(sealed / p, w / Path(p).name)
    lock = json.loads((Q / "locked_inputs.json").read_text())["assets"]
    lines = [ln for ln in (F / "r2.sha256").read_text().splitlines() if ln.split()[1] not in lock]
    (w / "r2.sha256").write_text("\n".join(lines) + "\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 'NVIDIA GeForce RTX 5090, 32607, 575.64.05, 5, 5, 16, 16, 575.00 W'\n")
    (bin_ / "nvidia-smi").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("Q235_", "TC1_"))}
    env.update(PATH=f"{bin_}:{env['PATH']}", Q235_W=str(w), TC1_RUN_NONCE="n0nce", E4B_SHA="a" * 40, GNF4_SHA="b" * 40,
               TC1_DEADLINE_EPOCH=str(int(__import__("time").time()) + 3600))
    return w, env


def _run(w, env):
    return subprocess.run(["bash", str(w / "r2_run.sh")], env=env, capture_output=True, text=True, timeout=120)


def _science_line():
    import re
    m = re.search(r"^  science\) CARD_RE='([^']*)'; MIN_VRAM_MIB=(\d+); MIN_RAM_GB=(\d+); MIN_DISK_GB=(\d+)",
                  (F / "r2_run.sh").read_text(), re.M)
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def test_the_science_host_floors_are_the_registration_s():
    """R2 Amendment 3: the RTX 5090 class (at least 32,000 MiB); Amendment 2's 134 GB of host RAM; 350 GB of free disk."""
    assert _science_line() == ("^NVIDIA GeForce RTX 5090", 32000, 134, 350)


def test_the_runner_refuses_a_card_that_is_not_the_registered_class(box):
    w, env = box
    (Path(env["PATH"].split(":")[0]) / "nvidia-smi").write_text(
        "#!/bin/sh\necho 'NVIDIA RTX A2000 12GB, 12282 MiB, 575.64.05, 1, 3, 8, 16, 70.00 W'\n")
    r = _run(w, env)
    assert r.returncode == 18 and (w / "REFUSAL").read_text().strip() == "card class", (r.returncode, r.stdout[-800:])
    assert json.loads((w / "receipts" / "run.json").read_text())["schema"] == "r2-run/1"


def test_the_runner_refuses_a_5090_host_below_the_ram_floor(box):
    w, env = box
    with open("/proc/meminfo") if os.path.exists("/proc/meminfo") else open(os.devnull) as f:
        total = next((int(ln.split()[1]) for ln in f if ln.startswith("MemTotal:")), None)
    if total is None or total * 1024 >= 134e9:
        pytest.skip("needs a Linux host with less than 134 GB of RAM")
    r = _run(w, env)                                                         # the fixture's stand-in card is an RTX 5090
    assert r.returncode == 18 and (w / "REFUSAL").read_text().strip() == "host ram floor", (r.returncode, r.stdout[-800:])


def test_the_sealed_caps_and_pull_floor_are_the_committed_fit_s():
    """R2 Amendment 3: each fitted cap is its budget rounded up to the minute, the fixed phases keep Q235's caps, the pull floor is
    the fit's at R2's fetch cap, and the fit (r2_fit.py at this commit, on the published rehearsal) fits inside the 6 h maximum."""
    import math
    import re
    text = (F / "r2_run.sh").read_text()
    caps = {k: int(v) for k, v in re.findall(r"\b(CAP_[A-Z]+)=(\d+)\b", text)}
    fit = json.loads((F / "r2-fit" / "fit-500-2.00.json").read_text())
    names = {"install": "CAP_INSTALL", "fetch": "CAP_FETCH", "box": "CAP_BOX", "repeat": "CAP_REPEAT"}
    for phase, var in names.items():
        assert caps[var] == 60 * math.ceil(fit["phases"][phase]["budget_s"] / 60), phase
    assert (caps["CAP_GIT"], caps["CAP_VENV"], caps["CAP_HOSTRATE"], caps["CAP_TOKENS"]) == (240, 120, 600, 300)
    assert float(re.search(r"^MIN_FETCH_GBS=([0-9.]+)$", text, re.M).group(1)) == fit["floors"]["fetch_gb_s"] == 0.063
    assert fit["fits"] and fit["within_policy_max"] and fit["wallclock_h"] <= 6.0 and fit["floors"]["sealed_match"]


def test_the_runner_voids_a_closure_that_does_not_match_its_manifest(box):
    w, env = box
    (w / "q235_box.py").write_text("# changed\n")
    r = _run(w, env)
    assert r.returncode == 9 and "r2.sha256" in r.stdout


def test_the_fetch_gate_reads_r2_s_lock_and_q235_s_wheels():
    g = _mod("r2_fetch_gate", F / "r2_fetch_gate.py").gate_module()
    assert g.HERE / "model_files.json" == F / "model_files.json"
    cq = _mod("r2_drift", F / "r2_drift.py").cited_tree() / "bench" / "q235"
    assert (g.HERE / "wheels.json").read_bytes() == (cq / "wheels.json").read_bytes()
    assert (g.HERE / "q235_bootstrap.py").name == "q235_bootstrap.py" and (g.HERE / "q235_bootstrap.py").parent.name == "q235"
    files = g.placeholder_files()
    names = {p.name for p in files}
    assert {"r2_run.sh", "r2_tiny.py", "model_files.json", "r2.sha256", "r2_drive.sh", "PREREG-r2-mixtral.md", "wheels.json"} <= names
    assert not names & {"q235_run.sh", "q235_tiny.py", "PREREG-q235.md"}
    assert g.scan_placeholders(files) == [], "an unfilled value in a file the box runs or the registration"
    now = 1_000_000.0
    rep = {"schema": "r2-fetch-gate/1", "passed": True, "refusals": [], "e4b_sha": "a" * 40, "generated_at": now - 60}
    assert g.report_ok(rep, "a" * 40, now) is None
    assert "schema" in g.report_ok({**rep, "schema": "q235-fetch-gate/1"}, "a" * 40, now)    # Q235's report never admits R2


def test_the_drive_refuses_to_launch_without_its_own_gate_report(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("Q235_", "R2_", "TC1_", "E4B_"))}
    env.update(TC1_DRIVE_DRYRUN="1", E4B_RENT_RUN_DIR=str(tmp_path))
    r = subprocess.run(["bash", str(F / "r2_drive.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 78 and "R2_FETCH_GATE unset" in r.stdout, (r.returncode, r.stdout, r.stderr)
    q235_report = tmp_path / "gate.json"
    q235_report.write_text(json.dumps({"schema": "q235-fetch-gate/1", "passed": True, "refusals": [], "e4b_sha": "0" * 40,
                                       "generated_at": 0}))
    r = subprocess.run(["bash", str(F / "r2_drive.sh")], env={**env, "R2_FETCH_GATE": str(q235_report)}, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 78 and "does not let" in r.stdout, (r.returncode, r.stdout, r.stderr)


def test_the_tiny_mixtral_is_the_subject_s_architecture(tmp_path):
    pytest.importorskip("transformers.models.mixtral", reason="needs transformers with Mixtral")
    src = tmp_path / "tok"
    src.mkdir()
    (src / "tokenizer.json").write_text("{}")
    (src / "special_tokens_map.json").write_text("{}")
    _mod("r2_tiny", F / "r2_tiny.py").build(tmp_path / "tiny", src)
    cfg = json.loads((tmp_path / "tiny" / "config.json").read_text())
    assert cfg["model_type"] == "mixtral" and cfg["num_hidden_layers"] == 4 and cfg["num_local_experts"] == 8
    assert cfg["num_experts_per_tok"] == 2 and cfg["vocab_size"] == 32768
    assert {p.name for p in (tmp_path / "tiny").iterdir()} >= {"tokenizer.json", "special_tokens_map.json", "config.json"}
    sys.modules.pop("r2_tiny", None)


def test_amendment_4_takes_the_dx_rules_and_not_the_heldout_read():
    """R2 Amendment 4: at the cite, Q235's box records dX with its two STRICT rules, which reach R2's box whatever the flags. R2's
    box line passes no --heldout, so Amendment 8's held-out read and rule 10 do not."""
    drift = _mod("r2_drift", F / "r2_drift.py")
    box = (drift.cited_tree() / "bench" / "q235" / "instrument" / "q235_box.py").read_text()
    assert 'STRICT_RULES_A8 = ("step0_dx_finite_nonzero", "step1_dx_finite_nonzero")' in box
    assert "probe = DxProbe(model)" in box
    line = next(ln for ln in (F / "r2_run.sh").read_text().splitlines() if ln.startswith("  extra=("))
    assert line.startswith("  extra=(); ") and "--heldout" not in line


def test_every_file_a_wrapped_q235_module_reads_is_classified():
    """The Python twin of the bash drift check: a file a wrapped Q235 module newly reads through its HERE fails until declared."""
    drift = _mod("r2_drift", F / "r2_drift.py")
    assert drift.check_wrapped(drift.CITED) == {src: [] for src in drift.WRAPPED}
    src = "bench/q235/q235_fetch_gate.py"
    text = drift.source_at(src, drift.CITED)
    assert drift.unclassified(text + '\nnew = json.loads((HERE / "new_lock.json").read_text())\n', drift.WRAPPED[src]) == \
        ["reads 'new_lock.json' through HERE: not classified"]
    assert "declares 'wheels.json', which it no longer reads" in drift.unclassified(text.replace('"wheels.json"', '"w.json"'),
                                                                                      drift.WRAPPED[src])
    assert drift.unclassified(text + "\nx = HERE.parent\n", drift.WRAPPED[src]) == ["uses HERE.parent: not declared"]


def test_the_fetch_gate_maps_each_classified_name_where_it_says():
    drift = _mod("r2_drift", F / "r2_drift.py")
    g = _mod("r2_fetch_gate", F / "r2_fetch_gate.py").gate_module()
    for name, where in drift.WRAPPED["bench/q235/q235_fetch_gate.py"]["names"].items():
        if where == "r2":
            assert (g.HERE / name).parent == F, name
        elif where == "q235":
            got = g.HERE / name
            assert got.parent.name == "q235" and F not in got.parents and Q not in got.parents, name      # the cited tree's


def test_the_rehearsal_depth_is_r2_s_own():
    """A Q235 re-cite may move Q235's slice; R2's stays 4 layers unless R2 changes it, and then the drift check names it."""
    drift = _mod("r2_drift", F / "r2_drift.py")
    rule = drift.PAIRS["bench/fam235/r2_run.sh"]
    source, copy = drift.source_at(rule["source"], drift.CITED), (F / "r2_run.sh").read_text()
    deeper = source.replace("--slice 4 --slice-dir \"$W/hf-cache/slice4\"", "--slice 8 --slice-dir \"$W/hf-cache/slice8\"") \
        .replace('MODEL="$W/hf-cache/slice4"', 'MODEL="$W/hf-cache/slice8"')
    assert deeper != source and drift.undeclared(deeper, copy, rule) == []
    assert drift.undeclared(source, copy.replace("--slice 4 ", "--slice 8 "), rule)


def test_q235_s_fetch_accepts_r2_s_lock(tmp_path):
    """The lock is Q235's format, read by Q235's fetch unchanged: its own check must pass (here a pattern that names no file then
    refuses, before any download). The first A2000 leg VOIDed on this check (a lock schema named for R2, not for its format)."""
    pytest.importorskip("huggingface_hub")
    fetch = _mod("r2_drift", F / "r2_drift.py").cited_tree() / "bench" / "q235" / "q235_fetch.py"
    r = subprocess.run([sys.executable, str(fetch), "--lock", str(F / "model_files.json"), "--dest", str(tmp_path / "d"),
                        "--only", "no-such-file", "--out", str(tmp_path / "f.json")], capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "--only names no locked file" in out and "not a complete" not in out, out[-400:]


def test_the_cpu_rehearsal_s_patterns_each_name_a_locked_file():
    import fnmatch
    import re
    line = next(ln for ln in (F / "r2_run.sh").read_text().splitlines() if ln.startswith("  cpu)     must fetch"))
    patterns = [p.strip("'") for p in re.search(r"--only (.*?) --out", line).group(1).split()]
    names = [r["file"] for r in json.loads((F / "model_files.json").read_text())["files"]]
    assert patterns == ["tokenizer*", "special_tokens_map.json"]
    assert all(any(fnmatch.fnmatch(n, p) for n in names) for p in patterns), patterns


def test_launch_staging_refuses_a_q235_tree_that_has_moved_past_the_citation(tmp_path):
    """CI reads Q235 at the cited commit, but launching reads the working tree: once Q235 moves, R2 refuses to stage until it
    re-cites, so it never ships Q235 bytes it has not cited."""
    stage = _mod("r2_stage", F / "r2_stage.py")
    tree = tmp_path / "repo"
    shutil.copytree(stage.sealing_tree(), tree)
    stage.q235_stage(tree)                                                   # loads; the sealed tree itself is what r2.sha256 says
    box = tree / "bench" / "q235" / "instrument" / "q235_box.py"
    box.write_text(box.read_text() + "\n# a Q235 change after the cited commit\n")
    with pytest.raises(ValueError, match="q235_box.py differs"):
        stage.q235_stage(tree).stage(tmp_path / "assets")


def test_every_wrapper_runs_q235_at_the_cited_commit():
    """Each wrapper's Q235 module is loaded from a tree extracted at ``CITED``, never from the working tree, byte for byte."""
    drift = _mod("r2_drift", F / "r2_drift.py")
    mods = {"bench/q235/q235_stage.py": _mod("r2_stage", F / "r2_stage.py").q235_stage(),
            "bench/q235/q235_fetch_gate.py": _mod("r2_fetch_gate", F / "r2_fetch_gate.py").gate_module(),
            "bench/q235/instrument/q235_verdict.py": _mod("r2_verdict", F / "r2_verdict.py").q235_verdict(),
            "bench/q235/q235_fit.py": _mod("r2_fit", F / "r2_fit.py").fit_module()}
    for path, m in mods.items():
        where = Path(m.__file__).resolve()
        assert ROOT.resolve() not in where.parents and any(p.name.startswith(f"r2-cited-{drift.CITED[:8]}-") for p in where.parents), path
        assert where.read_text() == drift.source_at(path, drift.CITED), path


def test_a_closure_file_outside_r2_and_the_cited_paths_is_refused(monkeypatch):
    stage = _mod("r2_stage", F / "r2_stage.py")
    monkeypatch.setattr(stage, "CLOSURE", stage.CLOSURE + ("bench/p999/somewhere_else.py",))
    with pytest.raises(ValueError, match="neither R2's own"):
        stage.sealing_tree()
