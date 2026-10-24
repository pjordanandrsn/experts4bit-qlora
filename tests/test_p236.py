# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P236: grouped-nf4-gemm 0.51.0's decode GEMV vs bitsandbytes 0.50.2, op level (bench/p236/PREREG-p236.md). The
self-tests; every runner call through its script's own argument check; the staged pins; the run ids across runner,
driver and PREREG; the pins; the pre-install fit and its boot slack; the budget; every pre-start exit classified; the
fetch imports no links; and the bench run end to end on CPU (``--dry``) into the reducer, which reads it VOID as a dry
run with no cell fault but the dry rep count. (Private names are the repository guards' job, not this file's.)"""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p236"
PREREG = (LANE / "PREREG-p236.md").read_text(encoding="utf-8")
RUN = (LANE / "p236_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p236_drive.sh").read_text(encoding="utf-8")
STAGED = ("p236_bench.py", "p236_reduce.py", "p236_run.sh", "release_proof.py")
RUN_IDS = ["p236-5090-1", "p236-5090-2", "p236-5090-r1", "p236-5090-r2", "p236-5090-r3"]
SSH_READY_S, STAGING_MARGIN_S, GUARD_H = 420, 300, 0.85
ENV = {**os.environ, "PYTHONPATH": str(LANE)}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, str(LANE))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(LANE))
    return mod


@pytest.mark.parametrize("name", ["p236_reduce", "p236_bench"])
def test_self_tests_pass(name):
    r = subprocess.run([sys.executable, str(LANE / f"{name}.py"), "--self-test"], capture_output=True, text=True,
                       timeout=300, cwd=str(LANE), env=ENV)
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("script", ["p236_bench.py", "p236_reduce.py"])
def test_every_runner_call_passes_the_scripts_own_argument_check(script):
    calls = []
    for line in RUN.splitlines():
        m = re.search(r"python \$W/" + re.escape(script) + r"(?P<rest>.*)", line)
        if m:
            rest = re.split(r"\s(?:\d?>|>>|\||;|&&|\|\|)", " " + m["rest"])[0]
            calls.append([re.sub(r'"?\$\{?[A-Za-z_][A-Za-z0-9_]*\}?(/[^ "]*)?"?', "X", t) for t in shlex.split(rest)])
    assert calls, script
    for argv in calls:
        r = subprocess.run([sys.executable, str(LANE / script), *argv, "--check-args"], capture_output=True, text=True,
                           timeout=60, cwd=str(LANE), env=ENV)
        assert r.returncode == 0 and "ARGS OK" in r.stdout, (argv, r.stdout + r.stderr)


def test_staged_pins_match_and_the_driver_stages_them():
    lines = [ln.split() for ln in (LANE / "staged-p236.sha256").read_text(encoding="ascii").splitlines() if ln.strip()]
    assert sorted(n for _, n in lines) == sorted(STAGED)
    for want, name in lines:
        assert hashlib.sha256((LANE / name).read_bytes()).hexdigest() == want, name
    for name in (*STAGED, "staged-p236.sha256"):
        assert f"bench/p236/{name}" in DRIVE, name
        assert name in RUN[:RUN.index("sha256sum -c staged-p236.sha256")], name


def test_run_ids_agree_across_runner_driver_and_prereg():
    assert "|".join(RUN_IDS) + ") :;;" in RUN
    assert "p236-5090-[12]|p236-5090-r[1-3]) :;;" in DRIVE
    for bad in ("p236-5090-3", "p236-5090-r4"):
        assert bad not in RUN, bad
    assert "`p236-5090-1`" in PREREG and "`p236-5090-2`" in PREREG and "`p236-5090-r1` to `p236-5090-r3`" in PREREG


def test_the_pins_agree():
    R = _load("p236_reduce")
    assert f"GNF4_SHA={R.GNF4_SHA}" in RUN and f"GNF4_VERSION={R.GNF4_VERSION}; BNB_VERSION={R.BNB_VERSION}" in RUN
    assert (R.GNF4_VERSION, R.BNB_VERSION, R.TORCH_VERSION) == ("0.51.0", "0.50.2", "2.8.0+cu128")
    assert f"TORCH_WANT={R.TORCH_VERSION}" in RUN and R.GNF4_SHA[:8] in PREREG
    shapes = "[" + ", ".join(f"({n}, {k})" for n, k in R.SHAPES.values()) + "]"
    assert f"shapes = {shapes}" in RUN, "the tripwire checks K33 at exactly the registered shapes"
    for (n, k) in R.SHAPES.values():
        assert f"({n}, {k})" in PREREG
    assert (R.REPS, R.ITERS, R.WARM, R.TOL, R.MS) == (20, 50, 10, 1.5e-2, (1, 2, 4, 8, 16))
    for s in ("**20 reps**", "**50 eager calls**", "**0.015**", "gnf4 bnb bnb gnf4"):
        assert s in PREREG, s


def test_the_pre_install_fit_and_the_budget():
    fit = 600 + 600 + 600
    need = fit * 125 // 100
    guard_s = round(GUARD_H * 3600)
    assert (fit, need, guard_s) == (1800, 2250, 3060)
    assert guard_s - need >= SSH_READY_S + STAGING_MARGIN_S, guard_s - need          # slack 810 s
    box = GUARD_H * 1.05 + 8 * 0.02
    nonstart = (guard_s - need) * 1.05 / 3600
    lane = 2 * box + 3 * nonstart
    assert abs(box - 1.0525) < 1e-9 and abs(nonstart - 0.23625) < 1e-9 and abs(lane - 2.81375) < 1e-9 and lane <= 3.0
    for s in ("1,800 s", "2,250 s", "Guard 0.85 h (3,060 s)", "810 s", "USD 1.0525", "USD 0.23625", "USD 2.81375"):
        assert s in PREREG, s
    for s in ("INSTALL_TO_GATE_S=600", "NEED_BENCH=600", "RESERVE=600", "FIT_HEADROOM_PCT=125"):
        assert s in RUN, s
    assert RUN.index('[ "$LEFT" -ge "$FIT_NEED" ]') < RUN.index(": > $W/STARTED") < RUN.index("python -m pip install")


def test_every_exit_before_the_start_is_classified():
    R = _load("p236_reduce")
    pre = RUN[:RUN.index(": > $W/STARTED")]
    codes = {int(c) for c in re.findall(r"finish (\d+)", pre)} - {130}
    assert codes and not [c for c in codes if c not in R.HOST_CODES and c not in R.FIT_CODES and c not in R.KIT_CODES]
    for rc in R.HOST_CODES:
        assert R.classify(rc, False)["event"] == "HOST_REFUSED"
    assert R.classify(41, False)["event"] == "FIT_REFUSED" and R.classify(0, True)["event"] == "ANSWER"
    assert R.classify(27, True)["event"] == "NO_ANSWER"
    assert 'case "$V" in READ) finish 0;; *) say "P236 $V -- no reading"; finish 27;; esac' in RUN


def test_the_fetch_imports_no_links():
    rsyncs = [ln for ln in DRIVE.splitlines() if re.search(r"(^|[;&|(]\s*)rsync\b", ln.split("#", 1)[0])]
    assert rsyncs and all("--no-links" in ln.split() for ln in rsyncs)
    assert "rsync --no-links" in PREREG


def test_the_bench_runs_end_to_end_on_cpu_into_the_reducer(tmp_path):
    """The bench's control flow and record, on CPU stand-ins (``--dry``), read by the reducer: VOID as a dry run, and no
    cell fault but the dry rep count, so a reading box's record will be read by the same code."""
    pytest.importorskip("nf4_grouped")
    pytest.importorskip("nf4_pack_ref")
    out = tmp_path / "dry.json"
    r = subprocess.run([sys.executable, str(LANE / "p236_bench.py"), "--dry", "--out", str(out)], capture_output=True,
                       text=True, timeout=600, cwd=str(LANE), env=ENV)
    assert r.returncode == 0 and "P236_BENCH rc=0 status=ok cells=30" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    R = _load("p236_reduce")
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["dry"] is True and len(rec["cells"]) == 30
    faults = {k: [f for f in R.cell_faults(k, c) if "reps, registered" not in f] for k, c in rec["cells"].items()}
    assert not any(faults.values()), faults
    v = R.reduce(rec)
    assert v["verdict"] == "VOID" and "a dry run is never a reading" in v["reasons"]
    routes = {k: next(x for x in R.BNB_ROUTES if c["engaged"]["bnb"][x]) for k, c in rec["cells"].items()}
    assert routes["mixtral.gate_up|M1"] == "custom" and routes["mixtral.gate_up|M16"] == "dequant"


def _fake_site(root, gnf4_direct_url, bnb_direct_url=None, gnf4_version="0.51.0"):
    """Installed-dist metadata for both libraries, as pip writes it, under ``root`` (prepended to sys.path)."""
    for name, dirname, version, du in (("grouped-nf4-gemm", "grouped_nf4_gemm", gnf4_version, gnf4_direct_url),
                                       ("bitsandbytes", "bitsandbytes", "0.50.2", bnb_direct_url)):
        d = root / f"{dirname}-{version}.dist-info"
        d.mkdir(parents=True)
        (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
        if du is not None:
            (d / "direct_url.json").write_text(json.dumps(du), encoding="utf-8")
    return root


def _provenance_in(site):
    code = ("import json, sys; import p236_reduce as R; "
            "d = {n: R.dist_record(n) for n in R.RELEASES}; print(json.dumps(R.provenance_faults(d)))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=str(LANE),
                       env={**os.environ, "PYTHONPATH": os.pathsep.join([str(site), str(LANE)])})
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.parametrize("case", ["editable", "wrong-commit", "file-wheel", "bnb-from-git", "right-release"])
def test_the_release_proof_reads_installed_dists(tmp_path, case):
    """The maintainer's rule for every "at the released pins" row: the box proves each library IS its release. Fake
    installed dists, read through importlib.metadata exactly as the box's tripwire reads them: an editable install and a
    wrong-commit git install are refused, as are a local wheel and a git install of bitsandbytes; the release commit
    from git plus bitsandbytes' index wheel is accepted."""
    R = _load("p236_reduce")
    git = {"url": "https://" + R.GNF4_REPO + ".git", "vcs_info": {"vcs": "git", "commit_id": R.GNF4_SHA}}
    gnf4, bnb = {
        "editable": ({"url": "file:///src/grouped-nf4-gemm", "dir_info": {"editable": True}}, None),
        "wrong-commit": ({**git, "vcs_info": {"vcs": "git", "commit_id": "f" * 40}}, None),
        "file-wheel": (git, {"url": "file:///tmp/bitsandbytes-0.50.2-py3-none-any.whl", "archive_info": {}}),
        "bnb-from-git": (git, {"url": "https://github.com/bitsandbytes-foundation/bitsandbytes.git",
                               "vcs_info": {"vcs": "git", "commit_id": "a" * 40}}),
        "right-release": (git, None),
    }[case]
    faults = _provenance_in(_fake_site(tmp_path / "site", gnf4, bnb))
    if case == "right-release":
        assert faults == [], faults
    else:
        assert len(faults) == 1, faults
    assert "provenance_faults(dists)" in RUN and RUN.index("provenance_faults(dists)") < RUN.index("p236_bench.py --out")


def test_a_cell_that_fails_a_gate_is_never_timed(tmp_path, monkeypatch):
    """The gates come before any timing: with grouped-nf4-gemm's output pushed off the reference (a stand-in fault),
    every cell records its gate failure and no rep, and the reducer VOIDs."""
    pytest.importorskip("nf4_grouped")
    pytest.importorskip("nf4_pack_ref")
    B = _load("p236_bench")
    real = B.Dry.gnf4

    def off(self, x, w, eids):
        return real(self, x, w, eids) * 1.1
    monkeypatch.setattr(B.Dry, "gnf4", off)
    out = tmp_path / "gated.json"
    import argparse
    assert B.run(argparse.Namespace(dry=True, out=str(out), release_proof=None)) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert len(rec["cells"]) == 30
    assert all(c["reps"] == [] and any("fp32 reference" in g for g in c["gate_failed"]) for c in rec["cells"].values())
    assert _load("p236_reduce").reduce(rec)["verdict"] == "VOID"


def test_the_release_proof_is_the_shared_helper_and_runs_before_any_timing():
    """bitsandbytes' bytes are proven by the shared helper (loggetta bench/release_proof.py), staged BYTE-IDENTICAL: its
    sha256 is the reducer's constant and the PREREG's; the runner calls it after the installs and before the self-tests
    and the bench, and a refusal stops the box; the bench records its output and the reducer requires it."""
    R = _load("p236_reduce")
    got = hashlib.sha256((LANE / "release_proof.py").read_bytes()).hexdigest()
    assert got == R.RELEASE_PROOF_SHA256 == "199d4f2faa49419090ef57e73fc34a86787ad5dcfe67060e92aaaf88a9bf6a63"
    assert R.RELEASE_PROOF_SHA256 in PREREG
    i = RUN.index('python $W/release_proof.py "bitsandbytes==$BNB_VERSION"')
    bench = RUN.index("python $W/p236_bench.py --out")
    assert RUN.index("tripwire OK") < RUN.index("p236_reduce.py --self-test") < RUN.index("can_run $NEED_BENCH") < i < bench
    between = RUN[i:bench]
    assert "RELEASE PROOF FAILED" in between and "finish 9" in between
    assert "pip" not in between and "install" not in between, "nothing is installed between the proof and the bench"
    assert RUN[RUN.rindex(chr(10), 0, i):i].strip().startswith("RELEASE_PROOF_NETWORK=1 PYTHONPATH= ")
    assert RUN[RUN.rindex(chr(10), 0, bench):bench].strip().startswith("PYTHONPATH= "), "the same PYTHONPATH as the proof"
    line_start = RUN.rindex(chr(10), 0, i)
    assert "RELEASE_PROOF_NETWORK=1" in RUN[line_start:i], "the network proof is mandatory on the box"
    assert "--release-proof $W/release_proof.json" in RUN
    rp = importlib.util.spec_from_file_location("release_proof_p236", LANE / "release_proof.py")
    mod = importlib.util.module_from_spec(rp)
    rp.loader.exec_module(mod)
    assert mod.main([]) == 2 and mod.main(["bitsandbytes"]) == 2, "bad usage exits 2"
    with pytest.raises(mod.ReleaseProofError, match="not installed"):
        mod.prove("no-such-dist-p236", "0.0.0")


def test_bytecode_only_under_a_fresh_prefix_from_the_first_python_call():
    """R2: every python the box runs reads and writes bytecode only under a fresh, empty PYTHONPYCACHEPREFIX, exported
    before the first python call and recorded; the bench records it and the reducer requires it."""
    i = RUN.index("export PYTHONPYCACHEPREFIX=$(mktemp -d /tmp/p236-pycache.XXXXXX)")
    assert RUN.index("export RELEASE_PROOF_FRESH_PYCACHE=$PYTHONPYCACHEPREFIX") < RUN.index("python $W/release_proof.py"),         "the helper records pycache_fresh only under its marker, set to the same fresh prefix"
    offsets, pos = [], 0
    for ln in RUN.splitlines(keepends=True):                 # the first python CALL (comment lines are not calls)
        if not ln.lstrip().startswith("#") and re.search(r"(^|[ (])python[ -]", ln):
            offsets.append(pos)
        pos += len(ln)
    first_python = min(offsets)
    assert i < first_python and "PYTHONPYCACHEPREFIX $PYTHONPYCACHEPREFIX" in RUN
    assert 'rec["pycache_prefix"] = os.environ.get("PYTHONPYCACHEPREFIX")' in (LANE / "p236_bench.py").read_text(
        encoding="utf-8")
    R = _load("p236_reduce")
    assert R.reduce(R.fake_record(pycache_prefix=None))["verdict"] == "VOID"


def test_no_staged_file_shadows_a_library_module():
    """The proof checks bitsandbytes' bytes and the git pin gnf4's; neither holds if a file the box stages in $W (which
    is sys.path[0] for every python there) shadows one of their top-level modules, or torch's."""
    from importlib import metadata
    staged = {pathlib.PurePosixPath(n).stem for n in STAGED if n.endswith(".py")}
    for dist in ("grouped-nf4-gemm", "bitsandbytes", "torch"):
        try:
            files = metadata.distribution(dist).files or []
        except metadata.PackageNotFoundError:
            pytest.skip(f"{dist} is not installed here")
        tops = {f.parts[0].removesuffix(".py") for f in files if not f.parts[0].endswith((".dist-info", ".data"))
                and (len(f.parts) > 1 or f.parts[0].endswith(".py"))}
        assert tops, dist
        assert not staged & tops, (dist, staged & tops)


def test_every_install_skips_bytecode_so_the_release_proof_can_hold():
    """The prefix is exported before the installs, so pip's default compile would write .pyc under it and list them in
    the installed RECORD as ../ entries, which the release proof refuses (seen live on bitsandbytes 0.50.2: refused
    with the default, "RECORD equal" with --no-compile). Every pip install on the box passes --no-compile, after the
    prefix is exported."""
    installs = [ln for ln in RUN.splitlines() if "pip install" in ln and not ln.lstrip().startswith("#")]
    assert installs and all("--no-compile" in ln.split() for ln in installs), installs
    assert RUN.index("export PYTHONPYCACHEPREFIX") < RUN.index("pip install")


def _dry_env(tmp_path, run_id):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P236_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800"}


def _drive(tmp_path, run_id):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    return subprocess.run(["bash", str(LANE / "p236_drive.sh")], capture_output=True, text=True,
                          env=_dry_env(tmp_path, run_id), timeout=120)


def _clean_or_skip(dirty):
    """The driver refuses a dirty tree by design, so a dirty local checkout skips the dry run, named. Under CI the
    checkout is fresh: a dirty one is broken and fails, never a silent skip that drops the dry-run asserts."""
    if dirty.strip():
        why = f"the checkout is dirty ({dirty.splitlines()[0][:80]}): the driver refuses a dirty tree by design"
        if os.environ.get("CI"):
            pytest.fail(why + " (CI must run on a clean checkout)")
        pytest.skip(why)


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_the_drivers_dry_run_stages_every_pinned_file(tmp_path, run_id):
    """The driver stages, checks staged-p236.sha256 and reaches its dry-run line on a clean checkout, for every
    registered id. Grepping DRIVE's text never ran its staging loop: a literal backslash-n in that loop (the
    registration's line 22) made bash stage a file named "n" and refuse every launch."""
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    _clean_or_skip(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                                  capture_output=True, text=True).stdout)
    out = _drive(tmp_path, run_id)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"DRYRUN stage {len(STAGED) + 1} files -> root@h:/root/p236" in out.stdout, out.stdout
    assert "run p236_run.sh" in out.stdout and f"fetch -> {tmp_path}/p236" in out.stdout


@pytest.mark.parametrize("run_id", ["p236-5090-3", "p236-5090-r4", "p235-5090-1", "p236-shake-1"])
def test_the_driver_refuses_an_unregistered_id(tmp_path, run_id):
    out = _drive(tmp_path, run_id)
    assert out.returncode == 78 and f"refusing unregistered run ID: {run_id}" in out.stdout, out.stdout + out.stderr


def test_every_lane_shell_script_parses():
    """``bash -n`` on every bench/p236/*.sh: a parse check catches the next defect of another shape (it does not catch
    the backslash-n, which parses as a word; the dry run and the check below do)."""
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("needs a POSIX bash")
    scripts = sorted(LANE.glob("*.sh"))
    assert {s.name for s in scripts} >= {"p236_drive.sh", "p236_run.sh"}, scripts
    for sh in scripts:
        out = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, f"{sh.name}: {out.stderr}"


def test_no_lane_script_carries_a_mangled_line_continuation():
    """A backslash-n followed by indentation is a line continuation an editor wrote as two characters: bash reads it as
    the word "n" (bash -n passes it), so it is refused in every shell script of the lane."""
    for sh in sorted(LANE.glob("*.sh")):
        for i, line in enumerate(sh.read_text(encoding="utf-8").splitlines(), 1):
            assert not re.search(r"\\n {2,}\S", line), f"{sh.name}:{i}: {line.strip()[:120]}"


# ---- Amendment 2: the gnf4 engagement gate under K33's split-K plan, on the VOID box's real records first
FIXTURE = LANE / "fixtures" / "p236-5090-1-engaged.json"
FIXTURE_SHA256 = "ff40fbb31802cc16dc8e3fa12f8978f5eca93c12d5333c2a4a44348ddd94d61e"
SOURCE_SHA256 = "6e60465f55c867c3eee832f306c7af4b6744b96d7a7c80caa4ea598c4116c0b2"   # the receipt's bench_p236.json


def _engagement_faults(R, key, g):
    return [f for f in R.cell_faults(key, {"engaged": {"gnf4": g}}) if "gnf4 engagement" in f]


def _real_records():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256, "the fixture is not the pinned extract"
    rec = json.loads(raw)
    assert rec["source_sha256"] == SOURCE_SHA256 and len(rec["engaged_gnf4"]) == 30
    return rec["engaged_gnf4"]


def test_the_void_boxs_real_records_pass_the_amended_gate():
    """p236-5090-1's 30 engagement records (the gate's input, no timing): under Amendment 2 every cell passes,
    mixtral.down's five with bw_splitk == calls beside bw_prmt32 == calls."""
    R = _load("p236_reduce")
    recs = _real_records()
    assert {k for k, g in recs.items() if g["bw_splitk"]} == {f"mixtral.down|M{m}" for m in R.MS}
    assert {k: _engagement_faults(R, k, g) for k, g in recs.items()} == {k: [] for k in recs}


def test_the_old_rule_fails_on_the_real_records(monkeypatch):
    """The registration's rule (no split-K shape) refuses exactly the five mixtral.down cells, as the box did."""
    R = _load("p236_reduce")
    monkeypatch.setattr(R, "SPLITK", {})
    recs = _real_records()
    assert {k for k, g in recs.items() if _engagement_faults(R, k, g)} == {f"mixtral.down|M{m}" for m in R.MS}


@pytest.mark.parametrize("case,key,edit", [
    ("the split mark missing", "mixtral.down|M4", {"bw_splitk": 0}),
    ("the split mark twice a call", "mixtral.down|M4", {"bw_splitk": 6}),
    ("the split mark short", "mixtral.down|M1", {"bw_splitk": 2}),
    ("two launches a call on the split shape", "mixtral.down|M8", {"bw_prmt32": 6}),
    ("a split mark on a split-1 shape", "qwen3.down|M4", {"bw_splitk": 3}),
    ("another route beside K33", "mixtral.down|M2", {"dotpad": 3}),
    ("no calls", "mixtral.down|M16", {"calls": 0, "bw_prmt32": 0, "bw_splitk": 0}),
])
def test_the_amended_gate_still_refuses(case, key, edit):
    R = _load("p236_reduce")
    g = {**_real_records()[key], **edit}
    assert _engagement_faults(R, key, g), case


def test_the_registered_split_is_the_pinned_kernels_plan(monkeypatch):
    """The check before any box: R.SPLITK equals grouped-nf4-gemm 0.51.0's own plan table at every read shape (CI
    installs the pinned commit), so a drift fails here, not on a rented card."""
    ng = pytest.importorskip("nf4_grouped")
    import importlib.metadata as md
    R = _load("p236_reduce")
    assert md.version("grouped-nf4-gemm") == R.GNF4_VERSION
    monkeypatch.delenv("GNF4_GEMV_BW_PLAN", raising=False)
    assert {name: ng._bw_plan(n, k)[3] for name, (n, k) in R.SHAPES.items()} == {
        name: R.SPLITK.get(name, 1) for name in R.SHAPES}
    assert R.SPLITK == {"mixtral.down": 2}


def test_the_box_checks_the_split_plans_after_the_install():
    """The runner's tripwire asserts the same table on the card, after the install and before anything is timed."""
    i = RUN.index("plans = {name: ng._bw_plan(n, k)[3] for name, (n, k) in R.SHAPES.items()}")
    assert RUN.index("pipx logs/pip_gnf4.log") < i < RUN.index("p236_bench.py --self-test")
    assert "assert plans == {name: R.SPLITK.get(name, 1) for name in R.SHAPES}" in RUN
