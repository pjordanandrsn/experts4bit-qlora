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
