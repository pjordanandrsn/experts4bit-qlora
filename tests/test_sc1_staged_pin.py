"""The SC1 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for SC1; the k22 pattern).

`bench/sc1/sc1_drive.sh` refuses to run when a staged file's sha256 differs from `bench/sc1/staged.sha256`. That guard runs
on the CONTROLLER after a box is rented; this test runs the same comparison in CI, where it costs nothing. It mirrors the
driver's resolution `case` (the lane's own files, the comparator driver directories, P39's harness pieces, P42's hook under
hook/, P88's premise test from tests/), asserts that the set of pinned names equals the set the driver stages, that the
harness is P86's (and P88's) pinned bytes, that `make_pin.sh` reproduces the pin, and that the driver reaches its dry run.
"""
import hashlib
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "sc1"
PIN = LANE / "staged.sha256"
OWN = ("sc1_run.sh", "sc1_e4b_sched.py", "sc1_prompts.py", "sc1_sampler.sh")
COMP_DIRS = ("vllm", "sglang", "llamacpp", "exl3", "lmdeploy")
SC1B = tuple("sc1b_census.py sc1b_e4b_census.py sc1b_vllm_census.py sc1b_serve_census.py sc1b_toy.py kernel_classes.json sc1b_box_d.sh".split())                         # bench/sc1b, staged flat on every box
SC2 = tuple("sc2_driver.py sc2_prompts.py sc2_reduce.py sc2_box_e.sh sc2_identity.py sc2b_box_f.sh sc2b_reduce.py sc2_trace.py sc2g_box_g.sh sc2g_reduce.py sc1g_box_i.sh sc1g_reduce.py sc1g_k8.py sc1g_gemv_check.py sc2c_box_h.sh sc2c_reduce.py sc2c_census.py".split())                                                                                            # bench/sc2, staged flat on every box
P39 = ("step_decomp.py", "k8_bake.py", "calib.json")


def resolve(name: str) -> pathlib.Path:
    """sc1_drive.sh's `case`, mirrored."""
    if name in OWN or name == "sc1_reduce.py":
        return LANE / name
    if name.split("/")[0] in COMP_DIRS:
        return LANE / name
    if name == "hook/usercustomize.py":
        return REPO / "bench" / "p42" / "hook" / "usercustomize.py"
    if name == "test_k19_row_exact_gpu.py":
        return REPO / "tests" / name
    if name in SC1B:
        return REPO / "bench" / "sc1b" / name
    if name in SC2:
        return REPO / "bench" / "sc2" / name
    return REPO / "bench" / "p39" / name


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            want, name = line.split(None, 1)
            yield want, name.strip()


def staged_names() -> set:
    """What make_pin.sh pins and sc1_drive.sh stages: the lane's own files, the reducer when present, P39's pieces, the hook,
    the premise test, and every file under each comparator directory that exists."""
    names = set(OWN) | set(P39) | set(SC1B) | set(SC2) | {"hook/usercustomize.py", "test_k19_row_exact_gpu.py"}
    if (LANE / "sc1_reduce.py").is_file() and any(n == "sc1_reduce.py" for _w, n in _entries()):
        names.add("sc1_reduce.py")            # pinned by default when present (make_pin.sh); staged by path either way
    for d in COMP_DIRS:
        if (LANE / d).is_dir():
            for p in (LANE / d).rglob("*"):
                if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts and p.name != ".DS_Store":
                    names.add(str(p.relative_to(LANE)))
    return names


@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and "." in v and len(v) < 60 else "sha")
def test_staged_file_matches_its_pin(want, name):
    src = resolve(name)
    assert src.is_file(), f"{name} resolves to {src}, which does not exist"
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, (f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); run bench/sc1/make_pin.sh -- "
                         f"the next SC1 launch would refuse ON A RENTED BOX")


def test_pinned_names_equal_the_staged_set():
    pinned = {n for _w, n in _entries()}
    assert pinned == staged_names(), {"pinned_not_staged": sorted(pinned - staged_names()), "staged_not_pinned": sorted(staged_names() - pinned)}
    assert "# sc1_reduce.py is pinned whenever it is present" in PIN.read_text()      # the reducer is pinned by default since integration


def test_every_pinned_name_is_staged_by_the_driver_and_resolves_the_same_way():
    driver = (LANE / "sc1_drive.sh").read_text()
    for name in OWN:
        assert f"$HERE/{name}" in driver, name
    assert "$HERE/sc1_reduce.py" in driver and '[ -s "$HERE/sc1_reduce.py" ] && STAGE=' in driver
    for var in ("$P39/step_decomp.py", "$P39/k8_bake.py", "$P39/calib.json", "$TESTS/test_k19_row_exact_gpu.py", "$P42/hook/usercustomize.py"):
        assert var in driver, var
    assert "for d in vllm sglang llamacpp exl3 lmdeploy; do" in driver and "COPYFILE_DISABLE=1 tar -C \"$HERE\"" in driver
    # the driver's case resolves every pinned name exactly as resolve() does
    case = driver[driver.index("while read -r want name; do"):driver.index('done < "$HERE/staged.sha256"')]
    assert "sc1_run.sh|sc1_e4b_sched.py|sc1_prompts.py|sc1_sampler.sh|sc1_reduce.py) src=\"$HERE/$name\"" in case
    assert "vllm/*|sglang/*|llamacpp/*|exl3/*|lmdeploy/*) src=\"$HERE/$name\"" in case
    assert 'hook/usercustomize.py) src="$P42/hook/usercustomize.py"' in case and 'test_k19_row_exact_gpu.py) src="$TESTS/$name"' in case
    assert 'sc1b_*|kernel_classes.json) src="$SC1B/$name"' in case
    assert 'sc2_*|sc2b_*|sc2g_*|sc1g_*|sc2c_*) src="$SC2/$name"' in case
    assert '*) src="$P39/$name"' in case
    # the box checks the same file with sha256sum -c (strict: a pinned file missing on the box is a stop)
    run = (LANE / "sc1_run.sh").read_text()
    assert "sha256sum -c staged.sha256 >/dev/null" in run and "--ignore-missing" not in run


def test_the_harness_is_p86s_pinned_bytes():
    """step_decomp.py / k8_bake.py / calib.json / the hook are the bytes P86 and P88 staged (the register's numbers were read on them);
    the premise test is P88's pinned bytes."""
    mine = {n: w for w, n in _entries()}
    p86 = {n: w for w, n in _entries(REPO / "bench" / "p86" / "staged.sha256")}
    p88 = {n: w for w, n in _entries(REPO / "bench" / "p88" / "staged.sha256")}
    for name in ("step_decomp.py", "k8_bake.py", "calib.json", "hook/usercustomize.py"):
        assert mine[name] == p86[name] == p88[name], name
    assert mine["test_k19_row_exact_gpu.py"] == p88["test_k19_row_exact_gpu.py"]


def test_make_pin_reproduces_the_pin(tmp_path):
    out = tmp_path / "staged.sha256"
    r = subprocess.run(["bash", str(LANE / "make_pin.sh"), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert dict(_entries(out)) == dict(_entries()), "make_pin.sh does not reproduce the committed pin -- re-run it and commit"


def _dry_env(tmp_path, **extra):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "sc1-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "SC1_DRIVE_DRYRUN": "1", "SC1_BOX": "A"}
    env.update(extra)
    return env


def test_the_driver_runs_to_its_dry_run(tmp_path):
    out = subprocess.run(["bash", str(LANE / "sc1_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, SC1_PROVE="1", SC1_CALIB_NSEQ="64", SC1_PROVE_SGLANG_MODEL="org/model@deadbeef"))
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.startswith("DRYRUN stage [sc1_run.sh sc1_e4b_sched.py sc1_prompts.py sc1_sampler.sh step_decomp.py k8_bake.py calib.json "
                                 "test_k19_row_exact_gpu.py staged.sha256 "), out.stdout
    assert "hook/usercustomize.py dirs: vllm sglang llamacpp exl3 lmdeploy] -> root@h:/root/sc1" in out.stdout
    assert " SC1_BOX=A SC1_RUN_ID=sc1-dry SC1_RUN_NONCE=" in out.stdout and " E4B_SHA=" + "0" * 40 in out.stdout
    assert " SC1_PROVE=1 " in out.stdout and " SC1_CALIB_NSEQ=64 " in out.stdout and " SC1_PROVE_SGLANG_MODEL=org/model@deadbeef " in out.stdout
    assert " bash sc1_run.sh ; poll TP_DONE." in out.stdout and out.stdout.rstrip().endswith("fetch -> " + str(tmp_path) + "/sc1")
    assert not out.stderr.strip(), out.stderr


def test_the_driver_refuses_without_a_box_or_with_a_bad_one(tmp_path):
    env = _dry_env(tmp_path)
    del env["SC1_BOX"]
    out = subprocess.run(["bash", str(LANE / "sc1_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 78 and "SC1_BOX is not set" in out.stdout
    out = subprocess.run(["bash", str(LANE / "sc1_drive.sh")], capture_output=True, text=True, env=_dry_env(tmp_path, SC1_BOX="K"))
    assert out.returncode == 78 and "SC1_BOX must be A, B, C, D, E, F, G, H, I or J" in out.stdout    # D SC1b, E SC2, F SC2b, G SC2g, H SC2c, I SC1g, J SC1g-diag
