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
COMP_DIRS = ("vllm", "sglang", "llamacpp", "exl3", "lmdeploy", "sc1g_ref")   # sc1g_ref: SC1g box R's registered artifacts (A5), once the sha-registration PR lands
SC1B = tuple("sc1b_census.py sc1b_e4b_census.py sc1b_vllm_census.py sc1b_serve_census.py sc1b_toy.py kernel_classes.json sc1b_box_d.sh".split())                         # bench/sc1b, staged flat on every box
SC2 = tuple("sc2_driver.py sc2_prompts.py sc2_reduce.py sc2_box_e.sh sc2_identity.py sc2b_box_f.sh sc2b_reduce.py sc2_trace.py sc2g_box_g.sh sc2g_reduce.py sc1g_box_i.sh sc1g_reduce.py sc1g_k8.py sc1g_gemv_check.py sc1g_attn_check.py sc1g_kl.py sc2c_box_h.sh sc2c_reduce.py sc2c_census.py sc2d_box_k.sh sc2d_reduce.py sc2e_box_l.sh sc2e_reduce.py sc2e_census.py sc2e_basis.py".split())                                                                                            # bench/sc2, staged flat on every box
P39 = ("step_decomp.py", "k8_bake.py", "calib.json")
P98 = ("p98_bake.py",)                                     # bench/p98: P98's Qwen3.6 arena bake, staged flat (SC2d, box K)
SC5 = tuple("sc5_box_m.sh sc5_driver.py sc5_quality.py sc5_e4b_quality.py sc5_ref.py sc5_windows.py sc5_reduce.py sc5_record.py sc5_windows_w64.json".split())   # bench/sc5, staged flat on every box (SC5, box M)
SC5_LOCKS = ("vllm.lock.txt", "sglang.lock.txt", "e4b-wheels.lock")   # bench/sc5/locks: SC5's provenance locks
SC5_REF = ("sc5_ref.json", "sc5_ref_chunked.json")   # bench/sc5/ref: pinned and staged once the amendment commits them
P117 = ("p117_box.py",)                                   # bench/p117: SC5's decode-shaped quality pass imports it


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
    if name in P98:
        return REPO / "bench" / "p98" / name
    if name in SC5_REF:
        return REPO / "bench" / "sc5" / "ref" / name
    if name in SC5:
        return REPO / "bench" / "sc5" / name
    if name in SC5_LOCKS:
        return REPO / "bench" / "sc5" / "locks" / name
    if name in P117:
        return REPO / "bench" / "p117" / name
    return REPO / "bench" / "p39" / name


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            want, name = line.split(None, 1)
            yield want, name.strip()


def staged_names() -> set:
    """What make_pin.sh pins and sc1_drive.sh stages: the lane's own files, the reducer when present, P39's pieces, the hook,
    the premise test, and every file under each comparator directory that exists."""
    names = (set(OWN) | set(P39) | set(P98) | set(SC1B) | set(SC2) | set(SC5) | set(SC5_LOCKS) | set(P117)
             | {"hook/usercustomize.py", "test_k19_row_exact_gpu.py"})
    names |= {n for n in SC5_REF if (REPO / "bench" / "sc5" / "ref" / n).is_file()}
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
    assert "for d in vllm sglang llamacpp exl3 lmdeploy sc1g_ref; do" in driver and "COPYFILE_DISABLE=1 tar -C \"$HERE\"" in driver
    # the driver's case resolves every pinned name exactly as resolve() does
    case = driver[driver.index("while read -r want name; do"):driver.index('done < "$HERE/staged.sha256"')]
    assert "sc1_run.sh|sc1_e4b_sched.py|sc1_prompts.py|sc1_sampler.sh|sc1_reduce.py) src=\"$HERE/$name\"" in case
    assert "vllm/*|sglang/*|llamacpp/*|exl3/*|lmdeploy/*|sc1g_ref/*) src=\"$HERE/$name\"" in case
    assert case.index("|sc1g_ref/*)") < case.index("|sc1g_*|")   # sc1g_ref/ resolves before bench/sc2's flat sc1g_* files
    assert 'hook/usercustomize.py) src="$P42/hook/usercustomize.py"' in case and 'test_k19_row_exact_gpu.py) src="$TESTS/$name"' in case
    assert 'sc1b_*|kernel_classes.json) src="$SC1B/$name"' in case
    assert 'sc2_*|sc2b_*|sc2g_*|sc1g_*|sc2c_*|sc2d_*|sc2e_*) src="$SC2/$name"' in case
    assert 'p98_bake.py) src="$P98/$name"' in case
    assert 'sc5_ref.json|sc5_ref_chunked.json) src="$SC5/ref/$name"' in case and 'sc5_*) src="$SC5/$name"' in case
    assert case.index("sc5_ref.json|") < case.index("sc5_*)")   # the reference files resolve before the flat sc5_* files
    assert 'vllm.lock.txt|sglang.lock.txt|e4b-wheels.lock) src="$SC5/locks/$name"' in case
    assert 'p117_box.py) src="$REPO/bench/p117/$name"' in case
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
    assert "hook/usercustomize.py dirs: vllm sglang llamacpp exl3 lmdeploy sc1g_ref] -> root@h:/root/sc1" in out.stdout
    assert " SC1_BOX=A SC1_RUN_ID=sc1-dry SC1_RUN_NONCE=" in out.stdout and " E4B_SHA=" + "0" * 40 in out.stdout
    assert " SC1_PROVE=1 " in out.stdout and " SC1_CALIB_NSEQ=64 " in out.stdout and " SC1_PROVE_SGLANG_MODEL=org/model@deadbeef " in out.stdout
    assert " bash sc1_run.sh ; poll TP_DONE." in out.stdout and out.stdout.rstrip().endswith("fetch -> " + str(tmp_path) + "/sc1")
    assert not out.stderr.strip(), out.stderr


def test_the_driver_refuses_without_a_box_or_with_a_bad_one(tmp_path):
    env = _dry_env(tmp_path)
    del env["SC1_BOX"]
    out = subprocess.run(["bash", str(LANE / "sc1_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 78 and "SC1_BOX is not set" in out.stdout
    out = subprocess.run(["bash", str(LANE / "sc1_drive.sh")], capture_output=True, text=True, env=_dry_env(tmp_path, SC1_BOX="Z"))   # a letter no lane will take next (H, then J, collided)
    assert out.returncode == 78 and "SC1_BOX must be A, B, C, D, E, F, G, H, I, J, K, L or M" in out.stdout    # D SC1b, E SC2, F SC2b, G SC2g, H SC2c, I SC1g, J SC1g-diag, K SC2d, L SC2e


def test_the_receipt_fetch_leaves_the_staged_reference_rows_on_the_box(tmp_path):
    """SC1g A5: pull_box's rsync filters (the mid-run pulls and the final fetch) keep box R's staged full-vocabulary rows and
    box I's .f16 copies of them out of the receipt, and still bring the receipts back. Run with the driver's own filter list
    against a box-shaped tree (sc1g-prove-a5-1 pulled 3.9 GB of rows into the receipt store without these)."""
    import re
    import shutil
    import subprocess
    drv = (LANE / "sc1_drive.sh").read_text()
    body = drv[drv.index("pull_box() {"):drv.index('"root@$HOST:$W/" "$1/"; }')]
    filt = re.findall(r"--(exclude|include) '([^']+)'", body)
    assert ("exclude", "sc1g_ref_full/") in filt and ("exclude", "sc1g/ref_full_*.f16") in filt
    assert shutil.which("rsync"), "rsync is needed to run the driver's filters (it is on every CI image and the mini)"
    box, got = tmp_path / "box", tmp_path / "got"
    for rel in ("sc1g_ref_full/ref_full_conv1.npy", "sc1g/ref_full_conv1.f16", "sc1g/kl_nll_llamacpp_q8_decode_conv1.npz",
                "sc1g/k8_window_conv1.json", "sc1g/ref/ref_full_shas.json", "summary.txt", "work_gptoss/bake.json",
                "work_gptoss/nf4.arena", "gguf/model.gguf", "sc1g_ref/ref_full_shas.json"):
        (box / rel).parent.mkdir(parents=True, exist_ok=True)
        (box / rel).write_text("x")
    args = [a for k, v in filt for a in (f"--{k}", v)]
    subprocess.run(["rsync", "-a", *args, f"{box}/", f"{got}/"], check=True, capture_output=True)
    have = sorted(str(p.relative_to(got)) for p in got.rglob("*") if p.is_file())
    assert have == ["sc1g/k8_window_conv1.json", "sc1g/kl_nll_llamacpp_q8_decode_conv1.npz", "sc1g/ref/ref_full_shas.json",
                    "sc1g_ref/ref_full_shas.json", "summary.txt", "work_gptoss/bake.json"], have

