"""The p102 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P102).

`bench/p102/p102_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p102/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test (14 cases) and the box instruments' (4), and pins
the A/B: P100's census, SC1's TTFT helpers and prompt dump, P39's step_decomp and calibration, P98's bake and P42's
hook staged at their lanes' registered bytes; the premise test staged from tests/; SC1's stack and token ids; the four
routes and their windows; the premise before the fetch; the tripwire on the knob; and the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p102"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p102_run.sh": LANE / "p102_run.sh",
    "p102_box.py": LANE / "p102_box.py",
    "p102_reduce.py": LANE / "p102_reduce.py",
    "p100_box.py": REPO / "bench" / "p100" / "p100_box.py",
    "sc1_e4b_sched.py": REPO / "bench" / "sc1" / "sc1_e4b_sched.py",
    "sc1_prompts.py": REPO / "bench" / "sc1" / "sc1_prompts.py",
    "step_decomp.py": REPO / "bench" / "p39" / "step_decomp.py",
    "p98_bake.py": REPO / "bench" / "p98" / "p98_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "p42_usercustomize.py": REPO / "bench" / "p42" / "hook" / "usercustomize.py",
    "test_int4_prefill_route_gpu.py": REPO / "tests" / "test_int4_prefill_route_gpu.py",
}
RUN = (LANE / "p102_run.sh").read_text()
BOX = (LANE / "p102_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p102 launch refuses ON A RENTED BOX"


def test_borrowed_pieces_run_at_their_lanes_registered_bytes():
    sc1 = dict((n, w) for w, n in _entries(REPO / "bench" / "sc1" / "staged.sha256"))
    p98 = dict((n, w) for w, n in _entries(REPO / "bench" / "p98" / "staged.sha256"))
    p100 = dict((n, w) for w, n in _entries(REPO / "bench" / "p100" / "staged.sha256"))
    mine = dict((n, w) for w, n in _entries())
    for name in ("sc1_e4b_sched.py", "sc1_prompts.py", "step_decomp.py", "calib.json"):
        assert mine[name] == sc1[name], name
    assert mine["p42_usercustomize.py"] == sc1["hook/usercustomize.py"]
    assert mine["p98_bake.py"] == p98["p98_bake.py"]
    assert mine["p100_box.py"] == p100["p100_box.py"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p102_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p102_run.sh":
            assert name in staged, name
    assert "cp $W/p42_usercustomize.py $W/hook/usercustomize.py" in staged
    assert "--include 'work_qwen3/bake.json' --exclude 'work_qwen3/*'" in driver   # the arena stays on the box


def test_the_reducer_and_the_box_instruments_pass_their_self_tests():
    for script, want in (("p102_reduce.py", "self-test OK (14 cases)"), ("p102_box.py", "self-test OK (4 cases)")):
        out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True)
        assert out.returncode == 0 and want in out.stdout, script + out.stdout + out.stderr


def test_the_stack_and_the_token_ids_are_sc1s_and_the_knob_starts_unset():
    sc1 = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()

    def value(text, var):
        return re.search(rf'^{var}="([^"]*)"', text, re.M).group(1)

    for var in ("FOLDS", "SPEEDENV", "ROUTEENV"):
        assert value(RUN, var) == value(sc1, var), var
    assert "SHA_B1=a8e6ea1d7d140dbe726c94f0e6325eb11457a93f196419483de79507272f95e3" in RUN
    assert "SHA_B1_4096=cd70a142d533eb88a3d2bec79bc4b11539f3a08b30c20ba5fbba248fb9dbafdd" in RUN
    assert "MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in RUN
    unset = RUN[RUN.index("# every serving lever starts unset"):RUN.index("# SC1's stack, byte for byte")]
    assert "E4B_INT4_PREFILL" in unset


def test_the_arms_and_the_windows_are_the_registered_ab():
    assert "ARMS=${P102_ARMS:-ttft nll}" in RUN
    ttft = RUN[RUN.index("ttft_arm(){"):RUN.index("nll_arm(){")]
    for kv in ("E4B_PAGED_MAX_SEQS=1", "E4B_PAGED_CHUNK_TOKENS=512", "E4B_PAGED_GRAPHS=1", "E4B_PAGED_BUCKETS=1",
               "E4B_PAGED_FUSE_QKV=1", "env PYTHONPATH= $ROUTEENV $SPEEDENV", "python $W/p102_box.py ttft"):
        assert kv in ttft, kv
    nll = RUN[RUN.index("nll_arm(){"):RUN.index("for N in $ARMS; do")]
    for flag in ("env PYTHONPATH=$W/hook $ROUTEENV $SPEEDENV", "--ppl-oracle eager", "--ppl-steps 2048", "--prompt-len 512",
                 "--no-fuse-qkv", "--b1d-loop eager", "--batch 1"):
        assert flag in nll, flag
    assert 'ROUTES = ("loop", "batched", "k19", "mtile")' in BOX
    assert 'WINDOWS = (("c4val1", (9, 10, 11, 12, 13, 14, 15, 16)), ("wikitext", (9, 10, 11, 12)))' in BOX
    assert "STRIDE, SPAN = 4096, 2600" in BOX and "ROUNDS = 3" in BOX


def test_the_order_the_premise_and_the_tripwire():
    assert RUN.index('say "install e4b @') < RUN.index("TRIPWIRE FAIL") \
        < RUN.index("python -m pytest test_int4_prefill_route_gpu.py") < RUN.index('say "fetch $MODEL @ $REV"') \
        < RUN.index('say "bake qwen3"') < RUN.index("PROMPTS_MATCH_SC1") < RUN.index("for N in $ARMS; do") \
        < RUN.index("python $W/p102_reduce.py --dir $W --out $W/verdict.json")
    assert 'grep -q "2 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    assert 'hr.INT4_PREFILL_ROUTES == ("loop", "batched", "k19", "mtile")' in RUN
    assert 'hr._int4_prefill_mode_env() == "loop"' in RUN
    assert "from int4_smallm import gemm_int4_b32_grouped_smallm" in RUN
    assert "PROVE" not in RUN                                                # the guard is 1 h: no proving run


def test_the_tripwire_strings_are_the_librarys_code():
    hr = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text()
    assert 'INT4_PREFILL_ROUTES = ("loop", "batched", "k19", "mtile")' in hr
    assert 'os.environ.get("E4B_INT4_PREFILL", "loop")' in hr
    assert 'w = dequant_int4_ref(st["packed"][e_],' in hr


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return|rec) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p102-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P102_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p102_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p102"), out.stdout + out.stderr


def test_the_install_carries_what_the_premise_runs():
    """A1: ``p102-5090-1`` stopped at the premise with "No module named pytest" -- the install line, derived from P100's,
    had none. Whatever the premise runs under must be installed, and imported by the tripwire before the premise."""
    install = RUN[RUN.index('say "install e4b @'):RUN.index("TRIPWIRE FAIL")]
    assert "python -m pytest test_int4_prefill_route_gpu.py" in RUN
    assert re.search(r'"huggingface_hub>=0\.23" pytest \|\|', install), "pytest is not on the install line"
    start = RUN.index("python - <<'PYT'")
    tripwire = RUN[start:RUN.index("\nPYT\n", start)]                       # the heredoc's body
    assert "import pytest" in tripwire
    assert RUN.index("import pytest") < RUN.index("python -m pytest test_int4_prefill_route_gpu.py")


def test_the_loop_engagement_check_is_structural():
    """A2: ``p102-5090-5`` was voided by a 9,600-decode floor that P100 had already measured as wrong (8,390). The check
    is two decodes per distinct expert over 48 loop calls, with a floor of 32 distinct experts per layer."""
    red = (LANE / "p102_reduce.py").read_text()
    assert "MIN_LOOP_DEQUANT = 2 * 32 * 48" in red and "9600" not in red
    assert 'c.get("loop_dequant_is_two_per_distinct") is True' in red
    assert '"dequant_total": 8390' in red                                   # the fixture is the measured census
