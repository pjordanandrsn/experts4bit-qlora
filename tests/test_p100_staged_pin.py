"""The p100 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P100).

`bench/p100/p100_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p100/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test (13 cases) and the box wrapper's (3), and pins the
confirmation: SC1's TTFT arm, prompt dump, P39's step_decomp and calibration, P98's bake and P42's hook staged at their
lanes' registered bytes; SC1's stack and token ids; the five arms, their chunks and their order; the tripwire on the
code the trace read; the timed requests running the library unwrapped; and the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p100"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p100_run.sh": LANE / "p100_run.sh",
    "p100_box.py": LANE / "p100_box.py",
    "p100_reduce.py": LANE / "p100_reduce.py",
    "sc1_e4b_sched.py": REPO / "bench" / "sc1" / "sc1_e4b_sched.py",
    "sc1_prompts.py": REPO / "bench" / "sc1" / "sc1_prompts.py",
    "step_decomp.py": REPO / "bench" / "p39" / "step_decomp.py",
    "p98_bake.py": REPO / "bench" / "p98" / "p98_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "p42_usercustomize.py": REPO / "bench" / "p42" / "hook" / "usercustomize.py",
}
RUN = (LANE / "p100_run.sh").read_text()
BOX = (LANE / "p100_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p100 launch refuses ON A RENTED BOX"


def test_borrowed_pieces_run_at_their_lanes_registered_bytes():
    sc1 = dict((n, w) for w, n in _entries(REPO / "bench" / "sc1" / "staged.sha256"))
    p98 = dict((n, w) for w, n in _entries(REPO / "bench" / "p98" / "staged.sha256"))
    mine = dict((n, w) for w, n in _entries())
    for name in ("sc1_e4b_sched.py", "sc1_prompts.py", "step_decomp.py", "calib.json"):
        assert mine[name] == sc1[name], name
    assert mine["p42_usercustomize.py"] == sc1["hook/usercustomize.py"]
    assert mine["p98_bake.py"] == p98["p98_bake.py"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p100_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p100_run.sh":
            assert name in staged, name
    assert "cp $W/p42_usercustomize.py $W/hook/usercustomize.py" in staged      # never importable by its staged name
    assert "--include 'work_qwen3/bake.json' --exclude 'work_qwen3/*'" in driver  # the arena stays on the box


def test_the_reducer_and_the_box_wrapper_pass_their_self_tests():
    for script, want in (("p100_reduce.py", "self-test OK (13 cases)"), ("p100_box.py", "self-test OK (3 cases)")):
        out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True)
        assert out.returncode == 0 and want in out.stdout, script + out.stdout + out.stderr


def test_the_stack_and_the_token_ids_are_sc1s():
    sc1 = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()

    def value(text, var):
        return re.search(rf'^{var}="([^"]*)"', text, re.M).group(1)

    for var in ("FOLDS", "SPEEDENV", "ROUTEENV"):
        assert value(RUN, var) == value(sc1, var), var
    assert "SHA_B1=a8e6ea1d7d140dbe726c94f0e6325eb11457a93f196419483de79507272f95e3" in RUN
    assert "SHA_B1_4096=cd70a142d533eb88a3d2bec79bc4b11539f3a08b30c20ba5fbba248fb9dbafdd" in RUN
    assert "MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in RUN
    assert "GNF4_SHA=34da93d6fe8d2a401b7001705658ce00b2b18213" in RUN
    ttft = RUN[RUN.index("ttft_arm(){"):RUN.index("prof_arm(){")]
    for kv in ("E4B_PAGED_PLACEMENT=all-vram", "E4B_PAGED_MAX_SEQS=1", "E4B_PAGED_CHUNK_TOKENS=$C", "E4B_PAGED_GRAPHS=1",
               "E4B_PAGED_BUCKETS=1", "E4B_PAGED_FUSE_QKV=1", "E4B_PAGED_TORCH_THREADS=8", "env PYTHONPATH= $ROUTEENV $SPEEDENV"):
        assert kv in ttft, kv
    assert "512) PF=$W/prompts_b1.json; MTS=2048;; 4096) PF=$W/prompts_b1_4096.json; MTS=4104;;" in ttft   # SC1's sizes


def test_the_arms_are_the_registered_confirmation_in_order():
    assert "ARMS=${P100_ARMS:-c512_t512 c2048_t4096 prof c512_t4096 c1024_t4096}" in RUN
    assert '[ "$NAME" = c512_t512 ] && LC=1' in RUN                        # the kernel count runs in one arm only
    prof = RUN[RUN.index("prof_arm(){"):RUN.index("for N in $ARMS; do")]
    assert "env PYTHONPATH=$W/hook $ROUTEENV $SPEEDENV" in prof
    for flag in ("--placement-override all-vram --amort off --batch 1 --prompt-len 512 --chunk 512 --gen-tokens 32 --fuse-qkv",
                 "--cprofile-out $W/cprofile_b1.txt"):
        assert flag in prof, flag


def test_the_order_and_the_tripwire():
    assert RUN.index('say "install e4b @') < RUN.index("TRIPWIRE FAIL") < RUN.index('say "fetch $MODEL @ $REV"') \
        < RUN.index('say "bake qwen3"') < RUN.index("python $W/sc1_prompts.py") < RUN.index("PROMPTS_MATCH_SC1") \
        < RUN.index("for N in $ARMS; do") < RUN.index("python $W/p100_reduce.py --dir $W --out $W/verdict.json")
    assert "w = dequant_int4_ref(st[\"packed\"][e_]," in RUN                  # the loop the trace read is the code installed
    assert "cfg.max_seqs > 1" in RUN and "hr.DEVICE_GROUPING == [False]" in RUN
    assert "site.ENABLE_USER_SITE" in RUN
    assert "PROVE" not in RUN                                                # the guard is 1 h: no proving run


def test_the_tripwire_strings_are_the_librarys_code():
    hr = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text()
    sp = (REPO / "experts4bit_qlora" / "serve_paged.py").read_text()
    assert 'w = dequant_int4_ref(st["packed"][e_],' in hr
    assert "DEVICE_GROUPING = [False]" in hr
    assert "if cfg.graphs and cfg.max_seqs > 1 and max(cfg.buckets) > 1:" in sp


def test_the_timed_requests_run_the_library_unwrapped():
    ttft = BOX[BOX.index("def _ttft()"):BOX.index("def _prof(")]
    assert ttft.index("rc = s.main([\"--ttft\"])") < ttft.index("census.install()")


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return|rec) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p100-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P100_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p100_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p100"), out.stdout + out.stderr
