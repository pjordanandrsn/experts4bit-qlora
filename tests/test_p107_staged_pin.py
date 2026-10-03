"""The p107 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P107).

`bench/p107/p107_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p107/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test (7 cases) and the box instrument's (4), and pins
the A/B: SC1's TTFT helpers and prompt dump, P39's step_decomp and calibration and P98's bake staged at their lanes'
registered bytes; the premise test staged from tests/; SC1's stack and token ids; the two routes, the windows and the
served-prefill scorer's shape; the premise before the fetch; the tripwire on the knob; and the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p107"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p107_run.sh": LANE / "p107_run.sh",
    "p107_box.py": LANE / "p107_box.py",
    "p107_reduce.py": LANE / "p107_reduce.py",
    "sc1_e4b_sched.py": REPO / "bench" / "sc1" / "sc1_e4b_sched.py",
    "sc1_prompts.py": REPO / "bench" / "sc1" / "sc1_prompts.py",
    "step_decomp.py": REPO / "bench" / "p39" / "step_decomp.py",
    "p98_bake.py": REPO / "bench" / "p98" / "p98_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_paged_prefill_attn_route_gpu.py": REPO / "tests" / "test_paged_prefill_attn_route_gpu.py",
}
RUN = (LANE / "p107_run.sh").read_text()
BOX = (LANE / "p107_box.py").read_text()
RED = (LANE / "p107_reduce.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p107 launch refuses ON A RENTED BOX"


def test_borrowed_pieces_run_at_their_lanes_registered_bytes():
    sc1 = dict((n, w) for w, n in _entries(REPO / "bench" / "sc1" / "staged.sha256"))
    p98 = dict((n, w) for w, n in _entries(REPO / "bench" / "p98" / "staged.sha256"))
    mine = dict((n, w) for w, n in _entries())
    for name in ("sc1_e4b_sched.py", "sc1_prompts.py", "step_decomp.py", "calib.json"):
        assert mine[name] == sc1[name], name
    assert mine["p98_bake.py"] == p98["p98_bake.py"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p107_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p107_run.sh":
            assert name in staged, name
    assert "usercustomize" not in RUN and "HOOK" not in driver               # no lever hook: the knob is the library's
    assert "--include 'work_qwen3/bake.json' --exclude 'work_qwen3/*'" in driver   # the arena stays on the box


def test_the_reducer_and_the_box_instrument_pass_their_self_tests():
    for script, want in (("p107_reduce.py", "self-test OK (7 cases)"), ("p107_box.py", "self-test OK (4 cases)")):
        out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True)
        assert out.returncode == 0 and want in out.stdout, script + out.stdout + out.stderr


def test_the_stack_and_the_token_ids_are_sc1s_and_the_knobs_start_unset():
    sc1 = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()

    def value(text, var):
        return re.search(rf'^{var}="([^"]*)"', text, re.M).group(1)

    for var in ("FOLDS", "SPEEDENV", "ROUTEENV"):
        assert value(RUN, var) == value(sc1, var), var
    assert "SHA_B1=a8e6ea1d7d140dbe726c94f0e6325eb11457a93f196419483de79507272f95e3" in RUN
    assert "SHA_B1_4096=cd70a142d533eb88a3d2bec79bc4b11539f3a08b30c20ba5fbba248fb9dbafdd" in RUN
    assert "MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in RUN
    unset = RUN[RUN.index("# every serving lever starts unset"):RUN.index("# SC1's stack, byte for byte")]
    for knob in ("E4B_PAGED_PREFILL_ATTN", "E4B_INT4_PREFILL"):
        assert knob in unset, knob
    for sha in ("a8e6ea1d7d140dbe726c94f0e6325eb11457a93f196419483de79507272f95e3",
                "cd70a142d533eb88a3d2bec79bc4b11539f3a08b30c20ba5fbba248fb9dbafdd"):
        assert sha in RED, sha                                               # the reducer checks the same digests


def test_the_arm_is_one_engine_and_the_registered_ab():
    assert "ARMS=${P107_ARMS:-ab}" in RUN
    ab = RUN[RUN.index("ab_arm(){"):RUN.index("for N in $ARMS; do")]
    for kv in ("E4B_PAGED_MAX_SEQS=1", "E4B_PAGED_CHUNK_TOKENS=512", "E4B_PAGED_GRAPHS=1", "E4B_PAGED_BUCKETS=1",
               "E4B_PAGED_FUSE_QKV=1", "E4B_PAGED_PLACEMENT=all-vram", "env PYTHONPATH= $ROUTEENV $SPEEDENV",
               "python $W/p107_box.py ab"):
        assert kv in ab, kv
    assert "E4B_PAGED_PREFILL_ATTN" not in ab                                # the box switches it per request
    assert 'ROUTES = ("math", "flash")' in BOX and 'ROUTES = ("math", "flash")' in RED
    assert 'WINDOWS = (("c4val1", (9, 10, 11, 12, 13, 14, 15, 16)), ("wikitext", (9, 10, 11, 12)))' in BOX
    assert 'WINDOWS = {"c4val1": (9, 10, 11, 12, 13, 14, 15, 16), "wikitext": (9, 10, 11, 12)}' in RED
    assert "STRIDE, SPAN = 4096, 2600" in BOX and "ROUNDS = 3" in BOX
    assert "PROMPT_LEN, STEPS, CHUNK = 512, 2048, 512" in BOX
    assert "BUDGET, LAYERS, GAIN = 0.05, 48, 0.9" in RED


def test_the_scorer_is_the_served_prefill_and_leaves_no_state():
    """The quality instrument runs the paged PREFILL path the runner's run_prefill runs (its mode switch, context and
    forward), scores 2,048 positions, and drops the staging instead of flushing it into the pool. Its numbers are
    tests/test_p107_served_prefill_scorer.py's."""
    score = BOX[BOX.index("def served_prefill_nll("):BOX.index("def _ab() -> int:")]
    for s in ("mode(True)", 'ctx.mode, ctx.slots = "prefill", [slot]', "prev = set_context(ctx)",
              "model(input_ids=x[None], position_ids=pos[None], use_cache=False)", "set_context(prev)",
              "ctx.drop(slot)", 'ctx.mode = "decode"', "mode(False)"):
        assert s in score, s
    assert "flush" not in score.replace("instead of flushed", "") and "kv.append" not in score
    ab = BOX[BOX.index("def _ab() -> int:"):]
    assert "served_prefill_nll(torch, runner.model, runner.ctx, set_context, runner._mode, ids, runner.device)" in ab
    assert "assert n == STEPS, n" in ab
    runner = (REPO / "experts4bit_qlora" / "engines" / "paged_runner.py").read_text()
    pre = runner[runner.index("    def run_prefill(self, chunks):"):runner.index("    def run_decode(self, rids):")]
    for s in ("self._mode(True)", 'self.ctx.mode = "prefill"', "self.ctx.slots = [slot]",
              "self.model(input_ids=ids[None],", "position_ids=pos[None], use_cache=False)",
              'self.ctx.mode = "decode"', "self._mode(False)"):
        assert s in pre, s                                                    # the path the scorer mirrors


def test_the_order_the_premise_and_the_tripwire():
    assert RUN.index('say "install e4b @') < RUN.index("TRIPWIRE FAIL") \
        < RUN.index("python -m pytest test_paged_prefill_attn_route_gpu.py") < RUN.index('say "fetch $MODEL @ $REV"') \
        < RUN.index('say "bake qwen3"') < RUN.index("PROMPTS_MATCH_SC1") < RUN.index("for N in $ARMS; do") \
        < RUN.index("python $W/p107_reduce.py --dir $W --out $W/verdict.json")
    assert 'grep -q "1 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    assert 'pa.PREFILL_ATTN_ROUTES == ("math", "flash")' in RUN
    assert 'pa._prefill_attn_mode_env() == "math"' in RUN
    assert 'hr._int4_prefill_mode_env() == "k19"' in RUN
    assert "from torch.nn.attention.bias import causal_lower_right" in RUN
    assert "PROVE" not in RUN                                                # the guard is 1 h: no proving run


def test_the_tripwire_strings_are_the_librarys_code():
    pa = (REPO / "experts4bit_qlora" / "engines" / "paged_attention.py").read_text()
    assert 'PREFILL_ATTN_ROUTES = ("math", "flash")' in pa
    assert 'os.environ.get("E4B_PAGED_PREFILL_ATTN", "math")' in pa
    assert "attn_mask=causal_lower_right(T, t_total), scale=scaling, enable_gqa=True" in pa
    hr = (REPO / "experts4bit_qlora" / "engines" / "hot_residency.py").read_text()
    assert 'os.environ.get("E4B_INT4_PREFILL", "auto")' in hr


def test_the_install_carries_what_the_premise_runs():
    install = RUN[RUN.index('say "install e4b @'):RUN.index("TRIPWIRE FAIL")]
    assert re.search(r'"huggingface_hub>=0\.23" pytest \|\|', install), "pytest is not on the install line"
    start = RUN.index("python - <<'PYT'")
    tripwire = RUN[start:RUN.index("\nPYT\n", start)]
    assert "import pytest" in tripwire
    assert RUN.index("import pytest") < RUN.index("python -m pytest test_paged_prefill_attn_route_gpu.py")


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return|rec) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p107-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P107_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p107_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p107"), out.stdout + out.stderr
