"""The p108 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P108).

`bench/p108/p108_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p108/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test (17 cases) and pins the lane's shape:
- P97's box (whose helpers P108's box imports) at P97's registered bytes;
- the order: install and tripwire (#964 and #966 present), the reducer self-test, the premise on the card, then (and only
  then) the fetch, the box and the reducer;
- the refusals: card class, disk, host RAM (#344), each before anything is installed;
- the box's registered shape and the rule's constants;
- the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p108"
P97 = REPO / "bench" / "p97"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p108_run.sh": LANE / "p108_run.sh",
    "p108_reduce.py": LANE / "p108_reduce.py",
    "p108_box.py": LANE / "p108_box.py",
    "p97_box.py": P97 / "p97_box.py",
    "test_gemma4_paged_window_gpu.py": REPO / "tests" / "test_gemma4_paged_window_gpu.py",
}
RUN = (LANE / "p108_run.sh").read_text()
REDUCE = (LANE / "p108_reduce.py").read_text()
BOX = (LANE / "p108_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p108 launch refuses ON A RENTED BOX"


def test_p97s_box_runs_at_p97s_registered_bytes():
    p97 = dict((n, w) for w, n in _entries(P97 / "staged.sha256"))
    mine = dict((n, w) for w, n in _entries())
    assert mine["p97_box.py"] == p97["p97_box.py"]


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p108_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p108_run.sh":
            assert name in staged, name
    assert "--exclude 'work'" in driver                                    # nothing but receipts travels


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p108_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (17 cases)" in out.stdout, out.stdout + out.stderr
    assert "TOL, SPREAD_X, SPREAD_MIN = 0.05, 2.0, 0.01" in REDUCE and "WINDOWS = 32" in REDUCE
    assert 'FLOORS = ("oneshot", "chunk", "batch")' in REDUCE
    rev = re.search(r'REV = "([0-9a-f]{40})"', REDUCE).group(1)
    assert f"REV={rev}" in RUN and "MODEL=google/gemma-4-26B-A4B-it;" in RUN


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["REFUSED: card is", "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @', "python - <<'PYT'",
             "p108_reduce.py --self-test", "python -m pytest test_gemma4_paged_window_gpu.py", 'echo "premise ok"',
             'if [ "$PROVE" = 1 ]; then', 'say "fetch $MODEL @ $REV"', "python $W/p108_box.py",
             "python $W/p108_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'grep -q "2 passed" && ! echo "$LASTL" | grep -q skipped' in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    assert 'hasattr(paged_attention, "_fallback_mask")' in trip and "_kv_geometry(cfg) == ([2, 1], [32, 64])" in trip
    prove = RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index("SRC=$MODEL;")]
    assert ": > PROVED; finish 0" in prove and "snapshot_download" not in prove


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes
    assert {16, 25} <= codes                                                # host RAM, premise


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p108-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P108_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p108_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p108"), out.stdout + out.stderr


def test_the_box_reads_the_registered_shape():
    for flag in ('"--windows", type=int, default=32', '"--group", type=int, default=8', '"--prompt", type=int, default=1280',
                 '"--cont", type=int, default=256', '"--chunk", type=int, default=256'):
        assert flag in BOX, flag
    assert 'FLOORS = ("oneshot", "chunk", "batch")' in BOX
    assert 'sm_scale = (sm_scale if sm_scale is not None else q.shape[-1] ** -0.5) * 0.5' in BOX
    assert "lps, _calls, _states, runner, step_ms = p97_box._paged_pass(" in BOX
