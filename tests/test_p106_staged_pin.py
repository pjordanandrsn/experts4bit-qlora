"""The p106 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P106).

`bench/p106/p106_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p106/staged.sha256`; this test
runs the same comparison in CI. It also runs the reducer's self-test (26 cases) and pins the lane's shape:
- P98's box (whose helpers the box imports) and bake at P98's registered bytes, P39's calibration, and P105's
  kernel-phase premise tests at P105's registered bytes;
- the kernel pins (flash-linear-attention 0.5.2, causal-conv1d 1.7.0) with torch held at the image's version;
- the order: install kernel-free, the toggle's reference saved in that process, both kernels, the engagement probe, the
  premise, the toggle check, then (and only then) the fetch, the bake, the box and the reducer;
- the box: one engine, the switch over both Qwen3.5 modules, the lockstep comparison on fp32 log-probs with a null
  pair and a mutant, TTFT alternated per round;
- the exit codes.
"""
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p106"
P98 = REPO / "bench" / "p98"
P105 = REPO / "bench" / "p105"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p106_run.sh": LANE / "p106_run.sh",
    "p106_reduce.py": LANE / "p106_reduce.py",
    "p106_box.py": LANE / "p106_box.py",
    "gdn_toggle.py": LANE / "gdn_toggle.py",
    "toggle_probe.py": LANE / "toggle_probe.py",
    "p98_box.py": P98 / "p98_box.py",
    "p98_bake.py": P98 / "p98_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_hybrid_decode_graphs_gpu.py": REPO / "tests" / "test_hybrid_decode_graphs_gpu.py",
    "test_linear_state_graph_gpu.py": REPO / "tests" / "test_linear_state_graph_gpu.py",
    "test_linear_state_chunk_matched_gpu.py": REPO / "tests" / "test_linear_state_chunk_matched_gpu.py",
    "test_linear_state_dense_parity_gpu.py": REPO / "tests" / "test_linear_state_dense_parity_gpu.py",
}
RUN = (LANE / "p106_run.sh").read_text()
REDUCE = (LANE / "p106_reduce.py").read_text()
BOX = (LANE / "p106_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p106 launch refuses ON A RENTED BOX"


def test_the_borrowed_pieces_run_at_their_registered_bytes():
    p98 = dict((n, w) for w, n in _entries(P98 / "staged.sha256"))
    p105 = dict((n, w) for w, n in _entries(P105 / "staged.sha256"))
    mine = dict((n, w) for w, n in _entries())
    for name in ("p98_box.py", "p98_bake.py", "calib.json"):
        assert mine[name] == p98[name], name
    for name in ("test_hybrid_decode_graphs_gpu.py", "test_linear_state_graph_gpu.py", "test_linear_state_chunk_matched_gpu.py",
                 "test_linear_state_dense_parity_gpu.py"):
        assert mine[name] == p105[name], name


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p106_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p106_run.sh":
            assert name in staged, name
    assert "--include 'work/' --include 'work/bake.json' --exclude 'work/*'" in driver   # the arena stays on the box


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p106_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (26 cases)" in out.stdout, out.stdout + out.stderr
    assert "KL_MAX, AGREE_MIN, DNLL_MAX = 0.05, 0.85, 0.01" in REDUCE and "NULL_MAX = KL_MAX / 10" in REDUCE
    assert 'REV = "995ad96eacd98c81ed38be0c5b274b04031597b0"' in REDUCE and "REV=995ad96eacd98c81ed38be0c5b274b04031597b0" in RUN


def test_the_kernels_are_pinned_and_torch_is_held():
    assert 'FLA_PIN="flash-linear-attention==0.5.2"; CC_PIN="causal-conv1d==1.7.0"' in RUN
    assert '"flash-linear-attention": "0.5.2", "causal-conv1d": "1.7.0", "kernels": None' in REDUCE
    assert 'echo "torch==$TORCH_PIN" > $W/constraints.txt' in RUN
    assert RUN.count("-c $W/constraints.txt") == 2                     # both pip invocations in pipx
    assert "assert not any(mods.values())" in RUN                      # the toggle's reference is a kernel-free process
    assert "USE_HUB_KERNELS" in RUN and "E4B_INT4_PREFILL" in RUN       # unset with the serving levers


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ['say "install e4b @', 'say "toggle SAVE (kernel-free process)"', 'for pin in "$FLA_PIN" "$CC_PIN"; do',
             "json.dump(rec, open(\"/root/p106/kernels_fc.json\"", 'echo "premise fc ok"', 'say "toggle COMPARE',
             'if [ "$PROVE" = 1 ]; then', 'say "fetch $MODEL @ $REV"', 'say "bake the NF4 arena"',
             "python $W/p106_box.py", "python $W/p106_reduce.py --dir $W --out $W/verdict.json"]
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    assert 'WANT="9 passed"' in RUN and 'grep -q "$WANT" && ! echo "$LASTL" | grep -q skipped' in RUN
    assert 'LINEAR="test_linear_state_chunk_matched_gpu.py::test_an_all_linear_pool_equals_transformers_prefilled_in_the_same_chunks"' in RUN
    prove = RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index('SRC=$MODEL;')]
    assert ": > PROVED; finish 0" in prove and "snapshot_download" not in prove and "p98_bake" not in prove


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes
    assert {25, 27, 28} <= codes                                         # premise, kernel install, toggle check


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p106-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P106_DRIVE_DRYRUN": "1"}
    out = subprocess.run(["bash", str(LANE / "p106_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p106"), out.stdout + out.stderr


def test_the_box_compares_both_paths_in_one_process():
    assert "toggle = GdnToggle([m_dense, m_moe])" in BOX
    assert 'paths = {"torch": "torch", "kernels": "resolved"}' in BOX and 'paths["torch_again"] = "torch"' in BOX
    assert "lr, la = r.float().log_softmax(-1), a.float().log_softmax(-1)" in BOX       # fp32 log-probs (P97's lesson)
    assert 'paths={"torch": "torch", "mutant": mutant}' in BOX
    assert 'order = ("torch", "resolved") if r % 2 == 0 else ("resolved", "torch")' in BOX
    for flag in ('"--windows", type=int, default=8', '"--null-windows", type=int, default=2', '"--prompt", type=int, default=2048',
                 '"--cont", type=int, default=64', '"--chunk", type=int, default=512', '"--mutant-cont", type=int, default=16',
                 '"--ttft-lengths", default="512,2048,4096"', '"--ttft-rounds", type=int, default=5'):
        assert flag in BOX, flag
    assert '"E4B_PAGED_GRAPHS": "0"' in BOX and '"E4B_PAGED_MAX_TOKENS_PER_SEQ": str(need)' in BOX
