"""The p95 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P95).

`bench/p95/p95_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p95/staged.sha256`; this test runs
the same comparison in CI. It also runs the reducer's self-test (12 cases) and pins the runner's registration: P94's
kernel pin and harness bytes, the families at their revisions with their licensed envs, the three arms, the windows
(and that the reducer reads the same ones), window 0 as P94's exact command line, the premise and the proving run
before the fetch, the window-major order, the exit codes -- and that the reducer's window-0 control values are P94's
receipts.
"""
import ast
import hashlib
import json
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p95"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p95_run.sh": LANE / "p95_run.sh",
    "p95_reduce.py": LANE / "p95_reduce.py",
    "p42_reduce.py": REPO / "bench" / "p42" / "p42_reduce.py",
    "step_decomp.py": REPO / "bench" / "p39" / "step_decomp.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "hook/usercustomize.py": REPO / "bench" / "p42" / "hook" / "usercustomize.py",
    "test_k25_row_exact_gpu.py": REPO / "tests" / "test_k25_row_exact_gpu.py",
    "test_nf4_t1_device_grouping_gpu.py": REPO / "tests" / "test_nf4_t1_device_grouping_gpu.py",
}
RUN = (LANE / "p95_run.sh").read_text()
REDUCE = (LANE / "p95_reduce.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _reducer_const(name):
    tree = ast.parse(REDUCE)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) == name for t in targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in p95_reduce.py")


@pytest.mark.parametrize("want,name", list(_entries()), ids=lambda v: v if isinstance(v, str) and "." in v and len(v) < 40 else "sha")
def test_staged_file_matches_its_pin(want, name):
    got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
    assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p95 launch refuses ON A RENTED BOX"


def test_every_pinned_name_is_staged_by_the_driver():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p95_drive.sh").read_text()
    for name in ("p95_run.sh", "p95_reduce.py", "p42_reduce.py", "test_k25_row_exact_gpu.py", "test_nf4_t1_device_grouping_gpu.py"):
        assert name in driver, name
    assert 'for v in P95_PROVE; do' in driver                                  # the proving switch travels


def test_the_harness_and_the_premise_are_p94s_pinned_bytes():
    p94 = {name: want for want, name in _entries(REPO / "bench" / "p94" / "staged.sha256")}
    mine = {name: want for want, name in _entries()}
    for name in ("step_decomp.py", "k8_bake.py", "calib.json", "hook/usercustomize.py", "p42_reduce.py",
                 "test_k25_row_exact_gpu.py", "test_nf4_t1_device_grouping_gpu.py"):
        assert mine[name] == p94[name], name


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p95_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (12 cases)" in out.stdout, out.stdout + out.stderr


def test_the_window0_control_values_are_p94s_receipts():
    p94 = _reducer_const("P94")
    rec = REPO / "bench" / "p94" / "receipts" / "p94-5090-2"
    for (fam, src), arms in p94.items():
        for arm, want in arms.items():
            got = json.loads((rec / f"{fam}_k8_{arm}_{src}.json").read_text())["ppl"]
            assert round(got, 5) == want, (fam, src, arm, got, want)


def test_the_families_run_their_licensed_envs_at_the_registered_revisions():
    sys.path.insert(0, str(REPO / "bench" / "p44"))
    try:
        from serve_stack import MODELS, arm_env
    finally:
        sys.path.pop(0)
    for fam, arm, var in (("granite", "r12epi", "GR"), ("olmoe", "nf4", "OL")):
        mid, rev = MODELS[fam]
        assert f"{var}={mid}; {var}_REV={rev}" in RUN, fam
        env = arm_env(fam, arm)
        want = " ".join(f"{k}={env[k]}" for k in ("E4B_SERVE_EXP_INT4", "E4B_SERVE_ATTN_INT4_CALIB", "E4B_CALIB_SOURCE",
                                                  "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI"))
        assert f'{var}_ENV="{want}"' in RUN, (fam, want)


def test_the_kernel_pin_the_arms_and_the_route_plan_are_p94s():
    p94_run = (REPO / "bench" / "p94" / "p94_run.sh").read_text()
    pin = re.search(r"^GNF4_SHA=([0-9a-f]{40})\b", RUN, re.M)
    assert pin and pin.group(1) == re.search(r"^GNF4_SHA=([0-9a-f]{40})\b", p94_run, re.M).group(1)
    for arm in ('ARM_G="E4B_NF4_GROUPED_SMALLM=0 E4B_NF4_T1_DEVICE_GROUPING=0"',
                'ARM_M="E4B_NF4_GROUPED_SMALLM=0 E4B_NF4_T1_DEVICE_GROUPING=1"',
                'ARM_T="E4B_NF4_GROUPED_SMALLM=1 E4B_NF4_T1_DEVICE_GROUPING=0"'):
        assert arm in RUN, arm
    assert "E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE" in RUN                  # every lever starts unset
    plan = ast.literal_eval(re.search(r"assert _K25_PLAN == (\{[^}]*\})", RUN).group(1))
    assert plan == {"block_n": 32, "kc": 64, "warps": 4, "stages": 3, "lut": "tree", "dot_bf16": False}, plan


def test_the_windows_are_registered_and_the_reducer_reads_the_same_ones():
    assert 'STRIDE=4096; SPAN=2600; C4_WINDOWS="0 1 2 3 4 5 6 7 8"; WT_WINDOWS="0 1 2 3 4"' in RUN
    fresh = _reducer_const("FRESH")
    assert fresh == {"c4val1": (1, 2, 3, 4, 5, 6, 7, 8), "wikitext": (1, 2, 3, 4)}, fresh
    assert _reducer_const("MIN_FRESH") == {"c4val1": 6, "wikitext": 3}
    assert _reducer_const("SIGMA_MAX") == 0.025 and _reducer_const("GATE") == 0.05
    # window 0 is P94's exact command line (no offset flags); k >= 1 slices SPAN tokens from k * STRIDE
    assert '[ "$K" = 0 ] || win=(--prompt-offset $((K * STRIDE)) --prompt-span $SPAN)' in RUN
    assert 'local name=${TAG}_k8_${ARM}_${SRC}_w$K' in RUN
    assert "--batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv" in RUN
    # the span covers the prompt and the scored window: 512 + 2048 + 1 <= SPAN <= STRIDE (disjoint windows)
    assert 512 + 2048 + 1 <= 2600 <= 4096


def test_the_premise_and_the_proof_precede_the_fetch_and_the_order_is_window_major():
    install = RUN.index('say "install e4b @')
    assert RUN.index("finish 15;") < install and RUN.index("finish 13;") < install
    premise = RUN.index("python -m pytest test_k25_row_exact_gpu.py test_nf4_t1_device_grouping_gpu.py")
    contract = RUN.index("python -m pytest test_nf4_grouped_smallm_interp.py")
    prove = RUN.index('if [ "$PROVE" = 1 ]; then')
    first_fetch = RUN.index('family granite "$GR"')
    assert install < premise < contract < prove < first_fetch
    assert RUN.index(": > PROVED; finish 0", prove) < first_fetch               # the proving run stops before any model
    loop = RUN[RUN.index("for K in $C4_WINDOWS; do"):RUN.index('say "reduce"')]
    assert RUN.index('family granite "$GR"') < RUN.index('family olmoe "$OL"') < RUN.index("for K in $C4_WINDOWS; do")
    assert loop.index("for TAG in granite olmoe; do") < loop.index("for SRC in c4val1 wikitext; do") \
        < loop.index('[ "$SRC" = wikitext ] && ! in_wt "$K" && continue') < loop.index("for ARM in g m t; do")
    fam = RUN[RUN.index("family(){"):RUN.index("# the registered order: both families prepared")]
    assert 'speed $TAG "$MID" "$QA" "$ENVS" m 1 1 1' in fam and "k8 " not in fam   # engagement census only, no K8 in prep
    assert RUN.index("for K in $C4_WINDOWS; do") < RUN.index("python $W/p95_reduce.py --dir $W --out $W/verdict.json")


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13}, codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts", "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p95-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P95_DRIVE_DRYRUN": "1", "P95_PROVE": "1"}
    out = subprocess.run(["bash", str(LANE / "p95_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p95"), out.stdout + out.stderr
    assert "P95_PROVE=1" in out.stdout                                          # the proving switch reaches the box
