"""The p97 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P97).

`bench/p97/p97_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p97/staged.sha256`; this test runs
the same comparison in CI. It also runs the reducer's self-test (26 cases) and pins the runner's registration: the
kernel pin (e4b CI's grouped-nf4-gemm), transformers 5.17.0, the two models at their revisions, the reducer reading the
same models, shape and layer plans, the premise and the proving run before the fetch, the order (control, then
subject), the measurement at its registered defaults, and the exit codes.
"""
import ast
import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p97"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p97_run.sh": LANE / "p97_run.sh",
    "p97_box.py": LANE / "p97_box.py",
    "p97_reduce.py": LANE / "p97_reduce.py",
    "test_linear_state_gpu.py": REPO / "tests" / "test_linear_state_gpu.py",
}
RUN = (LANE / "p97_run.sh").read_text()
REDUCE = (LANE / "p97_reduce.py").read_text()
BOX = (LANE / "p97_box.py").read_text()


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _const(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if getattr(t, "id", None) == name:
                    return ast.literal_eval(node.value)
                if isinstance(t, ast.Tuple) and name in [getattr(e, "id", None) for e in t.elts]:
                    return ast.literal_eval(node.value)[[e.id for e in t.elts].index(name)]
    raise AssertionError(f"{name} not found")


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p97 launch refuses ON A RENTED BOX"


def test_every_pinned_name_is_staged_by_the_driver():
    assert {n for _w, n in _entries()} == set(SOURCES)
    driver = (LANE / "p97_drive.sh").read_text()
    for name in SOURCES:
        assert name in driver, name
    assert 'for v in P97_PROVE; do' in driver                                  # the proving switch travels
    for f in ("p97_box.py", "p97_reduce.py", "test_linear_state_gpu.py", "staged.sha256"):
        assert f in RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")], f


def test_the_reducer_applies_the_registered_rule():
    out = subprocess.run([sys.executable, str(LANE / "p97_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (26 cases)" in out.stdout, out.stdout + out.stderr


def test_the_stack_is_e4b_cis_kernel_pin_on_transformers_5_17():
    pin = re.search(r"^GNF4_SHA=([0-9a-f]{40})\b", RUN, re.M).group(1)
    # CI's kernel pin when the lane registered: v0.34.1. CI moved to v0.35.0 in 0.42.0; a frozen lane keeps its own.
    assert pin == "34da93d6fe8d2a401b7001705658ce00b2b18213"
    assert '"transformers==5.17.0"' in RUN and 'assert transformers.__version__ == "5.17.0"' in RUN
    assert "import fp8_paged_attn" in RUN and "from experts4bit_qlora.engines.linear_state import" in RUN
    assert "E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE" in RUN                  # every serving lever starts unset


def test_the_runner_and_the_reducer_name_the_same_models_and_shape():
    subj, ctrl = _const(REDUCE, "SUBJECT"), _const(REDUCE, "CONTROL")
    assert f"SUBJ={subj['model']}; SUBJ_REV={subj['revision']}" in RUN
    assert f"CTRL={ctrl['model']}; CTRL_REV={ctrl['revision']}" in RUN
    assert (subj["n_layers"], subj["attn"], subj["linear"]) == (40, 10, 30)
    assert (ctrl["n_layers"], ctrl["attn"], ctrl["linear"]) == (16, 16, 0)
    # OLMoE is P96's revision (the lanes share a control checkpoint)
    assert f"OL={ctrl['model']}; OL_REV={ctrl['revision']}" in (REPO / "bench" / "p96" / "p96_run.sh").read_text()
    shape = _const(REDUCE, "SHAPE")
    assert shape == {"windows": 4, "prompt": 512, "cont": 256, "chunk": 128}
    for flag, key in (("--windows", "windows"), ("--prompt", "prompt"), ("--cont", "cont"), ("--chunk", "chunk")):
        assert f'ap.add_argument("{flag}", type=int, default={shape[key]})' in BOX, flag
    assert subj["pre_attention"] == [0, 1, 2]                                  # Qwen3.6's first attention layer is 3
    assert _const(REDUCE, "STATE_TOL") == 5e-2
    assert _const(REDUCE, "KL_CEIL") == 0.05 and _const(REDUCE, "AGREE_MIN") == 0.85


def test_the_loaded_commit_is_read_from_the_checkpoints_own_config():
    # a composite checkpoint (Qwen3.6) is built from its text_config, which carries no _commit_hash; the reducer voids
    # a record without the loaded commit, so the harness must read the config the loader returns
    assert "model, ckpt_cfg = load_moe_4bit_streaming(" in BOX
    assert '"loaded_commit": getattr(ckpt_cfg, "_commit_hash", None)' in BOX


def test_the_registered_run_passes_no_measurement_flags():
    box = RUN[RUN.index("box(){"):RUN.index("[ \"$CTRL_OK\" = 0 ]")]
    assert '--model "$SRC" --revision "$REV" --tag $TAG --out $W/p97_$TAG.json $BOX_EXTRA' in box
    assert "--offload" not in RUN and "--stand-in-attention" not in RUN          # rehearsal flags never in the runner
    rehearsal = RUN[RUN.index('if [ "$REHEARSAL" != 0 ]'):RUN.index("# ---- staged pieces")]
    for knob in ('[ -n "$MODEL_DIR" ]', '[ -n "$BOX_EXTRA" ]', '[ "$ALLOW_SKIP" != 0 ]', '[ "$GPU_CLASS" != 5090 ]',
                 '[ "$MIN_DISK_GB" != 130 ]'):
        assert knob in rehearsal, knob


def test_the_premise_and_the_proof_precede_the_fetch_and_the_order_is_control_then_subject():
    install = RUN.index('say "install e4b @')
    assert RUN.index("finish 15;") < install and RUN.index("finish 13;") < install
    premise = RUN.index("python -m pytest test_linear_state_gpu.py")
    prove = RUN.index('if [ "$PROVE" = 1 ]; then')
    first_fetch = RUN.index('fetch control "$CTRL" "$CTRL_REV"')
    assert install < premise < prove < first_fetch
    assert RUN.index(": > PROVED; finish 0", prove) < first_fetch               # the proving run stops before any model
    assert 'grep -q "1 passed" && ! echo "$LASTL" | grep -q skipped' in RUN       # a skipped premise is a failure
    assert first_fetch < RUN.index('fetch subject "$SUBJ" "$SUBJ_REV"') < RUN.index('box control "$CTRL"') \
        < RUN.index('box subject "$SUBJ"') < RUN.index("python $W/p97_reduce.py --dir $W --out $W/verdict.json")


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)} | {26}
    assert codes & {13, 14, 17, 18} == {13}, codes


def test_the_driver_runs_to_its_dry_run(tmp_path):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "p97-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "P97_DRIVE_DRYRUN": "1", "P97_PROVE": "1"}
    out = subprocess.run(["bash", str(LANE / "p97_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> root@h:/root/p97"), out.stdout + out.stderr
    assert "P97_PROVE=1" in out.stdout                                          # the proving switch reaches the box
