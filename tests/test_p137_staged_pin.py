"""The p137 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P137 from test_p133_staged_pin.py).

`bench/p137/p137_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p137/staged.sha256`; this test
runs the same comparison in CI. It also runs the box's and the reducer's self-tests and pins the lane's shape:
- the pins: e4b E4B_PIN, grouped-nf4-gemm 0.46.0 exact, torch 2.8.0+cu128, transformers 5.18.0, bitsandbytes 0.50.2;
- the declared run ids, and the order: refusals, install and tripwire, the self-tests, the premise (an exact pass count,
  no skip), the data, the fetch, the four arms and the reducer;
- the subject: TC3's fused offload configuration, with only the accumulation loop differing between arms;
- the rule's constants, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p137"
PIN = LANE / "staged.sha256"
SOURCES = {n: LANE / n for n in ("p137_run.sh", "p137_box.py", "p137_reduce.py", "p137_tiny.py")}
RUN = (LANE / "p137_run.sh").read_text()
DRIVE = (LANE / "p137_drive.sh").read_text()
REDUCE = (LANE / "p137_reduce.py").read_text()
BOX = (LANE / "p137_box.py").read_text()
PREREG = (LANE / "PREREG-p137.md").read_text(encoding="utf-8")
E4B_PIN = "b4366b9af2234730b5df783c5f3b0c7aef3ab41c"
GNF4_PIN = "6b7c2e7474d20c8843814dd46dad6cfc3d94b440"


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _drive(tmp_path, run_id, **extra):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "P137_DRIVE_DRYRUN": "1", **extra}
    return subprocess.run(["bash", str(LANE / "p137_drive.sh")], capture_output=True, text=True, env=env)


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p137 launch refuses ON A RENTED BOX"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p137_run.sh":
            assert name in staged, name
    assert "--exclude 'e4bsrc'" in DRIVE and "--exclude=./e4bsrc" in DRIVE


def test_the_self_tests_pass():
    for script, line in (("p137_reduce.py", "p137_reduce self-test OK (29 cases)"), ("p137_box.py", "p137_box self-test OK (10/10 cases)")):
        out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": ""})
        assert out.returncode == 0 and line in out.stdout, out.stdout + out.stderr


def test_the_pins_are_exact_and_agree():
    assert f"E4B_PIN={E4B_PIN}" in RUN and f"E4B_PIN={E4B_PIN}" in DRIVE and E4B_PIN in PREREG
    assert f"GNF4_SHA={GNF4_PIN}" in RUN and GNF4_PIN in PREREG
    assert '[ "$E4B_SHA" = "$E4B_PIN" ] ||' in RUN and "TORCH_WANT=2.8.0+cu128" in RUN
    trip = RUN[RUN.index("python - <<'PYT'"):RUN.index("PYT\ncat versions.txt")]
    for pin in ('md.version("grouped-nf4-gemm") == "0.46.0"', 'md.version("bitsandbytes") == "0.50.2"',
                'transformers.__version__ == "5.18.0"', 'torch.__version__ == os.environ["WANT_TORCH"]',
                "torch.cuda.get_device_capability() == (8, 9)", 'lm.SUPPORTED == ("Qwen3MoeForCausalLM",)',
                'hasattr(off._ExpertOffload, "_held")', "LayerMajorNotReproducibleError"):
        assert pin in trip, pin
    assert '"transformers==5.18.0" "bitsandbytes==0.50.2"' in RUN
    assert "MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in RUN


def test_the_declared_run_ids_and_nothing_else(tmp_path):
    ok = [("p137-4090-1", {}), ("p137-4090-2", {}), ("p137-dry-1", {"P137_DRY": "1", "P137_BOX_EXEC": "/bin/false"})]
    for run_id, extra in ok:
        out = _drive(tmp_path, run_id, **extra)
        assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> "), (run_id, out.stdout + out.stderr)
        assert f"E4B_SHA={E4B_PIN}" in out.stdout, out.stdout
    refused = [("p137-4090-3", {}), ("p137-4090-prove-1", {}), ("p137-4090-1", {"P137_DRY": "1", "P137_BOX_EXEC": "/bin/false"}),
               ("p137-dry-1", {}), ("p133-4090-1", {}), ("p137-4090-1", {"E4B_SHA": "0" * 40})]
    for run_id, extra in refused:
        out = _drive(tmp_path, run_id, **extra)
        assert out.returncode == 78 and "refusing" in out.stdout, (run_id, extra, out.stdout + out.stderr)
    assert "p137-4090-1" in PREREG and "p137-4090-2" in PREREG


def test_the_rule_is_the_registered_rule():
    assert 'TAGS = ("A0a", "L1a", "L1b", "A0b")' in REDUCE
    assert "SELF_LO, SELF_HI = 0.97, 1.03" in REDUCE and "S_MIN = 1.10" in REDUCE
    assert "BAND_LO, BAND_HI = 1.25, 2.0" in REDUCE and "MEM_TERM_GIB = 4.25" in REDUCE and "HELD_TOL = 0.01" in REDUCE
    for s in ("1.10", "[0.97, 1.03]", "[1.25, 2.0]", "4.25 GiB", "0.01"):
        assert s in PREREG, s
    assert 'LOOPS = {"A0": "stock", "L1": "layer_major"}' in BOX
    assert "SEQ, MB_ROWS, ACCUM = 2048, 2, 4" in BOX and "WARM, TIMED = 2, 8" in BOX
    assert "SEED, LR, R, ALPHA = 3407, 2e-4, 16, 16" in BOX
    assert "enable_fast_train(model, dgrad=True)" in BOX and "offload=True, pin=True" in BOX


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["CUDA_PROBE=$(python -", "REFUSED: the image's torch", "REFUSED: card is", "the licence covers sm_89",
             "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @', "python - <<'PYT'",
             "p137_reduce.py --self-test", "p137_box.py --self-test", "python -m pytest tests/test_layer_major.py",
             'echo "premise ok', "load_dataset('tatsu-lab/alpaca'", "python $W/p137_tiny.py", 'say "fetch $MODEL @ $REV"',
             "for TAG in A0a L1a L1b A0b; do", 'say "reduce"; python $W/p137_reduce.py --dir $W --out $W/verdict.json']
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))


def test_the_subject_only_the_loop_differs():
    unset = RUN[RUN.index("unset LAYER_MAJOR"):RUN.index(": > summary.txt")]
    for knob in ("LAYER_MAJOR", "E4B_OFFLOAD_STATS", "E4B_CKPT_OFFLOAD", "E4B_CHUNKED_LM_LOSS", "E4B_ABSMAX_DQ",
                 "E4B_OFFLOAD_PIN_SLAB", "TRAIN_ARENA", "OFFLOAD_EXPERTS"):
        assert re.search(rf"\b{knob}\b", unset), knob
    assert 'LOOP=stock; [ "${TAG:0:2}" = L1 ] && LOOP=layer_major' in RUN
    arm = RUN[RUN.index("for TAG in A0a L1a L1b A0b; do"):RUN.index("DF=")]
    assert arm.count("--loop $LOOP") == 1 and "GNF4" not in arm and "E4B_" not in arm.replace("E4B_SHA", "")


def test_every_time_left_check_fits_its_own_guard():
    assert "guard 2.5 h" in PREREG
    reading = dict(re.findall(r"NEED_(FETCH|ARM)=(\d+)", RUN[RUN.index("else\n  MODEL=Qwen"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(reading) == {"FETCH", "ARM"}
    for need in reading.values():
        assert int(need) + 600 <= 2.5 * 3600 - 900, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_ARM"]


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {9, 10, 11, 15, 16, 19, 21, 22, 25, 28, 40} <= codes
