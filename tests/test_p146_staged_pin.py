"""The p146 lane's staged-file pin must match the repo (the e4b#642 check, mirrored for P146 from test_p144_staged_pin.py).

`bench/p146/p146_drive.sh` refuses to run when a staged file's sha256 differs from `bench/p146/staged.sha256`; this test
runs the same comparison in CI. It also runs the box's and the reducer's self-tests and pins the lane's shape:
- the pins: e4b E4B_PIN, grouped-nf4-gemm 0.46.0 exact, torch 2.8.0+cu128, transformers 5.18.0, bitsandbytes 0.50.2;
- the declared run ids, and the order: refusals, install and tripwire, the self-tests, the premise (an exact pass count,
  no skip), the staged alpaca parquet, the fetch, the four arm processes and the reducer;
- the subject: train.py's own main() at its own recipe, with only LAYER_MAJOR differing between arms;
- the rule's constants, every time-left check inside its own guard, and the exit codes.
"""
import hashlib
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p146"
PIN = LANE / "staged.sha256"
SOURCES = {n: LANE / n for n in ("p146_run.sh", "p146_box.py", "p146_reduce.py", "p146_tiny.py")}
RUN = (LANE / "p146_run.sh").read_text()
DRIVE = (LANE / "p146_drive.sh").read_text()
REDUCE = (LANE / "p146_reduce.py").read_text()
BOX = (LANE / "p146_box.py").read_text()
PREREG = (LANE / "PREREG-p146.md").read_text(encoding="utf-8")
E4B_PIN = "4917506a22d960a6586bd5cefe9a0deca662fc46"
GNF4_PIN = "6b7c2e7474d20c8843814dd46dad6cfc3d94b440"
DATA_SHA = "06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073"


def _entries(pin=PIN):
    for line in pin.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        want, name = line.split(None, 1)
        yield want, name.strip()


def _drive(tmp_path, run_id, **extra):
    data = tmp_path / "alpaca.parquet"
    if not data.exists():
        data.write_text("stand-in")
    env = {"P146_DATA": str(data), "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path),
           "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "P146_DRIVE_DRYRUN": "1", **extra}
    return subprocess.run(["bash", str(LANE / "p146_drive.sh")], capture_output=True, text=True, env=env)


def test_staged_files_match_their_pins():
    for want, name in _entries():
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name} changed without re-pinning ({got[:12]} vs {want[:12]}); the next p146 launch refuses ON A RENTED BOX"


def test_every_pinned_name_is_staged_by_the_driver_and_checked_by_the_runner():
    assert {n for _w, n in _entries()} == set(SOURCES)
    for name in SOURCES:
        assert name in DRIVE, name
    staged = RUN[RUN.index("# ---- staged pieces"):RUN.index("# ---- refusals")]
    for name in list(SOURCES) + ["staged.sha256"]:
        if name != "p146_run.sh":
            assert name in staged, name
    assert "--exclude 'e4bsrc'" in DRIVE and "--exclude=./e4bsrc" in DRIVE
    assert "--exclude 'out_*'" in DRIVE and "--exclude='./out_*'" in DRIVE        # the arms' adapters stay on the box


def test_the_self_tests_pass():
    for script, line in (("p146_reduce.py", "p146_reduce self-test OK (43 cases)"),
                         ("p146_box.py", "p146_box self-test OK (11/11 cases)")):
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
                "torch.cuda.get_device_capability() == (8, 9)", '"Qwen3MoeForCausalLM" in lm.SUPPORTED',
                'hasattr(off._ExpertOffload, "_held")', "LayerMajorNotReproducibleError",
                'hasattr(train, "_FAST_TRAIN_ENV")', 'hasattr(census.PhaseClock, "step_end")'):
        assert pin in trip, pin
    assert '"transformers==5.18.0" "bitsandbytes==0.50.2"' in RUN and '"huggingface_hub>=0.23,<2"' in RUN   # P137 Amendment 1
    assert 'int(hub.split(".")[0]) < 2' in trip
    assert "MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in RUN
    assert "PREMISE_PASSED=57" in RUN and "57 passed" in PREREG


def test_the_declared_run_ids_and_nothing_else(tmp_path):
    ok = [("p146-4090-1", {}), ("p146-4090-2", {}), ("p146-4090-3", {}), ("p146-4090-4", {}),
          ("p146-dry-1", {"P146_DRY": "1", "P146_BOX_EXEC": "/bin/false"})]
    for run_id, extra in ok:
        out = _drive(tmp_path, run_id, **extra)
        assert out.returncode == 0 and out.stdout.startswith("DRYRUN stage -> "), (run_id, out.stdout + out.stderr)
        assert f"E4B_SHA={E4B_PIN}" in out.stdout, out.stdout
    refused = [("p146-4090-5", {}), ("p146-4090-10", {}), ("p146-5090-1", {}), ("p144-4090-1", {}), ("p146-4090-prove-1", {}),
               ("p146-4090-1", {"P146_DRY": "1", "P146_BOX_EXEC": "/bin/false"}), ("p146-dry-1", {}),
               ("p146-4090-1", {"E4B_SHA": "0" * 40})]
    for run_id, extra in refused:
        out = _drive(tmp_path, run_id, **extra)
        assert out.returncode == 78 and "refusing" in out.stdout, (run_id, extra, out.stdout + out.stderr)
    assert "p146-4090-1" in PREREG and "p146-4090-4" in PREREG


def test_the_rule_is_the_registered_rule():
    assert 'TAGS = ("A0a", "L1a", "L1b", "A0b")' in REDUCE and 'TAGS = ("A0a", "L1a", "L1b", "A0b")' in BOX
    assert "SELF_LO, SELF_HI = 0.97, 1.03" in REDUCE and "S_MIN = 1.10" in REDUCE
    assert "BAND_LO, BAND_HI = 1.5, 2.6" in REDUCE and "MEM_TERM_GIB = 4.25" in REDUCE and "HELD_TOL = 0.01" in REDUCE
    for s in ("1.10", "[0.97, 1.03]", "[1.5, 2.6]", "4.25 GiB", "0.01"):
        assert s in PREREG, s
    assert "WARM, TIMED = 2, 32" in BOX and "TIMED_REG = 32" in REDUCE and 'N_LAYERS = {"reading": 48, "dry": 2}' in REDUCE
    # train.py's own defaults are the recipe; the lane sets only STEPS, DO_GEN and the instruments
    for k in ('"seq": 192', '"token_budget": 2048', '"grad_accum": ACCUM', '"r": 8', '"alpha": 16', '"lr": 2e-4',
              '"n_train": 2000', '"seed": 0', '"train_attention": True'):
        assert k in REDUCE, k
    assert 'DATASET_CALLS = ["train[:2000]", "train[2000:2064]"]' in REDUCE


def test_the_order_puts_every_refusal_before_the_fetch():
    order = ["CUDA_PROBE=$(python -", "REFUSED: the image's torch", "REFUSED: card is", "the licence covers sm_89",
             "REFUSED: ${FREE_GB", "REFUSED: ${RAM_GB", 'say "install e4b @', "python - <<'PYT'",
             "p146_reduce.py --self-test", "p146_box.py --self-test", "python -m pytest tests/test_layer_major.py",
             'echo "premise ok', f"DATA_SHA={DATA_SHA}", "python $W/p146_tiny.py", 'say "fetch $MODEL @ $REV"',
             "--mode cli-abba --model", 'say "reduce"; python $W/p146_reduce.py --dir $W --out $W/verdict.json']
    at = [RUN.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))


def test_the_subject_is_the_cli_and_only_layer_major_differs():
    unset = RUN[RUN.index("unset LAYER_MAJOR"):RUN.index(": > summary.txt")]
    for knob in ("LAYER_MAJOR", "FAST_TRAIN", "E4B_OFFLOAD_STATS", "E4B_CKPT_OFFLOAD", "E4B_CHUNKED_LM_LOSS", "E4B_ABSMAX_DQ",
                 "E4B_OFFLOAD_PIN_SLAB", "TRAIN_ARENA", "OFFLOAD_EXPERTS", "SEQ", "TOKEN_BUDGET", "GRAD_ACCUM", "STEPS"):
        assert re.search(rf"\b{knob}\b", unset), knob
    arm = RUN[RUN.index("# ---- the four arms"):RUN.index("DF=")]
    assert arm.count("--mode cli-abba") == 1 and "GNF4" not in arm and "E4B_" not in arm.replace("E4B_SHA", "")
    assert 'ARM_ENV = {"A0": {"OFFLOAD_EXPERTS": "1", "FAST_TRAIN": "1"},' in BOX
    assert '"L1": {"OFFLOAD_EXPERTS": "1", "LAYER_MAJOR": "1"}}' in BOX
    assert "t.main()" in BOX and "subprocess.run(cmd, env=arm_env(" in BOX     # train.py's own main, a fresh process each


def test_every_time_left_check_fits_its_own_guard():
    assert "guard 3.0 h" in PREREG
    reading = dict(re.findall(r"NEED_(FETCH|ABBA)=(\d+)", RUN[RUN.index("else\n  MODEL=Qwen"):RUN.index("fi\nGPU_CLASS=")]))
    assert set(reading) == {"FETCH", "ABBA"}
    assert int(reading["FETCH"]) + int(reading["ABBA"]) + 600 <= 3.0 * 3600, reading
    assert re.findall(r"can_run (\S+)", RUN) == ["$NEED_FETCH", "$NEED_ABBA"]
    assert "ARM_TIMEOUT_S = 1500" in BOX and 4 * 1500 <= int(reading["ABBA"])


def test_lane_failures_avoid_the_machine_exclusion_codes():
    codes = {int(c) for c in re.findall(r"(?:finish|return) (\d+)", RUN)}
    assert codes & {13, 14, 17, 18} == {13, 18}, codes
    eighteen = [line for line in RUN.splitlines() if "finish 18" in line]
    assert len(eighteen) == 1 and "cuda unusable" in eighteen[0], eighteen
    assert {9, 10, 11, 15, 16, 19, 21, 22, 25, 28, 40} <= codes


def test_amendment_1_the_alpaca_parquet_is_staged_and_checked_never_downloaded(tmp_path):
    """P137 Amendment 1, carried: the driver stages the pinned alpaca parquet from the controller and checks it, the runner
    checks it again, and train.py's load_dataset("tatsu-lab/alpaca") is served from that file, never the hub."""
    assert f"DATA_SHA={DATA_SHA}" in RUN and f"DATA_SHA={DATA_SHA}" in DRIVE and DATA_SHA in PREREG
    assert '[ "$GOT_SHA" = "$DATA_SHA" ] || { say "DATA FAIL: alpaca.parquet is $GOT_SHA' in RUN
    assert '_load_dataset("parquet", data_files={"train": a.parquet}, split=split)' in BOX
    assert 'if path != ALPACA or args or kw:' in BOX
    out = _drive(tmp_path, "p146-4090-1", P146_DRIVE_DRYRUN="0")   # past the composition print: the stand-in's sha256 refuses
    assert out.returncode == 78 and "the registered alpaca parquet is" in out.stdout, out.stdout + out.stderr
    assert "P137 Amendment 1" in PREREG
