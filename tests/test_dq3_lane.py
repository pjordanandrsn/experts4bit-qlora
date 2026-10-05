"""Lane DQ3 (bench/dq3/DQ3-PREREG.md): the reducer's self-test passes, every rule mutant is caught by it, and the box,
the arm script and the reducer agree on the subject and the pins. CPU-only."""
import ast
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "dq3"
RUN = (LANE / "dq3_run.sh").read_text()
ARM = (LANE / "dq3_arm.py").read_text()


def _const(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def test_the_reducer_self_test_passes():
    out = subprocess.run([sys.executable, str(LANE / "dq3_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (25 cases)" in out.stdout, out.stdout + out.stderr


def test_the_registered_constants():
    r = LANE / "dq3_reduce.py"
    assert _const(r, "ORDER") == ("R", "S", "S0", "S0", "S", "R")
    assert _const(r, "N_LAYERS") == 64 and _const(r, "STEP_MAX") == 1.10 and _const(r, "SAVING_FRACTION") == 0.9
    assert _const(r, "BLOCKING_MAX") == 2 and _const(r, "SELF_PAIR_BAND") == (0.97, 1.03)
    # the seven projections' packed bytes for Qwen3-32B: (2*8192*5120 + 2*1024*5120 + 3*25600*5120) / 2
    assert _const(r, "LAYER_STREAMED_BYTES") == (2 * 8192 * 5120 + 2 * 1024 * 5120 + 3 * 25600 * 5120) // 2


def test_the_box_runs_the_registered_palindrome_and_pins():
    assert "for arm in R S S0 S0 S R; do" in RUN
    for pin in ("transformers==5.18.0", "peft==0.21.2", "bitsandbytes==0.50.2", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
                'case "$link" in *"RTX 5090, 5, 16")', "dq3_reduce.py --self-test", '"train_prefetch" in inspect'):
        assert pin in RUN, pin
    assert subprocess.run(["bash", "-n", str(LANE / "dq3_run.sh")]).returncode == 0
    for marker in ("> TC1_RUN_NONCE", "TC1_EXIT_CODE.$NONCE", "TC1_SUCCESS.$NONCE", "TP_DONE.$NONCE"):
        assert marker in RUN, marker


def test_the_arm_is_the_registered_subject():
    for s in ('"num_hidden_layers": args.layers', "compress_statistics=True", 'quant_type="nf4"', "blocksize=64",
              "model.is_loaded_in_4bit = True", "get_peft_model", "LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0",
              '"use_reentrant": False', "use_deterministic_algorithms(True", "SDPBackend.MATH",
              'train_prefetch=(args.arm == "S")', "load_lora_state(pm, init)"):
        assert s in ARM, s


MUTANTS = [
    ("STEP_MAX = 1.10", "STEP_MAX = 1.20"), ("SAVING_FRACTION = 0.9", "SAVING_FRACTION = 0.5"),
    ("BLOCKING_MAX = 2", "BLOCKING_MAX = 9"), ("SELF_PAIR_BAND = (0.97, 1.03)", "SELF_PAIR_BAND = (0.5, 2.0)"),
    ("and c[\"fwd_prefetch_issued\"] == c[\"bwd_prefetch_issued\"] == ISSUED_PER_STEP // 2", "and True"),
    ("and c[\"hwm_resident\"] <= 2", "and True"),
    ("if a.get(\"device\") != REGISTERED_DEVICE:", "if False:"), ("if a.get(\"rehearsal\"):", "if False:"),
    ("if not a.get(\"finished_at\"):", "if False:"),
    ("if a.get(\"config_overrides\") != {\"num_hidden_layers\": N_LAYERS}:", "if False:"),
    ("or e.get(\"lora_dtypes\") != [\"torch.float32\"]", ""),
    ("e.get(\"wrapper_kinds\") != [\"peft.tuners.lora.bnb.Linear4bit\"]", "False"),
    ("if [a.get(\"arm\") for a in arms] != list(ORDER):", "if False:"),
    ("if p.get(\"loss_bits\") != q.get(\"loss_bits\"):", "if False:"),
    ("bad = [n for n in q[\"grads\"] if p[\"grads\"].get(n) != q[\"grads\"][n]]", "bad = []"),
    ("if a[\"arm\"] != \"R\" and (off is None or off.get(\"layers\") != N_LAYERS):", "if False:"),
]


def test_every_rule_mutant_is_killed(tmp_path):
    src = (LANE / "dq3_reduce.py").read_text()
    survived = []
    for old, new in MUTANTS:
        assert src.count(old) == 1, f"mutant anchor not unique/absent: {old!r}"
        m = tmp_path / "mut.py"
        m.write_text(src.replace(old, new))
        out = subprocess.run([sys.executable, str(m), "--self-test"], capture_output=True, text=True)
        if "self-test FAILED" not in out.stdout:
            survived.append((old, (out.stdout + out.stderr)[-200:]))
    assert not survived, survived


def _fake_bin(tmp_path, link, probe_rc):
    """A PATH where nvidia-smi reports ``link`` and ``python dq3_vram_probe.py`` exits ``probe_rc``; any other python
    call (the install, the arms) records itself and fails, so reaching it is visible."""
    b = tmp_path / "bin"
    b.mkdir()
    (b / "nvidia-smi").write_text(f"#!/bin/sh\necho '{link}'\n")
    (b / "python").write_text(
        "#!/bin/sh\n"
        f'case "$*" in *dq3_vram_probe.py*) echo "VRAM_PROBE_FAIL test"; exit {probe_rc};; esac\n'
        'echo "$*" >> "$DQ3_W/reached"; exit 1\n')
    for f in b.iterdir():
        f.chmod(0o755)
    return b


def _run_box(tmp_path, link, probe_rc):
    w = tmp_path / "w"
    w.mkdir()
    env = {"PATH": f"{_fake_bin(tmp_path, link, probe_rc)}:/usr/bin:/bin", "DQ3_W": str(w),
           "TC1_RUN_NONCE": "n", "E4B_SHA": "0" * 40}
    rc = subprocess.run(["bash", str(LANE / "dq3_run.sh")], env=env, capture_output=True, text=True).returncode
    return rc, w


def test_a_host_that_refuses_the_subjects_memory_is_refused_at_18_before_any_install(tmp_path):
    rc, w = _run_box(tmp_path, "NVIDIA GeForce RTX 5090, 5, 16", probe_rc=3)
    assert rc == 18
    assert (w / "TC1_EXIT_CODE.n").read_text().strip() == "18" and (w / "TP_DONE.n").exists()
    assert (w / "REFUSAL").read_text().startswith("refused: vram")
    assert not (w / "reached").exists(), "the runner went past the probe on a refused host"
    assert "BOX_REFUSED vram" in (w / "summary.txt").read_text()


def test_the_link_check_still_comes_first(tmp_path):
    rc, w = _run_box(tmp_path, "NVIDIA GeForce RTX 5090, 4, 16", probe_rc=0)
    assert rc == 13 and not (w / "reached").exists()


def test_a_host_that_passes_the_probe_goes_on_to_the_install(tmp_path):
    rc, w = _run_box(tmp_path, "NVIDIA GeForce RTX 5090, 5, 16", probe_rc=0)
    assert rc != 18
    assert (w / "reached").exists(), "a passing probe must hand over to the tripwire/install"


@pytest.mark.parametrize("probe_rc", [1, 2])
def test_a_probe_error_that_is_not_an_oom_never_names_the_host(tmp_path, probe_rc):
    """1 = an uncaught exception (an image torch without kernels for the card, a launch failure), 2 = no CUDA device:
    neither is evidence against the host's memory, so neither may become rc 18 (rent.py would exclude a healthy machine)."""
    rc, w = _run_box(tmp_path, "NVIDIA GeForce RTX 5090, 5, 16", probe_rc=probe_rc)
    assert rc == 9 and not (w / "REFUSAL").exists() and not (w / "reached").exists()
    assert "not a host refusal" in (w / "summary.txt").read_text()
