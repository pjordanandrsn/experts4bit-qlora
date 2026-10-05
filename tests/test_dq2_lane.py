"""Lane DQ2 (bench/dq2/DQ2-PREREG.md): the reducer's self-test passes, every rule mutant is caught by it, and the box, the
layer census and the pre-registration agree on the pins and the subject. CPU-only."""
import ast
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "dq2"
RUN = (LANE / "dq2_run.sh").read_text()
PREREG = (LANE / "DQ2-PREREG.md").read_text()
LAYER = (LANE / "dq2_layer.py").read_text()


def _const(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def test_the_reducer_self_test_passes():
    out = subprocess.run([sys.executable, str(LANE / "dq2_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (27 cases)" in out.stdout, out.stdout + out.stderr


def test_the_registered_constants():
    r = LANE / "dq2_reduce.py"
    assert _const(r, "REGISTERED_ROWS") == [512, 1024, 2048, 4096, 8192]
    assert _const(r, "REGISTERED_LINK") == {"gen_max": 5, "width_max": 16}
    assert _const(r, "GRADED_ROW") == 2048 and _const(r, "SELF_PAIR_BAND") == (0.95, 1.05)
    assert _const(r, "NOISY_MAX_OUT") == 1 and _const(r, "MIN_GEN5_ALONE_GBS") == 35.0
    assert 'default="512,1024,2048,4096,8192"' in LAYER


def test_the_box_pins_what_the_prereg_registers():
    for pin in ("a5edec8789735bff1c0da4708ae5fc93260a1410", "bitsandbytes==0.50.2", "transformers==5.18.0", "peft==0.21.2"):
        assert pin in RUN, pin
    for pin in ("a5edec87", "0.50.2", "5.18.0", "0.21.2"):
        assert pin in PREREG, pin
    assert "dq2_reduce.py --self-test" in RUN and "pcie.link.gen.max" in RUN


def test_the_box_speaks_tc1_drives_contract():
    for marker in ("> TC1_RUN_NONCE", "TC1_EXIT_CODE.$NONCE", "TC1_SUCCESS.$NONCE", "TP_DONE.$NONCE", "summary.txt"):
        assert marker in RUN, marker
    assert subprocess.run(["bash", "-n", str(LANE / "dq2_run.sh")]).returncode == 0


def test_the_subject_is_hf_peft_bnb_qlora():
    for s in ("Qwen3DecoderLayer", "bnb.nn.Linear4bit", "compress_statistics=True", 'quant_type="nf4"',
              "inject_adapter_in_model", "LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0", 'use_reentrant=False',
              '_attn_implementation = "sdpa"', "layer.is_loaded_in_4bit = True",
              'cast_adapter_dtype(layer, adapter_name="default", autocast_adapter_dtype=True)'):
        assert s in LAYER, s


MUTANTS = [
    ('g["Rmin"] >= 1.25 and', 'g["Rmin"] >= 1.10 and'), ('g["Rmin"] < 1.0:', 'g["Rmin"] < 0.5:'),
    ('g["fwd_slowdown"] <= 1.05', "True"), ('g["copy_ratio"] >= 0.8', "True"),
    ("NOISY_MAX_OUT = 1 ", "NOISY_MAX_OUT = 3 "), ("SELF_PAIR_BAND = (0.95, 1.05)", "SELF_PAIR_BAND = (0.5, 2.0)"),
    ("MIN_GEN5_ALONE_GBS = 35.0", "MIN_GEN5_ALONE_GBS = 0.0"),
    ("if any(ln.get(k) != v for k, v in REGISTERED_LINK.items()):", "if False:"),
    ("if r.get(\"forensics\", {}).get(\"device\") != REGISTERED_DEVICE:", "if False:"),
    ('"Rmin": min(tf, tb) / x_ms', '"Rmin": tf / x_ms'),
    ("if r.get(\"rehearsal\"):", "if False:"), ("if not r.get(\"finished_at\"):", "if False:"),
    ("or r.get(\"frozen_sha256_before\") != r.get(\"frozen_sha256_after\"):", ":"),
    ('or p.get("integrity", {}).get("lora_grads_nonzero", 0) < LORA_TARGETS]', "]"),
    ('if not c.get("warm", {}).get("steady"):', "if False:"),
    ('not all("peft" in v and "bnb" in v.lower() for v in wr.values())', "False"),
    ('uncovered |= not all(', 'uncovered |= False and all('),
    ('or eng.get("lora_dtypes") != ["torch.float32"]):', "):"),
]


def test_every_rule_mutant_is_killed(tmp_path):
    src = (LANE / "dq2_reduce.py").read_text()
    survived = []
    for old, new in MUTANTS:
        assert src.count(old) == 1, f"mutant anchor not unique/absent: {old!r}"
        m = tmp_path / "mut.py"
        m.write_text(src.replace(old, new))
        out = subprocess.run([sys.executable, str(m), "--self-test"], capture_output=True, text=True)
        if "self-test FAILED" not in out.stdout:
            survived.append((old, (out.stdout + out.stderr)[-200:]))
    assert not survived, survived
