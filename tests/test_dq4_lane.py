"""Lane DQ4 (bench/dq4/DQ4-PREREG.md): the reducer's self-test passes, every rule mutant is caught by it, the box runner
refuses wrong hosts before any install, runs the graded configuration first, and pins what the prereg registers. CPU-only."""
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "dq4"
RUN = (LANE / "dq4_run.sh").read_text()
CAP = (LANE / "dq4_cap.py").read_text()


def test_the_reducer_self_test_passes():
    out = subprocess.run([sys.executable, str(LANE / "dq4_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


MUTANTS = [
    ("G_REAL, G_MARGINAL = 1.5, 1.1", "G_REAL, G_MARGINAL = 1.2, 1.1"),
    ("G_REAL, G_MARGINAL = 1.5, 1.1", "G_REAL, G_MARGINAL = 1.5, 0.9"),
    ('if a.get("rehearsal"):', "if False:"),
    ("if a.get(\"device\") != REGISTERED_DEVICE:", "if False:"),
    ('if a.get("config_overrides") != {"num_hidden_layers": N_LAYERS}:', "if False:"),
    ('if loss == "chunked" and not (a.get("chunked_loss") or {}).get("patched"):', "if False:"),
    ('if a.get("alloc_conf") != alloc:', "if False:"),
    ('if off.get("layers") != N_LAYERS or off.get("late_bound_4bit") != WRAPPED:', "if False:"),
    ('if r is None or not r["ok"]:', "if False:"),
    ('if r is None or r["ok"]:', "if False:"),
    ('if r["ok"] and not all(s["finite"] for s in r["steps"]):', "if False:"),
    ('if R.get("ceiling"):', "if False:"),
    ("if len(shas) > 1:", "if False:"),
    ('out["G"] = (S["L"] / R["L"])', 'out["G"] = (R["L"] / S["L"])'),
    ('out["void"].append(f"{arm}: no fresh confirmation at L*")', "pass"),
]


def test_every_rule_mutant_is_killed(tmp_path):
    src = (LANE / "dq4_reduce.py").read_text()
    survived = []
    for old, new in MUTANTS:
        assert src.count(old) == 1, f"mutant anchor not unique/absent: {old!r}"
        m = tmp_path / "mut.py"
        m.write_text(src.replace(old, new))
        out = subprocess.run([sys.executable, str(m), "--self-test"], capture_output=True, text=True)
        if "self-test FAILED" not in out.stdout:
            survived.append((old, (out.stdout + out.stderr)[-200:]))
    assert not survived, survived


def test_the_box_runs_the_registered_configurations_graded_first():
    i_c, i_e, i_s = RUN.index('"c_def chunked default 1"'), RUN.index('"c_exp chunked expandable 1"'), RUN.index('"s_def stock default 0"')
    assert i_c < i_e < i_s
    for pin in ('case "$link" in *"RTX 5090, 5, 16")', "transformers==5.18.0", "peft==0.21.2", "bitsandbytes==0.50.2",
                "_bnb_mirror_mismatches() == []", '"Qwen3ForCausalLM" in SUPPORTED', "dq4_reduce.py --self-test",
                "STEP=1024", 'conf="expandable_segments:True"'):
        assert pin in RUN, pin
    assert subprocess.run(["bash", "-n", str(LANE / "dq4_run.sh")]).returncode == 0
    for s in ("enable_dense_offload(pm, pin=True, prefetch=False, train_prefetch=True)", "enable_chunked_lm_loss(pm, chunk=512)",
              "except torch.OutOfMemoryError", "if not r[\"ok\"]:\n            break", '"late_bound_4bit"'):
        assert s in CAP, s


def _fake_bin(tmp_path, link, vram_rc=0, egress_rc=0):
    b = tmp_path / "bin"
    b.mkdir(parents=True)
    (b / "nvidia-smi").write_text(f"#!/bin/sh\necho '{link}'\n")
    (b / "python").write_text(
        "#!/bin/sh\n"
        f'case "$*" in *dq3_vram_probe.py*) exit {vram_rc};; esac\n'
        f'case "$*" in *dq3_egress_probe.py*) exit {egress_rc};; esac\n'
        'echo "$*" >> "$DQ4_W/reached"; exit 1\n')
    for f in b.iterdir():
        f.chmod(0o755)
    return b


def _run_box(tmp_path, link, **kw):
    w = tmp_path / "w"
    w.mkdir(parents=True)
    env = {"PATH": f"{_fake_bin(tmp_path, link, **kw)}:/usr/bin:/bin", "DQ4_W": str(w), "TC1_RUN_NONCE": "n",
           "E4B_SHA": "0" * 40}
    rc = subprocess.run(["bash", str(LANE / "dq4_run.sh")], env=env, capture_output=True, text=True).returncode
    return rc, w


def test_the_box_refuses_wrong_hosts_before_any_install(tmp_path):
    assert _run_box(tmp_path / "a", "NVIDIA GeForce RTX 5090, 4, 16")[0] == 13
    rc, w = _run_box(tmp_path / "b", "NVIDIA GeForce RTX 5090, 5, 16", vram_rc=3)
    assert rc == 18 and not (w / "reached").exists()
    rc, w = _run_box(tmp_path / "c", "NVIDIA GeForce RTX 5090, 5, 16", egress_rc=4)
    assert rc == 14 and not (w / "reached").exists()
    rc, w = _run_box(tmp_path / "d", "NVIDIA GeForce RTX 5090, 5, 16", vram_rc=1)
    assert rc == 9 and not (w / "REFUSAL").exists()
    rc, w = _run_box(tmp_path / "e", "NVIDIA GeForce RTX 5090, 5, 16")
    assert rc == 9 and (w / "reached").exists()          # a sound host proceeds to the install (which the fake fails)


def test_the_rehearsal_gate_asserts_the_capacity_direction():
    import importlib.util
    spec = importlib.util.spec_from_file_location("dq4_rehearsal_check", LANE / "dq4_rehearsal_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def arm(name, max_ok, first_oom, late=56):
        a = {"arm": name, "finished_at": "t", "chunked_loss": {"patched": True}, "max_ok": max_ok, "first_oom": first_oom,
             "rungs": [{"seq": max_ok, "ok": True, "steps": [{"finite": True}]}]}
        if name == "S":
            a["offload"] = {"layers": 8, "late_bound_4bit": late}
        return a

    assert mod.check(arm("R", 4096, 4608), arm("S", 5632, 6144))[1] == []
    assert any("CAPACITY DIRECTION" in b for b in mod.check(arm("R", 4096, 4608), arm("S", 4096, 4608))[1])
    assert any("CAPACITY DIRECTION" in b for b in mod.check(arm("R", 4096, 4608), arm("S", 3584, 4096))[1])
    assert any("late_bound_4bit" in b for b in mod.check(arm("R", 4096, 4608), arm("S", 5632, 6144, late=0))[1])
    assert any("ladder's top" in b for b in mod.check(arm("R", 4096, None), arm("S", 5632, None))[1])
