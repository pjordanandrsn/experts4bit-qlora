"""PGX shakedown kit (bench/pgx/PREREG-pgx.md, Amendment 1) at $0: every box step that needs no GPU, run against this commit.

- the driver stages exactly the files ``staged-pgx.sha256`` pins (a dry run in a clean copy of this tree), and its run ids are
  the runner's;
- the runner's exit codes: the registered set, the launcher's host codes only for genuine host faults (13 the disk, 18 the CUDA
  floor and the driver), no code that equals a driver or drive_tail literal (e4b#2141), and every box code passed by the driver;
- the tripwire's checks (prompt sets, fit profile, K table, ``serve_paged``'s configuration) on the repo's own modules;
- the shake verdict's and the box helper's self-tests;
- both hooks load where the box loads them: the SGLang routing hook inside a VENV (through ``hook/sitecustomize.py``; without it a
  venv never imports ``usercustomize``), and the E-hyb placement-proof hook wraps ``enable_hybrid_tier`` and records the tiers;
- the box's local safetensors-header reader, and the pins the runner and the tripwire share;
- Amendment 2: the image-python pins on the runner's line and in the tripwire; the tripwire's bake chain (the imports and calls
  read from ``k8_bake.py``) and calibrate steps, on this commit and on fixtures for each way they fail; ``bake_why`` copying
  ``work/bake.json`` into ``logs/bake.log``; the driver's fetch rules carrying ``work/*.json`` and nothing else from ``work/``;
  and the bake's exact load call on a shardless snapshot, which must stop only at the first shard read.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PGX = REPO / "bench" / "pgx"
RUN = (PGX / "pgx_run.sh").read_text()
DRIVE = (PGX / "pgx_drive.sh").read_text()
sys.path.insert(0, str(PGX))

REGISTERED = {0, 9, 10, 11, 12, 13, 15, 16, 18, 19, 25, 27, 28, 33, 40, 41, 78, 130}
HOST = {13, 14, 17, 18}
DRIVER_LITERALS = {20, 21, 22, 23, 24, 26}       # the driver's own exits and drive_tail's


def _finish_codes(text: str) -> set:
    return {int(x) for x in re.findall(r"\bfinish (\d+)\b", text)}


def test_the_runner_exits_only_registered_codes():
    codes = _finish_codes(RUN)
    assert codes == REGISTERED, sorted(codes ^ REGISTERED)
    assert not codes & DRIVER_LITERALS
    assert not codes & {14, 17}


def test_host_codes_only_for_host_faults():
    for line in RUN.splitlines():
        for code in _finish_codes(line) & HOST:
            if code == 13:
                assert "REFUSED:" in line and "GB free" in line, line
            else:   # 18: CUDA unusable on this host, or the driver below the wheels' floor
                assert "REFUSAL" in line and ("cuda unusable" in line or "driver" in line), line


def test_the_driver_passes_every_box_code():
    box = {int(x) for x in re.search(r'BOX_CODES="([\d ]+)"', DRIVE)[1].split()}
    assert box == REGISTERED - {0} - HOST
    assert "drive_finish" in DRIVE and ". \"$(cd \"$(dirname \"$0\")/../.." in DRIVE
    assert re.search(r"rsync -az --no-links ", DRIVE)


def test_run_ids_agree():
    pat = "pgx-shake-1|pgx-shake-r[1-3])"
    assert pat in DRIVE and pat in RUN


def test_the_pins_the_runner_and_tripwire_share():
    import pgx_tripwire as tw
    assert f"K_S4={tw.K_AT_8GB['S4']}; K_S8={tw.K_AT_8GB['S8']}; K_V8={tw.K_AT_8GB['V8']}; EHYB_VRAM_GIB={tw.EHYB_VRAM_GIB_8GB};" in RUN
    assert f"GNF4_VERSION={tw.GNF4_VERSION};" in RUN
    import pgx_bytes as pb
    for tag, (repo, rev) in (("BF16", pb.CHECKPOINTS["bf16"]), ("FP8", pb.CHECKPOINTS["fp8"]), ("INT4", pb.CHECKPOINTS["int4"])):
        assert re.search(rf"^{tag}={re.escape(repo)};\s+{tag}_REV={rev}$", RUN, re.M), tag


def test_the_guard_fit_is_amendment_1s():
    v = {k: int(x) for k, x in re.findall(r"\b(INSTALL_S|FETCH_S|BAKE_S|CALIB_S|LIFETIME_S|INSTR_S|RESERVE)=(\d+)", RUN)}
    fit = v["INSTALL_S"] + v["FETCH_S"] + v["BAKE_S"] + v["CALIB_S"] + 4 * v["LIFETIME_S"] + 2 * v["INSTR_S"] + v["RESERVE"]
    assert (fit, fit * 125 // 100) == (10_450, 13_062)
    assert "INSTALL_TO_GATE_S=$((INSTALL_S + FETCH_S + BAKE_S + CALIB_S))" in RUN
    assert "NEED_LIFETIMES=$((4 * LIFETIME_S + 2 * INSTR_S))" in RUN
    assert "FIT=$((INSTALL_TO_GATE_S + NEED_LIFETIMES + RESERVE))" in RUN


@pytest.mark.skipif(shutil.which("git") is None or shutil.which("shasum") is None, reason="needs git and shasum")
def test_the_driver_stages_exactly_the_pinned_files(tmp_path):
    """A dry run in a clean copy of this tree: the driver copies its files, checks them against staged-pgx.sha256, and the
    staged set is exactly the pinned set."""
    tree = tmp_path / "repo"
    for rel in ("bench/pgx", "bench/common", "bench/train-routing-trace/replay.py", "bench/sc1/sglang/install.sh",
                "bench/p39/k8_bake.py"):
        src, dst = REPO / rel, tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__")) if src.is_dir() else shutil.copy2(src, dst))
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t"]):
        subprocess.run(["git", "-C", str(tree), *cmd], check=True, env=env)
    env.update(E4B_RENT_SSH_HOST="h", E4B_RENT_SSH_PORT="1", E4B_RENT_SSH_OPTS="-q", E4B_RENT_RUN_DIR=str(tmp_path / "out"),
               E4B_RENT_RUN_ID="pgx-shake-1", E4B_RENT_DEADLINE_EPOCH="9999999999", PGX_DRIVE_DRYRUN="1")
    env.pop("E4B_SHA", None)
    r = subprocess.run(["bash", str(tree / "bench/pgx/pgx_drive.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    pinned = [ln.split(None, 1)[1] for ln in (PGX / "staged-pgx.sha256").read_text().splitlines()]
    assert f"DRYRUN stage {len(pinned) + 1} files" in r.stdout, r.stdout      # + staged-pgx.sha256 itself
    assert "sha256sum -c staged-pgx.sha256" in RUN, "the box re-checks every pinned file"
    env["E4B_RENT_RUN_ID"] = "pgx-4090-1"
    r = subprocess.run(["bash", str(tree / "bench/pgx/pgx_drive.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 78 and "unregistered run ID" in r.stdout


def test_the_tripwire_on_this_commit():
    import pgx_tripwire as tw
    assert tw.main(["--kit", str(PGX)]) == 0


def _web(version_f, version_u):
    import types
    f, u = types.ModuleType("fastapi"), types.ModuleType("uvicorn")
    f.__version__, u.__version__ = version_f, version_u
    return {"fastapi": f, "uvicorn": u}


@pytest.mark.parametrize("mods, want", [
    ({"fastapi": None}, "ModuleNotFoundError"),                  # a broken or skipped [serve] install
    ({"uvicorn": None}, "ModuleNotFoundError"),
    (_web("0.141.0", "0.54.0"), "fastapi 0.141.0 is not"),
    (_web("0.141.1", "0.53.9"), "uvicorn 0.53.9 is not"),
    (_web("0.141.1", "0.54.0"), None),
])
def test_the_box_tripwire_imports_the_serve_stack(monkeypatch, capsys, mods, want):
    """serve_paged imports fastapi and uvicorn lazily, so the --box tripwire must import both by name at their pins; a
    tripwire without those imports fails here (the maintainer's R1 on e4b#2172)."""
    import pgx_tripwire as tw
    monkeypatch.setattr(tw, "check_box", lambda *a: None)            # the CPU stand-ins for the installed-pin steps
    for step in ("check_image_pins", "check_bake_chain", "check_calibrate"):   # Amendment 2's steps: tested on their own below
        monkeypatch.setattr(tw, step, lambda *a: None)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    rc = tw.main(["--kit", str(PGX), "--box", "0" * 40, "2.8.0+cu128", "--bake", "k8_bake.py", "--calib", "calibrate.py"])
    out = capsys.readouterr().out
    if want is None:
        assert rc == 0 and "PGX_TRIPWIRE ok serve stack" in out, out
    else:
        assert rc == 9 and "PGX_TRIPWIRE FAIL serve stack" in out and want in out, out


def test_the_runner_installs_the_serve_stack_the_tripwire_checks():
    import pgx_tripwire as tw
    assert f'"fastapi=={tw.FASTAPI_VERSION}" "uvicorn=={tw.UVICORN_VERSION}"' in RUN
    assert '--box "$E4B_SHA" "$TORCH_WANT"' in RUN


def _pip_line() -> str:
    m = re.search(r'"\$PY_IMG" -m pip install -q --no-input -c \$W/constraints\.txt --prefer-binary \\\n(.*?)> logs/pip_e4b\.log', RUN, re.S)
    assert m, "the image-python install line moved"
    return m.group(1)


def test_the_runner_pins_every_image_package_the_tripwire_checks():
    """Amendment 2 A: every package on the image-python install line is pinned, and the tripwire checks each at that pin."""
    import pgx_tripwire as tw
    line = _pip_line()
    named = set(re.findall(r'"([A-Za-z0-9_.-]+)(?:\[[a-z,]+\])?(?:==|>=| @)', line))
    assert named == set(tw.IMAGE_PINS) | {"experts4bit-qlora"}, sorted(named ^ (set(tw.IMAGE_PINS) | {"experts4bit-qlora"}))
    for name, ver in tw.IMAGE_PINS.items():
        want = f'"{name}==$GNF4_VERSION"' if name == "grouped-nf4-gemm" else f'"{name}=={ver}"'
        assert want in line, want
    assert ">=" not in line and not re.search(r"(^|\s)(accelerate|safetensors|huggingface_hub)(\s|$)", line), line


def test_the_box_tripwire_runs_the_bake_and_calibrate_chains():
    assert '--bake $W/k8_bake.py \\\n  --calib $W/gnf4-src/bench/calibrate.py' in RUN
    import pgx_tripwire as tw
    with pytest.raises(SystemExit):
        tw.main(["--kit", str(PGX), "--box", "0" * 40, "2.8.0+cu128"])         # --box without --bake/--calib is a usage error


def test_the_bake_chain_reads_the_bakes_own_imports_and_calls():
    import pgx_tripwire as tw
    stmts, calls = tw.bake_imports(str(REPO / "bench/p39/k8_bake.py"))
    for st in ("from experts4bit_qlora.loader import load_moe_4bit_streaming", "from nvme_arena import bake_expert_tensors",
               "from experts4bit_qlora.engines.hot_residency import target_modules",
               "from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS", "from safetensors.torch import save_file"):
        assert st in stmts, st
    assert calls == {"load_moe_4bit_streaming": [(5, ["quant_type"])],
                     "bake_expert_tensors": [(2, ["name_template", "kinds", "align"])]}, calls


def test_the_bake_chain_on_this_commit():
    for mod in ("accelerate", "safetensors", "nvme_arena"):
        pytest.importorskip(mod)
    import pgx_tripwire as tw
    tw.check_bake_chain(str(REPO / "bench/p39/k8_bake.py"))


def _fake_bake(tmp_path, monkeypatch, imports: str, call: str):
    (tmp_path / "pgx_fake_loader.py").write_text("def load_moe_4bit_streaming(model_id, device, dtype, a, b, quant_type='nf4'):\n"
                                                 "    pass\n")
    (tmp_path / "pgx_fake_arena.py").write_text("def bake_expert_tensors(snap, arena, name_template, kinds, align=4096):\n"
                                                "    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    bake = tmp_path / "k8_fake.py"
    bake.write_text(f"{imports}\ntry:\n    {call}\n    bake_expert_tensors('s', 'a', name_template='t', kinds=(), align=4096)\n"
                    "except Exception:\n    pass\n")
    return str(bake)


@pytest.mark.parametrize("imports, call, want", [
    ("from pgx_fake_loader import load_moe_4bit_streaming\nfrom pgx_fake_arena import bake_expert_tensors",
     "load_moe_4bit_streaming('m', 'cuda', None, 8, 16, quant_type='nf4')", None),
    ("from pgx_fake_loader import load_moe_4bit_streaming\nfrom pgx_fake_arena import bake_expert_tensors\nimport pgx_no_such_module",
     "load_moe_4bit_streaming('m', 'cuda', None, 8, 16, quant_type='nf4')", "import pgx_no_such_module` fails"),
    ("from pgx_fake_loader import load_moe_4bit_streaming\nfrom pgx_fake_arena import bake_expert_tensors",
     "load_moe_4bit_streaming('m', 'cuda', None, 8, 16, 32, quant_type='nf4')", "does not bind"),
    ("from pgx_fake_loader import load_moe_4bit_streaming\nfrom pgx_fake_arena import bake_expert_tensors",
     "load_moe_4bit_streaming('m', 'cuda', None, 8, 16, quant='nf4')", "does not bind"),
])
def test_the_bake_chain_fails_each_way(tmp_path, monkeypatch, imports, call, want):
    import pgx_tripwire as tw
    bake = _fake_bake(tmp_path, monkeypatch, imports, call)
    if want is None:
        tw.check_bake_chain(bake)
    else:
        with pytest.raises(AssertionError, match=re.escape(want)):
            tw.check_bake_chain(bake)


@pytest.mark.parametrize("builds, src, want", [(True, True, None), (False, True, "does not build"), (True, False, "is missing")])
def test_the_calibrate_step(tmp_path, builds, src, want):
    import pgx_tripwire as tw
    bench = tmp_path / "bench"
    bench.mkdir()
    if src:
        (bench / "hybrid_calib.c").write_text("int main(void){return 0;}\n")
    body = ("    out = workdir / 'hybrid_calib'\n    out.write_text('bin')\n    return out, 'cc -O3'\n" if builds
            else "    return None, 'compile failed: no'\n")
    (bench / "calibrate.py").write_text("from pathlib import Path\nHERE = Path(__file__).resolve().parent\n"
                                        "C_SRC = HERE / 'hybrid_calib.c'\n\n\ndef build_microbench(workdir):\n" + body)
    if want is None:
        tw.check_calibrate(str(bench / "calibrate.py"))
    else:
        with pytest.raises(AssertionError, match=want):
            tw.check_calibrate(str(bench / "calibrate.py"))


@pytest.mark.parametrize("record", [True, False])
def test_bake_why_copies_the_bakes_own_error_into_its_log(tmp_path, record):
    """Amendment 2 C: k8_bake.py prints only "BAKE FAILED"; its error and traceback are in work/bake.json."""
    fn = re.search(r"^bake_why\(\)\{.*?^\}$", RUN, re.S | re.M)
    assert fn, "bake_why moved"
    (tmp_path / "logs").mkdir()
    (tmp_path / "work").mkdir()
    (tmp_path / "logs" / "bake.log").write_text("BAKE FAILED\n")
    if record:
        (tmp_path / "work" / "bake.json").write_text(json.dumps(
            {"step": "start", "status": "FAILED", "err": "ImportError('no module x')", "tb": "Traceback (most recent call last):\n  boom\n"}))
    script = f"W={tmp_path}; PY_IMG={sys.executable}; cd {tmp_path}\n{fn.group(0)}\nbake_why\n"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"})
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "logs" / "bake.log").read_text()
    assert log.startswith("BAKE FAILED\n")
    if record:
        assert "status=FAILED" in log and "err: ImportError('no module x')" in log and "boom" in log, log
    else:
        assert "bake_why: no work/bake.json" in log, log
    assert "PYTHONPATH= perl" in RUN and RUN.count("bake_why;") == 2


@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")
def test_the_fetch_carries_work_json_and_nothing_else_from_work(tmp_path):
    """Amendment 2 C: the driver's rsync rules, run on a fixture tree."""
    m = re.search(r"^PGX_RSYNC_EXCLUDES=\(.*?\)$", DRIVE, re.S | re.M)
    assert m, "PGX_RSYNC_EXCLUDES moved"
    src, dst = tmp_path / "box", tmp_path / "out"
    files = {"work/bake.json": True, "work/nf4.arena": False, "work/nf4snap/model-00000.safetensors": False,
             "work/nf4snap/model.safetensors.index.json": False, "work/notes.txt": False, "summary.txt": True,
             "logs/bake.log": True, "venv-sglang/x.py": False, "calib.json": True}
    for rel in files:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text("x")
    r = subprocess.run(["bash", "-c", f'{m.group(0)}\nrsync -a --no-links "${{PGX_RSYNC_EXCLUDES[@]}}" {src}/ {dst}/'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = {rel: (dst / rel).exists() for rel in files}
    assert got == files, got


def test_the_bakes_load_call_stops_only_at_the_first_shard_read(tmp_path, capfd):
    """Amendment 2 F: the bake's exact call into the loader, on a shardless snapshot (a small Qwen3-MoE config and an index
    naming shards that do not exist). It must log the host and the checkpoint (pgx-shake-1's bake.log had neither) and then
    fail on the first shard: nothing in the imports, the config, the admission or the meta build may stop it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
                         decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True, tie_word_embeddings=False,
                         max_position_embeddings=64)
    cfg.architectures = ["Qwen3MoeForCausalLM"]
    cfg.save_pretrained(tmp_path)
    with torch.device("meta"):
        names = [n for n, _ in Qwen3MoeForCausalLM(cfg).named_parameters() if ".mlp.experts." not in n]
    wm = {n: "model-00001-of-00002.safetensors" for n in names}
    for li in range(2):
        for e in range(8):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                wm[f"model.layers.{li}.mlp.experts.{e}.{proj}.weight"] = "model-00002-of-00002.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    with pytest.raises(FileNotFoundError, match=r"model-0000[12]-of-00002\.safetensors"):
        load_moe_4bit_streaming(str(tmp_path), "cuda", torch.bfloat16, 8, 16, quant_type="nf4")   # k8_bake.py's call
    err = capfd.readouterr()
    assert "host CPUs:" in err.out + err.err and "checkpoint: local directory" in err.out + err.err
    assert 'load_moe_4bit_streaming(MODEL, "cuda",\n                                       torch.bfloat16, 8, 16, quant_type="nf4")' in \
        (REPO / "bench/p39/k8_bake.py").read_text(), "the bake's call moved: update this test's call with it"


@pytest.mark.parametrize("script", ["pgx_shake.py", "pgx_box.py"])
def test_the_box_self_tests(script):
    r = subprocess.run([sys.executable, str(PGX / script), "--self-test"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _venv(tmp_path) -> str:
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(tmp_path / "v")], check=True)
    return str(tmp_path / "v" / ("Scripts" if os.name == "nt" else "bin") / "python")


def test_the_routing_hook_loads_in_a_venv(tmp_path):
    py = _venv(tmp_path)
    probe = "import sys; print('usercustomize' in sys.modules)"
    r = subprocess.run([py, "-c", probe], env={"PYTHONPATH": str(PGX / "hook"), "PATH": os.environ.get("PATH", "")},
                       capture_output=True, text=True)
    assert r.stdout.strip() == "True", r.stdout + r.stderr
    only = tmp_path / "only"
    only.mkdir()
    shutil.copy2(PGX / "hook" / "usercustomize.py", only)
    r = subprocess.run([py, "-c", probe], env={"PYTHONPATH": str(only), "PATH": os.environ.get("PATH", "")},
                       capture_output=True, text=True)
    assert r.stdout.strip() == "False", "a venv now runs usercustomize; hook/sitecustomize.py may be unneeded"


def test_the_placement_hook_records_the_tiers(tmp_path):
    out = tmp_path / "placement.json"
    code = ("from experts4bit_qlora.engines import hybrid\n"
            "man = {'tiers': {'vram': [[0, 1], [0, 2], [1, 5]], 'dram': [[1, 7]], 'nvme': []}, 'masses': {'vram_frac': 0.8}}\n"
            "try:\n    hybrid.enable_hybrid_tier(None, '/arena', man, hot_rows=64)\nexcept Exception:\n    pass\n")
    env = {**os.environ, "PYTHONPATH": f"{PGX / 'hook_e4b'}{os.pathsep}{REPO}", "PGX_PLACEMENT_OUT": str(out)}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert out.exists(), r.stdout + r.stderr
    rec = json.loads(out.read_text())
    assert rec["experts"] == {"vram": 3, "dram": 1, "nvme": 0}
    assert rec["per_layer"] == {"0": {"vram": 2, "dram": 0, "nvme": 0}, "1": {"vram": 1, "dram": 1, "nvme": 0}}


def test_the_local_header_reader(tmp_path):
    import pgx_bytes as pb
    hdr = {"model.layers.0.mlp.experts.0.up_proj.weight": {"dtype": "F8_E4M3", "shape": [4, 8], "data_offsets": [0, 32]}}
    blob = json.dumps(hdr).encode()
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 32)
    got = pb.local(str(tmp_path))
    assert got == {"model-00001-of-00001.safetensors": hdr}
    assert pb.expert_bytes(got) == {(0, 0): 32}


def test_the_staged_manifest_hashes_the_tree():
    rows = [ln.split(None, 1) for ln in (PGX / "staged-pgx.sha256").read_text().splitlines()]
    src = {"sglang_install.sh": REPO / "bench/sc1/sglang/install.sh", "replay.py": REPO / "bench/train-routing-trace/replay.py",
           "card_clock.py": REPO / "bench/common/card_clock.py", "k8_bake.py": REPO / "bench/p39/k8_bake.py"}
    for sha, name in rows:
        path = src.get(name, PGX / name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, name


# ---- Amendment 2, the root cause: pgx-shake-1's runner sourced SC1's sglang.env in its own shell (venv-sglang first on PATH),
# so the bake's bare `python` was SGLang's venv, which has no e4b: ModuleNotFoundError, "BAKE FAILED", rc 12.

def _code_lines(text: str) -> list[str]:
    """The runner's shell lines: heredoc bodies (python sources) and comment lines dropped."""
    out, hd = [], None
    for ln in text.splitlines():
        if hd is not None:
            if ln.strip() == hd:
                hd = None
            continue
        if ln.lstrip().startswith("#"):
            continue
        out.append(ln)
        m = re.search(r"<<-?\s*'?([A-Z_]+)'?", ln)
        if m:
            hd = m.group(1)
    return out


_BARE_PY = re.compile(r'(?<![\w/$"\'.-])python(?= +[-$/"])')    # an invocation: python then a flag, a path or a variable


def test_no_env_file_is_sourced_in_the_runners_own_shell():
    lines = _code_lines(RUN)
    top = [ln for ln in lines if re.match(r"\s*(\.|source)\s", ln)]
    assert not top, f"a top-level source changes the runner's own PATH: {top}"
    sgl = [ln for ln in lines if "sglang.env" in ln]
    assert len(sgl) == 2 and all(re.search(r"\(\s*\.\s+\$W/sglang\.env;\s", ln) for ln in sgl), sgl   # tripwire + server subshells


def test_every_python_call_is_the_bound_image_python_or_a_venv():
    bare = [ln.strip() for ln in _code_lines(RUN) if _BARE_PY.search(ln.replace("command -v python", ""))]
    assert not bare, f"a bare python resolves through PATH: {bare}"
    assert RUN.index("PY_IMG=$(command -v python)") < RUN.index('CUDA_PROBE=$("$PY_IMG"') < RUN.index("sglang_install.sh 2700")
    venv_calls = re.findall(r"\$W/(venv-[a-z]+)/bin/python", RUN)
    assert set(venv_calls) == {"venv-sglang", "venv-vllm"}, venv_calls


@pytest.mark.parametrize("step, marker", [("bake", '"$PY_IMG" $W/k8_bake.py'), ("calibrate", '"$PY_IMG" $W/gnf4-src/bench/calibrate.py'),
                                          ("place_preview", '"$PY_IMG" $W/pgx_place_preview.py'),
                                          ("e-hyb-server", '"$PY_IMG" -m experts4bit_qlora.serve_paged')])
def test_each_e4b_step_is_guarded_by_its_own_interpreter(step, marker):
    """The guard imports e4b and nvme_arena with "$PY_IMG", the interpreter the step runs, right before the step."""
    lines = RUN.splitlines()
    at = next(i for i, ln in enumerate(lines) if marker in ln)
    guard = next(i for i in range(at, -1, -1) if re.search(rf"\bimg_guard {re.escape(step)}\b", lines[i]))
    assert at - guard <= 6, (step, guard, at)
    assert "PYTHONPATH= \"$PY_IMG\" -c 'import experts4bit_qlora, nvme_arena'" in RUN


def _stand_in_venv(tmp_path) -> Path:
    """venv-sglang's python as pgx-shake-1 met it: torch and the stdlib import, e4b does not (the maintainer's repro form)."""
    vbin = tmp_path / "venv-sglang" / "bin"
    vbin.mkdir(parents=True)
    (vbin / "python").write_text(
        "#!/bin/sh\n"
        f"exec {sys.executable} -c '\n"
        "import os, sys\n"
        "class NotInstalled:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split(\".\")[0] in (\"experts4bit_qlora\", \"nvme_arena\"):\n"
        "            raise ModuleNotFoundError(\"No module named \" + repr(name.split(\".\")[0]))\n"
        "sys.meta_path.insert(0, NotInstalled())\n"
        "sys.argv = sys.argv[1:]\n"
        "exec(sys.argv[1]) if sys.argv[:1] == [\"-c\"] else None\n"
        "' \"$@\"\n")
    (vbin / "python").chmod(0o755)
    return vbin.parent


@pytest.mark.parametrize("sequence, want_ok", [("a1", False), ("a2", True)])
def test_the_bakes_interpreter_after_the_sglang_install(tmp_path, sequence, want_ok):
    """Dynamic, in pgx-shake-1's own order: bind PY_IMG (this runner's line), install SGLang (SC1's sglang.env, venv first on
    PATH), then run the bake's import in the bake's interpreter. The Amendment 1 sequence (the env sourced in the runner's
    shell, a bare `python` for the bake) reproduces the box's ModuleNotFoundError; this runner's sequence does not."""
    install = (REPO / "bench/sc1/sglang/install.sh").read_text()
    assert "export PATH=$VENV/bin:$CUDA_HOME/bin:\\$PATH" in install, "SC1's sglang.env no longer puts the venv first"
    venv = _stand_in_venv(tmp_path)
    img = tmp_path / "img"
    img.mkdir()
    (img / "python").write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")      # the image python (a symlink would drop the venv)
    (img / "python").chmod(0o755)
    w = tmp_path / "w"
    w.mkdir()
    (w / "sglang.env").write_text(f"export CUDA_HOME=/nonexistent-cuda\nexport PATH={venv}/bin:/nonexistent-cuda/bin:$PATH\n")
    bind = re.search(r"^PY_IMG=\$\(command -v python\).*$", RUN, re.M).group(0)
    runner_sources = [ln for ln in _code_lines(RUN) if re.match(r"\s*(\.|source)\s", ln)]
    if sequence == "a1":   # Amendment 1: `. $W/sglang.env` at the top level, then `python $W/k8_bake.py`
        body, bake = f". {w}/sglang.env", "python"
    else:                  # this runner: its own top-level sources (none), then "$PY_IMG" $W/k8_bake.py
        body, bake = "\n".join(ln.replace("$W", str(w)) for ln in runner_sources), '"$PY_IMG"'
    script = (f"say(){{ :; }}; finish(){{ exit \"$1\"; }}\nexport PATH={img}:$PATH\nW={w}\n{bind}\n{body}\n"
              f"{bake} -c 'import sys, experts4bit_qlora; print(\"IMPORTED\", sys.executable)'\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if want_ok:
        assert r.returncode == 0 and "IMPORTED" in r.stdout, r.stdout + r.stderr
    else:
        assert r.returncode != 0 and "No module named 'experts4bit_qlora'" in r.stderr, r.stdout + r.stderr
