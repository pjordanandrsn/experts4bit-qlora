"""PGX shakedown kit (bench/pgx/PREREG-pgx.md, Amendment 1) at $0: every box step that needs no GPU, run against this commit.

- the driver stages exactly the files ``staged-pgx.sha256`` pins (a dry run in a clean copy of this tree), and its run ids are
  the runner's;
- the runner's exit codes: the registered set, the launcher's host codes only for genuine host faults (13 the disk, 18 the CUDA
  floor and the driver), no code that equals a driver or drive_tail literal (e4b#2141), and every box code passed by the driver;
- the tripwire's checks (prompt sets, fit profile, K table, ``serve_paged``'s configuration) on the repo's own modules;
- the shake verdict's and the box helper's self-tests;
- both hooks load where the box loads them: the SGLang routing hook inside a VENV (through ``hook/sitecustomize.py``; without it a
  venv never imports ``usercustomize``), and the E-hyb placement-proof hook wraps ``enable_hybrid_tier`` and records the tiers;
- the box's local safetensors-header reader, and the pins the runner and the tripwire share.
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
    monkeypatch.setattr(tw, "check_box", lambda *a: None)            # the CPU stand-in for the installed-pin step
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    rc = tw.main(["--kit", str(PGX), "--box", "0" * 40, "2.8.0+cu128"])
    out = capsys.readouterr().out
    if want is None:
        assert rc == 0 and "PGX_TRIPWIRE ok serve stack" in out, out
    else:
        assert rc == 9 and "PGX_TRIPWIRE FAIL serve stack" in out and want in out, out


def test_the_runner_installs_the_serve_stack_the_tripwire_checks():
    import pgx_tripwire as tw
    assert f'"fastapi=={tw.FASTAPI_VERSION}" "uvicorn=={tw.UVICORN_VERSION}"' in RUN
    assert '--box "$E4B_SHA" "$TORCH_WANT"' in RUN


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
