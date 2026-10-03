"""Lane SC1 amendment A10 (#846): the defects box B's full run `sc1b-5090-1` exposed, and the stall box C's `sc1c-5090-1`
hit. Each test fails on the registered drivers and passes on the fix. CPU only: fake servers and stand-in packages."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "sc1"
LLAMACPP_BOX = LANE / "llamacpp" / "llamacpp_box.sh"
SGLANG_SERVER = LANE / "sglang" / "server.sh"
RUN = (LANE / "sc1_run.sh").read_text()

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash drivers")


def _bash(script: str, env=None, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **(env or {})})


# ---------------------------------------------------------------------------------------------------------- llama.cpp
FAKE_LLAMA_SERVER = textwrap.dedent("""\
    #!/bin/bash
    # a stand-in for llama-server at commit 552f18f: library INFO lines (the offload and flash_attn lines) are emitted only
    # at -lv >= 4 (common/log.cpp common_log_get_verbosity maps GGML INFO to LOG_LEVEL_TRACE = 4; default threshold 3)
    np=1; lv=3; prev=""
    for a in "$@"; do
      case "$prev" in -np) np=$a;; -lv) lv=$a;; esac; prev=$a
    done
    # the real order: load_tensors (offload) and llama_context (flash_attn) during the load, then the server's slot line
    if [ "$lv" -ge 4 ]; then
      echo "load_tensors: offloaded ${FAKE_OFFLOAD:-49/49} layers to GPU"
      echo "llama_context: flash_attn            = ${FAKE_FA:-enabled}"
    fi
    echo "0.00.10 I srv    load_model: initializing, n_slots = $np, n_ctx_slot = 1024"
    exec sleep 120
    """)
FAKE_CURL = textwrap.dedent("""\
    #!/bin/bash
    # like the real server: /health answers ok only once loading is done, i.e. after the startup lines are in the log
    for a in "$@"; do case "$a" in */health) grep -q "n_slots" "$FAKE_LOG" 2>/dev/null || exit 7; echo '{"status":"ok"}'; exit 0;; esac; done
    exit 0
    """)


def _fake_bin(tmp_path: Path) -> Path:
    b = tmp_path / "bin"
    b.mkdir()
    for name, body in (("llama-server", FAKE_LLAMA_SERVER), ("curl", FAKE_CURL)):
        (b / name).write_text(body)
        (b / name).chmod(0o755)
    return b


def _llamacpp_start(tmp_path: Path, extra_env=None) -> subprocess.CompletedProcess:
    b = _fake_bin(tmp_path)
    log = tmp_path / "server.log"
    script = (f'. "{LLAMACPP_BOX}"; llamacpp_server_dump(){{ :; }}; LLAMACPP_BIN="{b}"; '
              f'llamacpp_server_start /x/Qwen3-30B-A3B-Q4_K_M.gguf 1 18080 "{log}"; rc=$?; '
              f'LLAMACPP_STOP_TERM_S=2 LLAMACPP_STOP_WAIT_S=2 llamacpp_server_stop; echo "RC=$rc"')
    return _bash(script, env={"PATH": f"{b}:{os.environ['PATH']}", "FAKE_LOG": str(log), **(extra_env or {})})


def test_a10_llama_server_runs_at_a_verbosity_that_logs_the_engagement_lines(tmp_path):
    """sc1b-5090-1: every llama-server start was refused ('no offload line') because the pinned build logs the offload and
    flash_attn lines only at -lv >= 4. With -lv 4 in the flags the registered checks see them."""
    r = _llamacpp_start(tmp_path)
    assert "RC=0" in r.stdout, r.stdout + r.stderr
    assert "-lv 4" in (tmp_path / "server.log").read_text().splitlines()[0]


@pytest.mark.parametrize("env,needle", [({"FAKE_OFFLOAD": "48/49"}, "not every layer is on the GPU"),
                                        ({"FAKE_FA": "auto"}, "flash_attn = enabled")])
def test_a10_the_engagement_checks_still_refuse(tmp_path, env, needle):
    r = _llamacpp_start(tmp_path, env)
    assert "RC=1" in r.stdout and needle in r.stdout + r.stderr, r.stdout + r.stderr


# ------------------------------------------------------------------------------------- bounded stops (box C's stall)
# A process stuck in the GPU driver is in uninterruptible sleep and ignores SIGKILL. Stand-in: a real child (`sleep 300`)
# and a `kill` shell function that delivers nothing but answers `kill -0` truthfully. A registered `wait "$pid"` then
# blocks for the child's whole life; the fix gives up after the bound and reports the pid as stuck.
STUCK_KILL = 'kill(){ case "$1" in -0) builtin kill -0 "$2";; *) return 0;; esac; }'


def _stuck_stop(stop_call: str, env: dict) -> tuple[subprocess.CompletedProcess, float]:
    script = (f"sleep 300 & SPID=$!; {STUCK_KILL}; {stop_call}; rc=$?; "
              'echo "STUCK=${SGLANG_STOP_STUCK:-}"; builtin kill -9 $SPID 2>/dev/null; echo "RC=$rc"')
    t0 = time.monotonic()
    r = _bash(script, env=env, timeout=40)
    return r, time.monotonic() - t0


def test_a10_sglang_stop_never_waits_on_a_stuck_process():
    call = (f'. "{SGLANG_SERVER}"; SGLANG_SERVER_PID=$SPID; SGLANG_SERVER_PGID=""; SGLANG_SERVER_PORT=""; '
            "sglang_server_stop")
    r, wall = _stuck_stop(call, {"SGLANG_STOP_TERM_S": "1", "SGLANG_STOP_WAIT_S": "2"})
    assert "RC=0" in r.stdout and wall < 30, (wall, r.stdout, r.stderr)
    assert "STUCK=" in r.stdout and "STUCK=\n" not in r.stdout + "\n", r.stdout
    assert "STOP STUCK" in r.stdout + r.stderr


def test_a10_llamacpp_stop_never_waits_on_a_stuck_process():
    call = f'. "{LLAMACPP_BOX}"; LLAMACPP_SERVER_PID=$SPID; llamacpp_server_stop'
    r, wall = _stuck_stop(call, {"LLAMACPP_STOP_TERM_S": "1", "LLAMACPP_STOP_WAIT_S": "2"})
    assert "RC=0" in r.stdout and wall < 30, (wall, r.stdout, r.stderr)
    assert "STOP STUCK" in r.stdout + r.stderr


def _fn(name: str) -> str:
    """One function definition from sc1_run.sh: from `name(){` up to the line before the next top-level definition or
    comment (the script's one-function-per-block layout), so inner `{ ...; }` groups never end it early."""
    lines = RUN.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(f"{name}(){{"))
    j = next((k for k in range(i + 1, len(lines)) if lines[k] and not lines[k][0].isspace()), len(lines))
    return "\n".join(lines[i:j])


def test_a10_sglang_transitions_are_visible_and_a_stuck_stop_ends_sglang_on_the_box(tmp_path):
    """sc1c-5090-1 stalled for 1 h 52 min after an SGLang arm and the heartbeat could only show the arm before it. Every
    transition now writes a summary line before it starts, and a stop that leaves a stuck pid refuses later starts."""
    (tmp_path / "logs").mkdir()
    stubs = ('line(){ echo "LINE $*"; }; say(){ :; }; have(){ return 0; }; W=.; GPTQ_MID=m; GPTQ_REV=r; '
             'SGL_MODE=""; SGL_LOG=""; SGL_DEAD=""; sglang_server_start(){ echo SGLANG_ENGAGEMENT {}; return 0; }; ')
    stuck = 'sglang_server_stop(){ SGLANG_STOP_STUCK=4242; }; '
    script = (stubs + stuck + _fn("sglang_up") + "\n" + _fn("sglang_down") + "\n"
              'sglang_up matched; sglang_down; sglang_up native; echo "RC=$? DEAD=$SGL_DEAD MODE=$SGL_MODE"')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path, timeout=30)
    out = r.stdout
    assert "LINE sglang server mode=matched starting" in out, out + r.stderr
    assert "LINE sglang server mode=matched stopping" in out
    assert "STOP STUCK pid 4242" in out
    assert "LINE sglang server mode=native NOT STARTED" in out and "RC=46 DEAD=4242" in out


# ------------------------------------------------------------------------------------------------------- ExLlamaV3
def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def exl3_stand_in(tmp_path, monkeypatch):
    """ExLlamaV3 1.5.3's layout: __version__ lives in exllamav3/version.py, and __init__ does not import it."""
    pkg = tmp_path / "exllamav3"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "version.py").write_text('__version__ = "1.5.3"\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    for m in [m for m in sys.modules if m == "exllamav3" or m.startswith("exllamav3.")]:
        monkeypatch.delitem(sys.modules, m)
    import exllamav3  # noqa: F401
    assert not hasattr(sys.modules["exllamav3"], "version")       # the trap: unbound until imported
    return tmp_path


def test_a10_both_exl3_drivers_read_the_version_module(exl3_stand_in):
    """sc1b-5090-1: every nll_exl3_* run died on `exllamav3.version.__version__` AFTER loading and scoring, and the speed
    arms' receipts recorded versions.exllamav3 = None. Both drivers now import the version module."""
    for f in ("sc1_exl3_nll.py", "sc1_exl3_arm.py"):
        mod = _load(LANE / "exl3" / f, f"a10_{f[:-3]}")
        assert mod.exl3_version() == "1.5.3", f
        src = (LANE / "exl3" / f).read_text()
        assert "exllamav3.version.__version__" not in src and "self.exl.version" not in src, f
    nll = (LANE / "exl3" / "sc1_exl3_nll.py").read_text()
    assert nll.index("exl3_ver = exl3_version()") < nll.index("model.load("), "read the version BEFORE the load"


# ----------------------------------------------------------------------------------------------------- box B's proof
def _box_b_proof_block() -> str:
    prove = RUN[RUN.index('if [ "$PROVE" = 1 ]; then'):RUN.index(": > PROVED; finish 0")]
    return prove[prove.index('  if [ "$BOX" = B ]; then'):prove.index('  if [ "$BOX" = C ]; then')]


def _rec() -> str:
    import re
    m = re.search(r"rc_any=0; rec\(\)\{.*?\}", RUN)
    assert m, "rec() moved"
    return m.group(0)


def _run_box_b_proof(tmp_path: Path, stubs: str) -> str:
    (tmp_path / "logs").mkdir(exist_ok=True)
    base = (f'{_rec()}\nsay(){{ :; }}; line(){{ :; }}; BOX=B; W="{tmp_path}"; PY=python3; '
            'GGUF_REPO=r; GGUF_Q4KM=q.gguf; GGUF_REV=v; arm_alarm(){ echo 60; }; '
            'perl(){ return 0; }; llamacpp_up(){ return 0; }; llamacpp_down(){ :; }; can_run(){ return 0; }\n')
    script = base + stubs + "\n" + _box_b_proof_block() + '\nprintf "RC=%s STEPS=%s" "$rc_any" "$PB_STEPS"'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path, timeout=30).stdout


def test_a10_box_b_proof_starts_the_llama_server_it_could_not_start(tmp_path):
    """sc1b-prove-10 PROVED on installs + tripwires and Granite smokes; the full run then refused every llama-server. Box B's
    proof now starts llama-server through the registered checks on the lane's Q4_K_M. A skipped or failed fetch or start
    is NOT PROVED, never a pass."""
    assert _run_box_b_proof(tmp_path, "") == "RC=0 STEPS= llama_server"
    assert _run_box_b_proof(tmp_path, "llamacpp_up(){ return 1; }").startswith("RC=23")             # refused start
    assert _run_box_b_proof(tmp_path, "perl(){ return 9; }").startswith("RC=23")                    # GGUF fetch failed
    deadline = 'can_run(){ case $2 in prove_llama_server) return 1;; *) return 0;; esac; }'
    assert _run_box_b_proof(tmp_path, deadline).startswith("RC=23")                                 # skipped = not proved
    assert 'comparators=[${PB_STEPS# }]' in RUN


def test_a10_lmdeploy_is_unsupported_on_box_b_with_the_reason(tmp_path):
    """LMDeploy 0.18.0 TurboMind W4A16 runs its SM80 GEMM kernels on sm_120, where each kernel body is compiled out
    (Sm80 = Arch<800, 900>): every 4-bit GEMM is a no-op and sc1b-5090-1 aborted every run. Box B records P5b's stated
    alternative instead of installing it, and every LMDeploy arm receipt carries the reason."""
    import json
    import re
    assert 'B) install_vllm; install_llamacpp; install_exl3 cu128; unsupported lmdeploy "$LMD_UNSUPPORTED" ;;' in RUN
    m = re.search(r'^LMD_UNSUPPORTED="(.*)"$', RUN, re.M)
    assert m and "SM80 GEMM fallback compiles to empty kernels" in m.group(1)
    (tmp_path / "logs").mkdir()
    script = (f'{m.group(0)}\nW="{tmp_path}"; BASEPY={sys.executable}; GPTQ_DIR=""; line(){{ :; }}; have(){{ return 1; }}\n'
              + _fn("stem_of") + "\n" + _stub_fn() + "\n" + _fn("lmdeploy_arm") + "\n" + "lmdeploy_arm w4a16_b16_r1 16")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path, timeout=30)
    rec = json.loads((tmp_path / "lmdeploy_w4a16_b16_r1.json").read_text())
    assert rec["status"] == "unsupported" and "SM80 GEMM fallback" in rec["reason"], (rec, r.stderr)


def _stub_fn() -> str:
    lines = RUN.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("stub(){"))
    j = next(k for k in range(i, len(lines)) if lines[k] == "}")
    return "\n".join(lines[i:j + 1])




# ---------------------------------------------------------------------------------------- the sched decode estimator
def test_a10_sched_decode_reading_cancels_each_reps_prefill():
    """sc1b-5090-1's scheduler draws read 225.8 / 178.7 / 214.9 tok/s at B=1 (UNSTABLE): the registered slope differences
    two walls that each carry e4b's ~2.1 s prefill, so prefill jitter of a few percent became 10-25 % of their 0.45 s
    difference. The driver now subtracts each rep's own last-row Request.ttft before the slope (decode-only time); the
    registered wall slope is recorded beside it."""
    mod = _load(LANE / "sc1_e4b_sched.py", "a10_sched")
    pre_s, pre_l = [2.2, 2.3, 2.25], [2.0, 2.1, 2.05]                  # slow prefills on the short reps, fast on the long
    ws = [p + 31 * 0.0044 for p in pre_s]
    wl = [p + 127 * 0.0044 for p in pre_l]
    d = mod.decode_only_slope(ws, [[p] for p in pre_s], wl, [[p] for p in pre_l], batch=1)
    assert d["decode_only_decode_tok_s"] == round(1 / 0.0044, 1)        # 4.4 ms/step recovered exactly
    assert mod.slope(ws, wl, batch=1)["decode_tok_s"] > 400             # the registered wall slope, misled by the jitter
    b16 = mod.decode_only_slope(ws, [[p] * 16 for p in pre_s], wl, [[p - 0.1] + [p] * 15 for p in pre_l], batch=16)
    assert b16["decode_only_decode_tok_s"] == round(16 * 96 / (96 * 0.0044), 1)   # the LAST row's ttft is the cut, not the first
    assert mod.decode_only_slope(ws, [[2.2], None, [2.25]], wl, [[p] for p in pre_l], batch=1)["decode_only_status"] == "void"
    assert "DECODE-ONLY" in mod.METHOD
    src = (LANE / "sc1_e4b_sched.py").read_text()
    assert '"ttft": [getattr(q, "ttft", None) for q in ordered]' in src and 'estimator="decode-only slope (A10)' in src
    assert subprocess.run([sys.executable, str(LANE / "sc1_e4b_sched.py"), "--selftest"], capture_output=True, text=True).returncode == 0


# ------------------------------------------------------------------------------------------- incremental pulls (driver)
DRIVE = (LANE / "sc1_drive.sh").read_text()
SYSTEM_BASH = "/bin/bash" if Path("/bin/bash").exists() else "bash"


@pytest.mark.parametrize("mode,low", [("", False), ("low", True)])
def test_a10_pull_box_runs_under_the_controllers_bash(tmp_path, mode, low):
    """sc1c-5090-1's box stopped answering at 20:55Z and the single end-of-run fetch returned nothing. The driver now pulls
    the box's results every SC1_PULL_EVERY_S (low priority on the box) and keeps the last pull when the final fetch fails.
    The controller (the mini) runs /bin/bash 3.2 under `set -u`, where an empty "${arr[@]}" is an unbound-variable error:
    run the driver's own pull_box there with a fake rsync and check the arguments."""
    import re
    fn = re.search(r'^pull_box\(\) \{.*?"\$1/"; \}$', DRIVE, re.S | re.M)
    assert fn, "pull_box moved"
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "rsync").write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$PULL_ARGS"\n')
    (tmp_path / "bin" / "rsync").chmod(0o755)
    args = tmp_path / "args"
    script = (f"set -uo pipefail; E4B_RENT_SSH_OPTS='-o X=1'; PORT=22; HOST=h; W=/root/sc1\n{fn.group(0)}\n"
              f'pull_box "{tmp_path}/dest" {mode}; echo "RC=$?"')
    r = subprocess.run([SYSTEM_BASH, "-c", script], capture_output=True, text=True, timeout=30,
                       env={**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "PULL_ARGS": str(args)})
    assert "RC=0" in r.stdout, r.stdout + r.stderr
    got = args.read_text().splitlines()
    assert "-az" in got and "--timeout=120" in got and "root@h:/root/sc1/" in got
    assert any(a.startswith("--rsync-path=nice -n 19 ionice -c3 rsync") for a in got) is low
    assert "--exclude" in got and "venv*" in got and "gguf/" in got


def test_a10_the_receipt_keeps_the_last_pull_when_the_final_fetch_fails():
    tail = DRIVE[DRIVE.index('if ! pull_box "$RUN_DIR/sc1"; then'):]
    assert 'mv "$RUN_DIR/sc1.partial" "$RUN_DIR/sc1"' in tail and 'echo "$PARTIAL_AT" > "$RUN_DIR/sc1/PARTIAL_PULL_AT"' in tail
    assert tail.index("exit 22") < tail.index('rm -rf "$RUN_DIR/sc1.partial"')       # a kept partial is still exit 22
    loop = DRIVE[DRIVE.index("while :; do"):DRIVE.index("done\n", DRIVE.index("while :; do"))]
    assert 'pull_box "$RUN_DIR/sc1.partial" low' in loop and '[ -n "$hb" ]' in loop   # only while the box answers
