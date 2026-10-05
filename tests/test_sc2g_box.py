"""Lane SC2g's box G (#846): gpt-oss-20b on four engines. Executed, not grepped, wherever the property is behaviour: the
environment a child inherits, SGLang's engagement check in its `gptoss` mode, and box G's check of the e4b server's own
record."""
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
BOX = (REPO / "bench" / "sc2" / "sc2g_box_g.sh").read_text()
SGL = REPO / "bench" / "sc1" / "sglang" / "server.sh"


def _child_env(box):
    start = RUN.index("# every lever is unset; each arm names its own switches")
    end = RUN.index(": > summary.txt")
    script = f'BOX={box}\nexport E4B_INT4_PREFILL=batched E4B_PAGED_PREFILL_ATTN=bogus\n{RUN[start:end]}\nenv\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)


def test_box_g_children_inherit_neither_prefill_route_pin():
    env = _child_env("G")
    assert "E4B_INT4_PREFILL" not in env and "E4B_PAGED_PREFILL_ATTN" not in env


def test_box_g_is_wired_with_todays_kernel_package():
    assert 'case "$BOX" in A|B|C|D|E|F|G) ;;' in RUN and "G) box_g;;" in RUN and '[ "$BOX" = G ] && prove_g' in RUN
    assert 'G) GNF4_SHA=dc8f94abfd868f149178623f6eb403dc8b892b02;; esac' in RUN   # gnf4 v0.41.0's commit
    assert "G) . $W/sc2_box_e.sh; . $W/sc2g_box_g.sh; install_vllm; install_sglang; install_llamacpp; install_sc2_client ;;" in RUN
    assert 'G) PROVE_NEEDS="vllm sglang llamacpp sc2client";;' in RUN and "servers=[e4b_gptoss vllm sglang llamacpp]" in RUN


def test_box_g_pins_the_checkpoint_the_gguf_and_each_engines_path():
    assert "SC2G_MID=openai/gpt-oss-20b; SC2G_REV=6cee5e81ee83917806bbde320786a8fb61efebee" in BOX
    assert "SC2G_GGUF_REPO=ggml-org/gpt-oss-20b-GGUF; SC2G_GGUF_REV=ef9b12f2ff56c69cf32153a02784e7a3c88bf524; SC2G_GGUF=gpt-oss-20b-MXFP4.gguf" in BOX
    assert "ignore_patterns=['original/*', 'metal/*', 'consolidated*']" in BOX and "HF_HUB_DISABLE_XET=1" in BOX
    assert 'SC2G_E4B_ENV="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 GNF4_TRITON_PREBIND=1"' in BOX
    assert "setsid env -u E4B_INT4_PREFILL -u E4B_PAGED_PREFILL_ATTN PYTHONPATH= $ROUTEENV $SC2G_E4B_ENV $FOLDS" in BOX
    assert "--moe-backend marlin --attention-backend TRITON_ATTN" in BOX and "--no-enable-prefix-caching" in BOX
    assert '"$W/logs/sc2g_server_sglang.log" gptoss' in BOX
    # SC2's design note: both draws repeat ONE realisation (same seeds), so a draw disagreement is the engine, not the dice
    assert "drive ${E}_serial_d$D $PORT \"$MODEL\" $PROFILE serial 0 $SC2_SERIAL_N 1" in BOX
    assert "poisson $r $SC2_N $(( 100 + r ))" in BOX and "D * 100" not in BOX


def _sglang_engagement(mode, info, log_text, tmp_path):
    src = SGL.read_text()
    body = src[src.index("<<'PYT'\n") + len("<<'PYT'\n"): src.index("\nPYT\n", src.index("<<'PYT'\n"))]
    log = tmp_path / "srv.log"
    log.write_text(log_text)
    (tmp_path / "srv.log.server_info.json").write_text(json.dumps(info))
    env = {"SC1_MODE": mode, "SC1_MFS": "0.75", "SC1_LOG": str(log), "SC1_STARTUP_S": "30", "SC1_JIT": str(tmp_path / "jit"),
           "SC1_CMD": "x", "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True, env=env, timeout=60)
    return out.returncode, out.stdout


def test_sglang_gptoss_mode_reads_gpt_oss_engagement_and_still_refuses_the_wrong_attention(tmp_path):
    info = {"version": "0.5.20", "attention_backend": "triton", "disable_radix_cache": True, "max_running_requests": 16,
            "moe_runner_backend": "auto", "quantization": "mxfp4"}
    rc, out = _sglang_engagement("gptoss", info, "The server is fired up and ready to roll!\n", tmp_path)
    assert rc == 0 and '"mode": "gptoss"' in out, out              # no GPTQ banner, no Marlin JIT leaf: fine for gpt-oss
    rc, out = _sglang_engagement("gptoss", dict(info, attention_backend="flashinfer"), "", tmp_path)
    assert rc == 43 and "not triton" in out, out
    rc, out = _sglang_engagement("native", info, "", tmp_path)     # the GPTQ modes still demand their engagement
    assert rc == 43 and "gptq_marlin" in out, out


def _g_e4b_check(health, tmp_path):
    fn = BOX[BOX.index("g_e4b_check(){"):BOX.index("g_e4b_start(){")]
    p = tmp_path / "h.json"
    p.write_text(json.dumps(health))
    script = f'PY={sys.executable}\nSC2G_LAYERS=24\n{fn}\ng_e4b_check {p}\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    return out.returncode, out.stdout


def test_box_g_e4b_check_wants_the_native_mxfp4_store_on_every_layer(tmp_path):
    routes = {"device_grouping": True, "int4_prefill_env": None, "prefill_attn_env": None, "int4_prefill": "k19",
              "int4_prefill_above_256_rows": "k19", "prefill_attn": "flash"}
    ok = {"levers": {"int4_store_kinds": ["mxfp4"], "exp_int4_layers_enabled": 24}, "prefill_routes": routes,
          "prefill_graph": {"status": "on"}}
    rc, out = _g_e4b_check(ok, tmp_path)
    assert rc == 0 and '"ok": true' in out, out
    rc, out = _g_e4b_check(dict(ok, levers={"int4_store_kinds": ["int4_b32"], "exp_int4_layers_enabled": 24}), tmp_path)
    assert rc == 1 and "not ['mxfp4']" in out, out
    rc, out = _g_e4b_check(dict(ok, prefill_routes=dict(routes, prefill_attn="math")), tmp_path)   # a stray pin
    assert rc == 1 and "prefill_attn" in out, out


def test_the_sc2g_reducer_self_test_passes():
    out = subprocess.run([sys.executable, str(REPO / "bench" / "sc2" / "sc2g_reduce.py"), "--self-test"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "self-test OK (5 cases)" in out.stdout, out.stdout + out.stderr


def test_every_box_script_sources_under_set_u_with_only_what_sc1_run_defines_first():
    """A1 (sc2g-prove-1): sc1_run.sh sources the box scripts at its install step, under `set -u`, before FOLDS / SPEEDENV /
    GR_ENV / ROUTEENV exist. A top-level expansion of any of them kills the box there. Source every SC2 box script that way,
    with only $W defined."""
    sc2 = REPO / "bench" / "sc2"
    files = " ".join(f'. "{sc2 / f}";' for f in ("sc2_box_e.sh", "sc2b_box_f.sh", "sc2g_box_g.sh"))
    out = subprocess.run(["bash", "-c", f'set -uo pipefail; W=$(mktemp -d); {files} echo SOURCED'], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0 and "SOURCED" in out.stdout, out.stderr


def test_the_heartbeat_live_count_does_not_count_itself():
    """A1: the controller counts the lane's process inside an ssh shell whose own command line holds the pattern. Run the
    exact expression that way with no lane running: it must read 0 (the plain 'bash sc1_run.sh' read 1, forever)."""
    drive = (REPO / "bench" / "sc1" / "sc1_drive.sh").read_text()
    expr = "pgrep -f '[b]ash sc1_run.sh' | wc -l"
    assert expr in drive and "pgrep -f 'bash sc1_run.sh'" not in drive
    out = subprocess.run(["bash", "-c", f"echo live $({expr} | tr -d ' ')"], capture_output=True, text=True, timeout=30)
    assert out.stdout.strip() == "live 0", out.stdout
    if sys.platform.startswith("linux"):   # BSD pgrep (macOS) leaves out its own ancestors; procps (the box, CI) does not
        plain = subprocess.run(["bash", "-c", "echo live $(pgrep -f 'bash sc1_run.sh' | wc -l | tr -d ' ')"], capture_output=True,
                               text=True, timeout=30)
        assert plain.stdout.strip() != "live 0", plain.stdout              # the old form counts its own shell


def _run_head_then(tmp_path, tail):
    """sc1_run.sh's preamble (set -u, finish, both traps), run in a temp W, then `tail`. Returns (rc, files in W)."""
    head = RUN[RUN.index("set -uo pipefail\n"):RUN.index("for v in SC1_RUN_ID")]
    head = head.replace("W=/root/sc1;", f"W={tmp_path};")
    script = f"export SC1_RUN_NONCE=n1\n{head}\n{tail}\n"
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    return out.returncode, {f.name: f.read_text() for f in tmp_path.iterdir() if f.is_file()}, out.stdout


def test_an_exit_that_skips_finish_still_writes_the_markers(tmp_path):
    """A1: sc2g-prove-1's box died under set -u at source time and wrote no TP_DONE. Reproduce that death (an unbound
    expansion in a sourced file) after the preamble: the EXIT trap must record the rc and TP_DONE."""
    (tmp_path / "box.sh").write_text('X="a $NOT_DEFINED_HERE"\n')
    rc, files, out = _run_head_then(tmp_path, f". {tmp_path}/box.sh")
    assert rc != 0 and "TP_DONE.n1" in files and files["SC1_EXIT_CODE.n1"].strip() == str(rc), (rc, files, out)
    assert "SC1_SUCCESS.n1" not in files and "exit without finish" in out


def test_a_bare_exit_0_is_not_a_success_and_finish_still_owns_its_markers(tmp_path):
    rc, files, _ = _run_head_then(tmp_path, "exit 0")
    assert rc == 79 and files["SC1_EXIT_CODE.n1"].strip() == "79" and "SC1_SUCCESS.n1" not in files
    for f in tmp_path.iterdir():
        if f.is_file():
            f.unlink()
    rc, files, out = _run_head_then(tmp_path, "finish 0")
    assert rc == 0 and files["SC1_EXIT_CODE.n1"].strip() == "0" and "SC1_SUCCESS.n1" in files
    assert "exit without finish" not in out
