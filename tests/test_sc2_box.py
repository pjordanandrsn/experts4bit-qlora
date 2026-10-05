"""Lane SC2's box E (bench/sc2, #846): the box script, the wiring into SC1's run/drive scripts, and the registered shape.

No local card runs the four servers, so this pins what a rented box will do: the servers and their flags, the engine
order, the plan's constants, the proof's smokes, and that every SC2 file is staged and pinned. The driver's behaviour
is tested in ``tests/test_sc2_driver.py``; the reducer and the prompt pool run their own self-tests here.
"""
import importlib.util
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "sc2"
SC1 = REPO / "bench" / "sc1"
BOX = (LANE / "sc2_box_e.sh").read_text()
RUN = (SC1 / "sc1_run.sh").read_text()
PREREG = (LANE / "SC2-PREREG.md").read_text()


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_self_tests_pass():
    for script, want in (("sc2_reduce.py", "self-test OK (10 cases)"), ("sc2_prompts.py", "self-test OK (3/3 cases)"),
                         ("sc2_driver.py", "self-test OK (9/9 cases)")):
        out = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True)
        assert out.returncode == 0 and want in out.stdout, out.stdout + out.stderr


def test_box_e_is_wired_into_sc1s_box_script_and_controller():
    assert 'case "$BOX" in A|B|C|D|E|F|G|H|I|J|K) ;;' in RUN and "C|D|E|F|G|H|I|J|K) BASEPY=python3;;" in RUN
    assert "E) . $W/sc2_box_e.sh; install_vllm; install_sglang; install_llamacpp; install_sc2_client ;;" in RUN
    assert 'E) PROVE_NEEDS="vllm sglang llamacpp sc2client";;' in RUN and '[ "$BOX" = E ] && prove_e' in RUN
    assert "D) box_d;; E) box_e;; F) box_f;; G) box_g;; H) box_h;; I) box_i;; J) box_j;; K) box_k;; esac" in RUN
    drive = (SC1 / "sc1_drive.sh").read_text()
    assert 'case "$SC1_BOX" in A|B|C|D|E|F|G|H|I|J|K) ;;' in drive and 'sc2_*|sc2b_*|sc2g_*|sc1g_*|sc2c_*|sc2d_*) src="$SC2/$name";;' in drive
    pinned = {ln.split()[1] for ln in (SC1 / "staged.sha256").read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    assert {"sc2_driver.py", "sc2_prompts.py", "sc2_reduce.py", "sc2_box_e.sh"} <= pinned


def test_the_plan_is_the_registered_plan():
    assert 'SC2_RATES="1 2 4 8"; SC2_DRAWS=2; SC2_N=120; SC2_SERIAL_N=24; SC2_LO=64; SC2_HI=256; SC2_MAXLEN=2048' in BOX
    red, drv, pr = _mod("sc2_reduce"), _mod("sc2_driver"), _mod("sc2_prompts")
    assert red.RATES == (1, 2, 4, 8) and red.DRAWS == (1, 2)
    assert red.ENGINES == ("e4b_int4", "vllm", "sglang", "llamacpp", "e4b_nf4")
    assert (drv.SLO_TTFT_S, drv.SLO_TPOT_S) == (1.0, 0.100) and drv.PROFILES["llamacpp"] == {"cache_prompt": False}
    assert (pr.ROWS, pr.PROMPT, pr.OFFSET) == (64, 512, 2048)
    for phrase in ("r = 1, 2, 4 and 8 req/s", "120 requests each", "24 requests", "TTFT ≤ 1.0 s AND TPOT ≤ 100 ms",
                   "self-tested on 10 cases", "guard 3.0 h ≤ $2.25", "guard 1.25 h ≤ $0.94"):
        assert phrase in PREREG, phrase


def test_the_servers_carry_the_registered_flags():
    vllm = BOX[BOX.index("vllm_server_start(){"):BOX.index("sgl_server_start(){")]
    for flag in ("--max-num-seqs 16", "--max-model-len $SC2_MAXLEN", "--no-enable-prefix-caching", "--seed 0",
                 "--gpu-memory-utilization 0.90", "--served-model-name sc2", '--revision "$GPTQ_REV"'):
        assert flag in vllm, flag
    sgl = BOX[BOX.index("sgl_server_start(){"):BOX.index("ll_server_start(){")]
    assert "--max-running-requests 16 --dtype float16 --context-length $SC2_MAXLEN --mem-fraction-static $SC1_SGLANG_MEM_FRACTION_STATIC" in sgl
    assert 'sglang_server_start "$GPTQ_MID" "$GPTQ_REV" $PORT_SGL "$W/logs/sc2_server_sglang.log" native' in sgl
    assert 'llamacpp_server_start "$W/gguf/$GGUF_Q4KM" 16 $PORT_LL' in BOX
    e4b = BOX[BOX.index("e4b_server_start(){"):BOX.index("vllm_server_start(){")]
    assert 'int4) LEV="$SPEEDENV E4B_PAGED_FUSE_QKV=1";;' in e4b and 'nf4) LEV="";;' in e4b
    assert "E4B_PAGED_MAX_TOKENS_PER_SEQ=$SC2_MAXLEN" in e4b and "PYTHONPATH= $ROUTEENV" in e4b
    assert "E4B_INT4_PREFILL" not in e4b and "E4B_PAGED_PREFILL_ATTN" not in e4b      # main's defaults, not SC1's pins
    assert "E4B_KV_STEP_SELECT" not in BOX and "on by default since 0.44.0; P111" in PREREG   # the released default


def test_the_engine_order_drops_the_labelled_row_first():
    body = BOX[BOX.index("box_e(){"):BOX.index("prove_e(){")]
    order = ['phase E4B "', 'phase VLLM "', 'phase SGL "', 'phase LL "', 'phase NF4 "', 'phase RD "']
    at = [body.index(s) for s in order]
    assert at == sorted(at), list(zip(order, at))
    for e, port in (("e4b_int4", "$PORT_E4B"), ("vllm", "$PORT_VLLM"), ("sglang", "$PORT_SGL"), ("llamacpp", "$PORT_LL"),
                    ("e4b_nf4", "$PORT_E4B")):
        assert f"engine_runs {e} {port} " in body, e
    assert body.index("fetch_common || finish 11") < body.index("sc2_prompts || finish 19") < body.index('phase E4B "')


def test_the_proof_smokes_every_server_on_the_lane_checkpoints():
    prove = BOX[BOX.index("prove_e(){"):]
    assert 'e4b_server_start granite "$GR" "$GR_REV" "$W/work_granite/nf4.arena"' in prove
    assert 'fetch gptq "$GPTQ_MID" "$GPTQ_REV"' in prove and "vllm_server_start" in prove and "sgl_server_start" in prove
    assert "fetch_q4km && ll_server_start" in prove
    assert "serial 0 6 1" in prove and "poisson 4 16 2" in prove and "rec 23" in prove


def _health_server(states):
    """A local /health that answers HTTP 200 with each of ``states`` in turn, then the last one forever."""
    import http.server
    import json as _json
    import threading
    seq = list(states)

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            st = seq.pop(0) if len(seq) > 1 else seq[0]
            body = _json.dumps({"status": st, "error": None}, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _wait_e4b_ready(url, cap=30):
    fn = BOX[BOX.index("wait_e4b_ready(){"):BOX.index("stop_pid(){")]
    script = f'line(){{ echo "$*"; }}\n{fn}\nsleep 600 & P=$!\nwait_e4b_ready "{url}" {cap} $P /dev/null; rc=$?\nkill $P\nexit $rc\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)


def test_e4b_start_waits_for_ready_not_for_http_200():
    """A1: serve_paged's /health is HTTP 200 while the engine loads ({"status":"loading"}); generation is 503 until
    ready. The box's e4b start must wait for status ready, and fail on status error."""
    srv = _health_server(["loading", "loading", "loading", "ready"])
    try:
        out = _wait_e4b_ready(f"http://127.0.0.1:{srv.server_address[1]}/health")
        assert out.returncode == 0, out.stdout + out.stderr
    finally:
        srv.shutdown()
    srv = _health_server(["loading", "error"])
    try:
        out = _wait_e4b_ready(f"http://127.0.0.1:{srv.server_address[1]}/health")
        assert out.returncode == 45 and "engine error" in out.stdout, out.stdout + out.stderr
    finally:
        srv.shutdown()
    assert 'wait_e4b_ready "http://127.0.0.1:$PORT_E4B/health" 2400' in BOX
