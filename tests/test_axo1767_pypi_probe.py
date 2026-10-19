"""#1767's PyPI probe (bench/tc1/pypi_probe.py, AXO1767-PREREG "Box and budget"): the rate it reads, every refusal it can make, and
the tc1_run.sh hook that runs it only for a token that registers a floor. A local HTTP server stands in for PyPI; no network."""

import hashlib
import http.server
import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "bench" / "tc1" / "pypi_probe.py"
RUN_SH = REPO / "bench" / "tc1" / "tc1_run.sh"
DRIVE_SH = REPO / "bench" / "tc1" / "tc1_drive.sh"

spec = importlib.util.spec_from_file_location("pypi_probe_under_test", PROBE)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

BODY = os.urandom(3 * (1 << 20) + 12345)
SHA = hashlib.sha256(BODY).hexdigest()


class _Handler(http.server.BaseHTTPRequestHandler):
    delay = 0.0                                             # seconds per 64 KiB chunk: a slow host
    head_delay = 0.0                                        # seconds before the response head: a slow handshake

    def do_GET(self):
        if self.head_delay:
            time.sleep(self.head_delay)
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        for i in range(0, len(BODY), 1 << 16):
            self.wfile.write(BODY[i:i + (1 << 16)])
            if self.delay:
                time.sleep(self.delay)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/w.whl", _Handler
    srv.shutdown()
    _Handler.delay = 0.0
    _Handler.head_delay = 0.0


def test_a_fast_host_passes_with_the_bytes_verified(server):
    url, _ = server
    rc, rec = P.probe(url, len(BODY), SHA, min_mbps=1.0, cap_s=30)
    assert rc == 0 and rec["complete"] and rec["sha256_ok"] and rec["mb_per_s"] > 1.0 and rec["ttfb_s"] is not None, rec


def test_a_rate_below_the_floor_is_refused(server):
    url, _ = server
    rc, rec = P.probe(url, len(BODY), SHA, min_mbps=1e9, cap_s=30)
    assert rc == 3 and rec["complete"] and rec["sha256_ok"], rec


def test_a_slow_host_is_refused_on_a_partial_read_at_the_cap(server):
    url, h = server
    h.delay = 0.05                                          # about 1.3 MB/s
    rc, rec = P.probe(url, len(BODY), SHA, min_mbps=10.0, cap_s=1.0)
    assert rc == 3 and not rec["complete"] and 0 < rec["bytes"] < len(BODY) and rec["mb_per_s"] < 10.0, rec


def test_a_slow_handshake_is_recorded_but_never_counted_in_the_rate(server):
    url, h = server
    h.head_delay = 1.5                                      # 3.2 MB body: with the head inside, the rate would read under 2.2 MB/s
    rc, rec = P.probe(url, len(BODY), SHA, min_mbps=5.0, cap_s=30)
    assert rc == 0 and rec["ttfb_s"] >= 1.4 and rec["mb_per_s"] > 5.0 and rec["sha256_ok"], rec


def test_wrong_bytes_are_refused_even_when_fast(server):
    url, _ = server
    rc, rec = P.probe(url, len(BODY), "0" * 64, min_mbps=1.0, cap_s=30)
    assert rc == 4 and rec["sha256_ok"] is False, rec


def test_an_unreachable_host_reads_no_bytes():
    rc, rec = P.probe("http://127.0.0.1:9/none.whl", 10, "0" * 64, min_mbps=1.0, cap_s=3)
    assert rc == 5 and rec["bytes"] == 0 and "error" in rec and rec["mb_per_s"] is None, rec


def test_the_pinned_wheel_is_a_fixed_file():
    assert P.URL.startswith("https://files.pythonhosted.org/") and P.URL.endswith(".whl")
    assert P.SIZE == 35344199 and len(P.SHA256) == 64


def test_main_prints_one_line_and_its_exit_code(server):
    url, _ = server
    p = subprocess.run([sys.executable, str(PROBE), "--min-mbps", "1e9", "--url", url, "--size", str(len(BODY)), "--sha256", SHA],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 3 and p.stdout.startswith("PYPI PROBE ") and '"verdict": "BELOW_FLOOR"' in p.stdout, p.stdout + p.stderr


# ---------------------------------------------------------------- the tc1_run.sh hook


def _hook():
    src = RUN_SH.read_text()
    s = src.index("# The PyPI probe (#1767)")
    e = src.index("\nfi\n", s) + len("\nfi\n")
    return src[s:e]


def _run_hook(tmp_path, families, rc, env_floor=None):
    (tmp_path / "logs").mkdir(exist_ok=True)
    fake = tmp_path / "py"
    fake.write_text(f'#!/bin/bash\necho "PY $*" >> {tmp_path}/calls.log\necho "PYPI PROBE stub"\nexit {rc}\n')
    fake.chmod(0o755)
    script = (f'W={tmp_path}; PY_E4B={fake}; FAMILIES="{families}"\n'
              'say(){ echo "SAY: $*" >> "$W/say.log"; }\nfinish(){ echo "FINISH $1" >> "$W/finish.log"; exit "$1"; }\n'
              ': > summary.txt\n' + _hook() + 'echo "PAST THE PROBE" >> "$W/finish.log"\n')
    env = dict(os.environ)
    env.pop("TC1_PYPI_MIN_MBPS", None)
    if env_floor is not None:
        env["TC1_PYPI_MIN_MBPS"] = env_floor
    p = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)

    def read(n):
        return (tmp_path / n).read_text() if (tmp_path / n).exists() else ""
    return p.returncode, read("calls.log"), read("summary.txt"), read("finish.log")


def test_the_token_runs_the_probe_at_its_registered_floor_and_a_pass_goes_on(tmp_path):
    rc, calls, summary, fin = _run_hook(tmp_path, "qwen3axo1767", 0)
    assert rc == 0 and "--min-mbps 10" in calls and "pypi_probe.py" in calls and "PAST THE PROBE" in fin and "BOX_REFUSED" not in summary


@pytest.mark.parametrize("probe_rc", [3, 4, 5])
def test_a_probe_refusal_refuses_the_box_as_host_limited(tmp_path, probe_rc):
    rc, calls, summary, fin = _run_hook(tmp_path, "qwen3axo1767", probe_rc)
    assert rc == 15 and f"BOX_REFUSED pypi_probe rc={probe_rc}" in summary and "FINISH 15" in fin and "PAST THE PROBE" not in fin


def test_no_other_token_runs_the_probe(tmp_path):
    rc, calls, summary, fin = _run_hook(tmp_path, "qwen3hp1k", 3)
    assert rc == 0 and calls == "" and "PAST THE PROBE" in fin


def test_the_override_reaches_any_token_and_zero_skips(tmp_path):
    rc, calls, _, _ = _run_hook(tmp_path, "qwen3hp1k", 0, env_floor="25")
    assert rc == 0 and "--min-mbps 25" in calls
    (tmp_path / "calls.log").unlink()
    rc, calls, _, fin = _run_hook(tmp_path, "qwen3axo1767", 3, env_floor="0")
    assert rc == 0 and calls == "" and "PAST THE PROBE" in fin


def test_the_hook_runs_after_the_fetch_probe_and_before_any_pypi_venv():
    src = RUN_SH.read_text()
    assert src.index("BOX_REFUSED fetch_probe") < src.index("# The PyPI probe (#1767)") < src.index("venv-axolotl: uv venv")


def test_the_probe_is_staged_with_the_token():
    line = next(ln for ln in DRIVE_SH.read_text().splitlines() if '*" qwen3axo1767 "*)' in ln)
    assert "bench/tc1/pypi_probe.py" in line
