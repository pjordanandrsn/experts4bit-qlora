"""SC1g A3's two side checks, their pure-torch parts (CPU): the attention reference sc1g_attn_check.py grades the fp8 decode
kernel against, and the routing / grouping sc1g_mtile_check.py feeds the NF4 M-tile. The GPU halves ran on the A2000
(bench/sc2/sc1g-a2000/a3_*) and ride box J on the 5090."""
import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
SC2 = pathlib.Path(__file__).resolve().parents[1] / "bench" / "sc2"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, SC2 / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _mod("sc1g_attn_check")
M = _mod("sc1g_mtile_check")


def _qkv(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(A.HQ, A.D, generator=g), torch.randn(T, A.HKV, A.D, generator=g),
            torch.randn(T, A.HKV, A.D, generator=g), torch.randn(A.HQ, generator=g))


def test_attend_with_zero_query_and_no_sinks_averages_the_values():
    q, K, V, s = _qkv(40)
    out = A.attend(torch.zeros_like(q), K, V, 0, s, A.D ** -0.5, use_sinks=False)
    want = V.mean(0).repeat_interleave(A.HQ // A.HKV, dim=0)
    assert torch.allclose(out, want, atol=1e-6)


def test_attend_window_reads_only_the_last_keys():
    q, K, V, s = _qkv(300)
    K2, V2 = K.clone(), V.clone()
    K2[:-128] = torch.randn_like(K2[:-128]) * 50          # anything outside the window must not matter
    V2[:-128] = 1e3
    sc = A.D ** -0.5
    assert torch.allclose(A.attend(q, K, V, 128, s, sc), A.attend(q, K2, V2, 128, s, sc), atol=1e-6)
    assert not torch.allclose(A.attend(q, K, V, 0, s, sc), A.attend(q, K2, V2, 0, s, sc), atol=1e-3)


def test_attend_sink_takes_mass_and_never_contributes_a_value():
    q, K, V, s = _qkv(64)
    sc = A.D ** -0.5
    huge = torch.full_like(s, 80.0)                        # a dominant sink absorbs nearly all the softmax mass
    assert A.attend(q, K, V, 0, huge, sc).abs().max() < 1e-6
    low = torch.full_like(s, -80.0)                        # a negligible sink == no sink
    assert torch.allclose(A.attend(q, K, V, 0, low, sc), A.attend(q, K, V, 0, s, sc, use_sinks=False), atol=1e-6)


def test_attend_maps_query_heads_to_kv_heads_in_blocks():
    q, K, V, s = _qkv(32)
    V2 = V.clone()
    V2[:, 1] += 7.0                                        # only kv head 1 changes: only q heads 8..15 may move
    sc = A.D ** -0.5
    d = (A.attend(q, K, V2, 0, s, sc) - A.attend(q, K, V, 0, s, sc)).abs().amax(-1)
    G = A.HQ // A.HKV
    assert torch.all(d[G:2 * G] > 1e-3) and torch.all(torch.cat([d[:G], d[2 * G:]]) < 1e-6)


@pytest.mark.parametrize("kind", ["uniform", "skewed"])
def test_route_gives_top4_distinct_experts_per_token(kind):
    ids = M.route(500, 32, kind, torch.Generator().manual_seed(1))
    assert ids.shape == (500, M.TOPK)
    assert all(len(set(r.tolist())) == M.TOPK for r in ids)
    top = torch.bincount(ids.reshape(-1), minlength=32).max().item()
    assert (top > 400) == (kind == "skewed")               # the skewed draw puts one expert on most tokens


def test_grouped_sorts_by_expert_and_order_inverts_it():
    g = torch.Generator().manual_seed(2)
    ids = M.route(50, 32, "uniform", g)
    x = torch.randn(50, 8, generator=g)
    xs, sizes, eids, order = M.grouped(x, ids)
    assert sum(sizes) == 50 * M.TOPK and eids == sorted(eids) and len(set(eids)) == len(eids)
    back = torch.empty_like(xs)
    back[order] = xs
    assert torch.equal(back, x.repeat_interleave(M.TOPK, dim=0))
    r0 = 0
    flat = ids.reshape(-1)
    for m, e in zip(sizes, eids):                          # each group's rows really are routed to that expert
        assert torch.all(flat[order[r0:r0 + m]] == e)
        r0 += m


# ---- A4: the e4b named-token capture (sc1g_k8's torch proxy) ----------------------------------------------------------

K8 = _mod("sc1g_k8")
KLM = _mod("sc1g_kl")


def test_torch_proxy_forwards_tensor_and_everything_else():
    rows = []
    px = K8._TorchProxy(torch, rows.append)
    x = px.ones(2, 3)
    assert px.Tensor is torch.Tensor and isinstance(x, px.Tensor) and isinstance(x, torch.Tensor)
    assert px.float32 is torch.float32 and px.no_grad is torch.no_grad and px.__version__ == torch.__version__
    y = px.log_softmax(torch.zeros(1, 5), -1)
    assert torch.allclose(y, torch.log_softmax(torch.zeros(1, 5), -1)) and len(rows) == 1


def test_named_capture_records_the_served_rows_and_reproduces_the_nll(tmp_path):
    import types
    g = torch.Generator().manual_seed(3)
    V, P = 300, 12
    ref_logits = torch.randn(P, V, generator=g) * 2
    targets = torch.randint(0, V, (P,), generator=g).numpy()
    rows = KLM.reference_rows(ref_logits, targets)
    ref_path = str(tmp_path / "ref_conv1.npz")
    sha = KLM.save_artifact(ref_path, rows, {"source": "conv1"})
    fake = types.SimpleNamespace(torch=torch)            # stands in for step_decomp: its loop calls module.torch.log_softmax
    out = str(tmp_path / "named.npz")
    st = K8.named_capture(fake, ref_path, sha, out)
    eng_logits = ref_logits + 0.1 * torch.randn(P, V, generator=g)
    nll = 0.0
    for t in range(P):                                    # step_decomp's served loop, verbatim in shape
        lg = eng_logits[t:t + 1].float()
        nll += -fake.torch.log_softmax(lg, -1)[0, int(targets[t])].item()
    fake.torch.log_softmax(torch.zeros(3, V), -1)         # a non-[1, V] call is counted, not recorded
    st["save"]()
    z = __import__("numpy").load(out)
    meta = __import__("json").load(open(out + ".json"))
    assert meta["calls"] == P and meta["other_calls"] == 1 and meta["ref_sha"] == sha
    assert abs(-z["eng_target_lp"].mean() - nll / P) < 1e-9
    want = torch.log_softmax(eng_logits.double(), -1).gather(1, torch.as_tensor(rows["ids"]).long()).numpy()
    assert abs(z["eng_lp"] - want).max() < 1e-5          # fp32 log_softmax vs fp64
    r = KLM.window_read(rows, z["eng_lp"], z["eng_target_lp"])
    assert r["positions"] == P and r["kl65_mean"] > 0


def test_named_capture_refuses_a_wrong_reference_sha(tmp_path):
    import types
    rows = KLM.reference_rows(torch.randn(4, 100), [1, 2, 3, 4])
    ref_path = str(tmp_path / "ref.npz")
    KLM.save_artifact(ref_path, rows, {})
    with pytest.raises(SystemExit):
        K8.named_capture(types.SimpleNamespace(torch=torch), ref_path, "0" * 64, str(tmp_path / "o.npz"))


def test_sc1g_kl_and_ref_self_tests_pass():
    import subprocess
    import sys
    for f in ("sc1g_kl.py", "sc1g_ref.py"):
        r = subprocess.run([sys.executable, str(SC2 / f), "--self-test"], capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and "self-test OK" in r.stdout, f + r.stdout + r.stderr


def test_box_r_staging_pin_matches_its_sources():
    """bench/sc2/sc1g-r/staged-r.sha256 names every file box R stages; each must hash to its pin (make_pin.sh regenerates)."""
    import hashlib
    repo = SC2.parents[1]
    win = repo / "bench" / "h2h-2026-10-02" / "sc1g" / "receipts" / "sc1g-diag-2" / "sc1g"
    rdir = SC2 / "sc1g-r"
    names = []
    for ln in (rdir / "staged-r.sha256").read_text().splitlines():
        want, name = ln.split()
        names.append(name)
        if name.startswith("windows/"):
            src = win / name.split("/", 1)[1]
        elif name in ("sc1g_ref.py", "sc1g_kl.py"):
            src = SC2 / name
        elif name == "kl_fidelity.py":
            src = repo / "bench" / name
        else:
            src = rdir / name
        assert hashlib.sha256(src.read_bytes()).hexdigest() == want, name
    assert {"sc1g_r_run.sh", "sc1g_ref.py", "sc1g_kl.py", "kl_fidelity.py", "window_shas.json"} <= set(names)
    assert sum(n.startswith("windows/") for n in names) == 5
    run = (rdir / "sc1g_r_run.sh").read_text()
    assert "sha256sum -c staged-r.sha256" in run and "kl_fidelity.py --controls" in run and '"verdict": "R_OK"' in run


def test_box_r_egress_probe_is_python_and_says_why():
    """sc1g-r-6 read 0.0 MB/s with curl's stderr discarded, so the cause was unknowable. The probe is Python (no curl
    dependency), logs its status / final host / error to logs/egress.log, and every refusal carries that line."""
    run = (SC2 / "sc1g-r" / "sc1g_r_run.sh").read_text()
    probe = run[run.index('echo "curl $(command -v curl'):run.index('echo "refused: egress $MBPS MB/s" > REFUSAL; finish 14; fi')]
    assert "curl -s" not in probe and "urllib.request" in probe and "2>> logs/egress.log" in probe
    assert '"Range": "bytes=0-52428800"' in probe and "SC1G_R_EGRESS_TIMEOUT:-20" in probe and "SC1G_R_EGRESS_CAP_S:-20" in probe
    assert "bert-base-uncased/resolve/main/model.safetensors" in probe


def _egress_segment():
    run = (SC2 / "sc1g-r" / "sc1g_r_run.sh").read_text()
    end = 'echo "refused: egress $MBPS MB/s" > REFUSAL; finish 14; fi'
    return run[run.index('echo "curl $(command -v curl'):run.index(end) + len(end)]


@pytest.fixture(scope="module")
def egress_server():
    """A local server: /403 refuses, /stall sends 1 MiB then hangs, /fast sends 2 MiB."""
    import http.server
    import threading
    import time

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/403":
                self.send_error(403, "Forbidden")
                return
            self.send_response(206)
            self.send_header("Content-Length", str(2 << 20))
            self.end_headers()
            self.wfile.write(b"x" * (1 << 20))
            self.wfile.flush()
            if self.path == "/stall":
                time.sleep(4)
                return
            self.wfile.write(b"x" * (1 << 20))

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _run_egress(tmp_path, url, min_mbps, timeout=1, cap=20):
    import subprocess
    seg = tmp_path / "seg.sh"
    seg.write_text(_egress_segment())
    (tmp_path / "logs").mkdir(exist_ok=True)
    script = ('say(){ echo "SAY $*"; }; finish(){ echo "FINISH $1"; exit $1; }; '
              f'MIN_MBPS={min_mbps}; SC1G_R_EGRESS_URL={url}; SC1G_R_EGRESS_TIMEOUT={timeout}; SC1G_R_EGRESS_CAP_S={cap}; '
              '. ./seg.sh; echo PASSED')
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    log = (tmp_path / "logs" / "egress.log").read_text() if (tmp_path / "logs" / "egress.log").exists() else ""
    refusal = (tmp_path / "REFUSAL").read_text() if (tmp_path / "REFUSAL").exists() else ""
    return r.returncode, r.stdout, log, refusal


def test_egress_probe_raising_before_any_byte_is_rc9_not_a_slow_host(tmp_path, egress_server):
    rc, out, log, refusal = _run_egress(tmp_path, egress_server + "/403", 20)
    assert rc == 9 and "FINISH 9" in out and "HTTPError" in log and "egress probe error" in refusal
    assert "hf_cdn_mbps=ERROR" in (tmp_path / "forensics.txt").read_text()


def test_egress_probe_unreachable_host_is_rc9(tmp_path):
    rc, out, log, refusal = _run_egress(tmp_path, "http://127.0.0.1:9/x", 20)
    assert rc == 9 and "URLError" in log and "failed before reading any byte" in out


def test_egress_probe_timeout_mid_read_is_measured_slow_rc14(tmp_path, egress_server):
    rc, out, log, refusal = _run_egress(tmp_path, egress_server + "/stall", 100000, timeout=1)
    assert rc == 14 and "FINISH 14" in out and "egress probe error after 1048576 bytes" in log
    assert refusal.startswith("refused: egress ") and "error" not in refusal


def test_egress_probe_fast_read_passes(tmp_path, egress_server):
    rc, out, log, refusal = _run_egress(tmp_path, egress_server + "/fast", 0.001)
    assert rc == 0 and "PASSED" in out and not refusal and "egress status 206" in log


RULE_FUNCS_A5 = ("verdict_a5", "score", "load_reference", "window_ids", "fake_nf4", "fake_nf4_experts_", "gnf4_crosscheck")
RULE_CONSTS_A5 = ("SELF_CONSISTENCY_MAX", "SRCS", "NF4_LUT", "BLOCK", "A5_GRADED", "A5_MIN_GRADABLE", "A5_DROPPED")
RULE_SHA_A5 = "7f307c39c896e456f63e99f09ce33944cd12b4cf25275ae8801b761f152469b7"
# A4's registered rule, kept so the A4 read stays re-derivable: digest ee122b74... over A4's set at 8a1513a4 / 9dc7b59a, and
# verdict()'s own source (the function --reverdict uses for an A4 receipt) pinned below
A4_VERDICT_SHA = "0f521ac4d77d3a17daf342ca3cea7d20bf1f2f79b03d5c7adbffef6128c18102"


def _rule_digest(ref_src: str, kl_src: str, funcs=RULE_FUNCS_A5, consts=RULE_CONSTS_A5) -> str:
    """Box R's registered rule as one digest: the sources of every function its verdict depends on, the constants by
    repr of their literal values, and sc1g_kl.py whole."""
    import ast
    import hashlib
    tree = ast.parse(ref_src)
    fn = {n.name: ast.get_source_segment(ref_src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    cs = {t.id: repr(ast.literal_eval(n.value)) for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
          if isinstance(t, ast.Name) and t.id in consts}
    assert all(f in fn for f in funcs) and all(c in cs for c in consts)
    parts = [fn[f] for f in funcs] + [f"{c} = {cs[c]}" for c in consts] + [kl_src]
    return hashlib.sha256("\n\x00\n".join(parts).encode()).hexdigest()


def test_box_r_rule_is_the_registered_one():
    """A5's registered rule for box R: verdict_a5 and everything it and the scoring read, the constants, sc1g_kl.py."""
    ref_src, kl_src = (SC2 / "sc1g_ref.py").read_text(), (SC2 / "sc1g_kl.py").read_text()
    assert _rule_digest(ref_src, kl_src) == RULE_SHA_A5
    assert _rule_digest(ref_src.replace("A5_MIN_GRADABLE = 3", "A5_MIN_GRADABLE = 2"), kl_src) != RULE_SHA_A5
    assert _rule_digest(ref_src, kl_src.replace("STORAGE_F_FRACTION = 0.1", "STORAGE_F_FRACTION = 0.2")) != RULE_SHA_A5
    assert _rule_digest(ref_src, kl_src.replace("GRADABLE_F_MAX = 1e-2", "GRADABLE_F_MAX = 2e-2")) != RULE_SHA_A5


def test_the_a4_read_still_rederives_from_its_committed_receipt():
    """A5 changed the rule, not A4's: verdict() is byte-identical to A4's, and --reverdict on sc1g-r-8's committed receipt
    reproduces R_NOT_OK exactly."""
    import ast
    import hashlib
    import subprocess
    import sys
    src = (SC2 / "sc1g_ref.py").read_text()
    fn = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    assert hashlib.sha256(fn["verdict"].encode()).hexdigest() == A4_VERDICT_SHA
    rd = SC2.parents[1] / "bench" / "h2h-2026-10-02" / "sc1g" / "receipts" / "sc1g-r-8"
    r = subprocess.run([sys.executable, str(SC2 / "sc1g_ref.py"), "--reverdict", str(rd / "ref"), "--k0", str(rd / "k0.json")],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "R_NOT_OK rule=A4" in r.stdout and "matches_recorded=True" in r.stdout, r.stdout + r.stderr


def test_the_a5_read_rederives_and_the_registered_shas_are_rs():
    """A5's box R read (sc1g-r5-2): --reverdict on the committed receipt reproduces R_OK under rule A5, and the files box I
    stages from bench/sc1/sc1g_ref/ are R's own -- r_verdict.json and r_calib.json byte-identical to the receipt's, and
    ref_full_shas.json equal to the shas R recorded three ways (verdict, calibration, SHA256SUMS)."""
    import json
    import subprocess
    import sys
    rd = SC2.parents[1] / "bench" / "h2h-2026-10-02" / "sc1g" / "receipts" / "sc1g-r5-2"
    reg = SC2.parents[1] / "bench" / "sc1" / "sc1g_ref"
    r = subprocess.run([sys.executable, str(SC2 / "sc1g_ref.py"), "--reverdict", str(rd / "ref"), "--k0", str(rd / "k0.json")],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "R_OK rule=A5" in r.stdout and "matches_recorded=True" in r.stdout, r.stdout + r.stderr
    for f in ("r_verdict.json", "r_calib.json"):
        assert (reg / f).read_bytes() == (rd / "ref" / f).read_bytes(), f
    shas = json.loads((reg / "ref_full_shas.json").read_text())
    v, cal = json.loads((rd / "ref" / "r_verdict.json").read_text()), json.loads((rd / "ref" / "r_calib.json").read_text())
    sums = {ln.split()[1].rsplit("/", 1)[-1][len("ref_full_"):-len(".npy")]: ln.split()[0]
            for ln in (rd / "ref" / "SHA256SUMS").read_text().splitlines() if ln.strip()}
    assert set(shas) == {"conv1", "conv2", "conv3", "conv4", "wikitext"}
    assert shas == v["full_artifacts"] == sums == {s: a["sha256"] for s, a in cal["full_artifacts"].items()}
    assert v["rule"] == "A5" and v["verdict"] == "R_OK" and v["gradable_windows"] == ["conv1", "conv2", "conv3", "conv4"]

def test_the_a5_reading_rederives_from_its_committed_receipt(tmp_path):
    """A5's reading (sc1g-5090-a5-1): the reducer at main, run on the committed sc1g/ directory, reproduces the box's own
    verdict_sc1g.json A5 section -- every verdict, key, string and flag identical, every number to 1e-12 relative --
    K-A REFUTED (conv4 2.5x < 3x), L1 HOLDS (1.85x), L2 HOLDS, with every KL row VALID and zero masked reference mass.
    Not `==` on floats: CI's Linux numpy rounds three pooled means one ULP from the box's (0.07281677469236382 vs ...383)."""
    import json
    import subprocess
    import sys
    d = SC2.parents[1] / "bench" / "h2h-2026-10-02" / "sc1g" / "receipts" / "sc1g-5090-a5-1" / "sc1g"
    out = tmp_path / "rederived.json"
    r = subprocess.run([sys.executable, str(SC2 / "sc1g_reduce.py"), "--dir", str(d), "--out", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    got, box = json.loads(out.read_text())["a5"], json.loads((d / "verdict_sc1g.json").read_text())["a5"]
    def same(a, b, path="a5"):
        """Identical structure, keys, strings, bools and None; floats within 1e-12 relative (an ULP across platforms)."""
        if isinstance(a, dict) and isinstance(b, dict):
            assert a.keys() == b.keys(), (path, sorted(set(a) ^ set(b)))
            for k in a:
                same(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list) and isinstance(b, list):
            assert len(a) == len(b), (path, len(a), len(b))
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{path}[{i}]")
        elif isinstance(a, float) or isinstance(b, float):
            assert isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool), (path, a, b)
            assert a == b or abs(a - b) <= 1e-12 * max(abs(a), abs(b)), (path, a, b)
        else:
            assert a == b, (path, a, b)
    same(got, box)
    pred = got["predictions"]
    assert (pred["K-A"]["verdict"], pred["L1"]["verdict"], pred["L2"]["verdict"]) == ("REFUTED", "HOLDS", "HOLDS")
    assert got["instrument"] == {"box_r": "R_OK", "rule": "A5", "why": None}
    for lab in ("e4b_mxfp4", "e4b_nf4", "vllm", "llamacpp", "llamacpp_q8"):
        for s in ("conv1", "conv2", "conv3", "conv4", "wikitext"):
            row = got["rows"][lab][s]
            assert row["verdict"] == "VALID" and row["support"]["positions_masked"] == 0, (lab, s, row.get("why"))

def test_full_capture_reads_full_vocab_kl_and_reproduces_the_nll(tmp_path):
    """A5: the proxy computes KL(p_ref || p_e4b) over every token against R's fp16 full rows, per served step."""
    import json
    import types
    import numpy as np
    g = torch.Generator().manual_seed(11)
    V, P0, S = 300, 6, 10
    ids = torch.randint(0, V, (P0 + S + 1,), generator=g).tolist()
    win = tmp_path / "k8_window_conv1.json"
    win.write_text(json.dumps({"ids": ids, "prompt_len": P0, "steps": S}))
    ref_logits = torch.randn(S, V, generator=g) * 3
    rows = KLM.full_rows_fp16(ref_logits)
    ref_path = tmp_path / "ref_full_conv1.npy"
    np.save(ref_path, rows)
    sha = KLM.file_sha(str(ref_path))
    fake = types.SimpleNamespace(torch=torch)
    out = str(tmp_path / "kl.npz")
    st = K8.full_capture(fake, str(ref_path), sha, str(win), out)
    eng = ref_logits + 0.2 * torch.randn(S, V, generator=g)
    nll = 0.0
    for t in range(S):
        nll += -fake.torch.log_softmax(eng[t:t + 1].float(), -1)[0, ids[P0 + 1 + t]].item()
    st["save"]()
    z = np.load(out)
    meta = json.load(open(out + ".json"))
    want = KLM.kl_full_rows(rows, torch.log_softmax(eng.float(), -1))
    assert meta["calls"] == S and np.allclose(z["eng_kl"], want, atol=1e-9)
    assert abs(-z["eng_target_lp"].mean() - nll / S) < 1e-9
    with pytest.raises(SystemExit):
        K8.full_capture(types.SimpleNamespace(torch=torch), str(ref_path), "0" * 64, str(win), out)


def test_full_capture_records_masked_mass_and_void_positions_and_keeps_serving(tmp_path):
    """A5, #1223 review: a token e4b masks to -inf where R gives mass, and a row whose read fails (NaN), must not raise
    into the serving loop. The masked step records kl = inf, the reference mass and the common-support KL; the NaN step
    is a void position with its reason; every later step is still read; the reader then VOIDs the row with both."""
    import json
    import types
    import numpy as np
    g = torch.Generator().manual_seed(12)
    V, P0, S = 300, 6, 8
    ids = torch.randint(0, V, (P0 + S + 1,), generator=g).tolist()
    win = tmp_path / "k8_window_conv1.json"
    win.write_text(json.dumps({"ids": ids, "prompt_len": P0, "steps": S}))
    ref_logits = torch.randn(S, V, generator=g) * 3
    rows = KLM.full_rows_fp16(ref_logits)
    ref_path = tmp_path / "ref_full_conv1.npy"
    np.save(ref_path, rows)
    fake = types.SimpleNamespace(torch=torch)
    out = str(tmp_path / "kl.npz")
    st = K8.full_capture(fake, str(ref_path), KLM.file_sha(str(ref_path)), str(win), out)
    eng = ref_logits + 0.2 * torch.randn(S, V, generator=g)
    tok = next(i for i in range(V) if i != ids[P0 + 3])          # a non-target token the reference gives mass to
    eng[2, tok] = float("-inf")
    eng[5, :] = float("nan")
    for t in range(S):                                            # the served loop: every call returns, none raises
        fake.torch.log_softmax(eng[t:t + 1].float(), -1)
    st["save"]()
    z, meta = np.load(out), json.load(open(out + ".json"))
    pr = np.exp(rows[2].astype(np.float64) - np.logaddexp.reduce(rows[2].astype(np.float64)))
    assert meta["calls"] == S and meta["void_positions"] == 1 and meta["void_first"][0]["t"] == 5
    assert "NaN or +inf" in meta["void_first"][0]["why"]
    assert np.isposinf(z["eng_kl"][2]) and abs(z["eng_masked_mass"][2] - pr[tok]) < 1e-9 and z["eng_n_masked"][2] == 1
    assert np.isfinite(z["eng_kl_common"][2]) and np.isnan(z["eng_kl"][5])
    rest = [t for t in range(S) if t not in (2, 5)]
    assert np.all(np.isfinite(z["eng_kl"][rest])) and not z["eng_masked_mass"][rest].any()
    want = KLM.kl_full_rows(rows[rest], torch.log_softmax(eng[rest].float(), -1))
    assert np.allclose(z["eng_kl"][rest], want, atol=1e-9) and np.allclose(z["eng_kl_common"][rest], want, atol=1e-9)


def test_the_mxfp4_prompt_kernel_check_rederives_from_its_committed_results():
    """A6's conv2 lead, the $0 A2000 check (bench/sc2/sc1g-a2000/a6mx_*): both verdicts re-derive from the per-shape numbers
    by the script's own rule -- IN_LINE + AT_FLOOR on the real run, ABOVE_FLOOR on the nibble-order mutation (which the
    relative verdict alone cannot see) -- floats at 1e-12 relative, verdicts exactly."""
    import json
    import math

    def rederive(r):
        gt, others, ratio, eq = [], [], [], []
        for lr in r["layers"].values():
            for pr in lr.values():
                for mk, x in pr.items():
                    if not mk.startswith("M"):
                        continue
                    if int(mk[1:]) > 256:
                        gt.append(x["grouped_v1"]["mean_rel"])
                        ratio.append(x["grouped_v1"]["mean_rel"] / x["floor_bf16_output"]["mean_rel"])
                        eq.append(x["grouped_v1"]["eq_floor"])
                    others.append(max(x[p]["mean_rel"] for p in ("grouped_tile", "mxfp4_gemv", "nf4_mtile", "bf16_mm")))
        return ("OUT_OF_LINE" if max(gt) > 2.0 * max(others) else "IN_LINE", "ABOVE_FLOOR" if max(ratio) > 1.5 else "AT_FLOOR",
                max(gt), max(others), max(ratio), min(eq))

    for name, want in (("a6mx_a2000_check.json", ("IN_LINE", "AT_FLOOR")), ("a6mx_a2000_check_mutate.json", ("IN_LINE", "ABOVE_FLOOR"))):
        r = json.load(open(SC2 / "sc1g-a2000" / name))
        v, fv, g, o, ra, e = rederive(r)
        assert (v, fv) == want == (r["verdict"], r["floor_verdict"]), (name, v, fv)
        for got, key in ((g, "grouped_v1_gt256_mean_rel_max"), (o, "other_paths_mean_rel_max"), (ra, "grouped_v1_gt256_floor_ratio_max"),
                         (e, "grouped_v1_gt256_eq_floor_min")):
            assert math.isclose(got, r[key], rel_tol=1e-12), (name, key)
        assert r["device"]["cc"] == [8, 6] and r["mutate"] == name.endswith("_mutate.json")
