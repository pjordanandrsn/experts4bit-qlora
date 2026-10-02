"""The llama.cpp side of lane SC1 (experts4bit-qlora#846), CPU-only.

(a) ``bench/sc1/llamacpp/nll_teacher_forced.cpp`` compiles against the PINNED llama.cpp tree (b11327 =
    552f18f912a32ea86edf82e2b76431cb7131538d) when cmake and a C++ compiler exist here -- the tree is built CPU-only
    (``-DGGML_CUDA=OFF -DGGML_METAL=OFF``); otherwise the build-dependent tests skip with the reason.
(b) CALIBRATION: a tiny Q8_0 GGUF (Qwen/Qwen2.5-0.5B-Instruct-GGUF) is scored on identical token ids by the harness in
    decode mode AND prefill mode and, when torch + transformers import, by transformers in fp32 on the CPU using the
    GGUF's OWN (dequantised) weights -- the gold must run the same weights, see the note in ``_gold_from_gguf`` --
    with |dNLL| <= 0.02 asserted against the gold and |decode - prefill| <= 0.01 asserted between the harness's shapes.
    The measured deltas are printed (run with ``-s``). Without torch the gold is reported NOT RUN and only the
    decode-vs-prefill self-consistency is asserted.
(c) ``sc1_llamacpp_arm.py`` against a fake llama-server (threaded http.server, canned /completion responses): digest
    refusal, the N-tokens / cache-hit VOID predicates, the slope arithmetic, TTFT streaming.

Heavy parts (clone + build + a 676 MB download) are skipped under ``CI`` unless ``SC1_LLAMACPP_TESTS=1``; point
``SC1_LLAMACPP_SRC`` at an existing checkout of the pin to skip the clone.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # the xet download backend stalls at 0 bytes on this LAN

REPO = pathlib.Path(__file__).resolve().parents[1]
LC = REPO / "bench" / "sc1" / "llamacpp"
PIN = "552f18f912a32ea86edf82e2b76431cb7131538d"
TAG = "b11327"
UPSTREAM = "https://github.com/ggml-org/llama.cpp"
GGUF_REPO, GGUF_REV = "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "9217f5db79a29953eb74d5343926648285ec7e67"
GGUF_FILE = "qwen2.5-0.5b-instruct-q8_0.gguf"
HF_REPO, HF_REV = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"
CALIB_PROMPT, CALIB_STEPS = 32, int(os.environ.get("SC1_LLAMACPP_CALIB_STEPS", "64"))
GOLD_BUDGET, SELF_BUDGET = 0.02, 0.01


def _load_arm():
    spec = importlib.util.spec_from_file_location("sc1_llamacpp_arm", LC / "sc1_llamacpp_arm.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


arm = _load_arm()


# ======================================================================================================================
# (a) build against the pinned tree
# ======================================================================================================================
def _source_at_pin():
    env = os.environ.get("SC1_LLAMACPP_SRC")
    if env:
        src = pathlib.Path(env)
        head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
        if head.returncode != 0 or head.stdout.strip() != PIN:
            pytest.skip(f"SC1_LLAMACPP_SRC={src} is not at the pin {PIN[:12]} (got {head.stdout.strip()[:12] or head.stderr.strip()})")
        return src
    if not shutil.which("git"):
        pytest.skip("no git to clone llama.cpp")
    src = pathlib.Path(os.environ.get("SC1_LLAMACPP_CACHE", str(pathlib.Path.home() / ".cache" / "sc1-llamacpp"))) / f"llama.cpp-{TAG}"
    if not (src / ".git").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", TAG, UPSTREAM, str(src)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"could not clone llama.cpp {TAG} (network?): {r.stderr.strip()[:300]}")
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != PIN:
        r = subprocess.run(["git", "-C", str(src), "fetch", "--quiet", "--depth", "1", "origin", PIN], capture_output=True, text=True)
        subprocess.run(["git", "-C", str(src), "checkout", "--quiet", "--detach", "FETCH_HEAD"], capture_output=True, text=True)
        head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != PIN:
            pytest.skip(f"{src} could not be brought to the pin {PIN[:12]}: {r.stderr.strip()[:200]}")
    return src


@pytest.fixture(scope="session")
def harness():
    """Build the pinned tree CPU-only (libllama) and the harness against it; returns {bin, src, build, commit}."""
    if os.environ.get("CI") and os.environ.get("SC1_LLAMACPP_TESTS") != "1":
        pytest.skip("CI: set SC1_LLAMACPP_TESTS=1 to clone + build llama.cpp (minutes) and download the calibration GGUF")
    cmake = shutil.which("cmake")
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if not cmake or not cxx:
        pytest.skip(f"no cmake / C++ compiler on this machine (cmake={cmake}, cxx={cxx})")
    src = _source_at_pin()
    build = src / "build"
    cfg = [cmake, "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF",
           "-DGGML_OPENMP=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
           "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_COMMON=OFF"]
    r = subprocess.run(cfg, capture_output=True, text=True)
    assert r.returncode == 0, f"cmake configure failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    jobs = str(max(2, (os.cpu_count() or 2) - 1))
    r = subprocess.run([cmake, "--build", str(build), "--config", "Release", "--target", "llama", "-j", jobs],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"libllama build failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    r = subprocess.run(["bash", str(LC / "build_harness.sh"), str(src), str(build)], capture_output=True, text=True,
                       env={**os.environ, "CXX": cxx})
    assert r.returncode == 0, f"harness build failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    binary = build / "bin" / "nll_teacher_forced"
    assert binary.exists()
    return {"bin": binary, "src": src, "build": build, "commit": PIN}


def test_harness_builds_against_pinned_tree(harness):
    info = (harness["bin"].parent / "nll_teacher_forced.buildinfo").read_text()
    assert f"llama_commit: {PIN}" in info, info
    r = subprocess.run([str(harness["bin"]), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--mode decode|prefill" in r.stderr, (r.returncode, r.stderr)


def test_harness_refuses_too_few_ids(harness, tmp_path):
    """prompt_len + steps + 1 ids are needed; fewer must refuse (exit 3) BEFORE touching the model."""
    tf = tmp_path / "ids.json"
    tf.write_text(json.dumps({"ids": list(range(50))}))
    r = subprocess.run([str(harness["bin"]), "--model", str(tmp_path / "absent.gguf"), "--tokens", str(tf),
                        "--prompt-len", "32", "--steps", "64", "--mode", "decode", "--out", str(tmp_path / "o.json")],
                       capture_output=True, text=True)
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert "REFUSE" in r.stderr and "needs 97" in r.stderr, r.stderr
    assert not (tmp_path / "o.json").exists()


# ======================================================================================================================
# (b) calibration
# ======================================================================================================================
def _calib_ids():
    return json.load(open(LC / "calib_ids.json"))["ids"]


@pytest.fixture(scope="session")
def calib_files(harness):
    hub = pytest.importorskip("huggingface_hub", reason="huggingface_hub is needed to fetch the calibration GGUF")
    try:
        gguf = hub.hf_hub_download(GGUF_REPO, GGUF_FILE, revision=GGUF_REV)
        cfg = hub.hf_hub_download(HF_REPO, "config.json", revision=HF_REV)
    except Exception as e:  # noqa: BLE001 - no network is a skip, not a failure
        pytest.skip(f"could not fetch the calibration files: {e!r}"[:300])
    return {"gguf": pathlib.Path(gguf), "config": pathlib.Path(cfg)}


def _run_harness(harness, gguf, ids_file, mode, out):
    cmd = [str(harness["bin"]), "--model", str(gguf), "--tokens", str(ids_file), "--prompt-len", str(CALIB_PROMPT),
           "--steps", str(CALIB_STEPS), "--mode", mode, "--n-gpu-layers", "0", "--threads", "4", "--quiet", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"{mode}: rc {r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}"
    return json.load(open(out))


def _gold_from_gguf(harness, gguf_path, config_path, ids):
    """transformers fp32 on the CPU with the GGUF's OWN weights (dequantised through the pinned tree's gguf-py).

    The gold must run the weights the harness ran: the official Qwen2.5-0.5B-Instruct-GGUF files were NOT converted
    from the safetensors at the pinned HF revision (per-layer tensors differ, e.g. blk.11.attn_norm max|d| = 6.9;
    the embeddings are identical) and score ~0.04 nats apart on the same window -- a gold on the safetensors would
    measure that checkpoint gap, not the harness. Validated against transformers' own ``gguf_file=`` loader
    (identical NLL to 1e-6 on both the Q8_0 and the fp16 file)."""
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    import numpy as np
    try:
        import gguf  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(harness["src"] / "gguf-py"))
    import gguf
    from gguf import GGUFReader
    from gguf.quants import dequantize

    name_map = {"token_embd": "model.embed_tokens", "output_norm": "model.norm", "output": "lm_head",
                "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
                "attn_output": "self_attn.o_proj", "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
                "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}

    def hf_name(n):
        base, suffix = n.rsplit(".", 1)
        if base.startswith("blk."):
            _, i, rest = base.split(".", 2)
            return f"model.layers.{i}.{name_map[rest]}.{suffix}"
        return f"{name_map[base]}.{suffix}"

    cfg = tf.AutoConfig.from_pretrained(str(config_path.parent))
    model = tf.Qwen2ForCausalLM(cfg).eval()
    sd = model.state_dict()
    reader = GGUFReader(str(gguf_path))
    floats = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    loaded = set()
    for t in reader.tensors:
        arr = np.asarray(t.data, dtype=np.float32) if t.tensor_type in floats else dequantize(t.data, t.tensor_type)
        shape = tuple(int(x) for x in reversed(list(t.shape)))          # GGUF shapes list ne0 (fastest) first
        w = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).reshape(shape)
        k = hf_name(t.name)
        assert k in sd and sd[k].shape == w.shape, (t.name, k, tuple(sd.get(k, w).shape), tuple(w.shape))
        with torch.no_grad():
            sd[k].copy_(w)
        loaded.add(k)
    if "lm_head.weight" not in loaded:                                  # tied embeddings: no `output` tensor in the file
        with torch.no_grad():
            sd["lm_head.weight"].copy_(sd["model.embed_tokens.weight"])
        loaded.add("lm_head.weight")
    assert set(sd) <= loaded, sorted(set(sd) - loaded)
    P, S = CALIB_PROMPT, CALIB_STEPS
    with torch.no_grad():
        lg = model(torch.tensor([ids[:P + S]])).logits[0].float()
    lp = torch.log_softmax(lg, -1)
    pos = torch.arange(P, P + S)
    tgt = torch.tensor(ids[P + 1:P + S + 1])                             # the harness's targets: ids[P+1 .. P+S]
    nll = (-lp[pos, tgt]).mean().item()
    sha_torch = hashlib.sha256(torch.tensor(ids[:P + S + 1], dtype=torch.long).numpy().tobytes()).hexdigest()
    return nll, sha_torch, f"torch {torch.__version__} / transformers {tf.__version__} fp32 CPU on the GGUF's dequantised weights"


def test_calibration_decode_prefill_vs_gold(harness, calib_files, tmp_path):
    ids = _calib_ids()
    need = CALIB_PROMPT + CALIB_STEPS + 1
    assert len(ids) >= need
    ids_file = tmp_path / "ids.json"
    ids_file.write_text(json.dumps({"ids": ids}))
    dec = _run_harness(harness, calib_files["gguf"], ids_file, "decode", tmp_path / "decode.json")
    pre = _run_harness(harness, calib_files["gguf"], ids_file, "prefill", tmp_path / "prefill.json")

    # the K8 digest contract: little-endian int64 of the first prompt_len+steps+1 ids
    sha_struct = hashlib.sha256(struct.pack(f"<{need}q", *ids[:need])).hexdigest()
    assert dec["text_sha256"] == sha_struct == pre["text_sha256"]
    for rep, mode in ((dec, "decode"), (pre, "prefill")):
        assert rep["mode"] == mode and rep["steps"] == CALIB_STEPS and rep["prompt_len"] == CALIB_PROMPT
        assert rep["llama_commit"] == PIN
        assert rep["model_sha256"] == hashlib.sha256(calib_files["gguf"].read_bytes()).hexdigest()
        assert rep["n_gpu_layers_requested"] == 0 and rep["flash_attn_requested"] == "on"
        assert rep["n_ubatch"] >= 1 and rep["n_batch"] >= rep["n_ubatch"]
        assert rep["ppl"] > 1.0 and rep["mean_nll"] > 0 and rep["tokens_scored"] == CALIB_STEPS
    assert pre["n_batch"] >= CALIB_PROMPT + CALIB_STEPS, pre["n_batch"]
    d_self = dec["mean_nll"] - pre["mean_nll"]
    lines = [f"CALIBRATION {GGUF_FILE} prompt={CALIB_PROMPT} steps={CALIB_STEPS} sha={sha_struct[:12]} llama.cpp={PIN[:12]}",
             f"  harness decode  mean_nll={dec['mean_nll']:.6f} ppl={dec['ppl']:.4f} top1={dec['top1_agree_frac']:.4f} "
             f"tok/s={dec['scored_tok_s']:.1f} n_batch={dec['n_batch']} n_ubatch={dec['n_ubatch']}",
             f"  harness prefill mean_nll={pre['mean_nll']:.6f} ppl={pre['ppl']:.4f} top1={pre['top1_agree_frac']:.4f} "
             f"tok/s={pre['scored_tok_s']:.1f} n_batch={pre['n_batch']} n_ubatch={pre['n_ubatch']}",
             f"  |decode - prefill| = {abs(d_self):.6f} (budget {SELF_BUDGET})"]
    gold_ran = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    if gold_ran:
        gold, sha_torch, how = _gold_from_gguf(harness, calib_files["gguf"], calib_files["config"], ids)
        assert sha_torch == sha_struct, "torch int64 .numpy().tobytes() digest must equal the harness's little-endian int64 digest"
        d_dec, d_pre = dec["mean_nll"] - gold, pre["mean_nll"] - gold
        lines += [f"  gold ({how}) mean_nll={gold:.6f}",
                  f"  delta decode-gold = {d_dec:+.6f}  delta prefill-gold = {d_pre:+.6f} (budget {GOLD_BUDGET})"]
        print("\n".join(lines))
        assert abs(d_dec) <= GOLD_BUDGET, "\n".join(lines)
        assert abs(d_pre) <= GOLD_BUDGET, "\n".join(lines)
    else:
        lines.append("  gold: NOT RUN (torch/transformers not importable) -- only decode-vs-prefill self-consistency asserted")
        print("\n".join(lines))
    assert abs(d_self) <= SELF_BUDGET, "\n".join(lines)


# ======================================================================================================================
# (c) the arm driver against a fake llama-server
# ======================================================================================================================
class FakeLlamaServer:
    """Canned llama-server: /health, /props, /metrics and /completion (non-stream + SSE stream).

    short_by      : generate n_predict - short_by tokens (a row that stops early)
    prompt_n_delta: report timings.prompt_n = len(prompt) + delta (a cache hit when negative)
    base_s/per_tok_s : non-stream latency model wall = base + per_tok * n  (so the slope is per_tok)
    first_tok_s/stream_per_tok_s : streaming: delay before the first token event, then per token
    """

    def __init__(self, short_by=0, prompt_n_delta=0, base_s=0.0, per_tok_s=0.0, first_tok_s=0.0, stream_per_tok_s=0.0):
        self.short_by, self.prompt_n_delta = short_by, prompt_n_delta
        self.base_s, self.per_tok_s = base_s, per_tok_s
        self.first_tok_s, self.stream_per_tok_s = first_tok_s, stream_per_tok_s
        self.requests = []
        self.lock = threading.Lock()
        srv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # quiet
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    return self._json(200, {"status": "ok"})
                if self.path == "/props":
                    return self._json(200, {"total_slots": 16, "build_info": "b1-fake", "default_generation_settings": {"n_ctx": 1024}})
                if self.path == "/metrics":
                    body = b"llamacpp:tokens_predicted_total 0\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return None
                return self._json(404, {"error": "no"})

            def do_POST(self):
                n_bytes = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n_bytes))
                with srv.lock:
                    srv.requests.append(body)
                if self.path != "/completion":
                    return self._json(404, {"error": "no"})
                n = int(body["n_predict"]) - srv.short_by
                prompt_n = len(body["prompt"]) + srv.prompt_n_delta
                toks = [1000 + i for i in range(n)]
                timings = {"prompt_n": prompt_n, "prompt_ms": 12.5, "prompt_per_token_ms": 12.5 / max(1, prompt_n),
                           "predicted_n": n, "predicted_ms": 1000.0 * srv.per_tok_s * n, "predicted_per_token_ms": 1000.0 * srv.per_tok_s}
                final = {"index": 0, "content": "x" * n, "tokens": toks, "id_slot": 0, "stop": True, "model": "fake",
                         "tokens_predicted": n, "tokens_evaluated": len(body["prompt"]), "truncated": False,
                         "stop_type": "limit" if srv.short_by == 0 else "eos", "stopping_word": "", "timings": timings}
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    time.sleep(srv.first_tok_s)
                    # as llama-server streams (server-context.cpp:2046-2066, :2102-2105 @ b11327): one partial event per
                    # token carrying that token, then a SEPARATE final event with stop: true, EMPTY content/tokens, timings
                    for i, t in enumerate(toks):
                        ev = {"index": 0, "content": "x", "tokens": [t], "stop": False, "id_slot": 0,
                              "tokens_predicted": i + 1, "tokens_evaluated": len(body["prompt"])}
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                        if i < len(toks) - 1:
                            time.sleep(srv.stream_per_tok_s)
                    final.update(content="", tokens=[])
                    self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                    self.wfile.flush()
                    self.close_connection = True
                    return None
                time.sleep(srv.base_s + srv.per_tok_s * n)
                return self._json(200, final)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _prompts_file(path, batch, prompt_len=512, tamper=None):
    prompts = [[(i * 7919 + j * 104729) % 150000 + 10 for j in range(prompt_len)] for i in range(batch)]
    pf = {"batch": batch, "prompt_len": prompt_len, "prompts": prompts,
          "prompts_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
          "rows_sha256": [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]}
    if tamper:
        pf.update(tamper)
    path.write_text(json.dumps(pf))
    return pf


def test_prompt_file_digest_refusal(tmp_path):
    good = tmp_path / "prompts_b2.json"
    _prompts_file(good, 2)
    prompts, sha, rows = arm.load_prompts(str(good), 2)
    assert len(prompts) == 2 and len(rows) == 2 and len(sha) == 64
    bad = tmp_path / "prompts_b2_tampered.json"
    _prompts_file(bad, 2, tamper={"prompts_sha256": "0" * 64})
    with pytest.raises(SystemExit) as e:
        arm.load_prompts(str(bad), 2)
    assert e.value.code == 2 and "digest mismatch" in e.value.msg
    with pytest.raises(SystemExit) as e:                  # batch mismatch
        arm.load_prompts(str(good), 16)
    assert "--batch" in e.value.msg
    dup = tmp_path / "prompts_dup.json"
    pf = _prompts_file(dup, 1)
    pf["prompts"] = [pf["prompts"][0], pf["prompts"][0]]
    pf["batch"] = 2
    pf["prompts_sha256"] = hashlib.sha256(json.dumps(pf["prompts"]).encode()).hexdigest()
    dup.write_text(json.dumps(pf))
    with pytest.raises(SystemExit) as e:
        arm.load_prompts(str(dup), 2)
    assert "distinct" in e.value.msg
    # main() refuses before any request: a dead server address must not be contacted
    rc = None
    with pytest.raises(SystemExit) as e:
        rc = arm.main(["--server", "http://127.0.0.1:9", "--prompts", str(bad), "--batch", "2", "--arm", "x",
                       "--out", str(tmp_path / "r.json")])
    assert e.value.code == 2 and rc is None and not (tmp_path / "r.json").exists()


def test_slope_stats_arithmetic():
    s = arm.slope_stats([1.0, 1.1, 1.2], [2.92, 3.1, 3.0], batch=16)
    assert s["decode_ms_per_step"] == pytest.approx(20.0)                 # (2.92 - 1.0) / 96 s
    assert s["decode_tok_s"] == pytest.approx(16 * 96 / 1.92, abs=0.1)     # 800
    assert s["decode_ms_per_step_median"] == pytest.approx((3.0 - 1.1) / 96 * 1e3, abs=1e-3)
    assert s["decode_tok_s_median"] == pytest.approx(16 * 96 / 1.9, abs=0.1)
    assert s["end_to_end_tok_s_long"] == pytest.approx(128 * 16 / 2.92, abs=0.1)
    assert s["wall_short_s"] == 1.0 and s["wall_long_s"] == 2.92
    flat = arm.slope_stats([2.0, 2.0, 2.0], [2.0, 1.9, 2.1], batch=1)
    assert flat["decode_ms_per_step"] is None and "slope_note" in flat
    s2 = arm.slope_stats([0.5, 0.5, 0.5], [1.3, 1.3, 1.3], batch=2, short=8, long_=24)
    assert s2["decode_ms_per_step"] == pytest.approx(50.0) and s2["decode_tok_s"] == pytest.approx(2 * 16 / 0.8)


def test_arm_slope_against_fake_server(tmp_path):
    srv = FakeLlamaServer(base_s=0.02, per_tok_s=0.003)
    try:
        pfile = tmp_path / "prompts_b2.json"
        _prompts_file(pfile, 2, prompt_len=16)
        out = tmp_path / "r.json"
        rc = arm.main(["--server", srv.base, "--prompts", str(pfile), "--batch", "2", "--arm", "fake_b2", "--out", str(out),
                       "--prompt-len", "16", "--short", "8", "--long", "24", "--reps", "3", "--server-version", "fake 1"])
        assert rc == 0
        r = json.load(open(out))
        assert r["engine"] == "llamacpp" and r["arm"] == "fake_b2" and r["batch"] == 2 and r["verdict"] == "VALID"
        assert r["prompts_sha256"] == json.load(open(pfile))["prompts_sha256"] and r["prompt_tokens"] == [16, 16]
        assert r["server_version"] == "fake 1" and r["props"]["total_slots"] == 16 and '"ok"' in r["health"]
        # slope: per_tok_s = 3 ms/step; walls are B concurrent so the batch does not multiply the per-step cost
        assert 2.0 <= r["decode_ms_per_step"] <= 6.0, r["decode_ms_per_step"]
        assert r["decode_tok_s"] == pytest.approx(2 * 16 / (r["decode_ms_per_step"] * 16 / 1e3), rel=1e-3)
        assert len(r["walls_short_s"]) == 3 and len(r["walls_long_s"]) == 3
        assert r["tokens"] == {"0": [1000 + i for i in range(24)], "1": [1000 + i for i in range(24)]}
        assert r["server_timings_summary"]["long"]["predicted_per_token_ms_median"] == pytest.approx(3.0)
        assert len(r["server_timings"]["long"]) == 3 and len(r["server_timings"]["long"][0]) == 2
        # every request carried the registered knobs; (1 warm + 3 reps) x 2 lengths x B rows
        assert len(srv.requests) == (1 + 3) * 2 * 2
        for q in srv.requests:
            assert q["temperature"] == 0 and q["cache_prompt"] is False and q["ignore_eos"] is True
            assert q["return_tokens"] is True and q["stream"] is False and all(isinstance(t, int) for t in q["prompt"])
        assert sorted(q["n_predict"] for q in srv.requests) == [8] * 8 + [24] * 8
    finally:
        srv.close()


def test_arm_void_when_a_row_stops_early(tmp_path):
    srv = FakeLlamaServer(short_by=1)
    try:
        pfile = tmp_path / "prompts_b1.json"
        _prompts_file(pfile, 1, prompt_len=16)
        out = tmp_path / "r.json"
        rc = arm.main(["--server", srv.base, "--prompts", str(pfile), "--batch", "1", "--arm", "fake_short", "--out", str(out),
                       "--prompt-len", "16", "--short", "8", "--long", "24", "--reps", "1"])
        assert rc == 3
        r = json.load(open(out))
        assert r["verdict"] == "VOID" and "stopped early" in r["void_reason"] and "7" in r["void_reason"]
        assert "decode_ms_per_step" not in r
    finally:
        srv.close()


def test_arm_void_on_prompt_cache_hit(tmp_path):
    srv = FakeLlamaServer(prompt_n_delta=-5)
    try:
        pfile = tmp_path / "prompts_b1.json"
        _prompts_file(pfile, 1, prompt_len=16)
        out = tmp_path / "r.json"
        rc = arm.main(["--server", srv.base, "--prompts", str(pfile), "--batch", "1", "--arm", "fake_cache", "--out", str(out),
                       "--prompt-len", "16", "--short", "8", "--long", "24", "--reps", "1"])
        assert rc == 3
        r = json.load(open(out))
        assert r["verdict"] == "VOID" and "cache" in r["void_reason"] and "prompt_n 11" in r["void_reason"]
    finally:
        srv.close()


def test_arm_ttft_mode_measures_first_token(tmp_path):
    srv = FakeLlamaServer(first_tok_s=0.15, stream_per_tok_s=0.03)
    try:
        pfile = tmp_path / "prompts_b1.json"
        _prompts_file(pfile, 1, prompt_len=16)
        out = tmp_path / "r.json"
        rc = arm.main(["--server", srv.base, "--prompts", str(pfile), "--batch", "1", "--arm", "fake_ttft", "--out", str(out),
                       "--prompt-len", "16", "--ttft", "--ttft-n-predict", "8", "--reps", "3"])
        assert rc == 0
        r = json.load(open(out))
        assert r["mode"] == "ttft" and r["verdict"] == "VALID" and r["ttft_n_predict"] == 8
        assert len(r["ttft_s"]) == 3 and 0.12 <= r["ttft_s_median"] <= 0.45, r["ttft_s"]
        for rep in r["ttft_reps"]:
            assert rep["tokens"] == [1000 + i for i in range(8)]
            assert rep["total_s"] >= rep["ttft_s"] + 7 * 0.03 * 0.8      # the first token arrived well before the stream ended
            assert rep["timings"]["prompt_n"] == 16
        assert r["prefill_tok_s_from_ttft"] == pytest.approx(16 / r["ttft_s_median"], rel=1e-3)
        assert r["server_prompt_ms_median"] == 12.5
        assert len(srv.requests) == 4 and all(q["stream"] is True and q["n_predict"] == 8 for q in srv.requests)
    finally:
        srv.close()


def test_arm_ttft_refuses_batch_above_one(tmp_path):
    srv = FakeLlamaServer()
    try:
        pfile = tmp_path / "prompts_b2.json"
        _prompts_file(pfile, 2, prompt_len=16)
        out = tmp_path / "r.json"
        rc = arm.main(["--server", srv.base, "--prompts", str(pfile), "--batch", "2", "--arm", "x", "--out", str(out),
                       "--prompt-len", "16", "--ttft"])
        assert rc == 3 and json.load(open(out))["verdict"] == "VOID"
    finally:
        srv.close()


# --- A3 (sc1b-prove-5): the version check accepts git's abbreviation as a prefix of the pinned commit ---------------------

def _version_check(version_text: str, commit: str) -> int:
    src = (REPO / "bench" / "sc1" / "llamacpp" / "llamacpp_box.sh").read_text()
    a = src.index("llamacpp_version_names_commit() {")
    b = src.index("\n}\n", a) + 3
    with __import__("tempfile").NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(version_text)
    try:
        return subprocess.run(["bash", "-c", src[a:b] + f'\nllamacpp_version_names_commit "{f.name}" "{commit}"'],
                              capture_output=True, text=True).returncode
    finally:
        os.unlink(f.name)


def test_llamacpp_version_check_takes_the_printed_abbreviation_as_a_prefix():
    """sc1b-prove-5: llama.cpp b11327 BUILT, then the box refused it -- it grepped the 8-char commit 552f18f9 while
    `llama-server --version` prints git's 7-char abbreviation ("commit 552f18f")."""
    pin = "552f18f912a32ea86edf82e2b76431cb7131538d"
    real = "version: 0.5.0-dev (build 1, commit 552f18f)\nbuilt with GNU 11.4.0 for Linux x86_64\n"
    assert _version_check(real, pin) == 0                                                   # the box's actual output
    assert _version_check(real.replace("552f18f)", "552f18f9)"), pin) == 0                   # a longer abbreviation
    assert _version_check(real, "0" * 40) != 0                                              # a different commit
    assert _version_check(real.replace("552f18f)", "552f18)"), pin) != 0                    # 6 chars names nothing
    assert _version_check("version: 0.5.0-dev (build 1)\n", pin) != 0                       # no commit at all
    # the registered check, for the record: an 8-char grep cannot match the 7-char output
    assert "552f18f9" not in real
