"""Lane P127's instrument on CPU (bench/p127/PREREG-p127.md; e4b#1313).

- The staged-file pin matches the repo, as `bench/p127/p127_drive.sh` checks before it stages anything.
- The box's and the reducer's self-tests pass.
- The logits observer reads a step's logits BEFORE the next replay can overwrite the graph-owned storage, and only in
  the identity pass (the maintainer's two guards, 2026-10-09).
- The engagement wrappers keep the real signatures (e4b's feature detection reads them) and count the P127 calls; M's
  store rounds toward zero.
- The box's `_padded_step` replica is the repo's, byte for byte.
- The reducer reads records shaped as the box writes them (P115 Phase D's Amendment 4 lesson).
- The runner's shape: the arms and their order, the subject, the registered stacks, the audit, the staging.
"""
import hashlib
import importlib.util
import inspect
import json
import pathlib
import re
import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch")

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p127"
PIN = LANE / "staged.sha256"
SOURCES = {
    "p127_run.sh": LANE / "p127_run.sh",
    "p127_box.py": LANE / "p127_box.py",
    "p127_reduce.py": LANE / "p127_reduce.py",
    "p109_box.py": REPO / "bench" / "p109" / "p109_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py",
    "calib.json": REPO / "bench" / "p39" / "calib.json",
    "test_decode_graph_buckets.py": REPO / "tests" / "test_decode_graph_buckets.py",
    "test_t1_glue_host_casts.py": REPO / "tests" / "test_t1_glue_host_casts.py",
}
RUN = (LANE / "p127_run.sh").read_text()
DRIVE = (LANE / "p127_drive.sh").read_text()


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


box = _load("p127_box")
red = _load("p127_reduce")
from experts4bit_qlora.engines.paged_runner import PagedModelRunner as _PMR  # noqa: E402
_PRISTINE = {"_padded_step": _PMR.__dict__["_padded_step"], "run_decode": _PMR.__dict__["run_decode"]}


def _entries():
    for line in PIN.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            want, name = line.split(None, 1)
            yield want, name.strip()


def test_the_staged_pin_matches_the_repo():
    seen = {}
    for want, name in _entries():
        assert name in SOURCES, f"staged.sha256 names {name}, which the driver does not stage"
        got = hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
        assert got == want, f"{name}: {got} != pinned {want}"
        seen[name] = got
    assert set(seen) == set(SOURCES)
    staged = re.search(r'STAGE="([^"]+)"', DRIVE).group(1).split()
    assert {pathlib.PurePosixPath(s).name for s in staged} == set(SOURCES) | {"staged.sha256"}


@pytest.mark.parametrize("script", ["p127_box.py", "p127_reduce.py"])
def test_the_self_tests_pass(script):
    r = subprocess.run([sys.executable, str(LANE / script), "--self-test"], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_the_padded_step_replica_is_the_repos():
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    assert inspect.getsource(PagedModelRunner._padded_step) == box.PADDED_STEP_SRC


# ------------------------------------------------------------------------- the logits observer's two guards --
class _FakeRunner:
    """A bucketed runner in miniature: each bucket's logits live in ONE storage that a "replay" overwrites in place
    (as a captured graph's output does), and run_decode reads the step's tokens -- the sync -- before returning."""
    V = 8

    def __init__(self):
        from experts4bit_qlora.engines.paged_runner import bucket_for
        self.bucket_for = bucket_for
        self._buckets = [1, 2, 4]
        self._graphs = {b: object() for b in self._buckets}
        self._bufs = {b: {"ids": torch.zeros(b, 1, dtype=torch.long), "pos": torch.zeros(b, 1, dtype=torch.long),
                          "tok": torch.zeros(b, dtype=torch.long)} for b in self._buckets}
        self.store = {b: torch.zeros(b, 1, self.V, dtype=torch.bfloat16) for b in self._buckets}
        self.step = 0
        self.written = []

    def model(self, input_ids, position_ids, use_cache):
        return types.SimpleNamespace(logits=self.store[input_ids.shape[0]])

    def capture(self):
        for b in self._buckets:
            type(self)._padded_step(self, b)

    def run_decode(self, rids):
        n = len(rids)
        b = self.bucket_for(n, self._buckets)
        self.step += 1
        vals = (torch.arange(b * self.V, dtype=torch.float32).view(b, 1, self.V) + 100 * self.step).to(torch.bfloat16)
        self.store[b].copy_(vals)                                  # the replay, in place
        self._bufs[b]["tok"].copy_(self.store[b][:, -1].argmax(-1))
        self.written.append(self.store[b][:n, -1].clone())
        return dict(zip(rids, self._bufs[b]["tok"][:n].tolist()))


def _observed(monkeypatch, identity=True):
    obs = box.Observers("B")
    monkeypatch.setattr(_FakeRunner, "_padded_step", obs.padded_step(), raising=False)
    monkeypatch.setattr(_FakeRunner, "run_decode", obs.run_decode(_FakeRunner.run_decode, _FakeRunner().bucket_for))
    r = _FakeRunner()
    r.capture()
    obs.identity = identity
    return obs, r


def _sha(row):
    return hashlib.sha256(row.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


def test_each_steps_logits_are_digested_before_the_next_replay_overwrites_them(monkeypatch):
    obs, r = _observed(monkeypatch)
    for _ in range(3):
        r.run_decode([10, 11])
    want = {rid: [_sha(w[i]) for w in r.written] for i, rid in enumerate((10, 11))}
    assert obs.digests == want, "each digest is of its own step's values"
    assert len(set(obs.digests[10])) == 3, "three steps, three different logits: nothing was read late"


def test_a_timed_pass_does_no_digest_work(monkeypatch):
    obs, r = _observed(monkeypatch, identity=False)
    for _ in range(3):
        r.run_decode([10])
    assert obs.digests == {} and obs.steps_seen == 0


def test_the_identity_read_refuses_a_step_it_could_not_read(monkeypatch):
    obs, r = _observed(monkeypatch)
    r._graphs = None
    with pytest.raises(RuntimeError, match="identity pass"):
        r.run_decode([10])


# --------------------------------------------------------------------------- the engagement wrappers and M --
def _stub_kernels(monkeypatch):
    k = types.ModuleType("int4_b32")

    def router_epilogue(logits, k_, norm, *, select_on_logits=False, bias=None, weights_dtype=torch.float32):
        return logits, (logits[:, :k_] * 0.3).to(weights_dtype), None

    def combine_rows(dn, w, k_, *, residual=None):
        return dn

    def rope_norm_qk(q, k_, qw, kw, cos, sin, qe, ke):
        return q, k_
    k.router_epilogue, k.combine_rows, k.rope_norm_qk = router_epilogue, combine_rows, rope_norm_qk
    n = types.ModuleType("nf4_grouped")

    def gemm_4bit_grouped(a_cat, B, absmax, sizes, expert_ids, block_m=None, gather_div=1):
        return a_cat
    n.gemm_4bit_grouped = gemm_4bit_grouped
    k.__file__, n.__file__ = "/stub/kernel/int4_b32.py", "/stub/kernel/nf4_grouped.py"
    monkeypatch.setitem(sys.modules, "int4_b32", k)
    monkeypatch.setitem(sys.modules, "nf4_grouped", n)
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    # each arm is a fresh process on the box; here the class is shared, so start every arm from the pristine methods
    monkeypatch.setattr(PagedModelRunner, "_padded_step", _PRISTINE["_padded_step"])
    monkeypatch.setattr(PagedModelRunner, "run_decode", _PRISTINE["run_decode"])
    return k, n


@pytest.mark.parametrize("arm", ["A", "B", "M"])
def test_the_wrappers_keep_the_signatures_and_count_the_p127_calls(monkeypatch, arm):
    k, n = _stub_kernels(monkeypatch)
    real_sig = {f: inspect.signature(getattr(k, f)) for f in ("router_epilogue", "combine_rows", "rope_norm_qk")}
    real_gemm = inspect.signature(n.gemm_4bit_grouped)
    obs = box.Observers(arm)
    box.install_observers(obs, torch)
    for f, sig in real_sig.items():
        assert inspect.signature(getattr(k, f)) == sig, f
    assert inspect.signature(n.gemm_4bit_grouped) == real_gemm
    lg = torch.randn(2, 8)
    _f, w, _i = k.router_epilogue(lg, 4, True, weights_dtype=torch.bfloat16)
    k.router_epilogue(lg, 4, True)                                           # fp32: not a P127 call
    k.combine_rows(lg, lg, 2, residual=lg)
    k.combine_rows(lg, lg, 2)
    k.rope_norm_qk(lg, lg, lg, lg, lg, lg, 1e-6, 1e-6)
    n.gemm_4bit_grouped(lg, None, None, [1], torch.tensor([1], dtype=torch.int64), gather_div=8)
    n.gemm_4bit_grouped(lg, None, None, [1], torch.tensor([1], dtype=torch.int32))
    assert obs.counts == {"router_weights_dtype": 1, "rope_norm_qk": 1, "combine_residual": 1, "gather_div": 1,
                          "int64_ids": 1}
    want = box.rtz_bf16(lg[:, :4] * 0.3, torch) if arm == "M" else (lg[:, :4] * 0.3).to(torch.bfloat16)
    assert torch.equal(w, want)
    if arm == "M":
        assert not torch.equal(w, (lg[:, :4] * 0.3).to(torch.bfloat16)), "M's store differs from nearest-even"


def test_the_box_refuses_a_padded_step_it_does_not_reproduce(monkeypatch):
    _stub_kernels(monkeypatch)
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner

    def other(self, b):
        return None
    monkeypatch.setattr(PagedModelRunner, "_padded_step", other)
    with pytest.raises(SystemExit, match="_padded_step"):
        box.install_observers(box.Observers("B"), torch)


# ---------------------------------------------------------------- the reducer reads the box's own records --
class _Sched:
    def __init__(self):
        self.done, self.active, self.queue, self._rid, self._pending = [], [], [], 0, []

    def add_request(self, row, max_new_tokens):
        self._rid += 1
        self._pending.append(types.SimpleNamespace(rid=self._rid, prompt_len=len(row),
                                                   out=[(row[0] + i) % 97 for i in range(max_new_tokens)]))
        return self._rid

    def run_until_idle(self):
        self.done.extend(self._pending)
        n, self._pending = len(self._pending), []
        return n


def _box_record(monkeypatch, tmp_path, arm, tag, e4b, gnf4, prompts):
    _stub_kernels(monkeypatch)
    from experts4bit_qlora import serve_paged
    cfg = types.SimpleNamespace(graphs=True, placement="all-vram", buckets=(1, 2, 4, 8, 16), max_seqs=16,
                                model="Qwen/Qwen3-30B-A3B", revision="rev")
    parts = types.SimpleNamespace(info={"graph_status": {1: "graph"}, "moe_layers": 48,
                                        "moe_residual": {"rows": [1, 2, 4, 8, 16], "licensed": 48}},
                                  scheduler=_Sched(), runner=types.SimpleNamespace(graph_stats={}))
    monkeypatch.setattr(serve_paged.PagedServeConfig, "from_env", staticmethod(lambda: cfg))
    monkeypatch.setattr(serve_paged, "build_engine", lambda c: parts)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1, 2))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 7)
    monkeypatch.setenv("P127_ARM", arm)
    monkeypatch.setenv("E4B_SHA", e4b)
    monkeypatch.setenv("GNF4_SHA", gnf4)
    out = tmp_path / f"arm_{tag}.json"
    argv = ["--prompts", str(prompts), "--out", str(out), "--tag", tag, "--short", "4", "--long", "8", "--reps", "2"]
    if arm == "M":
        argv.append("--identity-only")
    walls = iter([0.0, 1.0, 1.0, 2.0] * 400)
    monkeypatch.setattr(box.time, "perf_counter", lambda: next(walls))
    assert box.main(argv) == 0
    return json.loads(out.read_text())


def test_the_reducer_reads_the_boxs_own_records(monkeypatch, tmp_path):
    rows = [[k + 1] * 4 for k in range(16)]
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"rows": rows, "prompts_sha256": box.digest(rows)}))
    want = {"e4b_a": "a" * 40, "e4b_b": "b" * 40, "gnf4_a": "c" * 40, "gnf4_b": "d" * 40, "revision": "rev"}
    recs = {}
    for tag in ("A1", "B1", "B2", "A2", "M1"):
        side = "a" if tag[0] == "A" else "b"
        recs[tag] = _box_record(monkeypatch, tmp_path, tag[0], tag, want[f"e4b_{side}"], want[f"gnf4_{side}"], prompts)
    assert recs["M1"]["identity_only"] and recs["M1"]["workloads"] == {}
    res = red.reduce(recs, {"ok": True, "unlisted": []}, want)
    # the stand-ins engage nothing and have no logits, so the verdict is VOID -- for those reasons, not a missing key
    assert res["verdict"] == "VOID"
    assert any("not engaged" in r for r in res["reasons"]) and any("blind" in r for r in res["reasons"])


# --------------------------------------------------------------------------------------- the runner's shape --
def test_the_runner_runs_the_registered_arms_on_the_registered_stacks():
    assert re.search(r"for TAG in A1 B1 B2 A2 M1; do", RUN)
    assert "E4B_PAGED_MAX_SEQS=16" in RUN and 'EXTRA="--identity-only"' in RUN
    assert "E4B_A=a8c01d426bc3e489c8bd11c7c8ced091da557582" in RUN
    assert "GNF4_A=d1f64ba50afce94533e0166ba3332ef075aa43bb" in RUN
    assert re.search(r"^GNF4_B=d769d5022c0fb7a2ada847f69a3cba6e0f45c77f", RUN, re.M)
    assert "E4B_B=$E4B_SHA" in RUN
    assert 'case "$GNF4_B $E4B_P127" in *__*) say "refusing: a registered SHA is still a placeholder"' in RUN
    assert "finish 31" in RUN and "audit.json" in RUN
    assert '-k real_kernels' in RUN and '"7 passed"' in RUN
    order = [RUN.index(s) for s in ("AUDIT FAIL", "TRIPWIRE FAIL", "SELF-TEST FAILED", "PREMISE FAILED", "fetch $MODEL",
                                    "bake the NF4 arena", "P127_PROMPTS", "for TAG in", "say \"reduce\"")]
    assert order == sorted(order), "refusals, audit and premise before the fetch; the arms before the reducer"


def test_the_audit_allows_exactly_the_registered_commits():
    prereg = (LANE / "PREREG-p127.md").read_text(encoding="utf-8")
    for short in ("ce3dfb54", "1ddcb0ae", "058f98eb", "f5398962", "abe7d772", "14b1f23", "e21a712", "7d4163b", "18f5bda", "f69adcc",
                  "d0a2e56", "b64a39b", "d769d502"):
        assert short in prereg, f"{short} is not in the PREREG's audit table"
        assert re.search(short + r"[0-9a-f]{%d}" % (40 - len(short)), RUN), f"{short} is not a full SHA in the runner"
