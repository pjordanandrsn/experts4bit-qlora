"""CPU composition checks; GPU kernels/capture still need the on-card premise."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench/ra" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


env = load("ra_env")
load("ra_stage")
load("ra_quality")
load("ra_fallback")
load("ra_routes")
serving = load("ra_serving")


@pytest.fixture(autouse=True)
def synthetic_route_composition(monkeypatch):
    # Composition controls only; independent route mutation tests use their own
    # declared synthetic source adapter. Runtime never accepts an adapter spec.
    class FakeRoutes:
        unscoped_calls = 0
        def __init__(self, nf4):
            self.nf4 = nf4
        def start(self):
            return self
        def close(self):
            pass
        def candidates(self, model):
            return []
        def assemble(self, model, candidates, modes):
            pass
        def snapshot(self):
            return self.nf4.dispatch_counts()
        def evidence(self, info, calls, before=None):
            count = sum(v - (before or {}).get(k, 0) for k, v in self.snapshot().items())
            return {"qkv": {"fallback_calls": 0}, "decode_gemv": {"calls": count, "fallback_calls": 0}}
    monkeypatch.setattr(serving.ra_routes.RouteObserver, "native", lambda nf4: FakeRoutes(nf4))


@pytest.mark.parametrize("name", ["E4B_PAGED_MAX_SEQS", "E4B_PAGED_BUCKETS", "E4B_FUSE_T1_GLUE",
                                 "GNF4_GEMV_BW", "NF4_QLORA_SINGLE_LADDER", "TC1_PAD_CENSUS", "P115D_ARM"])
def test_no_speed_flags_survive_cleaning_or_direct_invocation(name, tmp_path):
    base = {name: "1", "PATH": "/usr/bin", "HF_TOKEN": "synthetic-secret", "PYTHONPATH": "/foreign"}
    clean, removed = env.clean(base, component="decode", fixture={}, venv=tmp_path / "old",
                               cache=tmp_path / "old-cache", threads=8, allocator="expandable_segments:True")
    assert name not in clean and name in removed
    assert "HF_TOKEN" not in clean and "PYTHONPATH" not in clean
    assert clean["TRITON_CACHE_DIR"] != str(tmp_path / "new-cache" / "triton")
    with pytest.raises(ValueError, match="unregistered"):
        env.check_current("decode", {name: "1"})


def test_fixtures_cannot_force_decode_graphs_or_change_quality_mode():
    with pytest.raises(ValueError):
        env.validate("decode", {"E4B_PAGED_GRAPHS": "1"})
    with pytest.raises(ValueError):
        env.validate("quality", {"E4B_PAGED_GRAPHS": "1"})
    assert env.validate("quality", {"E4B_PAGED_GRAPHS": "0"})
    with pytest.raises(ValueError):
        env.validate("capacity", {"E4B_PAGED_MAX_TOKENS_PER_SEQ": "4096", "E4B_PAGED_CHUNK_TOKENS": "512"})
    assert env.validate("capacity", {"E4B_PAGED_MAX_TOKENS_PER_SEQ": "2048", "E4B_PAGED_CHUNK_TOKENS": "512"})


@pytest.mark.parametrize("mode,patched,calls", [("auto", 1, 0), ("1", 0, 0), ("0", 0, 1), ("auto", 0, 1), ("0", 1, 0)])
def test_feature_census_and_calls_cannot_disagree(mode, patched, calls):
    with pytest.raises(ValueError, match="engagement"):
        serving.feature(mode, patched, calls, scope="synthetic")


def test_auto_unmatched_family_is_explicitly_inapplicable():
    assert serving.feature("auto", 0, 0, scope="synthetic")["mode"] == "inapplicable"


def fake_build(*, fail=False):
    counter = types.SimpleNamespace(snapshot=lambda: {"rmsnorm_rows": 0, "rmsnorm_resid_rows": 0,
                                                      "rope_norm_heads": 0, "rope_heads": 0, "router_epilogue": 0})
    instrument = types.SimpleNamespace(KernelCounters=lambda: types.SimpleNamespace(install=lambda: counter),
                                       CENSUS_KEYS=("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n"))

    def forward(model):
        model.observed = True
        return types.SimpleNamespace(qkv_calls=0, snapshot=lambda: {"forwards": 1, "qkv_calls": 0})

    instrument.ForwardCounter = forward
    modes = {k: "auto" for k in ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")}
    info = {"fusion_modes": modes, "fuse_qkv_n": 0, "fuse_t1_glue_n": 0, "fuse_t1_glue_r2_n": [0, 0],
            "fuse_router_epilogue_n": 0, "moe_layers": 1, "graph_status": {1: "graph", 16: "graph"}}
    info["fusion_report"] = {k: {"mode": "auto", "patched": [0, 0] if k == "E4B_FUSE_T1_GLUE_R2" else 0}
                             for k in modes if k != "E4B_PAGED_FUSE_QKV"}
    parts = types.SimpleNamespace(info=info, runner=types.SimpleNamespace(
        graph_stats={1: {"replays": 0, "eager_steps": 0}, 16: {"replays": 0, "eager_steps": 0}}))
    server = types.SimpleNamespace(_apply_fusions=lambda model, cfg: None)
    state = {"scalar": 0}
    nf4 = types.SimpleNamespace(dispatch_counts=lambda: dict(state), state=state)

    def build(cfg):
        model = types.SimpleNamespace(observed=False, eval=lambda: None, named_modules=lambda: [])
        server._apply_fusions(model, cfg)
        assert model.observed  # Hooks must exist before the simulated graph capture.
        if fail:
            raise RuntimeError("capture failed")
        state["scalar"] += 1
        parts.runner.model = model
        return parts

    server.build_engine = build
    return server, instrument, parts, nf4


def test_capture_hook_precedes_build_and_restores_on_failure():
    server, instrument, _, _ = fake_build(fail=True)
    original = server._apply_fusions
    with pytest.raises(RuntimeError, match="capture failed"):
        serving.build_instrumented(server, instrument, None)
    assert server._apply_fusions is original


def test_real_decode_wrapper_preserves_helper_samples_and_capture_scope(tmp_path):
    import hashlib
    import json

    server, instrument, parts, nf4 = fake_build()
    cfg = types.SimpleNamespace(model="synthetic", revision="synthetic", graphs=True, buckets=(1, 16), token="synthetic-secret")
    cfg.fusion_modes = parts.info["fusion_modes"]
    server.PagedServeConfig = types.SimpleNamespace(from_env=lambda: cfg)
    rows = [[i] * 512 for i in range(16)]
    digest = lambda x: hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest()  # noqa: E731
    prompt = tmp_path / "prompts.json"
    prompt.write_text(json.dumps({"rows": rows, "prompts_sha256": digest(rows)}))
    helper = types.SimpleNamespace(WORKLOADS={"W16": 16, "W1": 1}, digest=digest)
    observed = []

    def run(parts, torch, rows, n):
        observed.append((len(rows), n))
        parts.runner.graph_stats[len(rows)]["replays"] += 1
        return n / 7, 1, [[row[0]] * n for row in rows]

    helper.run_pass = run
    helper.slope = lambda *args: {"diagnostic": "synthetic"}
    rec = serving.decode({"prompts": str(prompt), "short": 8, "long": 24, "reps": 1}, helper, instrument, server, None, nf4)
    assert observed == [(16, 8), (16, 8), (16, 24), (16, 24), (1, 8), (1, 8), (1, 24), (1, 24)]
    assert rec["workloads"]["W1"]["walls"]["8"] == [8 / 7]  # No display rounding enters raw samples.
    assert rec["dispatch_build"] == rec["dispatch_total"] == {"scalar": 1}
    assert rec["graph_stats"]["1"]["replays"] == 4
    assert rec["features"]["decode_gemv"]["calls_scope"] == "build-capture+warm+timed"
    assert "token" not in rec["config"]


@pytest.mark.parametrize("group", [1, 12])
def test_quality_wrapper_records_per_pass_dispatch_and_restores_hooks(tmp_path, group):
    import json

    server, instrument, parts, nf4 = fake_build()
    cfg = types.SimpleNamespace(model="synthetic", revision="synthetic", graphs=False, device="cuda")
    cfg.fusion_modes = parts.info["fusion_modes"]
    server.PagedServeConfig = types.SimpleNamespace(from_env=lambda: cfg)
    instrument.p108_box = types.SimpleNamespace(_score=lambda lp, tokens: (lp, tokens))

    def paged_pass(model, ws, prompt, cont, chunk, device, **kwargs):
        nf4.state["scalar"] += int(len(ws) == 1)
        return [[2.] * cont for _ in ws], {"kernels": {"rmsnorm_rows": 0, "rmsnorm_resid_rows": 0,
                                                     "rope_norm_heads": 0, "rope_heads": 0, "router_epilogue": 0},
                                          "qkv_calls": 0}

    instrument.p110_box = types.SimpleNamespace(paged_pass=paged_pass)

    def measure(model, windows, *, group, cont, **kwargs):
        rows, passes = [], []
        for offset in range(0, 12, group):
            ws = windows["wikitext"][offset:offset + group]
            lps, e = instrument.p110_box.paged_pass(model, ws, 512, cont, 512, "cuda")
            passes.append(e)
            for i, (lp, w) in enumerate(zip(lps, ws)):
                nll, am = instrument.p108_box._score(lp, w[512:])
                rows.append({"window": offset + i, "nll": sum(nll) / len(nll), "argmax_agree": 1.})
        return {"per_window": {"wikitext": {"R": rows}}, "engagement": {"wikitext": {"R": passes}}}

    instrument.measure_phase = measure
    window = tmp_path / "windows.json"
    window.write_text(json.dumps({"wikitext": [[i] * 544 for i in range(12)]}))
    original_score = instrument.p108_box._score
    result = serving.quality({"windows": str(window), "group": group, "cont": 32, "ref_dir": str(tmp_path / "ref")},
                              instrument, server, None, nf4)
    passes = result["engagement"]["wikitext"]["R"]
    assert len(passes) == 12 // group
    assert all(e["gemv_dispatch"]["scalar"] == int(group == 1) for e in passes)
    assert all(e["features"]["decode_gemv"]["mode"] == ("on" if group == 1 else "inapplicable") for e in passes)
    assert result["per_window"]["wikitext"]["R"][11]["argmax_ids"] == [11] * 32
    assert instrument.p110_box.paged_pass is paged_pass and instrument.p108_box._score is original_score


def test_unmeasured_fallback_is_explicitly_unknown():
    f = serving.feature("auto", 1, 2, scope="synthetic")
    assert f["fallback_calls"] is None and f["fallback_coverage"] == "UNVERIFIED"
    assert serving.feature("auto", 1, 2, scope="synthetic", fallback_calls=1)["fallback_calls"] == 1
    with pytest.raises(ValueError, match="fallback"):
        serving.feature("auto", 1, 2, scope="synthetic", fallback_calls=True)


@pytest.mark.parametrize("fallback_calls", [None, 1])
def test_registered_reducer_refuses_unknown_or_observed_fallback(fallback_calls):
    reducer = load("ra_reduce")
    rec = {"features": {"decode_gemv": serving.feature("auto", 1, 2, scope="synthetic",
                                                        fallback_calls=fallback_calls)}}
    with pytest.raises(reducer.Invalid, match="fallback"):
        reducer.features(rec)
