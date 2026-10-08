"""CPU checks of retained native binding; synthetic data never counts as GPU proof."""
import copy
import importlib.util
import json
import shutil
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


reducer, stage, quality = load("ra_reduce"), load("ra_stage"), load("ra_quality")
normalizer = load("ra_normalize")
spec = importlib.util.spec_from_file_location("ra_test_driver", ROOT / "bench/sc2/sc2_driver.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.fixture(scope="session")
def staged(tmp_path_factory):
    path = tmp_path_factory.mktemp("ra") / "instruments"
    stage.stage(ROOT, ROOT / "bench/ra/source-pins.json", path)
    return path


def native_sc2(mode, rate, seed, n):
    plan = [list(row) for row in driver.plan(mode, rate, n, seed, 16)]
    rows = [{"i": i, "prompt_index": p, "planned_s": off, "max_tokens": m, "status": 200,
             "valid": True, "prompt_tokens": 512, "prompt_len": 512, "completion_tokens": m,
             "finish_reason": "length", "ttft_s": .5, "tpot_s": .05} for i, (off, p, m) in enumerate(plan)]
    return {"mode": mode, "rate": rate, "seed": seed, "n": n, "profile": "e4b", "max_tokens_range": [64, 256],
            "prompts_sha256": "synthetic", "plan": plan, "summary": {"valid": n, "invalid": 0}, "requests": rows}


def raw_fixture(tag):
    run, arms = reducer.fixture()
    arm = arms[tag]
    raw = {"metadata": {k: copy.deepcopy(arm[k]) for k in normalizer.META},
           **{k: copy.deepcopy(arm[k]) for k in ("training", "training_profile", "decode", "quality_1", "quality_12")},
           "capacity": native_sc2("poisson", 12, 112, 120), "capacity_warm": native_sc2("serial", 0, 999, 4),
           "capacity_burst": native_sc2("poisson", 1000, 998, 64)}
    cap = arm["capacity"]
    raw["capacity_extra"] = {"slo": cap["slo"], "features": cap["features"]}
    for label, n in (("warm", 4), ("burst", 68), ("end", 188)):
        raw[f"capacity_health_{label}"] = {
            "status": "ready", "error": None, "queue_depth": 0,
            "engine": {"buckets": [1, 2, 4, 8, 16], "graphs": True, "chunk_tokens": 512,
                       "max_tokens_per_seq": 2048, "max_tokens_limit": 2047,
                       "graph_status": {str(b): "graph" for b in (1, 2, 4, 8, 16)},
                       "graph_stats": {"16": {"replays": n, "eager_steps": 0}}},
            "prefill_graph": {"status": "on", "T": 512, "replays": n, "eager_chunks": 0},
            "kv_bookkeeping": {"requested": True, "bulk": True, "flush_layers": 0, "ready_layers": 0,
                               "ready_bulk": 0, "flush_bulk": n, "ready_at_flush": n}}
    run["identity"]["capacity_plan_sha256"] = reducer.digest(raw["capacity"]["plan"])
    raw["metadata"]["identity"] = copy.deepcopy(run["identity"])
    for group in (1, 12):
        raw[f"quality_{group}"].update(phase="off", arms=["R"], windows={"wikitext": 12},
                                        windows_sha256={"wikitext": "synthetic"})
    return run, raw


def test_native_projection_preserves_scores_and_samples():
    _, raw = raw_fixture("old_a")
    arm = normalizer.normalize(raw, driver)
    assert arm["training"] == raw["training"]
    assert arm["decode"] == raw["decode"]
    assert arm["quality_1"]["per_window"] == raw["quality_1"]["per_window"]
    assert arm["capacity"]["requests"][0]["completion_tokens"] == raw["capacity"]["plan"][0][2]


@pytest.mark.parametrize("component,key,value,match", [
    ("capacity", "status", 503, "HTTP"), ("capacity", "error", "HTTP failure", "HTTP"),
    ("capacity", "valid", False, "validity"), ("capacity", "completion_tokens", 1, "usage"),
    ("capacity", "prompt_tokens", 1, "usage"), ("capacity", "planned_s", 999, "plan/order"),
    ("capacity", "prompt_index", 999, "plan/order"), ("capacity", "i", 999, "plan/order"),
    ("capacity", "request_id_conflict", True, "identity"),
    ("capacity_burst", "status", 503, "HTTP"), ("capacity_warm", "completion_tokens", 1, "usage"),
])
def test_native_request_mutations(component, key, value, match):
    _, raw = raw_fixture("old_a")
    raw[component]["requests"][0][key] = value
    with pytest.raises(reducer.Invalid, match=match):
        normalizer.normalize(raw, driver)


def test_coherently_changed_plan_cannot_pass_seed_check():
    _, raw = raw_fixture("old_a")
    raw["capacity"]["plan"][0][0] += .1
    raw["capacity"]["requests"][0]["planned_s"] = raw["capacity"]["plan"][0][0]
    with pytest.raises(reducer.Invalid, match="seeded plan"):
        normalizer.normalize(raw, driver)


@pytest.mark.parametrize("block,key,value", [
    ("prefill_graph", "replays", 187), ("prefill_graph", "eager_chunks", 1),
    ("kv_bookkeeping", "flush_bulk", 187), ("kv_bookkeeping", "flush_layers", 1),
    ("kv_bookkeeping", "ready_at_flush", 187), ("engine", "graphs", False),
])
def test_native_health_mutations(block, key, value):
    _, raw = raw_fixture("old_a")
    raw["capacity_health_end"][block][key] = value
    with pytest.raises(reducer.Invalid):
        normalizer.normalize(raw, driver)


def test_burst_cannot_reuse_warm_replays():
    _, raw = raw_fixture("old_a")
    raw["capacity_health_burst"]["engine"]["graph_stats"]["16"]["replays"] = 4
    with pytest.raises(reducer.Invalid, match="widest bucket"):
        normalizer.normalize(raw, driver)


@pytest.mark.parametrize("path,value,want", [
    (("training", "kernel_calls_per_step"), [0] * 20, "VOID"),
    (("decode", "graph_status", "16"), "eager", "VOID"),
    (("quality_1", "rehearsal", "stand_in_attention"), True, "VOID"),
    (("metadata", "feature_env", "E4B_PAGED_GRAPHS"), "1", "VOID"),
    (("metadata", "identity", "checkpoint_sha256"), "changed", "VOID"),
    (("metadata", "identity", "wheel_lock_sha256"), "changed", "VOID"),
    (("metadata", "release", "e4b", "version"), "changed", "VOID"),
    (("quality_1", "per_window", "wikitext", "R", 0, "nll"), 3.2, "REGRESSION"),
])
def test_native_mutants_reach_frozen_reducer(path, value, want):
    arms = {}
    for tag in reducer.TAGS:
        run, raw = raw_fixture(tag)
        if tag.startswith("new"):
            dest = raw
            for part in path[:-1]:
                dest = dest[part]
            dest[path[-1]] = value
        arms[tag] = normalizer.normalize(raw, driver)
    assert reducer.reduce(run, arms)["verdict"] == want


def test_raw_to_envelope_and_checksum_binding(tmp_path, staged):
    shutil.copytree(staged, tmp_path / "instruments")
    for tag in reducer.TAGS:
        run, raw = raw_fixture(tag)
        run["identity"]["source_pins_sha256"] = stage.sha((staged / "source-pins.json").read_bytes())
        raw["metadata"]["identity"] = copy.deepcopy(run["identity"])
        dest = tmp_path / "raw" / tag
        dest.mkdir(parents=True)
        for name, obj in raw.items():
            (dest / f"{name}.json").write_text(json.dumps(obj))
    (tmp_path / "run.json").write_text(json.dumps(run))
    assert normalizer.bind(tmp_path)["verdict"] == "CLEAR"
    assert normalizer.bind(tmp_path, check=True)["verdict"] == "CLEAR"
    envelope = tmp_path / "arm_new_a.json"
    obj = json.loads(envelope.read_bytes())
    obj["training"]["step_ms"][0] *= 2
    envelope.write_text(json.dumps(obj))
    with pytest.raises(reducer.Invalid, match="differs from raw"):
        normalizer.bind(tmp_path, check=True)


def test_staged_mutation_even_after_resealing(tmp_path, staged):
    shutil.copytree(staged, tmp_path / "instruments")
    root = tmp_path / "instruments"
    path = root / "p109_box.py"
    path.write_bytes(path.read_bytes() + b"\n# mutant\n")
    with pytest.raises(stage.Refused, match="staged bytes changed"):
        stage.verify(root)
    sums = root / "SHA256SUMS"
    sums.write_text("\n".join(f"{stage.sha(path.read_bytes())}  p109_box.py" if line.endswith("  p109_box.py")
                              else line for line in sums.read_text().splitlines()) + "\n")
    with pytest.raises(stage.Refused, match="faithfully staged"):
        stage.verify(root)


def fake_quality(*, fail=False, altered=False):
    module = types.SimpleNamespace(p108_box=types.SimpleNamespace(_score=lambda lp, tokens: (lp, tokens)))

    def measure(model, windows, **kwargs):
        if fail:
            raise RuntimeError("measurement failed")
        rows = []
        for i, w in enumerate(windows["wikitext"]):
            nll, am = module.p108_box._score([2.] * kwargs["cont"], w)
            rows.append({"window": i, "nll": sum(nll) / len(nll) + int(altered), "argmax_agree": 1.})
        return {"per_window": {"wikitext": {"R": rows}}}

    module.measure_phase = measure
    return module


def test_score_capture_keeps_original_scores_and_restores_hook():
    module = fake_quality()
    original = module.p108_box._score
    result = quality.measure_r(module, None, {"wikitext": [[1, 2], [3, 4]]}, phase="off", cont=2)
    rows = result["per_window"]["wikitext"]["R"]
    assert [r["nll"] for r in rows] == [2., 2.]
    assert [r["argmax_ids"] for r in rows] == [[1, 2], [3, 4]]
    assert module.p108_box._score is original


@pytest.mark.parametrize("fail,altered,error", [(True, False, RuntimeError), (False, True, ValueError)])
def test_score_capture_refuses_changes_and_restores_after_failure(fail, altered, error):
    module = fake_quality(fail=fail, altered=altered)
    original = module.p108_box._score
    with pytest.raises(error):
        quality.measure_r(module, None, {"wikitext": [[1, 2]]}, phase="off", cont=2)
    assert module.p108_box._score is original
