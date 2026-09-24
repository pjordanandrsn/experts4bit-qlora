"""Lane P68's reducer applies the registration literally (bench/p68/p68_reduce.py; bench/p68/P68-PREREG.md): every
gate and verdict branch fires on synthetic receipts built to trigger it, and the prediction tables name only arms and
modes the probe registers."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "bench" / "p68" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("p68_reduce")


def _mode(fds, kl=1e-3, flips=0):
    """A mode record in p63_compare.summarize_mode's shape (slimmed): per position, its first difference."""
    return {"positions": len(fds), "exact_positions": sum(1 for f in fds if f is None), "argmax_flips": flips,
            "kl_mean": kl, "per_position": [{"first_diff": (None if f is None else list(f))} for f in fds]}


def _counts(**split):
    c = {k: {"t1_calls": 10, "split_calls": 0, "split_rows": 0, "t1_mask_present": 0, "wrapped_modules": 1}
         for k in ("proj", "core", "router", "lm_head", "norms")}
    for k, v in split.items():
        c[k]["split_calls"] = v
    return c


def _ablate(name, force, modes, counts=None):
    return {"arm": {"name": name, "kind": "ablate", "attn": name.split(".")[0], "force": force},
            "modes": modes, "counts": counts or _counts(**{k: 3 for k in force})}


def _size(name, rows):
    return {"arm": {"name": name, "kind": "size", "attn": name.split(".")[1], "force": []}, "rows": rows}


def _size_rows(n_rows=4, windows=15, width=17, kl=5e-3, flip_every=40):
    rows = []
    k = 0
    for r in range(n_rows):
        m = {"kl": [], "flip": [], "window": []}
        for w in range(windows):
            for i in range(width):
                m["kl"].append(kl)
                m["flip"].append(int(k % flip_every == 0))
                m["window"].append(64 + 32 * w)
                k += 1
        rows.append({"row": r, "modes": {f"verify{width}": m}})
    return rows


def _write(tmp, stack, arm):
    d = tmp / stack
    d.mkdir(parents=True, exist_ok=True)
    (d / "p68_arm.json").write_text(json.dumps(arm))


A = ("attn_core",)


def test_the_prediction_tables_name_only_registered_arms_and_modes():
    src = (REPO / "bench" / "p68" / "p68_probe.py").read_text()
    names = {a for a in ("hf.base", "hf.proj", "hf.core", "hf.proj_core", "hf.all", "hf.fp32red", "paged.base",
                         "paged.proj") if f'"name": "{a}"' in src}
    assert len(names) == 8
    for stack, table in R.PRED.items():
        assert stack in R.STACKS
        for (arm, mode) in table:
            assert arm in names and mode in ("verify17", "verify16", "prefill"), (stack, arm, mode)


def test_judge_exact_sites_and_not_branches():
    ex = _mode([None] * 5)
    assert R.judge("exact", ex)["verdict"] == "HELD"
    assert R.judge("exact", _mode([None, (0, "mlp_out")]))["verdict"] == "REFUTED"
    core = _mode([(0, "attn_core")] * 4)
    assert R.judge({"sites": {"attn_core"}}, core)["verdict"] == "HELD"
    assert R.judge({"sites": {"attn_core"}}, _mode([(0, "attn_core"), (0, "attn_in")]))["verdict"] == "REFUTED"
    assert R.judge({"sites": {"attn_core"}}, _mode([(1, "attn_core")]))["verdict"] == "REFUTED"   # layer 1 is not layer 0
    assert R.judge({"sites": {"attn_core"}}, _mode([None, (0, "attn_core")]))["verdict"] == "REFUTED"  # one was exact
    moved = _mode([(0, "mlp_out"), (3, "attn_core"), None, (None, "logits")])
    assert R.judge({"not": R.ATTN_SITES_L0}, moved)["verdict"] == "HELD"
    assert R.judge({"not": R.ATTN_SITES_L0}, _mode([(0, "mlp_out"), (0, "attn_out")]))["verdict"] == "REFUTED"
    assert R.judge(None, core)["verdict"] == "INFO"
    maj = {"majority": {"attn_core", "attn_out"}}
    assert R.judge(maj, _mode([(0, "attn_core")] * 3 + [(0, "mlp_out")]))["verdict"] == "HELD"
    assert R.judge(maj, _mode([(0, "attn_core")] * 2 + [(0, "mlp_out")] * 2))["verdict"] == "REFUTED"   # half is not more
    assert R.judge(maj, _mode([(0, "attn_core")] * 3 + [None]))["verdict"] == "REFUTED"                # one was exact


def _good_arm(**over):
    arms = {
        "hf.base": _ablate("hf.base", [], {"verify17": _mode([(0, "attn_core")] * 4)}),
        "hf.all": _ablate("hf.all", ["proj", "core", "router", "lm_head", "norms"],
                          {"verify17": _mode([None] * 4), "verify16": _mode([None] * 4)}),
        "size.hf": _size("size.hf", _size_rows()),
    }
    arms.update(over)
    return {"g0": {"hf.base control repeat bit-identical": True, "hf.all control == hf.base control": True},
            "arms": arms}


def test_a_clean_stack_is_read(tmp_path):
    _write(tmp_path, "nf4", _good_arm())
    rep = R.reduce(str(tmp_path))
    assert rep["gates"]["nf4"]["ok"]
    assert rep["verdicts"]["exact_verify_held"] == {"nf4": True}
    s = rep["size"][0]
    assert s["positions"] == 4 * 15 * 17 and s["clusters"] == 60
    assert s["class"] == "WITHIN-BAR" and s["verdict"] == "HELD"


@pytest.mark.parametrize("breaker,needle", [
    (lambda a: a["g0"].update({"hf.base control repeat bit-identical": False}), "G0 failed"),
    (lambda a: a["arms"]["hf.all"]["counts"]["core"].update({"split_calls": 0}), "never engaged"),
    (lambda a: a["arms"]["hf.all"]["counts"]["core"].update({"t1_mask_present": 3}), "carried a mask"),
    (lambda a: a["arms"].update({"hf.proj": {"error": "RuntimeError('boom')"}}), "errored"),
    (lambda a: a.update({"g0": {}}), "G0 not recorded"),
])
def test_a_failed_gate_voids_the_stack(tmp_path, breaker, needle):
    arm = _good_arm()
    breaker(arm)
    _write(tmp_path, "int4", arm)
    rep = R.reduce(str(tmp_path))
    assert not rep["gates"]["int4"]["ok"]
    assert any(needle in w for w in rep["gates"]["int4"]["why"]), rep["gates"]["int4"]["why"]
    assert rep["ablation"] == [] and rep["size"] == []
    assert R.main([str(tmp_path)]) == 14


def test_size_classes_and_the_prediction():
    within = R._clusters(_size_rows(flip_every=50), "verify17")
    assert R.size_cell(within)["class"] == "WITHIN-BAR"
    over = R._clusters(_size_rows(flip_every=5), "verify17")                    # top-1 0.80
    c = R.size_cell(over)
    assert c["class"] == "OVER-BAR" and c["verdict"] == "REFUTED"
    kl_over = R._clusters(_size_rows(kl=0.2, flip_every=50), "verify17")
    assert R.size_cell(kl_over)["class"] == "OVER-BAR"
    # near the line with few clusters: the interval straddles 0.93
    edge = R._clusters(_size_rows(n_rows=1, windows=4, flip_every=14), "verify17")
    assert R.size_cell(edge)["class"] == "UNRESOLVED"


def test_the_bootstrap_is_deterministic():
    cs = R._clusters(_size_rows(flip_every=13), "verify17")
    assert R.boot_ci(cs, R._top1) == R.boot_ci(cs, R._top1)


def test_no_receipt_is_rc_43(tmp_path):
    assert R.main([str(tmp_path)]) == 43
