"""P64's reducer (bench/p64/p64_reduce.py) fires every registered branch on synthetic runs built to trigger it.

The run directory is written in exactly the layout kl_a16.py and p64_run.sh write (`out/<pass>.<text>.pt` +
`.census.json`, `out/build.json`, `prompts_<text>.json`, `pack.json`), with logits constructed so the answer is
known: an int8 step far below, or far above, the arithmetic-order floor; a determinism control that is not exactly
zero; a pass whose decode counts are not the registered ones; a floor that reads exactly zero.
"""
import hashlib
import importlib.util
import json
import pathlib

import pytest
import torch

LANE = pathlib.Path(__file__).resolve().parents[1] / "bench" / "p64"


def _reducer():
    spec = importlib.util.spec_from_file_location("p64_reduce", LANE / "p64_reduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R, T, PREFIX, V = 4, 16, 8, 40
BUILD = {"moe_layers": 2, "top_k": 3, "int4_attn_projections": 5, "int4_expert_layers": 2}


def _write_run(d: pathlib.Path, *, gap: float, floor: float, rep_noise: float = 0.0, a16_effect: str = "noise",
               bad_counts: str | None = None, seed: int = 0):
    red = _reducer()
    out = d / "out"
    out.mkdir(parents=True)
    (d / "out_nf4").mkdir()
    json.dump(BUILD, open(out / "build.json", "w"))
    json.dump({"fingerprint": "sha256:" + "0" * 64, "licensed": True}, open(d / "pack.json", "w"))
    g = torch.Generator().manual_seed(seed)
    for text in ("wikitext", "c4val1"):
        ids = torch.randint(0, V, (R, T), generator=g)
        rows = ids.tolist()
        sha = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
        json.dump({"prompts": rows, "prompts_sha256": sha}, open(d / f"prompts_{text}.json", "w"))
        base = torch.randn(R, T - PREFIX, V, generator=g) * 2.0
        pre = torch.randn(R, V, generator=g)
        onehot = torch.zeros(R, T - PREFIX, V)
        tgt = torch.cat([ids[:, PREFIX + 1:], ids[:, -1:]], dim=1)          # last position has no target; any value
        onehot.scatter_(-1, tgt.unsqueeze(-1), 1.0)
        a16_delta = {"helps": onehot * gap, "hurts": -onehot * gap,
                     "noise": torch.randn(R, T - PREFIX, V, generator=g) * gap}[a16_effect]   # helps: a8 is the worse arm
        logits = {"a8": base, "a8_rep": base + rep_noise * torch.randn(R, T - PREFIX, V, generator=g),
                  "a16": base + a16_delta, "a16_all": base + a16_delta,
                  "a8_pc64": base + floor * torch.randn(R, T - PREFIX, V, generator=g),
                  "a8_pc384": base + floor * torch.randn(R, T - PREFIX, V, generator=g),
                  "nf4": base + 0.5 * torch.randn(R, T - PREFIX, V, generator=g)}
        steps = R * (T - PREFIX)
        for p, lg in logits.items():
            dest = d / "out_nf4" if p == "nf4" else out
            torch.save({"prefill_last": pre.clone(), "decode": lg, "prefix": PREFIX, "prompts_sha256": sha}, dest / f"{p}.{text}.pt")
            counts = red.expected_counts(p, BUILD, steps)
            if bad_counts == p:
                counts = dict(counts, expert_quant_decode=counts["expert_quant_decode"] + 1)
            json.dump({"pass": p, "text": text, "decode_steps": steps, "counts": counts, "prompts_sha256": sha,
                       "rows": R, "prefill_chunk": 128}, open(dest / f"{p}.{text}.census.json", "w"))


def test_rule_primitives():
    red = _reducer()
    assert red.classify(0.001, 0.002) == "BELOW FLOOR"
    assert red.classify(0.003, 0.002) == "NOT DISTINGUISHABLE"
    assert red.classify(0.0041, 0.002) == "DISTINGUISHABLE"
    assert red.is_material("DISTINGUISHABLE", 0.01, 0.001, 0.005)
    assert not red.is_material("DISTINGUISHABLE", 0.01, -0.001, 0.005)      # interval includes 0
    assert not red.is_material("DISTINGUISHABLE", 0.004, 0.001, 0.005)      # below the NLL floor
    assert not red.is_material("NOT DISTINGUISHABLE", 1.0, 0.5, 0.005)
    assert red.lane_verdict({"a": {"class": "DISTINGUISHABLE", "material": True}, "b": {"class": "BELOW FLOOR", "material": False}}) == "MATERIAL"
    assert red.lane_verdict({"a": {"class": "DISTINGUISHABLE", "material": False}, "b": {"class": "BELOW FLOOR", "material": False}}) == "DISTINGUISHABLE, NOT MATERIAL"
    assert red.lane_verdict({"a": {"class": "NOT DISTINGUISHABLE", "material": False}, "b": {"class": "BELOW FLOOR", "material": False}}) == "INDISTINGUISHABLE"
    lo, hi = red.paired_bootstrap(torch.full((16,), 0.25))
    assert lo == pytest.approx(0.25) and hi == pytest.approx(0.25)


def test_row_nll_reads_the_next_token():
    red = _reducer()
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    dec = torch.full((1, 3, 6), -30.0)            # prefix 3: logit j predicts ids[3 + j + 1]
    dec[0, 0, 4] = 30.0
    dec[0, 1, 5] = 30.0
    nll = red.row_nll(dec, ids, 3)
    assert nll.shape == (1,) and float(nll[0]) < 1e-6          # both scored targets certain; the last position unscored


def test_indistinguishable_when_the_gap_is_inside_the_floor(tmp_path):
    _write_run(tmp_path, gap=0.01, floor=0.05)
    rep, lines = _reducer().reduce(str(tmp_path))
    assert rep["validity"]["status"] == "VALID", rep["validity"]
    assert rep["verdicts"]["G_exp"] == "INDISTINGUISHABLE"
    assert any("VERDICT (expert int8 step, G_exp): INDISTINGUISHABLE" in ln for ln in lines)


def test_material_when_far_above_the_floor_and_a8_is_worse(tmp_path):
    _write_run(tmp_path, gap=4.0, floor=0.01, a16_effect="helps")
    rep, _ = _reducer().reduce(str(tmp_path))
    assert rep["validity"]["status"] == "VALID"
    s = rep["texts"]["wikitext"]["stats"]["G_exp"]
    assert s["class"] == "DISTINGUISHABLE" and s["dNLL"] > 0 and s["dNLL_ci95"][0] > 0
    assert rep["verdicts"]["G_exp"] == "MATERIAL"


def test_distinguishable_not_material_when_the_int8_step_moves_but_does_not_hurt(tmp_path):
    _write_run(tmp_path, gap=1.0, floor=0.01, a16_effect="hurts")
    rep, _ = _reducer().reduce(str(tmp_path))
    assert rep["validity"]["status"] == "VALID"
    assert rep["verdicts"]["G_exp"] == "DISTINGUISHABLE, NOT MATERIAL"
    for t in ("wikitext", "c4val1"):
        assert rep["texts"][t]["stats"]["G_exp"]["class"] == "DISTINGUISHABLE"


def test_a_nonzero_determinism_control_reads_nothing(tmp_path):
    _write_run(tmp_path, gap=4.0, floor=0.01, a16_effect="helps", rep_noise=1e-3)
    rep, lines = _reducer().reduce(str(tmp_path))
    assert rep["validity"]["status"] == "NOTHING READ" and not rep["validity"]["determinism"]
    assert rep["verdicts"]["G_exp"].startswith("NOTHING READ")


def test_wrong_decode_counts_read_nothing(tmp_path):
    _write_run(tmp_path, gap=0.01, floor=0.05, bad_counts="a16")
    rep, _ = _reducer().reduce(str(tmp_path))
    assert rep["validity"]["status"] == "NOTHING READ" and not rep["validity"]["engagement"]


def test_a_zero_floor_falls_back_to_the_record_and_says_so(tmp_path):
    _write_run(tmp_path, gap=0.001, floor=0.0)
    red = _reducer()
    rep, _ = red.reduce(str(tmp_path))
    f = rep["texts"]["wikitext"]["floor"]
    assert f["F"] == red.RECORD_FLOOR and f["source"].startswith("ON RECORD")
