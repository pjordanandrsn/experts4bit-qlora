"""The P67 reducer's rule and refusals, driven on synthetic receipts (bench/p67/P67-PREREG.md).

The lane turns on three things that each fail silently if they are wrong: which loss is read (TRAIN, never
held-out), which floor draws are admitted (same session, proven perturbation, not bit-identical), and the band
(max(TOL, K * F_hi), with no floor meaning "undecidable" rather than FAIL). Every branch is exercised here,
including the ones that REFUSE, because a reducer that has never rejected anything is not evidence.
"""

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("p67_reduce", REPO / "bench" / "p67" / "p67_reduce.py")
p67 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(p67)

N = 20
FAM = "gemma4"


def _curve(delta, wiggle=0.0, n=N):
    """A falling train-loss curve shifted by `delta` at every step (so D_med == |delta|, D_final == |delta|)."""
    return [round(2.0 - 0.05 * i + delta + (wiggle if i % 2 else 0.0), 6) for i in range(n)]


def _arm(tag, losses, *, eval_final=1.0, order=None, **kw):
    r = {"framework": "e4b", "fam": FAM, "tag": tag, "arm": "reference" if tag.startswith("reference") else tag.split("_")[0],
         "status": "ok", "C1_bit_exact": True, "init_sha": "aa" * 32, "trainable_params": 487280640,
         "tokens": {"sha256": "bb" * 32}, "steps": len(losses), "seq": 2048, "micro_batch": 2, "accum": 4, "r": 16,
         "alpha": 16, "lr": 0.0002, "seed": 3407, "optimizer": "adamw_8bit", "attn_4bit": True,
         "losses": losses, "loss_last": losses[-1], "eval_loss_final": eval_final, "n_patched": 0,
         "kernel_calls_per_step_min": 0}
    if tag.startswith("reference"):
        m = re.match(r"^reference_attn4_perm(\d+)$", tag)
        want = f"perm:{m.group(1)}" if m else None
        o = order if order is not None else want
        r["reference_order"] = {"requested": o, "order": o, "calls": (600 if o else 0)}
    else:
        r.update(n_patched=30, kernel_calls_per_step_min=480)
    if tag == "batched_attn4":
        r["batched_stats"] = {"modules": 30, "calls": 7680, "batched": 7680, "fallback_calls": 0, "by_reason": {}}
    r.update(kw)
    return r


def _session(tmp_path, arms, name="s1"):
    d = tmp_path / name
    d.mkdir()
    for a in arms:
        (d / f"{a['fam']}_e4b_{a['tag']}.json").write_text(json.dumps(a))
    return str(d)


def _floors(deltas):
    """A reference, a repeat and one perm draw per delta (the first delta goes to the repeat)."""
    out = [_arm("reference_attn4", _curve(0.0)), _arm("reference_attn4_repeat", _curve(deltas[0]))]
    out += [_arm(f"reference_attn4_perm{i}", _curve(d)) for i, d in enumerate(deltas[1:], start=1)]
    return out


def _judged(res, tag):
    return next(j for j in res[FAM]["judged"] if j["tag"] == tag)


# ----------------------------------------------------------------------------- the registered constants
def test_registered_constants_match_the_prereg_text():
    """The rule's three numbers live in two places; they must be the SAME numbers, or the reducer is not the
    rule that was registered. Read the prereg's own registration lines, not a paraphrase."""
    txt = (REPO / "bench" / "p67" / "P67-PREREG.md").read_text()
    assert re.search(r"\bTOL\s*=\s*0\.05\b", txt) and p67.TOL == 0.05
    assert re.search(r"\bK\s*=\s*3\b", txt) and p67.K == 3.0
    assert re.search(r"\bN_MIN\s*=\s*3\b", txt) and p67.N_MIN == 3


# ----------------------------------------------------------------------------- the band
def test_floor_carries_a_pass_the_constant_band_fails(tmp_path):
    """D above 0.05 but within K * F_hi on both quantities: PASS, carried by the floor."""
    d = _session(tmp_path, _floors([0.030, 0.020, 0.025, 0.028]) + [_arm("fused_attn4", _curve(-0.08))])
    res = p67.reduce_session(d)
    j = _judged(res, "fused_attn4")
    assert res[FAM]["floor"]["exists"] and res[FAM]["floor"]["n_admissible"] == 4
    assert j["constant"] == "FAIL" and j["verdict"] == "PASS" and j["carried_by"] == "floor"
    assert j["band_final"] == pytest.approx(3 * 0.030)


def test_above_the_floor_band_and_the_tolerance_is_a_fail(tmp_path):
    d = _session(tmp_path, _floors([0.010, 0.012, 0.008]) + [_arm("fused_attn4", _curve(-0.10))])
    j = _judged(p67.reduce_session(d), "fused_attn4")
    assert j["verdict"] == "FAIL" and j["carried_by"] is None and j["detectable"] is True


def test_either_quantity_alone_fails_the_arm(tmp_path):
    """D_final inside, D_med outside: FAIL. The two quantities are ANDed, as tp1 registered them."""
    ref = _curve(0.0)
    arm = [x - 0.2 for x in ref[:-1]] + [ref[-1] - 0.01]     # final close, every other step far
    d = _session(tmp_path, _floors([0.010, 0.012, 0.008]) + [_arm("fused_attn4", arm)])
    j = _judged(p67.reduce_session(d), "fused_attn4")
    assert j["d_final_train"] < 0.05 < j["d_med_train"] and j["verdict"] == "FAIL"


def test_a_floor_never_narrows_the_tolerance(tmp_path):
    """A tiny floor cannot FAIL an arm inside 0.05: the verdict is PASS (tolerance), marked DETECTABLE."""
    d = _session(tmp_path, _floors([0.0002, 0.0003, 0.0001]) + [_arm("fused_attn4", _curve(0.004))])
    j = _judged(p67.reduce_session(d), "fused_attn4")
    assert j["verdict"] == "PASS" and j["carried_by"] == "tolerance" and j["detectable"] is True


def test_no_floor_leaves_a_constant_pass_standing_and_a_constant_fail_undecided(tmp_path):
    arms = [_arm("reference_attn4", _curve(0.0)), _arm("fused_attn4", _curve(0.002)), _arm("batched_attn4", _curve(0.07))]
    res = p67.reduce_session(_session(tmp_path, arms))
    assert not res[FAM]["floor"]["exists"]
    assert _judged(res, "fused_attn4")["verdict"] == "PASS"
    assert _judged(res, "batched_attn4")["verdict"] == "NO-FLOOR"     # undecidable -- never FAIL
    assert res[FAM]["reading"] == "NO-FLOOR"


def test_two_floor_draws_are_not_a_floor(tmp_path):
    arms = _floors([0.03, 0.03]) + [_arm("fused_attn4", _curve(0.07))]
    res = p67.reduce_session(_session(tmp_path, arms))
    assert res[FAM]["floor"]["n_admissible"] == 2 and not res[FAM]["floor"]["exists"]
    assert _judged(res, "fused_attn4")["verdict"] == "NO-FLOOR"


# ----------------------------------------------------------------------------- which loss
def test_the_band_reads_the_train_loss_never_the_heldout(tmp_path):
    """A held-out delta of 0.5 nats changes nothing; it is reported beside the verdict, under its own name."""
    arms = _floors([0.010, 0.012, 0.008]) + [_arm("fused_attn4", _curve(0.004), eval_final=1.5)]
    j = _judged(p67.reduce_session(_session(tmp_path, arms)), "fused_attn4")
    assert j["d_heldout"] == pytest.approx(0.5) and j["verdict"] == "PASS"
    assert j["d_final_train"] == pytest.approx(0.004)


# ----------------------------------------------------------------------------- floor-draw admission
def test_a_bit_identical_floor_draw_is_refused_and_the_repeat_answers_determinism(tmp_path):
    arms = [_arm("reference_attn4", _curve(0.0)), _arm("reference_attn4_repeat", _curve(0.0))] + \
           [_arm(f"reference_attn4_perm{i}", _curve(d)) for i, d in enumerate([0.01, 0.02, 0.015], start=1)]
    res = p67.reduce_session(_session(tmp_path, arms))
    rep = next(d for d in res[FAM]["floor_draws"] if d["kind"] == "repeat")
    assert rep["identical"] and not rep["admissible"]
    assert res[FAM]["reference_bit_identical_on_repeat"] is True
    assert res[FAM]["floor"]["n_admissible"] == 3                 # the perms still make a floor


def test_a_perm_draw_without_proof_is_refused(tmp_path):
    """The env was set but the loop never ran it (calls 0), or it ran a different order: a plain repeat
    wearing a perturbed label. Refused, like P56's batched arm that fell back on every call."""
    arms = _floors([0.01, 0.02, 0.015, 0.012])
    arms[2]["reference_order"] = {"requested": "perm:1", "order": None, "calls": 0}
    arms[3]["reference_order"] = {"requested": "perm:2", "order": "perm:9", "calls": 600}
    res = p67.reduce_session(_session(tmp_path, arms))
    bad = {d["tag"]: d for d in res[FAM]["floor_draws"]}
    assert not bad["reference_attn4_perm1"]["admissible"] and not bad["reference_attn4_perm2"]["admissible"]
    assert res[FAM]["floor"]["n_admissible"] == 2 and not res[FAM]["floor"]["exists"]


def test_a_draw_missing_the_reference_order_field_is_refused(tmp_path):
    arms = _floors([0.01, 0.02, 0.015])
    del arms[3]["reference_order"]
    res = p67.reduce_session(_session(tmp_path, arms))
    assert res[FAM]["floor"]["n_admissible"] == 2


def test_a_draw_on_a_different_fixture_is_refused(tmp_path):
    arms = _floors([0.01, 0.02, 0.015, 0.012])
    arms[2]["tokens"] = {"sha256": "cc" * 32}
    arms[3]["init_sha"] = "dd" * 32
    arms[4]["trainable_params"] = 1
    res = p67.reduce_session(_session(tmp_path, arms))
    assert res[FAM]["floor"]["n_admissible"] == 1


def test_a_reference_that_itself_ran_a_perturbed_order_disqualifies_every_draw(tmp_path):
    arms = _floors([0.01, 0.02, 0.015, 0.012])
    arms[0]["reference_order"] = {"requested": "descending", "order": "descending", "calls": 600}
    res = p67.reduce_session(_session(tmp_path, arms))
    assert res[FAM]["floor"]["n_admissible"] == 0


def test_cross_session_pairs_are_disclosed_never_admitted(tmp_path):
    """Two sessions' reference arms of one fixture differ -- that is evidence a floor is not zero, and it is
    printed -- but it is not a floor draw: the pair spans boxes and commits."""
    s1 = _session(tmp_path, [_arm("reference_attn4", _curve(0.0)), _arm("fused_attn4", _curve(0.08))], "a")
    s2 = _session(tmp_path, [_arm("reference_attn4", _curve(0.02))], "b")
    res = p67.reduce_reread([s1, s2])
    g = res["groups"][0]
    assert g["n_admissible_floor_draws"] == 0
    assert any(p["tag"] == "reference_attn4" and p["d_final_train"] == pytest.approx(0.02) for p in g["cross_session_pairs"])
    assert all(not p["admissible"] for p in g["cross_session_pairs"])
    assert g["judged"][0]["verdict"] == "NO-FLOOR"


def test_step0_movement_is_flagged_not_refused(tmp_path):
    arms = _floors([0.01, 0.02, 0.015])
    arms[2]["losses"] = [arms[2]["losses"][0] + 0.01] + arms[2]["losses"][1:]
    res = p67.reduce_session(_session(tmp_path, arms))
    d = next(x for x in res[FAM]["floor_draws"] if x["tag"] == "reference_attn4_perm1")
    assert d["step0_flag"] and d["admissible"]


# ----------------------------------------------------------------------------- judged-arm validity
def test_a_batched_arm_that_fell_back_is_void(tmp_path):
    b = _arm("batched_attn4", _curve(0.05))
    b["batched_stats"] = {"calls": 7680, "batched": 7000, "fallback_calls": 680, "by_reason": {"pad_waste": 680}}
    j = _judged(p67.reduce_session(_session(tmp_path, _floors([0.03, 0.03, 0.03]) + [b])), "batched_attn4")
    assert j["status"] == "VOID" and "fell back" in "; ".join(j["why"])


def test_a_fused_arm_below_its_kernel_call_count_is_void(tmp_path):
    f = _arm("fused_attn4", _curve(0.001), kernel_calls_per_step_min=100)
    j = _judged(p67.reduce_session(_session(tmp_path, _floors([0.03, 0.03, 0.03]) + [f])), "fused_attn4")
    assert j["status"] == "VOID"


def test_consistency_pair_that_disagrees_makes_the_family_mixed(tmp_path):
    here = _session(tmp_path, _floors([0.030, 0.020, 0.025]) + [_arm("fused_attn4", _curve(0.07))], "here")
    other = _session(tmp_path, [_arm("reference_attn4", _curve(0.0)), _arm("fused_attn4", _curve(0.12))], "other")
    res = p67.reduce_session(here, [other])
    assert _judged(res, "fused_attn4")["verdict"] == "PASS"
    assert res[FAM]["consistency"][0]["verdict"] == "FAIL" and res[FAM]["reading"] == "MIXED"


def test_a_consistency_session_on_another_fixture_is_listed_not_read(tmp_path):
    here = _session(tmp_path, _floors([0.030, 0.020, 0.025]) + [_arm("fused_attn4", _curve(0.07))], "here")
    ref = _arm("reference_attn4", _curve(0.0))
    ref["tokens"] = {"sha256": "ee" * 32}
    other = _session(tmp_path, [ref, _arm("fused_attn4", _curve(0.12), tokens={"sha256": "ee" * 32})], "other")
    res = p67.reduce_session(here, [other])
    assert res[FAM]["consistency"] == [] and res[FAM]["reading"] == "READ"
    assert "tokens_sha" in res[FAM]["consistency_skipped"][0]["why"]


# ----------------------------------------------------------------------------- real committed receipts
def test_tp1_bundle_reads_as_its_own_table_did(tmp_path):
    """tp1's committed bundle through the tp1 adapter: every fused row PASSes by tolerance, the batched rows are
    PASS on Granite and Mixtral and VOID on OLMoE, Qwen3 and Gemma-4 -- tp1's own verdicts, and its numbers."""
    tp1 = REPO / "bench" / "train-parity-20260905" / "tp1"
    res = p67.reduce_reread([], str(tp1))
    rows = {(g["fam"], j["tag"]): j for g in res["groups"] for j in g["judged"]}
    for fam in ("granite", "olmoe", "qwen3", "gemma4", "mixtral"):
        assert rows[(fam, "fused_attn4")]["verdict"] == "PASS" and rows[(fam, "fused_attn4")]["carried_by"] == "tolerance"
    for fam in ("granite", "mixtral"):
        assert rows[(fam, "batched_attn4")]["verdict"] == "PASS"
    for fam in ("olmoe", "qwen3", "gemma4"):
        assert rows[(fam, "batched_attn4")]["status"] == "VOID"
    g4 = rows[("gemma4", "fused_attn4")]
    assert round(g4["d_final_train"], 5) == 0.02385 and round(g4["d_med_train"], 5) == 0.04742   # RESULTS-tp1.md


def test_cli_writes_both_outputs(tmp_path):
    d = _session(tmp_path, _floors([0.030, 0.020, 0.025]) + [_arm("fused_attn4", _curve(0.07))])
    md, js = tmp_path / "o.md", tmp_path / "o.json"
    assert p67.main(["--session", d, "--md", str(md), "--json", str(js)]) == 0
    assert "floor band" in md.read_text() and json.loads(js.read_text())[FAM]["reading"] == "READ"
    assert os.path.getsize(js) > 0
