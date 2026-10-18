"""HP1's reducer (bench/hp1/hp1_reduce.py, HP1-PREREG.md "Readings and decision rules"): the t distribution against table values,
the 2x2 ANOVA against a hand-computed example, the K selection rule (CAUSAL, regime switch, cheapest within 1 sigma, K_INCOMPLETE)
and the P verdicts (BEATS / PARITY / BEHIND / INCONCLUSIVE, C*, OWN_BASE_ONLY, duplicate tags), on synthetic evaluator JSONs."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "hp1"))
import hp1_reduce as hr  # noqa: E402

SEL = hr.SELECTION_SEEDS
CONF = hr.CONFIRMATION_SEEDS
BASE = 0.77
NOISE = (-0.001, 0.0, 0.001)          # per-seed offsets: within-arm SD exactly 0.001


# ------------------------------------------------------------------------------------------------------------ helpers
def _order(seed):
    return {"order_seed": seed, "n_rows": 1200, "order_sha": f"{seed:04d}" + "ab" * 30}


def _arm(tag, seed, y, status="SCORED", framework="e4b"):
    arm = {"tag": tag, "framework": framework, "seed": seed, "status": status, "inloop_final": None, "order": _order(seed), "steps": 200}
    if status == "SCORED":
        arm["evaluator"] = {"row_mean": y, "token_weighted": y}
        arm["inloop_final"] = y + 0.0003
        arm["adapter"] = f"/root/tc1/hp1/canonical_qwen3hp1_{tag}.safetensors"
        arm["adapter_sha256"] = f"sha-{tag}"
        arm["base"] = f"/root/tc1/hp1/base_{framework}_qwen3hp1"
    elif status == "OWN_BASE_ONLY":
        arm["inloop_final"] = y
    else:
        arm["reason"] = "synthetic VOID"
    return arm


def _k_arms(means, noise=NOISE, skip=()):
    out = []
    for cell, mu in means.items():
        for s, eps in zip(SEL, noise):
            if (cell, s) not in skip:
                out.append(_arm(f"hp1_{cell}_s{s}", s, mu + eps))
    return out


def _mf(tmp_path, *eval_paths, drop=()):
    """['--manifest', path]: a bulk manifest pinning every SCORED arm's adapter (by its receipt sha256) and base export in eval_paths,
    minus the rel paths in `drop` (the injected failures)."""
    rows = {}
    for p in eval_paths:
        for a in json.loads(Path(p).read_text())["arms"]:
            if a.get("status") == "SCORED":
                rows[hr._rel_hp1(a["adapter"])] = a["adapter_sha256"]
                rows[hr._rel_hp1(a["base"]) + "/index.json"] = "idx"
    path = tmp_path / "bulk_manifest.json"
    path.write_text(json.dumps({"hp1_bulk_manifest": 1, "files": [{"path": k, "bytes": 1, "sha256": v} for k, v in rows.items() if k not in drop]}))
    return ["--manifest", str(path)]


def _write(tmp_path, name, arms, eval_n=48):
    path = tmp_path / name
    path.write_text(json.dumps({"evaluator": "bench/hp1/hp1_eval.py", "tokens": {"eval_n": eval_n}, "arms": arms}))
    return str(path)


def _select(tmp_path, arms, capsys):
    ev = _write(tmp_path, "k.json", arms)
    out = tmp_path / "sel.json"
    code = hr.main(["select", ev, "--out", str(out), *_mf(tmp_path, ev)])
    return code, json.loads(out.read_text()), capsys.readouterr().out


K10_BEST = {"k00": BASE + 0.03, "k10": BASE, "k01": BASE + 0.03, "k11": BASE}   # init CAUSAL, no interaction -> k10


def _p_arms(e_cell, e_deltas, unsloth_off=0.0, axolotl_off=0.0, k00_off=0.03, comp_status="SCORED", skip=()):
    """Competitor y = BASE + off + NOISE[s]; E = axolotl-or-C* + delta_s (deltas are against the lower-mean competitor)."""
    arms = []
    lower = min(unsloth_off, axolotl_off)
    for i, s in enumerate(CONF):
        for name, off in (("unsloth", unsloth_off), ("axolotl", axolotl_off)):
            if (name, s) not in skip:
                arms.append(_arm(f"hp1_{name}_s{s}", s, BASE + off + NOISE[i], status=comp_status, framework=name))
        if ("e", s) not in skip:
            arms.append(_arm(f"hp1_e_{e_cell}_s{s}", s, BASE + lower + NOISE[i] + e_deltas[i]))
        if e_cell != "k00" and ("k00", s) not in skip:
            arms.append(_arm(f"hp1_k00_s{s}", s, BASE + k00_off + NOISE[i]))
    return arms


def _verdict(tmp_path, k_arms, p_arms, capsys):
    kf = _write(tmp_path, "k.json", k_arms)
    pf = _write(tmp_path, "p.json", p_arms)
    out, md = tmp_path / "v.json", tmp_path / "v.md"
    code = hr.main(["verdict", kf, pf, "--out", str(out), "--md", str(md), *_mf(tmp_path, kf, pf)])
    res = json.loads(out.read_text())
    return code, res, md.read_text(), capsys.readouterr().out


def _reading(res, role):
    return next(r for r in res["p"]["readings"] if r["role"] == role)


# ----------------------------------------------------------------------------------------------------------- t and F
@pytest.mark.parametrize("df,want", [(2, 2.919986), (8, 1.859548), (12, 1.782288), (1, 6.313752), (30, 1.697261)])
def test_t_quantile_table(df, want):
    assert hr.t_ppf(0.95, df) == pytest.approx(want, abs=2e-6)
    assert hr.t_cdf(want, df) == pytest.approx(0.95, abs=1e-6)
    assert hr.t_ppf(0.05, df) == pytest.approx(-want, abs=2e-6)


def test_t_cdf_closed_forms():
    for t in (-3.0, -0.4, 0.0, 0.7, 2.0, 9.0):
        assert hr.t_cdf(t, 1) == pytest.approx(0.5 + math.atan(t) / math.pi, abs=1e-12)          # Cauchy
        assert hr.t_cdf(t, 2) == pytest.approx(0.5 + t / (2 * math.sqrt(2 + t * t)), abs=1e-12)  # df 2


def test_t_sf_df8_matches_simpson_integral_of_density():
    df = 8

    def pdf(x):
        c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
        return c * (1 + x * x / df) ** (-(df + 1) / 2)

    for t in (0.5, 1.859548, 3.2733):
        n = 2000
        h = t / n
        area = pdf(0) + pdf(t) + sum((4 if i % 2 else 2) * pdf(i * h) for i in range(1, n))
        assert hr.t_sf(t, df) == pytest.approx(0.5 - area * h / 3, abs=1e-10)


def test_f_sf_is_two_sided_t():
    for t in (0.3, 1.5, 3.0):
        assert hr.f_sf(t * t, 1, 8) == pytest.approx(2 * hr.t_sf(t, 8), abs=1e-12)


# --------------------------------------------------------------------------------------------------------- K readout
def test_anova_hand_computed_example():
    cells = {"k00": [4.0, 5.0, 6.0], "k10": [1.0, 2.0, 3.0], "k01": [3.0, 4.0, 5.0], "k11": [0.0, 2.0, 4.0]}
    # by hand: cell means 5, 2, 4, 2; grand 3.25; within SS 2 + 2 + 2 + 8 = 14 on df 8 -> MSE 1.75
    m = {c: sum(v) / 3 for c, v in cells.items()}
    grand = sum(m.values()) / 4
    ss_err = sum((x - m[c]) ** 2 for c, v in cells.items() for x in v)
    mse = ss_err / 8
    init = (m["k00"] + m["k01"]) / 2 - (m["k10"] + m["k11"]) / 2
    dtype = (m["k00"] + m["k10"]) / 2 - (m["k01"] + m["k11"]) / 2
    inter = m["k00"] - m["k10"] - m["k01"] + m["k11"]
    total = sum((x - grand) ** 2 for v in cells.values() for x in v)
    assert (mse, init, dtype, inter, grand, total) == (1.75, 2.5, 0.5, 1.0, 3.25, 34.25)

    got = hr.two_way_anova(cells)
    an = got["anova"]
    assert an["mse"] == pytest.approx(mse) and an["df_error"] == 8
    assert an["ss_init"] == pytest.approx(3 * init ** 2)          # 18.75
    assert an["ss_dtype"] == pytest.approx(3 * dtype ** 2)        # 0.75
    assert an["ss_interaction"] == pytest.approx(3 * inter ** 2 / 4)   # 0.75
    assert an["ss_init"] + an["ss_dtype"] + an["ss_interaction"] + an["ss_error"] == pytest.approx(total)
    se = math.sqrt(mse * (1 / 6 + 1 / 6))
    assert got["init"]["effect"] == pytest.approx(init) and got["init"]["se"] == pytest.approx(se)
    assert got["init"]["t"] == pytest.approx(init / se)           # 3.2733
    assert got["init"]["F"] == pytest.approx(got["init"]["t"] ** 2)
    assert got["init"]["p_one_sided"] == pytest.approx(hr.t_sf(init / se, 8))
    assert got["init"]["p_one_sided"] == pytest.approx(got["init"]["p_F_two_sided"] / 2)
    assert 0.005 < got["init"]["p_one_sided"] < 0.01 and got["init"]["reading"] == "CAUSAL"
    assert got["dtype"]["t"] == pytest.approx(dtype / se) and got["dtype"]["reading"] == "NOT-SHOWN"
    assert got["interaction"]["estimate"] == pytest.approx(inter)
    assert got["interaction"]["se"] == pytest.approx(math.sqrt(mse * 4 / 3))
    assert got["interaction"]["F"] == pytest.approx(3 * inter ** 2 / (4 * mse))     # 0.428571
    assert got["interaction"]["p_two_sided"] == pytest.approx(2 * hr.t_sf(inter / math.sqrt(mse * 4 / 3), 8))
    assert got["sigma_K"] == pytest.approx(math.sqrt(1.75))


def test_planted_init_effect_is_causal_main_effects_regime(tmp_path, capsys):
    code, res, out = _select(tmp_path, _k_arms(K10_BEST), capsys)
    k = res["k"]
    assert code == 0 and res["verdict"] == "OK"
    assert k["init"]["reading"] == "CAUSAL" and k["init"]["effect"] == pytest.approx(0.03)
    assert k["dtype"]["reading"] == "NOT-SHOWN"
    assert k["interaction"]["p_two_sided"] == pytest.approx(1.0)
    assert k["regime"] == "main-effects" and k["main_effects_decide"]
    assert k["sigma_K"] == pytest.approx(0.001)
    assert k["selected"] == "k10"
    assert out.strip().splitlines() == ["HP1_SELECTED=k10"]


def test_negative_effect_is_not_shown(tmp_path, capsys):
    means = {"k00": BASE, "k10": BASE + 0.03, "k01": BASE, "k11": BASE + 0.03}     # native is BETTER: not causal
    _, res, out = _select(tmp_path, _k_arms(means), capsys)
    assert res["k"]["init"]["reading"] == "NOT-SHOWN" and res["k"]["init"]["p_one_sided"] > 0.99
    assert out.strip() == "HP1_SELECTED=k00"


def test_planted_interaction_switches_regime(tmp_path, capsys):
    means = {"k00": BASE + 0.03, "k10": BASE, "k01": BASE, "k11": BASE}             # only the shipped cell is worse
    _, res, out = _select(tmp_path, _k_arms(means), capsys)
    k = res["k"]
    assert k["interaction"]["estimate"] == pytest.approx(0.03) and k["interaction"]["p_two_sided"] < 0.05
    assert k["regime"] == "cell-means" and not k["main_effects_decide"]
    assert k["init"]["effect"] == pytest.approx(0.015)     # still reported
    assert k["selected"] == "k10" and out.strip() == "HP1_SELECTED=k10"


def test_selection_cheapest_within_one_sigma(tmp_path, capsys):
    means = {"k00": BASE + 0.03, "k10": BASE + 0.0005, "k01": BASE + 0.02, "k11": BASE}
    _, res, out = _select(tmp_path, _k_arms(means), capsys)
    assert res["k"]["best_cell"] == "k11" and res["k"]["eligible"] == ["k10", "k11"]
    assert res["k"]["selected"] == "k10" and out.strip() == "HP1_SELECTED=k10"


def test_selection_outside_one_sigma_takes_the_best(tmp_path, capsys):
    means = {"k00": BASE + 0.03, "k10": BASE + 0.003, "k01": BASE + 0.02, "k11": BASE}   # k10 is 3 sigma off
    _, res, out = _select(tmp_path, _k_arms(means), capsys)
    assert res["k"]["selected"] == "k11" and out.strip() == "HP1_SELECTED=k11"


def test_select_cell_unit():
    assert hr.select_cell({"k00": 1.0, "k10": 0.9, "k01": 0.95, "k11": 0.9}, 0.1)[0] == "k00"
    assert hr.select_cell({"k00": 1.0, "k10": 0.9, "k01": 0.95, "k11": 0.9}, 0.05)[0] == "k10"


@pytest.mark.parametrize("how", ["missing", "void"])
def test_k_incomplete_exits_3_and_selects_nothing(tmp_path, capsys, how):
    if how == "missing":
        arms = _k_arms(K10_BEST, skip={("k01", 3408)})
    else:
        arms = _k_arms(K10_BEST)
        arms[7] = _arm(arms[7]["tag"], arms[7]["seed"], None, status="VOID")
    code, res, out = _select(tmp_path, arms, capsys)
    assert code == 3 and res["verdict"] == "K_INCOMPLETE" and res["k"]["selected"] is None
    assert len(res["k"]["missing"]) == 1
    assert "HP1_SELECTED" not in out


def test_select_ignores_p_block_and_unknown_tags(tmp_path, capsys):
    arms = _k_arms(K10_BEST) + _p_arms("k10", (0, 0, 0)) + [_arm("hp1_k10_s3410", 3410, 9.0), _arm("tc1_x", 1, 9.0)]
    code, res, out = _select(tmp_path, arms, capsys)
    assert code == 0 and out.strip() == "HP1_SELECTED=k10"
    assert {x["tag"] for x in res["ignored"]} == {"hp1_k10_s3410", "tc1_x"}


# ---------------------------------------------------------------------------------------------------------- P readout
def test_verdict_beats(tmp_path, capsys):
    code, res, md, out = _verdict(tmp_path, _k_arms(K10_BEST), _p_arms("k10", (-0.02, -0.021, -0.019)), capsys)
    head = _reading(res, "headline")
    assert code == 0 and res["verdict"] == "BEATS" and out.strip() == "HP1_VERDICT=BEATS"
    assert head["mean"] == pytest.approx(-0.02)
    assert head["se"] == pytest.approx(0.001 / math.sqrt(3))
    assert head["t_crit"] == pytest.approx(2.919986, abs=2e-6)
    assert head["ci90"][1] == pytest.approx(-0.02 + 2.919986 * 0.001 / math.sqrt(3), abs=1e-8)
    assert "BEATS" in md and "| reading |" in md


def test_verdict_parity(tmp_path, capsys):
    code, res, _, _ = _verdict(tmp_path, _k_arms(K10_BEST), _p_arms("k10", (0.0001, -0.0001, 0.0)), capsys)
    head = _reading(res, "headline")
    assert code == 0 and res["verdict"] == "PARITY"
    assert head["m"] == pytest.approx(0.005)              # sigma_seed ~ 0.001 -> floor
    assert head["sigma_seed"] == pytest.approx(res["p"]["sigma_seed"])


def test_verdict_behind_and_k00_reading(tmp_path, capsys):
    code, res, _, _ = _verdict(tmp_path, _k_arms(K10_BEST), _p_arms("k10", (0.02, 0.021, 0.019), k00_off=0.03), capsys)
    assert code == 0 and res["verdict"] == "BEHIND"
    k00 = _reading(res, "k00")
    assert k00["verdict"] == "BEHIND" and k00["mean"] == pytest.approx(0.03)


def test_verdict_inconclusive(tmp_path, capsys):
    code, res, md, _ = _verdict(tmp_path, _k_arms(K10_BEST), _p_arms("k10", (0.01, -0.01, 0.004)), capsys)
    head = _reading(res, "headline")
    assert code == 0 and res["verdict"] == "INCONCLUSIVE"
    assert head["ci90"][0] < -head["m"] and head["ci90"][1] > head["m"]
    assert "INCONCLUSIVE" in md


def test_sigma_seed_pools_every_arm():
    # K cells: SS 2e-6 each (df 2); P arms here likewise; E has its own spread
    groups = [("a", [1.0, 2.0, 3.0]), ("b", [0.0, 2.0, 4.0]), ("c", [5.0])]
    sd, df, _ = hr.pooled_sd(groups)
    assert df == 4 and sd == pytest.approx(math.sqrt((2.0 + 8.0) / 4))


def test_verdict_sigma_seed_counts_all_groups(tmp_path, capsys):
    _, res, _, _ = _verdict(tmp_path, _k_arms(K10_BEST), _p_arms("k10", (0.01, -0.01, 0.004)), capsys)
    p = res["p"]
    names = {g["group"] for g in p["sigma_seed_groups"]}
    assert names == {"K:k00", "K:k10", "K:k01", "K:k11", "P:E:k10", "P:k00", "P:unsloth", "P:axolotl"}
    e_vals = [BASE + n + d for n, d in zip(NOISE, (0.01, -0.01, 0.004))]
    mu = sum(e_vals) / 3
    ss = 7 * 2e-6 + sum((v - mu) ** 2 for v in e_vals)
    assert p["sigma_seed_df"] == 16 and p["sigma_seed"] == pytest.approx(math.sqrt(ss / 16))
    assert p["m"] == pytest.approx(min(max(p["sigma_seed"], 0.005), 0.02))


def test_c_star_is_lower_mean_competitor_and_per_competitor_readings(tmp_path, capsys):
    p_arms = _p_arms("k10", (0.0, 0.0, 0.0), unsloth_off=0.01, axolotl_off=0.0)
    _, res, _, _ = _verdict(tmp_path, _k_arms(K10_BEST), p_arms, capsys)
    assert res["p"]["c_star"] == "axolotl"
    per = {r["reading"]: r for r in res["p"]["readings"] if r["role"] == "per-competitor"}
    assert per["E (k10) vs unsloth"]["mean"] == pytest.approx(-0.01)
    assert per["E (k10) vs axolotl"]["mean"] == pytest.approx(0.0, abs=1e-12)
    p_arms = _p_arms("k10", (0.0, 0.0, 0.0), unsloth_off=-0.01, axolotl_off=0.0)
    _, res, _, _ = _verdict(tmp_path, _k_arms(K10_BEST), p_arms, capsys)
    assert res["p"]["c_star"] == "unsloth"


def test_an_own_base_only_arm_gives_no_y(tmp_path, capsys):
    """Every verdict uses evaluator numbers only: an OWN_BASE_ONLY competitor contributes no y, so its cell is incomplete and the
    headline reads P_INCOMPLETE -- an in-loop number never enters a verdict."""
    p_arms = _p_arms("k10", (0.0001, -0.0001, 0.0), unsloth_off=0.005, comp_status="OWN_BASE_ONLY")
    code, res, _, _ = _verdict(tmp_path, _k_arms(K10_BEST), p_arms, capsys)
    assert code == 3 and res["verdict"] == "P_INCOMPLETE" and "axolotl" in res["p"]["excluded_competitors"]

def test_an_incomplete_competitor_gives_no_headline(tmp_path, capsys):
    """Registered: the headline needs BOTH competitors' confirmation cells. With one incomplete it reads P_INCOMPLETE (exit 3), never
    a headline against the other; the complete competitor's reading is reported, labelled, beside it."""
    p_arms = _p_arms("k10", (0.0, 0.0, 0.0), unsloth_off=-0.05, skip={("unsloth", 3411)})
    code, res, _, _ = _verdict(tmp_path, _k_arms(K10_BEST), p_arms, capsys)
    assert code == 3 and res["verdict"] == "P_INCOMPLETE" and res["p"]["c_star"] is None
    assert res["p"]["excluded_competitors"] == ["unsloth"] and any("unsloth" in x for x in res["p"]["problems"])
    side = [r for r in res["p"]["readings"] if r["role"].startswith("per-competitor")]
    assert len(side) == 1 and "axolotl" in side[0]["reading"] and side[0]["role"] == "per-competitor (no headline)"


def test_p_incomplete_when_e_or_both_competitors_missing(tmp_path, capsys):
    code, res, md, out = _verdict(tmp_path, _k_arms(K10_BEST), _p_arms("k10", (0, 0, 0), skip={("e", 3412)}), capsys)
    assert code == 3 and res["verdict"] == "P_INCOMPLETE" and "P_INCOMPLETE" in md
    skip = {("unsloth", 3410), ("axolotl", 3412)}
    code, res, _, _ = _verdict(tmp_path, _k_arms(K10_BEST), _p_arms("k10", (0, 0, 0), skip=skip), capsys)
    assert code == 3 and res["verdict"] == "P_INCOMPLETE"


def test_e_k00_uses_shipped_runs_and_k00_reading_is_the_headline(tmp_path, capsys):
    k_means = {"k00": BASE, "k10": BASE + 0.0005, "k01": BASE + 0.02, "k11": BASE + 0.0002}
    p_arms = []
    for i, s in enumerate(CONF):
        p_arms.append(_arm(f"hp1_k00_s{s}", s, BASE - 0.02 + NOISE[i]))
        p_arms += [_arm(f"hp1_{c}_s{s}", s, BASE + NOISE[i], framework=c) for c in hr.COMPETITORS]
    code, res, _, _ = _verdict(tmp_path, _k_arms(k_means), p_arms, capsys)
    assert code == 0 and res["p"]["selected"] == "k00" and res["p"]["e_arm"] == "k00"
    assert res["verdict"] == "BEATS"
    assert _reading(res, "k00")["same_as_headline"]


def test_e_tag_naming_another_cell_is_refused(tmp_path, capsys):
    kf, pf = _write(tmp_path, "k.json", _k_arms(K10_BEST)), _write(tmp_path, "p.json", _p_arms("k11", (0, 0, 0)))
    code = hr.main(["verdict", kf, pf, "--out", str(tmp_path / "v.json"), "--md", str(tmp_path / "v.md"), *_mf(tmp_path, kf, pf)])
    assert code == 2 and "selects k10" in capsys.readouterr().err


def test_duplicate_tags_across_files_refused(tmp_path, capsys):
    k_arms = _k_arms(K10_BEST)
    f1 = _write(tmp_path, "a.json", k_arms)
    f2 = _write(tmp_path, "b.json", [k_arms[0]])
    code = hr.main(["select", f1, f2, "--out", str(tmp_path / "s.json"), "--manifest", str(tmp_path / "none.json")])
    captured = capsys.readouterr()
    assert code == 2 and "duplicate tag" in captured.err and "HP1_SELECTED" not in captured.out
    with pytest.raises(hr.RefusedInput):
        hr.load_arms([f1, f2])


def test_tag_seed_mismatch_refused(tmp_path):
    arms = _k_arms(K10_BEST)
    arms[0]["seed"] = 3409
    with pytest.raises(hr.RefusedInput, match="names seed"):
        hr.load_arms([_write(tmp_path, "k.json", arms)])


# ------------------------------------------------------------------------------------------------------- data order
def _break_order(arm, how):
    if how == "null":
        arm["order"] = None
    elif how == "seed":
        arm["order"] = dict(arm["order"], order_seed=arm["seed"] + 1)
    else:
        arm["order"] = dict(arm["order"], order_sha="ff" * 32)


@pytest.mark.parametrize("how", ["null", "seed", "sha"])
def test_select_refuses_unpaired_data_order(tmp_path, capsys, how):
    arms = _k_arms(K10_BEST)
    victim = next(a for a in arms if a["tag"] == "hp1_k01_s3408")
    _break_order(victim, how)
    code, res, out = _select(tmp_path, arms, capsys)
    assert code == 3 and res["verdict"] == "K_INCOMPLETE" and res["k"]["selected"] is None
    probs = res["k"]["order_problems"]
    assert probs and all(x["seed"] == 3408 for x in probs)
    assert "hp1_k01_s3408" in {x["tag"] for x in probs}
    assert "HP1_SELECTED" not in out


@pytest.mark.parametrize("victim_tag", ["hp1_axolotl_s3411", "hp1_e_k10_s3410", "hp1_k00_s3412", "hp1_unsloth_s3410"])
@pytest.mark.parametrize("how", ["null", "seed", "sha"])
def test_verdict_refuses_unpaired_data_order(tmp_path, capsys, victim_tag, how):
    p_arms = _p_arms("k10", (-0.02, -0.021, -0.019))
    victim = next(a for a in p_arms if a["tag"] == victim_tag)
    _break_order(victim, how)
    code, res, md, out = _verdict(tmp_path, _k_arms(K10_BEST), p_arms, capsys)
    assert code == 3 and res["verdict"] == "P_INCOMPLETE" and out.strip() == "HP1_VERDICT=P_INCOMPLETE"
    probs = res["p"]["order_problems"]
    assert victim_tag in {x["tag"] for x in probs} and {x["seed"] for x in probs} == {victim["seed"]}
    assert any(victim_tag in msg and str(victim["seed"]) in msg for msg in res["p"]["problems"])
    assert victim_tag in md


def test_order_problems_unit():
    recs = [{"tag": "a", "seed": 1, "order": {"order_seed": 1, "order_sha": "x"}},
            {"tag": "b", "seed": 1, "order": {"order_seed": "1", "order_sha": "x"}},
            {"tag": "c", "seed": 2, "order": {"order_seed": 2, "order_sha": "y"}}]
    assert hr.order_problems(recs) == []
    recs.append({"tag": "d", "seed": 2, "order": {"order_seed": 2, "order_sha": "z"}})
    assert {x["tag"] for x in hr.order_problems(recs)} == {"c", "d"}


def test_voids_lists_exactly_the_void_arms(tmp_path, capsys):
    """The driver's redraw rule reads `voids`: every RUN-level VOID arm's tag; a base-level VOID (a systematic miss) is not redrawn."""
    path = _write(tmp_path, "e.json", [_arm("hp1_k00_s3407", 3407, 1.0), dict(_arm("hp1_k10_s3407", 3407, 1.0), status="VOID"),
                                       dict(_arm("hp1_k01_s3407", 3407, 1.0), status="OWN_BASE_ONLY"),
                                       dict(_arm("hp1_k11_s3407", 3407, 1.0), status="VOID", void_level="base")])
    capsys.readouterr()
    assert hr.main(["voids", path]) == 0
    assert capsys.readouterr().out.split() == ["hp1_k10_s3407"]


def test_a_scored_arm_on_its_own_base_is_usable(tmp_path, capsys):
    """(a) A competitor scored by the evaluator on its own exported base (status SCORED, a different `base`) is an ordinary,
    usable y; (b), an in-loop-only arm giving no y, is test_an_own_base_only_arm_gives_no_y."""
    p_arms = _p_arms("k10", (0.0001, -0.0001, 0.0), unsloth_off=0.005)
    for a in p_arms:
        if a["tag"].startswith("hp1_axolotl"):
            a["base"] = "base_axolotl_qwen3hp1"
    code, res, _, _ = _verdict(tmp_path, _k_arms(K10_BEST), p_arms, capsys)
    assert code == 0 and res["p"]["c_star"] == "axolotl" and res["p"]["competitors"]["axolotl"]["included"]


def test_a_non_registered_eval_n_is_refused(tmp_path, capsys):
    """A forwarded TC1_HP1_EVAL_N cannot silently change the instrument: an evaluator JSON that read other than 48 held-out rows is
    refused (exit 2), fail-closed."""
    path = _write(tmp_path, "k.json", _k_arms(K10_BEST), eval_n=16)
    assert hr.main(["select", path, "--out", str(tmp_path / "s.json"), *_mf(tmp_path, path)]) == hr.EXIT_REFUSED
    assert "48" in capsys.readouterr().err


def test_a_run_at_another_n_gives_no_y(tmp_path, capsys):
    """A run whose receipt says steps != 200 (a forwarded TC1_CURVE_STEPS) is not this registration's run: it gives no y, so its
    cell is incomplete."""
    arms = _k_arms(K10_BEST)
    arms[0]["steps"] = 100
    code, res, _ = _select(tmp_path, arms, capsys)
    assert code == hr.EXIT_INCOMPLETE and res["verdict"] != "OK"


# ----------------------------------------------------------------------------- Amendment 1: the bulk manifest pins what was scored
@pytest.mark.parametrize("case", ["missing", "unparsable", "adapter_absent", "adapter_sha", "base_absent"])
def test_a_reading_whose_manifest_does_not_pin_what_was_scored_is_void(tmp_path, capsys, case):
    kf = _write(tmp_path, "k.json", _k_arms(K10_BEST))
    mf = _mf(tmp_path, kf)
    man = Path(mf[1])
    if case == "missing":
        man.unlink()
    elif case == "unparsable":
        man.write_text("{not json")
    elif case == "adapter_absent":
        mf = _mf(tmp_path, kf, drop={"hp1/canonical_qwen3hp1_hp1_k10_s3408.safetensors"})
    elif case == "adapter_sha":
        doc = json.loads(man.read_text())
        doc["files"][0]["sha256"] = "tampered"
        man.write_text(json.dumps(doc))
    elif case == "base_absent":
        mf = _mf(tmp_path, kf, drop={"hp1/base_e4b_qwen3hp1/index.json"})
    out = tmp_path / "s.json"
    assert hr.main(["select", kf, "--out", str(out), *mf]) == hr.EXIT_INCOMPLETE
    res = json.loads(out.read_text())
    assert res["verdict"] == "VOID" and res["manifest_problems"] and "HP1_SELECTED" not in capsys.readouterr().out
