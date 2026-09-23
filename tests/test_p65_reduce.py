"""P65's reducer applies the pre-registration literally (bench/p65/p65_reduce.py; bench/p65/P65-PREREG.md).

Synthetic census rows with a KNOWN structure -- a signal that is a stable property of each expert, one that the text
re-draws, one that only resampling moves -- and each branch of the decision rule fires on the rows built to trigger
it. Also: the routing ranking is the package's own ``hot_sets_from_profile``, and the cosine statistic Colla-Q reports
cannot tell an ordering from noise, which is why the rule does not read it."""
import importlib.util
import json
import pathlib
import random

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p65_reduce", REPO / "bench" / "p65" / "p65_reduce.py")
red = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(red)

L, E = 8, 24


def census(*, entropy="stable", rel="stable", freq="text", ent_tracks_rel=False, seed=0, family="granite",
           selfcheck=True, noise=0.01):
    """Rows for every (text, half, layer, expert, role). Each signal is one of:
      stable -- a fixed property of the expert plus small per-census noise (survives resampling AND domain);
      text   -- a property of (expert, text): the halves of one text agree, the two texts are independent;
      random -- redrawn for every census (survives nothing)."""
    rng = random.Random(seed)
    true = {k: {(la, e): rng.random() for la in range(L) for e in range(E)} for k in ("ent", "rel", "freq")}
    per_text = {k: {t: {(la, e): rng.random() for la in range(L) for e in range(E)} for t in red.TEXTS}
                for k in ("ent", "rel", "freq")}

    def draw(kind, key, t, le):
        if kind == "stable":
            return true[key][le] + rng.gauss(0, noise)
        if kind == "text":
            return per_text[key][t][le] + rng.gauss(0, noise)
        return rng.random()

    rows = []
    for t in red.TEXTS:
        for half in (0, 1, "full"):
            for la in range(L):
                for e in range(E):
                    r = 0.05 + 0.2 * max(0.0, draw(rel, "rel", t, (la, e)))
                    rho = (0.2 + r) if ent_tracks_rel else 0.1 + 0.8 * min(1.0, max(0.0, draw(entropy, "ent", t, (la, e))))
                    n = 40 + int(2000 * max(0.0, draw(freq, "freq", t, (la, e))))
                    base = {"layer": la, "expert": e, "text": t, "half": half, "rows": n, "recipe_method": "rtn"}
                    rows.append({**base, "role": "gu", "rtn": {"rel_act": r, "sq_err_act": r * r, "rel_frob": 0.1},
                                 "entropy": {"rho": 0.5}})
                    rows.append({**base, "role": "dn", "rtn": {"rel_act": r, "sq_err_act": r * r, "rel_frob": 0.1},
                                 "entropy": {"rho": rho, "h_energy": 0.9}})
    return {"family": family, "experts": E, "layers_censused": list(range(L)), "rows": rows,
            "selfcheck": {"ok": selfcheck}}


def test_the_colla_q_world_gives_two_arms():
    """Entropy and error are stable, independent properties; routing is re-drawn by the text."""
    v = red.reduce_family("granite", census())
    st = v["stability"]
    assert st["entropy"]["survives"] and st["rel_act"]["survives"] and st["rho_x_err"]["survives"]
    assert not st["freq"]["survives"]
    assert st["freq"]["r_split"] > 0.9 and st["freq"]["r_cross_half"] < 0.5 and st["freq"]["penalty"] > 0.15
    assert not v["entropy_redundant_with_rel_act"]
    assert v["collaq_claim"]["verdict"] == "REPLICATES"
    assert v["premise"]["premise"] and v["selector"] == "TWO_ARMS"


def test_entropy_that_is_the_error_again_is_redundant():
    v = red.reduce_family("granite", census(ent_tracks_rel=True))
    assert v["entropy_redundant_with_rel_act"] and v["selector"] == "REL_ACT_ONLY (entropy redundant)"


def test_entropy_that_resampling_moves_is_refused():
    v = red.reduce_family("granite", census(entropy="random"))
    assert not v["stability"]["entropy"]["survives"] and v["selector"] == "REL_ACT_ONLY (entropy refused)"


def test_entropy_that_the_text_moves_is_refused_even_though_resampling_does_not():
    v = red.reduce_family("granite", census(entropy="text"))
    st = v["stability"]["entropy"]
    assert st["r_split"] > 0.9 and st["penalty"] > 0.15 and not st["survives"]
    assert v["selector"] == "REL_ACT_ONLY (entropy refused)"


def test_error_that_the_text_moves_leaves_entropy_alone():
    v = red.reduce_family("granite", census(rel="text"))
    assert not v["stability"]["rel_act"]["survives"] and v["stability"]["entropy"]["survives"]
    assert v["selector"] == "ENTROPY_ONLY"


def test_nothing_survives():
    v = red.reduce_family("granite", census(entropy="random", rel="random"))
    assert v["selector"] == "NO_ONE_TEXT_SELECTOR"


def test_routing_that_survives_better_reverses_the_claim():
    v = red.reduce_family("granite", census(entropy="text", freq="stable"))
    assert v["collaq_claim"]["verdict"] == "REVERSED"


def test_no_premise_no_selector():
    """Mixtral's error is flat (P44-a P3 refuted): the selector is not written there, whatever the rankings do."""
    v = red.reduce_family("mixtral", census(family="mixtral"))
    assert not v["premise"]["premise"] and v["selector"].startswith("NOT_WRITTEN")
    assert v["selector_if_premise"] == "TWO_ARMS"


def test_a_family_p44_did_not_read_gets_its_premise_from_this_census():
    v = red.reduce_family("olmoe", census(family="olmoe"))
    assert "this census" in v["premise"]["source"] and v["premise"]["fraction_for_share"] is not None


def test_run_dir_missing_or_failed_selfcheck_is_not_read(tmp_path):
    (tmp_path / "census_granite.json").write_text(json.dumps(census(selfcheck=False)))
    (tmp_path / "census_olmoe.json").write_text(json.dumps(census(family="olmoe")))
    v = red.reduce(str(tmp_path))
    assert v["granite"]["verdict"] == "NOT_READ" and "selfcheck" in v["granite"]["reason"]
    assert v["mixtral"]["verdict"] == "NOT_READ" and v["olmoe"]["selector"] in {"TWO_ARMS", "NOT_WRITTEN (no per-expert premise)"}
    md = red.render_md(v)
    assert "| granite | — | NOT_READ" in md and "| olmoe | entropy |" in md


def test_underrouted_experts_are_excluded():
    c = census()
    for r in c["rows"]:
        if r["layer"] == 0 and r["expert"] < 5 and r["text"] == "c4val1" and r["half"] == 1:
            r["rows"] = 3
    cs = red.censuses(c)
    keep = red._eligible(cs, (("wikitext", 0), ("c4val1", 1)), "entropy")
    assert not any((0, e) in keep for e in range(5)) and (0, 5) in keep


def test_frequency_hot_sets_are_the_packages():
    c = census()
    k = 3
    hs = red.freq_hot_sets(c, "wikitext", "full", k)
    sig = red.expert_signals(c, "wikitext", "full")
    for layer in range(L):
        want = sorted(range(E), key=lambda e: (-sig[(layer, e)]["rows"], e))[:k]
        assert hs[layer] == want


def test_cosine_cannot_tell_order_from_noise():
    """Two INDEPENDENT positive vectors near a constant: cosine ~ 1, rank correlation ~ 0. Colla-Q's Table 4 reports
    the cosine; the rule reads Spearman."""
    rng = random.Random(3)
    a = [0.9 + 0.1 * rng.random() for _ in range(400)]
    b = [0.9 + 0.1 * rng.random() for _ in range(400)]
    assert red.cosine(a, b) > 0.99 and abs(red.spearman(a, b)) < 0.15


def test_ranks_average_ties_and_spearman_edges():
    assert red.ranks([3.0, 1.0, 3.0, 2.0]) == [3.5, 1.0, 3.5, 2.0]
    assert red.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert red.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert red.spearman([1, 2], [1, 2]) is None and red.spearman([1, 1, 1], [1, 2, 3]) is None


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    d = {i: 0.1 * i for i in range(10)}
    a, b = red.bootstrap_ci(d), red.bootstrap_ci(d)
    assert a == b and a[0] < red.mean_over(d) < a[1]


def test_chance_jaccard():
    assert red.chance_jaccard(10, 100) == pytest.approx(1.0 / 19.0)
