"""Lane P70's reducer applies the registration literally (bench/p70/p70_reduce.py; bench/p70/P70-PREREG.md): a clean run
reads, every validity gate voids the read when it fails, the floor is the two same-router-function samples, and the
reducer's restated counts are the scorer's own on every registered pass."""
import importlib.util
import json
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

REPO = pathlib.Path(__file__).resolve().parents[1]
for sub in ("bench", "bench/p59", "bench/p64", "bench/p70"):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load(name, sub):
    spec = importlib.util.spec_from_file_location(name, REPO / "bench" / sub / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("p70_reduce", "p70")
V, ROWS, S = 64, 2, 6                     # tiny vocabulary, rows, decode positions stored per row
BUILD = {"moe_layers": 3, "top_k": 2, "int4_expert_layers": 3, "int4_attn_projections": 5, "fuse_router_epilogue_n": 3,
         "counters": {"router_hooks": 3}}
STEPS = ROWS * (R.TOKENS - R.PREFIX)


def _write_run(tmp, perturb=None, counts_override=None, rep_noise=0.0, prefill_noise=None):
    """A run dir the reducer reads: logits per pass (ep32 base; ep16 a small perturbation; floors bigger ones)."""
    out = tmp / "out"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(BUILD, open(out / "build.json", "w"))
    json.dump({"fingerprint": "sha256:test", "licensed": False}, open(tmp / "pack.json", "w"))
    g = torch.Generator().manual_seed(70)
    rows = torch.randint(0, V, (ROWS, R.TOKENS), generator=g).tolist()
    for text in R.TEXTS:
        json.dump({"prompts": rows}, open(tmp / f"prompts_{text}.json", "w"))
        base_dec = torch.randn(ROWS, S, V, generator=g)
        base_pre = torch.randn(ROWS, V, generator=g)
        scale = {"ep32": 0.0, "ep32_rep": rep_noise, "ep16": 0.02, "ep32_pc96": 0.05, "ep32_pc384": 0.05, "ep32_pc64": 0.06}
        if perturb:
            scale.update(perturb)
        for p in R.PASSES:
            dec = base_dec + scale[p] * torch.randn(ROWS, S, V, generator=g)
            pre = base_pre.clone()
            if prefill_noise and p in prefill_noise:
                pre = pre + 0.1
            torch.save({"prefill_last": pre, "decode": dec, "prefix": R.PREFIX, "prompts_sha256": "x", "pass": p, "text": text},
                       out / f"{p}.{text}.pt")
            counts = R.expected_counts(p, BUILD, STEPS)
            if counts_override and p in counts_override:
                counts.update(counts_override[p])
            json.dump({"pass": p, "text": text, "decode_steps": STEPS, "counts": counts, "prompts_sha256": "x", "rows": ROWS,
                       "prefill_chunk": R.CHUNK[p], "wall_s": 1.0, "toggle_state_during_pass": {"CAST_WEIGHTS": p in R.CAST}},
                      open(out / f"{p}.{text}.census.json", "w"))
    return tmp


def test_a_clean_run_reads_and_the_floor_is_the_two_same_function_samples(tmp_path):
    rep, lines = R.reduce(str(_write_run(tmp_path)))
    assert rep["validity"]["status"] == "VALID", rep["validity"]["problems"]
    for text in R.TEXTS:
        T = rep["texts"][text]
        assert len(T["floor"]["samples"]) == 2
        assert T["floor"]["F"] == pytest.approx(sum(T["floor"]["samples"]) / 2)
        assert T["stats"]["class"] == "BELOW FLOOR"
        assert T["p64_floor_informational"]["ratio_to_F"] > 0
    assert rep["verdict"] == "INDISTINGUISHABLE"
    assert any("VERDICT (the cast, G): INDISTINGUISHABLE" in ln for ln in lines)


def test_a_large_cast_effect_is_distinguishable(tmp_path):
    rep, _ = R.reduce(str(_write_run(tmp_path, perturb={"ep16": 0.5})))
    assert rep["validity"]["status"] == "VALID"
    assert all(rep["texts"][t]["stats"]["class"] == "DISTINGUISHABLE" for t in R.TEXTS)
    assert rep["verdict"] in ("DISTINGUISHABLE, NOT MATERIAL", "MATERIAL")


@pytest.mark.parametrize("kw,needle", [
    ({"rep_noise": 1e-3}, "ep32 || ep32_rep"),
    ({"prefill_noise": {"ep16"}}, "prefill-last logits of ep16"),
    ({"counts_override": {"ep16": {"router_le64_bf16_decode": 0, "router_le64_fp32_decode": 3 * STEPS}}}, "ep16/"),
    ({"counts_override": {"ep32": {"router_gt64_bf16_prefill": 0}}}, "ep32/"),
])
def test_every_validity_gate_voids_the_read(tmp_path, kw, needle):
    rep, _ = R.reduce(str(_write_run(tmp_path, **kw)))
    assert rep["validity"]["status"] == "NOTHING READ"
    assert rep["verdict"] == "NOTHING READ (validity)"
    assert any(needle in p for p in rep["validity"]["problems"]), rep["validity"]["problems"]


def test_a_router_left_unhooked_voids_the_read(tmp_path):
    d = _write_run(tmp_path)
    b = dict(BUILD, counters={"router_hooks": 2})
    json.dump(b, open(d / "out" / "build.json", "w"))
    rep, _ = R.reduce(str(d))
    assert rep["validity"]["status"] == "NOTHING READ" and not rep["validity"]["router_hooks"]


def test_no_out_dir_is_rc_43(tmp_path):
    assert R.main([str(tmp_path)]) == 43


def test_the_reducers_counts_are_the_scorers_on_every_registered_pass():
    """p70_reduce.expected_counts restates kl_router.expected_counts; the two must agree pass by pass."""
    pytest.importorskip("experts4bit_qlora")
    pytest.importorskip("kl_b16")
    kr = _load("kl_router", "p70")
    assert set(kr.PASSES) == set(R.PASSES)
    for p, (cast, attn, chunk) in kr.PASSES.items():
        assert R.CHUNK[p] == chunk and (p in R.CAST) == cast and attn is False
        assert kr.expected_counts(p, BUILD, STEPS) == R.expected_counts(p, BUILD, STEPS), p
    assert (kr.PREFIX, kr.TOKENS) == (R.PREFIX, R.TOKENS)
    assert set(p for p, _ in kr.SCHEDULE) == set(kr.PASSES)


def test_the_primary_pair_and_the_floor_keep_one_router_function_in_the_prefill():
    """The registration's design point: ep32, ep16 and both floor samples prefill in > 64-row forwards (the original
    router); only the informational pc64 pass prefills through the fused router."""
    over = {p for p, c in R.CHUNK.items() if c <= R.MAX_DECODE_ROWS}
    assert over == {"ep32_pc64"}
    assert set(R.FLOOR_PASSES) <= set(R.PASSES) - over
