"""Lane P115 Phase C's SANE instrument builds and scores both Phase C families (bench/p115/PREREG-p115.md, Amendment 2;
e4b#1313).

SANE runs Phase B's teacher-forced instrument (``p115_quality.measure_phase`` over ``p110_box.paged_pass``) on
gpt-oss-20b and Qwen3.6-35B-A3B. Neither family has been through that instrument: gpt-oss's attention carries sinks and
alternates sliding with full layers, and Qwen3.6 is a hybrid whose Gated DeltaNet layers keep no K/V (the pool holds its
attention layers only). The pre-registration says a model the instrument cannot build fails SANE, so FLIP_HELD; a
harness gap would then hold the flip for a reason that is not the fused stack's. This test runs both phases, R then ON,
on tiny models of each family on CPU (P108's stand-in attention: SDPA over the pool's own dequantized K/V, which drops
the sinks, so the scores are not the real kernel's -- what is checked is that the instrument builds, steps every
window and writes records the Phase C reducer reads), then hands the records to ``p115c_reduce.model_gates``.
"""
import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

REPO = pathlib.Path(__file__).resolve().parents[1]
for sub in ("p115", "p110", "p108", "p97", "p109"):
    d = str(REPO / "bench" / sub)
    if d not in sys.path:
        sys.path.insert(0, d)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "bench" / "p115" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gpt_oss(layers=2):
    pytest.importorskip("transformers.models.gpt_oss", reason="needs transformers with gpt-oss")
    from transformers import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
    cfg = GptOssConfig(vocab_size=256, hidden_size=128, intermediate_size=64, num_hidden_layers=layers, head_dim=32,
                       num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2,
                       max_position_embeddings=512, sliding_window=16)
    torch.manual_seed(0)
    return GptOssForCausalLM(cfg).float().eval()


def _qwen3_5_moe(layers=4):
    pytest.importorskip("transformers.models.qwen3_5_moe", reason="needs transformers with Qwen3.5-MoE")
    from hybrid_reference import reference_modeling
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    m = reference_modeling("qwen3_5_moe")
    cfg = Qwen3_5MoeTextConfig(vocab_size=256, hidden_size=128, num_hidden_layers=layers, num_attention_heads=4,
                               num_key_value_heads=2, head_dim=32, num_experts=4, num_experts_per_tok=2,
                               moe_intermediate_size=64, shared_expert_intermediate_size=64,
                               max_position_embeddings=512)
    torch.manual_seed(0)
    return m.Qwen3_5MoeForCausalLM(cfg).float().eval()


P, C, GROUP = 24, 6, 4
UNPADDED = {"device_grouping": True}


@pytest.mark.parametrize("tag,build", [("gptoss", _gpt_oss), ("qw36", _qwen3_5_moe)])
def test_sane_builds_and_scores_the_phase_c_family(tmp_path, tag, build):
    q = _load("p115_quality")
    r = _load("p115c_reduce")
    n = r.SANE_WINDOWS
    g = torch.Generator().manual_seed(7)
    windows = {"wikitext": [torch.randint(0, 256, (P + C,), generator=g).tolist() for _ in range(n)]}
    model = build()
    kw = dict(prompt=P, cont=C, chunk=P, floor_chunk=P, group=GROUP, device="cpu", ref_dir=str(tmp_path / "ref"),
              ref_kw=UNPADDED, stand_in=True)
    off = q.measure_phase(model, windows, phase="off", arms=("R",), **kw)
    on = q.measure_phase(model, windows, phase="on", arms=("ON",), **kw)
    assert off["arms"] == ["R"] and on["arms"] == ["ON"]
    assert [x["window"] for x in off["per_window"]["wikitext"]["R"]] == list(range(n))
    assert [x["window"] for x in on["per_window"]["wikitext"]["ON"]] == list(range(n))
    assert all("kl" in x for x in on["per_window"]["wikitext"]["ON"])
    calls = [e["decode_calls"] for e in on["engagement"]["wikitext"]["ON"]]
    assert all(c > 0 for c in calls), calls                     # the paged decode attention ran in every group
    # the same model scored twice through the same arithmetic: ON reproduces R, so SANE holds with no fold applied
    on_nll = [x["nll"] for x in on["per_window"]["wikitext"]["ON"]]
    assert on_nll == [x["nll"] for x in off["per_window"]["wikitext"]["R"]]
    model_id, rev = r.MODELS[tag]
    base = {"model_tag": tag, "model": model_id, "revision": rev, "status": "ok"}
    recs = {f"sane_{tag}_off": {**base, **off}, f"sane_{tag}_on": {**base, **on}}
    gates, why, rep = r.model_gates(tag, recs, new=4)
    assert gates["SANE"], why
    assert rep["sane"]["n"] == n and rep["sane"]["bias"] == 0.0 and rep["sane"]["argmax_agree"] == 1.0


def test_sane_fails_when_a_window_is_missing(tmp_path):
    r = _load("p115c_reduce")
    n = r.SANE_WINDOWS
    base = {"model_tag": "qw36", "status": "ok"}
    off = {"per_window": {"wikitext": {"R": [{"window": w, "nll": 2.0} for w in range(n)]}}}
    on = {"per_window": {"wikitext": {"ON": [{"window": w, "nll": 2.0, "argmax_agree": 1.0} for w in range(n - 1)]}}}
    gates, why, _ = r.model_gates("qw36", {"sane_qw36_off": {**base, **off}, "sane_qw36_on": {**base, **on}}, new=4)
    assert not gates["SANE"] and "short" in why["SANE"]
