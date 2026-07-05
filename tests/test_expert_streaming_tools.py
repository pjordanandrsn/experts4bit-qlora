"""CPU-safe tests for the expert-streaming profile summarizer + manifest (no torch, no GPU).

The profiler module itself (experts4bit_qlora.expert_profile) needs a live offloaded model, so it
is exercised on the pod; here we pin the summarizer's concentration/decision math and the
profile-job manifest, which are the parts that decide whether hot-static gets built.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import make_expert_streaming_manifest as prof_manifest  # noqa: E402
import summarize_expert_streaming as summ  # noqa: E402


def _profile(layer_stalls, routing):
    """layer_stalls: {layer_id: total_ms}; routing: {(layer,expert): tokens}. per_expert_bytes
    fixed so budget math is checkable."""
    layers = {lid: {"row": "layer", "layer_id": lid, "storage_mode": "int8", "num_experts": 8,
                    "h2d_ms_total": ms, "h2d_bytes": int(1e8), "per_expert_bytes": int(1e7)}
              for lid, ms in layer_stalls.items()}
    experts = [{"row": "expert", "layer_id": lid, "expert_id": e, "hits": tok, "tokens_routed": tok}
               for (lid, e), tok in routing.items()]
    return layers, experts


def test_concentration_and_decision_diffuse():
    # 10 layers, 8 experts each, uniform tokens + uniform layer stall -> diffuse, DO NOT build.
    layer_stalls = {lid: 100.0 for lid in range(10)}
    routing = {(lid, e): 50 for lid in range(10) for e in range(8)}
    layers, experts = _profile(layer_stalls, routing)
    pairs = summ.build_pairs(layers, experts)
    conc, total = summ.concentration(pairs, "projected_stall_ms")
    shares = {pct: s for pct, _, s in conc}
    assert total > 0
    assert shares[20] < 0.6  # uniform -> top 20% holds ~its proportional share, not >=60%
    assert "DO NOT build" in summ.decide(pairs)


def test_concentration_and_decision_hot():
    # One layer carries almost all stall and its tokens sit on 1-2 experts -> concentrated, BUILD.
    layer_stalls = {lid: (1000.0 if lid == 0 else 1.0) for lid in range(10)}
    routing = {}
    for lid in range(10):
        for e in range(8):
            routing[(lid, e)] = 900 if (lid == 0 and e == 0) else 1
    layers, experts = _profile(layer_stalls, routing)
    pairs = summ.build_pairs(layers, experts)
    # layer 0 / expert 0 should dominate projected stall.
    hottest = max(pairs, key=lambda p: p["projected_stall_ms"])
    assert (hottest["layer_id"], hottest["expert_id"]) == (0, 0)
    assert "BUILD hot-static" in summ.decide(pairs)


def test_budget_projection_monotonic():
    layer_stalls = {lid: 100.0 for lid in range(4)}
    routing = {(lid, e): (e + 1) * 10 for lid in range(4) for e in range(8)}
    layers, experts = _profile(layer_stalls, routing)
    pairs = summ.build_pairs(layers, experts)
    total = sum(v["h2d_ms_total"] for v in layers.values())
    rows = summ.budget_projection(pairs, total)
    covered = [frac for _, _, _, frac in rows]
    assert covered == sorted(covered)  # more budget never covers less
    assert all(0.0 <= f <= 1.0 for f in covered)


def test_decision_insufficient_data():
    # No staging stall recorded (e.g. a non-CUDA or non-offload profile) -> explicit message.
    layers, experts = _profile({0: 0.0}, {(0, 0): 10})
    assert "INSUFFICIENT DATA" in summ.decide(summ.build_pairs(layers, experts))


def test_profile_manifest_ids_and_env():
    jobs = [prof_manifest._train_job("int8"), prof_manifest._train_job("nf4"),
            prof_manifest._decode_job("int8"), prof_manifest._decode_job("nf4")]
    ids = {j["job_id"] for j in jobs}
    assert "profile_olmoe_int8_offload_train_seed1337_steps100" in ids
    assert "profile_olmoe_nf4_offload_decode_repeat5" in ids
    assert all("qwen3" not in i for i in ids)  # OLMoE first, not Qwen3
    train = prof_manifest._train_job("int8")
    assert "OFFLOAD_EXPERTS=1" in train["command"]  # profiling the offload path
    assert any(a.startswith("E4B_EXPERT_PROFILE=") for a in train["command"])
