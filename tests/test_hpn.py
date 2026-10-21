"""bench/hpn (HPN-DESIGN.md): the fp32 reference reader's arithmetic and the decomposition's calls, on HP2's real receipt files."""

import json
import math
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench" / "hpn"))
sys.path.insert(0, str(REPO / "bench" / "hp1"))                    # hp1_lib / hp1_eval, for tests that run alone (-k)
import hpn_reduce as R  # noqa: E402

RCPT = REPO / "bench" / "hp2" / "receipts" / "hp2-5090-1"
REAL_ARM = json.loads((RCPT / "qwen3hp2_e4b_hp2cal_e4b_k00_s3407.json").read_text())       # e4b's real one-step calibration arm
REAL_EVAL = json.loads((RCPT / "hp2" / "eval_cal.json").read_text())                         # the evaluator's real zero-adapter pass
E4B_BASE = next(b for b in REAL_EVAL["bases"] if b["dir"].endswith("base_e4b_qwen3hp2"))
I_REAL = [c for c in REAL_ARM["eval_curve"] if c["step"] == 0][0]["row_losses"]            # 300 real in-loop step-0 rows
E_REAL = E4B_BASE["zero_adapter"]["row_losses"]                                               # 300 real evaluator rows


def _write_depth(tmp, L, In, E, F):
    """One depth's files in the real formats: two receipts (eval_curve step 0), two evaluator outputs (one base), one F output."""
    d = tmp / f"L{L}"
    d.mkdir()
    for k, rows in enumerate(In):
        arm = dict(REAL_ARM, eval_curve=[dict(c, row_losses=rows) if c["step"] == 0 else c for c in REAL_ARM["eval_curve"]])
        (d / f"arm{k}.json").write_text(json.dumps(arm))
    for k, rows in enumerate(E):
        base = dict(E4B_BASE, zero_adapter=dict(E4B_BASE["zero_adapter"], row_losses=rows))
        (d / f"eval{k}.json").write_text(json.dumps(dict(REAL_EVAL, bases=[base])))
    (d / "f.json").write_text(json.dumps({"reader": "F", "runs": F}))
    return {"L": L, "inloop": [f"L{L}/arm0.json", f"L{L}/arm1.json"], "eval": [f"L{L}/eval0.json", f"L{L}/eval1.json"], "fref": f"L{L}/f.json"}


def _synth(n, se, si, seed, base=None):
    """F = a real-shaped base; E = F + N(0, se); I = F + N(0, si), independent: the side share is about se^2 / (se^2 + si^2)."""
    rng = random.Random(seed)
    F = base or [round(1.9 + rng.gauss(0, 0.2), 5) for _ in range(n)]
    E = [round(f + rng.gauss(0, se), 5) for f in F]
    In = [round(f + rng.gauss(0, si), 5) for f in F]
    return [In, list(In)], [E, list(E)], [F, list(F)]


def _write_unsloth(tmp, L, In, E, F, shift_i, shift_e, shift_f, noise=0.0):
    """Stage 1b's Unsloth readers at depth L (one run each): e4b's rows shifted by a constant per reader (plus optional per-row
    noise on Unsloth's in-loop rows), so delta0_E = -shift_e + shift_i and delta0_F = -shift_f + shift_i up to that noise."""
    rng = random.Random(L)
    jit = [rng.gauss(0, noise) for _ in In[0]]
    sh = lambda rows, c, j=None: [round(x + c + (j[k] if j else 0.0), 5) for k, x in enumerate(rows)]   # noqa: E731
    In = [sh(In[0], 0.0, jit)]
    d = _write_depth(tmp / f"L{L}", "u", [sh(In[0], shift_i)] * 2, [sh(E[0], shift_e)] * 2, [sh(F[0], shift_f)] * 2)
    return {"inloop": [f"L{L}/{d['inloop'][0]}"], "eval": [f"L{L}/{d['eval'][0]}"], "fref": f"L{L}/{d['fref']}"}


def _run(tmp, depth_noise, n=300, shifts=(0.0019, -0.0008, 0.0019), unsloth=True, noise=0.0):
    """shifts = (in-loop, E, F) offsets of Unsloth's readers: by default delta0_E = 0.0027 (HP2's size) and delta0_F = 0 (not MOOT)."""
    spec = {"framework": "e4b", "free_mib_at_claim": 10600, "depths": []}
    for k, (L, se, si) in enumerate(depth_noise):
        In, E, F = _synth(n, se, si, seed=k + 1)
        d = _write_depth(tmp, L, In, E, F)
        if unsloth:
            d["unsloth"] = _write_unsloth(tmp, L, In, E, F, *shifts, noise=noise)
        spec["depths"].append(d)
    (tmp / "spec.json").write_text(json.dumps(spec))
    out = tmp / "hpn.json"
    rc = R.main([str(tmp / "spec.json"), "--out", str(out)])
    return rc, (json.loads(out.read_text()) if out.exists() else None)


def test_the_real_receipts_load_as_the_readers_see_them(tmp_path):
    """I from the receipt's eval_curve step 0 (as hp1_eval.receipt_arm reads it), E from the evaluator's one base."""
    spec = _write_depth(tmp_path, 2, [I_REAL, I_REAL], [E_REAL, E_REAL], [E_REAL, E_REAL])
    assert R._inloop_rows(tmp_path / spec["inloop"][0]) == I_REAL and R._eval_rows(tmp_path / spec["eval"][0]) == E_REAL
    rec = R.depth_record([I_REAL, I_REAL], [E_REAL, E_REAL], [E_REAL, E_REAL])           # F := E: all of g sits on the I side
    assert rec["rows"] == 300 and rec["call"] == "I" and rec["e"]["sd"] == 0.0
    assert rec["g"]["sd"] == pytest.approx(0.030206, abs=2e-4)               # HP2's real e4b per-row sd (b_e4b's sd)


@pytest.mark.parametrize("se, si, call", [(0.03, 0.005, "E"), (0.005, 0.03, "I"), (0.02, 0.02, "comparable")])
def test_the_side_call_reads_the_variance_share(se, si, call):
    In, E, F = _synth(300, se, si, seed=3)
    assert R.depth_record(In, E, F)["call"] == call


def test_the_call_boundaries_are_a_variance_ratio_of_two():
    assert R._call(2 / 3) == "E" and R._call(0.66) == "comparable" and R._call(1 / 3) == "I" and R._call(0.34) == "comparable"


def test_e_dominant_at_the_two_deepest_depths_leads_to_stage_two(tmp_path):
    rc, rec = _run(tmp_path, [(2, 0.004, 0.002), (4, 0.008, 0.003), (8, 0.015, 0.004)])
    assert rc == 0 and rec["outcome"] == "STAGE2" and rec["two_deepest_calls"] == {"4": "E", "8": "E"}


def test_comparable_at_the_two_deepest_depths_is_a_stop(tmp_path):
    rc, rec = _run(tmp_path, [(2, 0.003, 0.003), (4, 0.006, 0.006), (8, 0.012, 0.012)])
    assert rec["outcome"] == "STOP" and rec["N2_comparable"] and "no lower-noise reference" in rec["reason"]


def test_a_call_that_flips_between_the_two_deepest_depths_is_depth_dependent(tmp_path):
    rc, rec = _run(tmp_path, [(2, 0.004, 0.002), (4, 0.008, 0.002), (8, 0.004, 0.015)])
    assert rec["outcome"] == "STOP_DEPTH_DEPENDENT" and rec["two_deepest_calls"] == {"4": "E", "8": "I"}


def test_the_decision_uses_the_two_DEEPEST_depths_not_the_shallow_ones(tmp_path):
    rc, rec = _run(tmp_path, [(2, 0.002, 0.008), (4, 0.010, 0.003), (8, 0.020, 0.005)])  # L2 reads I; 4 and 8 read E
    assert rec["depths"]["2"]["call"] == "I" and rec["outcome"] == "STAGE2"


def test_a_base_term_beyond_the_bound_is_moot_whatever_the_side_call(tmp_path):
    """Stage 1b is read first: E-dominant at both deciding depths, but F leaves |delta0_F| = 0.004 > 0.0025, so stage 2 is moot."""
    rc, rec = _run(tmp_path, [(2, 0.004, 0.002), (4, 0.008, 0.003), (8, 0.015, 0.004)], shifts=(0.004, 0.0, 0.0))
    assert rec["two_deepest_calls"] == {"4": "E", "8": "E"} and rec["outcome"] == "MOOT"
    assert rec["base_term"]["8"]["delta0_F"] == pytest.approx(0.004, abs=1e-5)


def test_a_small_absolute_delta0_f_is_still_moot_when_f_did_not_move_it(tmp_path):
    """The vacuous-pass case from review: at shallow depth |delta0_F| = 0.001 sits under the 48-layer 0.0025, but F moved nothing
    (ratio 1.0 > 0.93), so 1b reads MOOT, not a pass."""
    rc, rec = _run(tmp_path, [(4, 0.008, 0.003), (8, 0.015, 0.004)], shifts=(0.001, 0.0, 0.0))
    b = rec["base_term"]["8"]
    assert b["abs_delta0_F"] == pytest.approx(0.001, abs=1e-5) and b["abs_delta0_F"] < 0.0025
    assert b["ratio_F_over_E"] == pytest.approx(1.0, abs=1e-3)
    assert rec["outcome"] == "MOOT" and b["shift_call"] == "no measurable shift"


def test_a_delta0_e_inside_two_se_reads_indeterminate_not_a_pass(tmp_path):
    """|delta0_E| = 0.0005 with per-row noise 0.03 (SE ~0.0017): dividing by it means nothing, so 1b is INDETERMINATE there."""
    rc, rec = _run(tmp_path, [(4, 0.008, 0.003), (8, 0.015, 0.004)], shifts=(0.0005, 0.0, 0.0005), noise=0.03)
    b = rec["base_term"]["8"]
    assert b["abs_delta0_E"] < 2 * b["se_delta0_E"] and b["read_1b"] == "INDETERMINATE" and rec["outcome"] == "INDETERMINATE"


def test_a_ratio_under_the_bound_without_a_measurable_shift_is_indeterminate():
    """Ruled: stage 2 never opens on a move inside its own noise. F moves delta0 from -0.0100 to -0.0085 (ratio 0.85 <= 0.93), but
    the shift (0.0015) sits inside 2 x its paired SE (zero-mean per-row noise of sd 0.02 -> SE ~0.00115)."""
    rng = random.Random(4)
    n = 300
    base = [1.9 + rng.gauss(0, 0.2) for _ in range(n)]
    z = [rng.gauss(0, 0.02) for _ in range(n)]
    z = [x - sum(z) / n for x in z]                                  # zero-mean: the means are exact, the noise sits in the SEs
    e4b = {"I": base, "E": base, "F": base}
    uns = {"I": [x + 0.01 for x in base], "E": base, "F": [x + 0.0015 + zk for x, zk in zip(base, z)]}
    b = R.base_term(e4b, uns)
    assert b["delta0_E"] == pytest.approx(0.01, abs=1e-9) and b["ratio_F_over_E"] == pytest.approx(0.85, abs=1e-6)
    assert b["shift_call"] == "no measurable shift" and b["read_1b"] == "INDETERMINATE"
    uns["F"] = [x + 0.0015 for x in base]                            # the same move without the noise: a measurable shift, OPEN
    b = R.base_term(e4b, uns)
    assert b["shift_call"] == "SHIFT" and b["read_1b"] == "OPEN"


def test_the_ratio_bound_is_derived_from_g1s_half_floor_and_hp2s_delta0():
    assert R.MOOT_RATIO == round(0.0025 / 0.002691, 2) == 0.93 and R.GUARD_Z == 2.0


def test_the_base_term_arithmetic_is_delta0_under_each_reader(tmp_path):
    _, rec = _run(tmp_path, [(4, 0.008, 0.003), (8, 0.015, 0.004)], shifts=(0.0019, -0.0008, 0.0019))
    b = rec["base_term"]["8"]
    assert b["inloop_term"] == pytest.approx(-0.0019, abs=1e-5) and b["base_term_E"] == pytest.approx(0.0008, abs=1e-5)
    assert b["delta0_E"] == pytest.approx(0.0027, abs=1e-5) and b["delta0_F"] == pytest.approx(0.0, abs=1e-5)
    assert b["shift_E_to_F"] == pytest.approx(-0.0027, abs=1e-5) and b["shift_call"] == "SHIFT" and b["read_1b"] == "OPEN"


def test_no_decision_without_stage_1b(tmp_path):
    rc, rec = _run(tmp_path, [(2, 0.004, 0.002), (4, 0.008, 0.003), (8, 0.015, 0.004)], unsloth=False)
    assert rec["outcome"] == "INCOMPLETE" and "1b" in rec["reason"]


def test_one_depth_is_insufficient(tmp_path):
    rc, rec = _run(tmp_path, [(2, 0.004, 0.002)])
    assert rec["outcome"] == "INSUFFICIENT"


def test_the_drivers_climb_stop_reason_reaches_insufficient_depth(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec = {"framework": "e4b", "climb_stop": "df did not settle before L8 (two reads 30 s apart within 0.5 GB, 600 s)", "depths": []}
    for k, (L, se, si) in enumerate([(2, 0.004, 0.002), (4, 0.008, 0.003)]):
        spec["depths"].append(_write_depth(tmp_path, L, *_synth(300, se, si, seed=k + 1)))
    spec_path.write_text(json.dumps(spec))
    assert R.main([str(spec_path), "--out", str(tmp_path / "o.json")]) == 0
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["outcome"] == "INSUFFICIENT_DEPTH" and "df did not settle before L8" in out["reason"] and out["climb_stop"]


def test_a_decision_below_the_l8_floor_waits_for_depth(tmp_path):
    rc, rec = _run(tmp_path, [(2, 0.004, 0.002), (4, 0.008, 0.003)])                       # only L2 and L4 fit
    assert rec["outcome"] == "INSUFFICIENT_DEPTH" and rec["deepest_fit"] == 4


def test_where_e_and_i_correlate_the_call_reads_what_the_swap_leaves():
    """N3 fails: e and i share a large component c, which cancels in g = e - i. The share sees two equal sides ('comparable'), but
    swapping E for F would leave var(i), which is far MORE than var(g): the swap makes it worse, so the call is I-dominant."""
    rng = random.Random(9)
    F = [1.9 + rng.gauss(0, 0.2) for _ in range(300)]
    c = [rng.gauss(0, 0.02) for _ in range(300)]
    E = [round(f + ck + rng.gauss(0, 0.004), 5) for f, ck in zip(F, c)]
    In = [round(f + ck + rng.gauss(0, 0.004), 5) for f, ck in zip(F, c)]
    rec = R.depth_record([In, In], [E, E], [[round(f, 5) for f in F]] * 2)
    assert R._call(rec["share_e"]) == "comparable"                                              # what the share alone would say
    assert abs(rec["corr_e_i"]) > R.N3_RHO and rec["call_basis"] == "1 - var(i)/var(g)" and rec["call"] == "I"


def test_n1_needs_sd_g_rising_with_depth_and_below_003_at_l2(tmp_path):
    _, rec = _run(tmp_path, [(2, 0.003, 0.003), (4, 0.006, 0.006), (8, 0.012, 0.012)])
    assert rec["N1"] is True
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    _, rec = _run(tmp2, [(2, 0.003, 0.003), (4, 0.002, 0.002), (8, 0.012, 0.012)])            # falls at L4
    assert rec["N1"] is False


def test_n0_reads_each_readers_rerun(tmp_path):
    In, E, F = _synth(50, 0.01, 0.01, seed=5)
    rec = R.depth_record(In, E, F)
    assert rec["n0_holds"] == {"I": True, "E": True, "F": True}
    I2 = [In[0], [x + (0.00003 if k == 7 else 0.0) for k, x in enumerate(In[0])]]               # one row moves 3e-5 between I's runs
    rec = R.depth_record(I2, E, F)
    assert rec["n0_holds"]["I"] is False and rec["n0_max_abs_rerun"]["I"] == pytest.approx(3e-5, abs=1e-9)


def test_the_48_layer_figure_is_labelled_an_extrapolation_with_a_range(tmp_path):
    _, rec = _run(tmp_path, [(2, 0.003, 0.003), (4, 0.006, 0.006), (8, 0.012, 0.012), (16, 0.024, 0.024)])
    x = rec["extrapolation"]["g"]
    assert x["label"].startswith("EXTRAPOLATION") and x["at_L"] == 48 and x["loo_range"][0] <= x["value"] <= x["loo_range"][1]
    assert x["slope"] == pytest.approx(1.0, abs=0.15)                                           # sd doubles with depth here


def test_the_fit_recovers_a_power_law():
    v, p = R._fit([2, 4, 8, 16], [0.01 * math.sqrt(L / 2) for L in (2, 4, 8, 16)])
    assert p == pytest.approx(0.5, abs=1e-9) and v == pytest.approx(0.01 * math.sqrt(24), rel=1e-9)


def test_unusable_inputs_refuse(tmp_path):
    spec = _write_depth(tmp_path, 2, [I_REAL, I_REAL], [E_REAL, E_REAL], [E_REAL])         # F with one run only
    (tmp_path / "spec.json").write_text(json.dumps({"depths": [spec]}))
    assert R.main([str(tmp_path / "spec.json"), "--out", str(tmp_path / "o.json")]) == 2
    with pytest.raises(ValueError, match="row count"):
        R.depth_record([I_REAL, I_REAL], [E_REAL[:299], E_REAL[:299]], [E_REAL, E_REAL])
    bad = dict(REAL_EVAL)                                                                        # two bases: refused
    (tmp_path / "two.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="exactly one base"):
        R._eval_rows(tmp_path / "two.json")


# ------------------------------------------------------------------------------------------------------------- hpn_read
torch = pytest.importorskip("torch")


def _hpn_read():
    import importlib
    return importlib.import_module("hpn_read")


def test_f32_experts_match_a_float64_reference_and_the_bf16_evaluator_does_not():
    """F's expert forward is the evaluator's with the bf16 casts removed: it tracks a float64 computation to fp32 precision, and the
    evaluator's own EagerExperts (bf16) does not."""
    hr = _hpn_read()
    import hp1_lib
    g = torch.Generator().manual_seed(0)
    T, H, M, NE, K = 6, 16, 8, 4, 2
    W = {("gate_up", e): torch.randn(2 * M, H, generator=g) * 0.3 for e in range(NE)}
    W.update({("down", e): torch.randn(H, M, generator=g) * 0.3 for e in range(NE)})
    x = torch.randn(T, H, generator=g)
    idx = torch.stack([torch.randperm(NE, generator=g)[:K] for _ in range(T)])
    wts = torch.softmax(torch.randn(T, K, generator=g), -1)
    act = torch.nn.functional.silu

    def ref64():
        out = torch.zeros(T, H, dtype=torch.float64)
        for t in range(T):
            for j in range(K):
                e = int(idx[t, j])
                h = W[("gate_up", e)].double() @ x[t].double()
                gate, up = h.chunk(2)
                out[t] += wts[t, j].double() * (W[("down", e)].double() @ (act(gate) * up))
        return out
    f32 = hr.F32Experts(lambda kind, e: W[(kind, e)].float(), NE, act)(x, idx, wts)
    bf = hp1_lib.EagerExperts(lambda kind, e: W[(kind, e)].to(torch.bfloat16), NE, act)(x, idx, wts)
    r = ref64()
    err32, errbf = (f32.double() - r).abs().max().item(), (bf.double() - r).abs().max().item()
    assert err32 < 1e-5 and errbf > 100 * err32


def test_f32_linear_is_an_fp32_matmul():
    hr = _hpn_read()
    g = torch.Generator().manual_seed(1)
    w, x = torch.randn(5, 7, generator=g), torch.randn(3, 7, generator=g)
    assert torch.equal(hr.F32Linear(lambda: w)(x), x @ w.t())


def test_deq32_upcasts_bf16_slots_exactly():
    hr = _hpn_read()
    w = torch.randn(4, 4).to(torch.bfloat16)
    out = hr.deq32({"fmt": "bf16", "w": w}, "cpu")
    assert out.dtype == torch.float32 and torch.equal(out, w.float())


def test_deq32_keeps_nf4_in_fp32_where_the_evaluator_rounds_to_bf16():
    """The same NF4 bytes through both dequantizers: deq32 is code x absmax in fp32; the evaluator's dequant_raw is that, rounded to
    bf16. They agree to bf16 precision and differ below it."""
    bnbf = pytest.importorskip("bitsandbytes.functional")
    hr = _hpn_read()
    import hp1_lib
    w = torch.randn(64, 64) * 0.05
    try:
        packed, st = bnbf.quantize_4bit(w, blocksize=64, quant_type="nf4")
    except Exception as e:                                                                       # no CPU 4-bit path in this bnb
        pytest.skip(f"bitsandbytes cannot quantize on CPU here: {e}")
    raw = {"fmt": "nf4", "packed": packed, "absmax": st.absmax.float(), "code": st.code, "shape": list(w.shape), "blocksize": 64,
           "transposed": False, "dtype": "float32"}
    a, b = hr.deq32(raw, "cpu"), hp1_lib.dequant_raw(raw, "cpu")
    assert a.dtype == torch.float32 and b.dtype == torch.bfloat16
    assert torch.allclose(a, b.float(), atol=0, rtol=2 ** -7) and not torch.equal(a, b.float())


def test_the_fp32_reference_model_reproduces_hfs_own_fp32_forward_end_to_end(tmp_path):
    """build_model_f32 over a store that serves a tiny Qwen3-MoE's own weights: the same per-row loss as HF's fp32 eager model, so the
    wiring (q/k/v/o, the gate_up chunk order, the routing, the norms, embeddings and lm_head keys) is the model's, not just plausible."""
    hr = _hpn_read()
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3MoeConfig(vocab_size=64, hidden_size=16, intermediate_size=32, moe_intermediate_size=8, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=4, num_experts=4, num_experts_per_tok=2,
                                      max_position_embeddings=64, decoder_sparse_step=1, mlp_only_layers=[])
    cfg.save_pretrained(tmp_path)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    ref = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.3) if p.dim() > 1 else p.uniform_(0.8, 1.2)
    w = {(-1, "embed", -1): ref.model.embed_tokens.weight, (-1, "lm_head", -1): ref.lm_head.weight, (-1, "final_norm", -1): ref.model.norm.weight}
    for L, layer in enumerate(ref.model.layers):
        a, mlp = layer.self_attn, layer.mlp
        for kind in "qkvo":
            w[(L, kind, -1)] = getattr(a, f"{kind}_proj").weight
        for e in range(cfg.num_experts):
            w[(L, "gate_up", e)], w[(L, "down", e)] = mlp.experts.gate_up_proj[e], mlp.experts.down_proj[e]
        w[(L, "router", -1)] = mlp.gate.weight
        for name, m in layer.named_modules():
            last = name.split(".")[-1]
            if "norm" in last.lower() and isinstance(getattr(m, "weight", None), torch.nn.Parameter):
                w[(L, "norm:" + last, -1)] = m.weight

    class Store:
        dir, device = str(tmp_path), "cpu"

        def keys(self):
            return list(w)
    f32 = hr.build_model_f32(Store(), str(tmp_path), None, deq=lambda key: w[key].detach().clone())
    ids = torch.randint(0, 64, (1, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        a, b = ref(input_ids=ids, labels=ids).loss.item(), f32(input_ids=ids, labels=ids).loss.item()
    assert a == pytest.approx(b, abs=1e-5) and math.isfinite(a)
    w[(1, "gate_up", 2)] = w[(1, "gate_up", 2)].flip(0)                 # swap gate/up halves on one expert: the loss must move
    f32b = hr.build_model_f32(Store(), str(tmp_path), None, deq=lambda key: w[key].detach().clone())
    with torch.no_grad():
        assert abs(f32b(input_ids=ids, labels=ids).loss.item() - a) > 1e-4


# ------------------------------------------------------------- review: every bar reads the UNROUNDED value (a flip test per bar)
def test_n0_flips_on_a_rerun_gap_just_past_1e5_that_rounds_to_1e5():
    In, E, F = _synth(40, 0.01, 0.01, seed=6)
    over = [In[0], [x + (1.04e-5 if k == 3 else 0.0) for k, x in enumerate(In[0])]]
    at = [In[0], [x + (1.0e-5 if k == 3 else 0.0) for k, x in enumerate(In[0])]]
    rec_over, rec_at = R.depth_record(over, E, F), R.depth_record(at, E, F)
    assert rec_over["n0_max_abs_rerun"]["I"] == 1e-05 and rec_over["n0_holds"]["I"] is False     # it rounds to the bar, still fails
    assert rec_at["n0_holds"]["I"] is True


def test_n3_reads_the_same_unrounded_rho_in_the_depth_call_and_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_corr", lambda a, b: 0.20004)                                       # rounds to 0.2, which is past 0.2
    rec = R.depth_record(*_synth(50, 0.01, 0.01, seed=7))
    assert rec["corr_e_i"] == 0.2 and rec["n3_holds"] is False and rec["call_basis"] == "1 - var(i)/var(g)"
    _, out = _run(tmp_path, [(2, 0.003, 0.003), (4, 0.006, 0.006), (8, 0.012, 0.012)])
    assert out["N3"] is False


def _gscaled(n, a, seed):
    """I = F; E = F + a z with z standardized (mean 0, sd 1), so sd_r(g) = a exactly."""
    rng = random.Random(seed)
    F = [1.9 + rng.gauss(0, 0.2) for _ in range(n)]
    z = [rng.gauss(0, 1) for _ in range(n)]
    m, s = sum(z) / n, (sum((x - sum(z) / n) ** 2 for x in z) / (n - 1)) ** 0.5
    E = [f + a * (x - m) / s for f, x in zip(F, z)]
    return [F, F], [E, E], [[f + 0.0 for f in F]] * 2


def test_n1_reads_a_rise_finer_than_the_stored_rounding():
    """sd(g) = 0.0100001 at L4 and 0.0100004 at L8: both store as 0.01, but it rises, and N1 holds."""
    data = {2: _gscaled(300, 0.005, 1), 4: _gscaled(300, 0.0100001, 2), 8: _gscaled(300, 0.0100004, 3)}
    spec = {"depths": [{"L": L, "inloop": [f"{L}i0", f"{L}i1"], "eval": [f"{L}e0", f"{L}e1"], "fref": f"{L}f"} for L in data]}
    load = {"inloop": lambda p: data[int(p[:-2])][0][int(p[-1])], "eval": lambda p: data[int(p[:-2])][1][int(p[-1])],
            "fref": lambda p: data[int(p[:-1])][2]}
    out = R.reduce(spec, load=load)
    assert out["depths"]["4"]["g"]["sd"] == out["depths"]["8"]["g"]["sd"] == 0.01 and out["N1"] is True
    assert not any(k.startswith("_") for k in out["depths"]["4"])                                # the private fields stay inside


def test_n1_is_unreadable_without_l2():
    data = {4: _gscaled(300, 0.006, 2), 8: _gscaled(300, 0.012, 3)}
    spec = {"depths": [{"L": L, "inloop": [f"{L}i0", f"{L}i1"], "eval": [f"{L}e0", f"{L}e1"], "fref": f"{L}f"} for L in data]}
    load = {"inloop": lambda p: data[int(p[:-2])][0][int(p[-1])], "eval": lambda p: data[int(p[:-2])][1][int(p[-1])],
            "fref": lambda p: data[int(p[:-1])][2]}
    assert R.reduce(spec, load=load)["N1"] is None


def test_e_and_f_build_their_rotary_with_the_same_helper_and_it_matches_the_versions_own_construction():
    """F must differ from E in precision only: both build the rotary embedding through hp1_lib.make_rotary (the call-site pin is what
    guarantees E == F), and the helper's buffers equal the installed version's own construction (5.18 takes device=, later do not)."""
    import inspect

    import hp1_eval
    import hp1_lib
    hr = _hpn_read()
    transformers = pytest.importorskip("transformers")
    assert "hp1_lib.make_rotary(" in inspect.getsource(hp1_eval.build_model) and "hp1_lib.make_rotary(" in inspect.getsource(hr.build_model_f32)
    cfg = transformers.Qwen3MoeConfig(hidden_size=16, num_attention_heads=4, head_dim=4, max_position_embeddings=64)
    cls = type(transformers.AutoModelForCausalLM.from_config(cfg).model.rotary_emb)
    # the helper reproduces the INSTALLED version's own construction (review: comparing the helper with itself could not fail)
    native = cls(config=cfg, device="cpu") if "device" in inspect.signature(cls.__init__).parameters else cls(config=cfg).to("cpu")
    assert torch.equal(hp1_lib.make_rotary(cls, cfg, "cpu").inv_freq, native.inv_freq)


def test_make_rotary_builds_on_a_transformers_without_the_device_argument():
    import hp1_lib

    class NoDevice(torch.nn.Module):                     # the later releases' signature: config only
        def __init__(self, config):
            super().__init__()
            self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))

    class WithDevice(torch.nn.Module):                   # 5.18's signature
        def __init__(self, config, device=None):
            super().__init__()
            self.got = device
            self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32, device=device))
    assert torch.equal(hp1_lib.make_rotary(NoDevice, None, "cpu").inv_freq, torch.arange(4.0))
    assert hp1_lib.make_rotary(WithDevice, None, "cpu").got == "cpu"

    class Broken(torch.nn.Module):                       # takes device=, then fails inside for an unrelated reason
        def __init__(self, config, device=None):
            super().__init__()
            self.register_buffer("inv_freq", torch.zeros(4))
            if device is not None:                       # the bug only on the device= path: a blanket except would hide it by retrying
                raise TypeError("an unrelated bug in the constructor")
    with pytest.raises(TypeError, match="unrelated bug"):           # surfaced, never retried silently without device=
        hp1_lib.make_rotary(Broken, None, "cpu")


# ------------------------------------------------------------------------------------------------------------ hpn_local.sh
def _local_repo(tmp_path):
    import subprocess
    r = tmp_path / "repo"
    files = ["bench/tc1/tc1_arm.py", "bench/tc1/tc1_reduce.py", "bench/tp4/tp4_alpaca.py", "bench/flagship-matrix/drivers/n9_datasets.py",
             "bench/flagship-matrix/ds_manifest.json", "bench/hpn/hpn_read.py", "bench/hpn/hpn_reduce.py", "bench/hpn/HPN-DESIGN.md"]
    files += [f"bench/hp1/{f}" for f in ("hp1_lib.py", "hp1_eval.py", "hp1_caltokens.py", "hp1_trunc.py", "hp1_reduce.py")]
    for f in files:
        (r / f).parent.mkdir(parents=True, exist_ok=True)
        (r / f).write_text(f"# {f}\n")
    (r / "bench/tc1/tc1_run.sh").write_text("env | grep -E '^(TC1_|HPN_|E4B_SHA|GNF4_SHA)' | sort > env.txt\necho 'HPN=STOP depths=[2]' > summary.txt\n")
    g = lambda *a: subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True)   # noqa: E731
    g("init", "-q")
    g("add", "-A")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text("#!/bin/bash\necho 10612\n")
    (bin_ / "nvidia-smi").chmod(0o755)
    return r, bin_


def test_hpn_local_stages_the_pieces_and_runs_the_local_box_token(tmp_path):
    import os
    import subprocess
    r, bin_ = _local_repo(tmp_path)
    out = tmp_path / "out"
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}")
    p = subprocess.run(["bash", str(REPO / "bench/hpn/hpn_local.sh"), str(r), str(out)], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    for f in ("tc1_run.sh", "tc1_arm.py", "tp4_alpaca.py", "n9_datasets.py", "ds_manifest.json", "hp1_eval.py", "hp1_trunc.py",
              "hpn_read.py", "hpn_reduce.py", "hpn/HPN-DESIGN.md"):
        assert (out / f).exists(), f
    got = dict(ln.split("=", 1) for ln in (out / "env.txt").read_text().splitlines())
    sha = subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert got["TC1_LOCAL_BOX"] == "1" and got["TC1_FAMILIES"] == "qwen3hpn" and got["TC1_LOCAL_OUT"] == str(out)
    assert got["HPN_FREE_MIB"] == "10612" and got["E4B_SHA"] == sha and got["HPN_UNS_BACKEND"] == "native_torch"
    assert "HPN free GPU memory at start: 10612 MiB" in p.stdout and "HPN=STOP" in p.stdout


def test_hpn_local_refuses_an_unclean_checkout(tmp_path):
    import os
    import subprocess
    r, bin_ = _local_repo(tmp_path)
    (r / "bench/hpn/hpn_read.py").write_text("# edited after the commit\n")
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}")
    p = subprocess.run(["bash", str(REPO / "bench/hpn/hpn_local.sh"), str(r), str(tmp_path / "out")], env=env, capture_output=True,
                       text=True, timeout=60)
    assert p.returncode == 78 and "uncommitted changes" in p.stdout and not (tmp_path / "out").exists()


def test_hpn_local_refuses_a_non_empty_out(tmp_path):
    import os
    import subprocess
    r, bin_ = _local_repo(tmp_path)
    out = tmp_path / "out"
    (out / "hpn").mkdir(parents=True)
    (out / "hpn" / "eval_e4b_L8_1.json").write_text("{}")                     # an earlier run's file
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}")
    p = subprocess.run(["bash", str(REPO / "bench/hpn/hpn_local.sh"), str(r), str(out)], env=env, capture_output=True, text=True,
                       timeout=60)
    assert p.returncode == 78 and "is not empty" in p.stdout and not (out / "env.txt").exists()


def test_hpn_local_refuses_an_untracked_file_in_a_staged_dir(tmp_path):
    """Review: find bench/tc1 -type f would ship an untracked file into a run that cites a clean commit."""
    import os
    import subprocess
    r, bin_ = _local_repo(tmp_path)
    (r / "bench/tc1/stray_helper.py").write_text("# not committed\n")
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}")
    p = subprocess.run(["bash", str(REPO / "bench/hpn/hpn_local.sh"), str(r), str(tmp_path / "out")], env=env, capture_output=True,
                       text=True, timeout=60)
    assert p.returncode == 78 and "untracked files in the staged dirs" in p.stdout


# ---- the committed record (bench/hpn/receipts/hpn-a2000-read8): the registered call, rebuilt from public files -------------
READ8 = REPO / "bench" / "hpn" / "receipts" / "hpn-a2000-read8"


def test_the_registered_reducer_rebuilds_read_8s_result_from_the_public_projections(tmp_path):
    """The reducer's inputs stay in the private store; their projections (hpn_public_inloop.py) and the public spec let
    hpn_reduce rebuild hpn/result.json byte-identically: MOOT at L4 and L8."""
    import subprocess
    import sys
    out = tmp_path / "result.json"
    p = subprocess.run([sys.executable, str(REPO / "bench" / "hpn" / "hpn_reduce.py"), str(READ8 / "hpn_spec.public.json"),
                        "--out", str(out)], capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.startswith("HPN=MOOT depths=[4, 8]"), p.stdout + p.stderr
    assert out.read_bytes() == (READ8 / "hpn" / "result.json").read_bytes()


def test_read_8s_projections_hold_only_the_fields_the_reducer_reads():
    for p in (READ8 / "proj").glob("qwen3hpn*.json"):
        assert set(json.loads(p.read_text())) == {"eval_curve"}, p.name
    for p in (READ8 / "proj" / "hpn").glob("eval_*.json"):
        bases = json.loads(p.read_text())["bases"]
        assert all(set(b) <= {"digest", "n_slots", "digest_matches_training_run", "status", "zero_adapter"} for b in bases), p.name
    for p in (READ8 / "proj" / "hpn").glob("f_*.json"):
        assert set(json.loads(p.read_text())) <= {"reader", "digest", "n_slots", "runs", "row_mean"}, p.name


def test_read_8s_committed_files_carry_no_identifier_and_no_run_path():
    import re
    pat = re.compile(r"GPU-[0-9a-f]{8}-|instance_id|vast_instance|\"uuid\"|/home/|/Users/|/workspace/scratch/")
    hits = [p.name for p in READ8.rglob("*") if p.is_file() and pat.search(p.read_text(errors="replace"))]
    assert not hits, hits


def test_read_8_is_moot_at_both_deciding_depths_with_identical_bases():
    r = json.loads((READ8 / "hpn" / "result.json").read_text())
    assert r["outcome"] == "MOOT" and r["fitted_depths"] == [4, 8]
    assert {k: v["read_1b"] for k, v in r["base_term"].items()} == {"4": "MOOT", "8": "MOOT"}
    assert all(v["ratio_F_over_E"] == 1.0 and v["base_term_E"] == v["base_term_F"] == 0.0 for v in r["base_term"].values())
