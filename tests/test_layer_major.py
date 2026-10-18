"""Layer-major accumulation (:mod:`experts4bit_qlora.engines.layer_major`), against the stock accumulation loop.

STRICT, as registered for the layer-major work:
- each micro-batch's loss is ``torch.equal`` to the stock forward's (the model's own forward, call for call);
- under expert offload a step stages exactly 2n layers from cold and 2n - 1 in steady state, against M x (2n - 1) (+1 cold)
  for the stock loop;
- gradients are the same weighted sum: fp32 here, so summation order only; the bf16 band is :func:`test_bf16_gradient_band`;
- other families, router logits and a weights/batches mismatch are refused.
Both checkpoint kinds: the reentrant one is ``enable_fast_train``'s default.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers")

from experts4bit_qlora.engines.layer_major import layer_major_step  # noqa: E402
from experts4bit_qlora.engines.offload import _ExpertOffload  # noqa: E402

VOCAB, M = 97, 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # the tiny model too: the drift guard must hold on CUDA
WIDTHS = (9, 14, 6)  # unequal micro-batches, right-padded rows inside each


@pytest.fixture(autouse=True)
def _reset_resident_slot():
    _ExpertOffload._resident = None
    yield
    _ExpertOffload._resident = None


def _tiny(seed=0, layers=3, reentrant=False, dtype=torch.float32):
    cfg = tr.Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_experts=8,
                            num_experts_per_tok=2, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, vocab_size=VOCAB, max_position_embeddings=64, decoder_sparse_step=1,
                            norm_topk_prob=True)
    torch.manual_seed(seed)
    m = tr.Qwen3MoeForCausalLM(cfg).to(dtype).to(DEVICE)
    _train_mode(m, reentrant)
    return m


def _train_mode(m, reentrant):
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    if reentrant:
        m.enable_input_require_grads()
    m.config.use_cache = False
    m.train()


def _batches(seed=3, device=DEVICE):
    g = torch.Generator().manual_seed(seed)
    out = []
    for w in WIDTHS:
        ids = torch.randint(1, VOCAB, (2, w), generator=g)
        att = torch.ones(2, w, dtype=torch.long)
        att[1, w - 2:] = 0  # the second row is right-padded
        lbl = ids.masked_fill(att == 0, -100)
        out.append({"input_ids": ids.to(device), "labels": lbl.to(device), "attention_mask": att.to(device)})
    return out


WEIGHTS = [1.0 / M] * M  # the stock loop's 1 / GRAD_ACCUM


def _stock(model, batches, weights):
    """train.py's loop: one forward + backward per micro-batch, (loss * w).backward() accumulating into .grad."""
    losses = []
    for b, w in zip(batches, weights):
        out = model(**b)
        (out.loss * w).backward()
        losses.append(out.loss.detach())
    return losses


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_losses_equal_the_stock_forward_and_grads_are_the_same_sum(reentrant):
    model = _tiny(reentrant=reentrant)
    ref_losses = _stock(model, _batches(), WEIGHTS)
    ref = _grads(model)
    model.zero_grad(set_to_none=True)
    losses = layer_major_step(model, _batches(), WEIGHTS, handles=[])
    for got, want in zip(losses, ref_losses):
        assert torch.equal(got, want)
    got = _grads(model)
    assert got.keys() == ref.keys() and len(got) > 0
    for n in ref:
        torch.testing.assert_close(got[n], ref[n], rtol=1e-5, atol=1e-7, msg=n)
    assert len(model.model.layers) == 3  # the head-only call restored the decoder stack


def test_refuses_other_families_router_logits_and_a_mismatch():
    cfg = tr.GraniteMoeConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                              num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2, vocab_size=VOCAB)
    with pytest.raises(NotImplementedError, match="GraniteMoeForCausalLM is not supported"):
        layer_major_step(tr.GraniteMoeForCausalLM(cfg), _batches(), WEIGHTS, handles=[])
    model = _tiny()
    model.config.output_router_logits = True
    with pytest.raises(NotImplementedError, match="output_router_logits"):
        layer_major_step(model, _batches(), WEIGHTS, handles=[])
    model.config.output_router_logits = False
    with pytest.raises(ValueError, match="weights"):
        layer_major_step(model, _batches(), WEIGHTS[:2], handles=[])


# ---------------------------------------------------- the real path: NF4 experts, offloaded --

E, TOP_K, N_LAYERS = 8, 2, 3
GA_BAR = 0.0231  # bench/q235/instrument/q235_verdict.py (Amendment 6); the registered band's (a)


def _qwen3_moe_checkpoint(d):
    """A random 3-layer Qwen3-MoE in its on-disk (per-expert) layout, at sizes the NF4 blocksize accepts."""
    from safetensors.torch import save_file

    torch.manual_seed(0)
    model = tr.Qwen3MoeForCausalLM(tr.Qwen3MoeConfig(
        hidden_size=128, intermediate_size=256, moe_intermediate_size=128, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_experts=E, num_experts_per_tok=TOP_K,
        vocab_size=VOCAB, decoder_sparse_step=1, mlp_only_layers=[], max_position_embeddings=64))
    new = {}
    for k, v in model.state_dict().items():
        if k.endswith("experts.gate_up_proj"):
            base = k[: -len("gate_up_proj")]
            for e in range(v.shape[0]):
                g, u = v[e].chunk(2, dim=0)
                new[f"{base}{e}.gate_proj.weight"] = g
                new[f"{base}{e}.up_proj.weight"] = u
        elif k.endswith("experts.down_proj"):
            base = k[: -len("down_proj")]
            for e in range(v.shape[0]):
                new[f"{base}{e}.down_proj.weight"] = v[e]
        else:
            new[k] = v
    new = {k: v.to(torch.bfloat16).contiguous().clone() for k, v in new.items()}
    save_file(new, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in new}}, f)
    model.config.save_pretrained(d)




@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("safetensors")
    d = tmp_path_factory.mktemp("qwen3moe")
    _qwen3_moe_checkpoint(str(d))
    return str(d)


def _load(path, dtype, reentrant):
    from quant_guard import load_or_skip

    torch.manual_seed(0)
    model, _ = load_or_skip(path, DEVICE, dtype, r=4, alpha=8, offload=True, pin=False, quant_type="nf4")
    for n, p in model.named_parameters():
        if "lora_B" in n:  # B = 0 at init would leave every A gradient zero
            torch.nn.init.normal_(p, std=0.02)
    _train_mode(model, reentrant)
    return model


@pytest.fixture
def copies(monkeypatch):
    log = []
    real = _ExpertOffload._copy_home_to_device

    def counted(self, policy="sync"):
        log.append(self)
        return real(self, policy)

    monkeypatch.setattr(_ExpertOffload, "_copy_home_to_device", counted)
    return log


@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_offloaded_step_stages_2n_cold_and_2n_minus_1_steady(ckpt, copies, reentrant):
    model = _load(ckpt, torch.float32, reentrant)
    n = N_LAYERS
    ref_losses = _stock(model, _batches(device=DEVICE), WEIGHTS)
    assert len(copies) == M * (2 * n - 1) + 1  # cold: the first forward also stages layer 0
    layer_major_step(model, _batches(device=DEVICE), WEIGHTS)  # the first call runs the drift guard's stock forward too
    for h in _handles(model):
        h.evict()
    copies.clear()
    model.zero_grad(set_to_none=True)
    losses = layer_major_step(model, _batches(device=DEVICE), WEIGHTS)
    assert len(copies) == 2 * n
    for got, want in zip(losses, ref_losses):
        assert torch.equal(got, want)
    copies.clear()
    layer_major_step(model, _batches(device=DEVICE), WEIGHTS)  # steady: layer 0 resident from the last recompute
    assert len(copies) == 2 * n - 1


def _handles(model):
    from experts4bit_qlora.engines.offload import offload_handles

    hs = offload_handles(model)
    assert len(hs) == N_LAYERS
    return hs


def _adapter_grads(model):
    return {n: p.grad.detach().double().clone() for n, p in model.named_parameters()
            if "lora" in n and p.grad is not None}


def _rel_and_cos(ref, got):
    """Rule 8's views (q235_box.compare_grads): the largest per-tensor relative L2 difference, and the global cosine."""
    rel, dot, nr, ng = [], 0.0, 0.0, 0.0
    for n, r in ref.items():
        g = got.get(n, torch.zeros_like(r))
        dot += float((r * g).sum())
        nr += float(r.norm()) ** 2
        ng += float(g.norm()) ** 2
        if float(r.norm()) > 0:
            rel.append(float((g - r).norm()) / float(r.norm()))
    return max(rel), dot / (nr ** 0.5 * ng ** 0.5)


@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_bf16_gradient_band(ckpt, reentrant):
    """The registered band (maintainer's rulings on the layer-major packet). Both arms against a host fp64 sum of each
    micro-batch's own gradient, on every adapter tensor, with the same weights, over 5 input seeds (the same seeds both
    arms). Layer-major passes iff:
    (a) max_rel <= GA_BAR on every seed;
    (b) per seed, cos >= 0.9999; where the as-is arm is ALSO under 0.9999 on a seed, the absolute bar is reading bf16
        rounding, not the order, and (b) becomes paired: median(1 - cos_LM) <= max(2 x median(1 - cos_asis), 1e-4).
        Layer-major under 0.9999 while as-is is at or over it on the same seed fails;
    (c) median(max_rel_LM) <= max(2 x median(max_rel_asis), 1e-3).
    Set E4B_LAYER_MAJOR_BAND_OUT to a path to write both arms' per-seed values and the (b) branch taken (the receipt)."""
    model = _load(ckpt, torch.bfloat16, reentrant)
    rows = []
    for seed in range(5):
        batches = _batches(seed=100 + seed, device=DEVICE)
        ref = None
        for b, w in zip(batches, WEIGHTS):  # the reference: each micro-batch alone, summed on the host in fp64
            model.zero_grad(set_to_none=True)
            model(**b).loss.backward()
            g = {n: t * w for n, t in _adapter_grads(model).items()}
            ref = g if ref is None else {n: ref[n] + g[n] for n in ref}
        model.zero_grad(set_to_none=True)
        _stock(model, batches, WEIGHTS)
        a_rel, a_cos = _rel_and_cos(ref, _adapter_grads(model))
        model.zero_grad(set_to_none=True)
        layer_major_step(model, batches, WEIGHTS)
        l_rel, l_cos = _rel_and_cos(ref, _adapter_grads(model))
        rows.append({"seed": 100 + seed, "asis_max_rel": a_rel, "asis_cos": a_cos, "lm_max_rel": l_rel, "lm_cos": l_cos})

    def med(xs):
        return sorted(xs)[len(xs) // 2]

    paired = any(r["asis_cos"] < 0.9999 for r in rows)
    verdict = {
        "a": all(r["lm_max_rel"] <= GA_BAR for r in rows),
        "b_branch": "paired" if paired else "absolute",
        "b": (not any(r["lm_cos"] < 0.9999 <= r["asis_cos"] for r in rows)
              and (med([1 - r["lm_cos"] for r in rows]) <= max(2 * med([1 - r["asis_cos"] for r in rows]), 1e-4)
                   if paired else all(r["lm_cos"] >= 0.9999 for r in rows))),
        "c": med([r["lm_max_rel"] for r in rows]) <= max(2 * med([r["asis_max_rel"] for r in rows]), 1e-3),
    }
    out = os.environ.get("E4B_LAYER_MAJOR_BAND_OUT")
    if out:
        with open(f"{out}.{'reentrant' if reentrant else 'nonreentrant'}.json", "w") as f:
            json.dump({"device": DEVICE, "torch": torch.__version__, "reentrant": reentrant, "ga_bar": GA_BAR,
                       "rows": rows, "verdict": verdict}, f, indent=1)
    assert verdict["a"] and verdict["b"] and verdict["c"], (verdict, rows)


def test_trainer_refuses_layer_major_without_offload(monkeypatch):
    """LAYER_MAJOR=1 saves expert stagings; resident training stages none, so the trainer refuses it before loading."""
    import experts4bit_qlora.train as train

    monkeypatch.setattr(train, "LAYER_MAJOR", True)
    monkeypatch.setattr(train, "OFFLOAD_EXPERTS", False)
    monkeypatch.setattr(train, "ABSMAX_DQ_REQUIRED", False)
    monkeypatch.setattr(train.sys, "argv", ["train"])
    with pytest.raises(SystemExit, match="LAYER_MAJOR=1 needs OFFLOAD_EXPERTS=1"):
        train.main()


@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_drift_guard_refuses_a_diverged_preamble_before_any_backward(monkeypatch, reentrant):
    """If the reproduced preamble stops matching the model's forward (here: the step's causal mask also hides each row's
    first token, which the stock forward's mask does not), the first call refuses before its backward: no gradient is
    written."""
    import transformers.masking_utils as mu

    from experts4bit_qlora.engines import layer_major as lm

    model = _tiny(reentrant=reentrant)
    real = mu.create_causal_mask

    def diverged(**kw):
        att = kw["attention_mask"].clone()
        att[:, 0] = 0
        return real(**{**kw, "attention_mask": att})

    monkeypatch.setattr(mu, "create_causal_mask", diverged)
    with pytest.raises(lm.LayerMajorDriftError, match="drifted"):
        layer_major_step(model, _batches(), WEIGHTS, handles=[])
    assert all(p.grad is None for p in model.parameters())
    assert model not in lm._VERIFIED


@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_drift_guard_runs_once_per_model(reentrant):
    from experts4bit_qlora.engines import layer_major as lm

    model = _tiny(reentrant=reentrant)
    calls = {"n": 0}
    model.register_forward_pre_hook(lambda m, a: calls.__setitem__("n", calls["n"] + 1))
    layer_major_step(model, _batches(), WEIGHTS, handles=[])
    first = calls["n"]
    layer_major_step(model, _batches(), WEIGHTS, handles=[])
    assert first == 2 * M + 1 and calls["n"] - first == M  # first: M head calls + M stock checks + 1 repeat; then M
    assert model in lm._VERIFIED


@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_drift_guard_refuses_a_non_reproducible_forward_as_such(monkeypatch, reentrant):
    """If the stock forward itself is not bit-reproducible here (simulated: each call's loss nudged by a counter), the
    guard says so, rather than blaming drift, and writes no gradient."""
    from experts4bit_qlora.engines import layer_major as lm

    model = _tiny(reentrant=reentrant)
    real = type(model).forward
    n = {"k": 0}

    def wobbly(self, *a, **kw):
        out = real(self, *a, **kw)
        if kw.get("input_ids") is not None:  # only the stock calls (the head-only call passes inputs_embeds)
            n["k"] += 1
            out.loss = out.loss + n["k"] * 1e-6
        return out

    monkeypatch.setattr(type(model), "forward", wobbly)
    with pytest.raises(lm.LayerMajorNotReproducibleError, match="not bit-reproducible"):
        layer_major_step(model, _batches(), WEIGHTS, handles=[])
    assert all(p.grad is None for p in model.parameters())


def test_refuses_dropout_in_training():
    """Layer-major draws the random stream in another order than the stock loop, so any active dropout is refused: a
    config float (Qwen3-MoE's attention dropout) and an ``nn.Dropout`` module (as a PEFT ``lora_dropout`` is)."""
    from experts4bit_qlora.engines import layer_major as lm

    model = _tiny()
    model.config.attention_dropout = 0.05
    with pytest.raises(NotImplementedError, match="dropout is on"):
        layer_major_step(model, _batches(), WEIGHTS, handles=[])
    model.config.attention_dropout = 0.0
    model.model.layers[1].self_attn.q_proj_dropout = torch.nn.Dropout(0.05)
    with pytest.raises(NotImplementedError, match="q_proj_dropout"):
        layer_major_step(model, _batches(), WEIGHTS, handles=[])
    assert model not in lm._VERIFIED
