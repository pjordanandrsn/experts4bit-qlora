# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Golden reference: e4b's chunked LM loss against the stock loss THROUGH e4b's QLoRA preparation, on CPU.

``test_chunked_lm_loss.py`` pins the chunked loss on full-precision models with every parameter trainable. This module pins it
where the 4090-fit work (TC1 amendment 74's read) needs it: a Qwen3-MoE prepared by ``prepare_qlora_training``, with frozen NF4
experts, the attention and expert LoRA adapters trainable in fp32, a masked prompt and right padding, the router auxiliary loss,
two accumulated micro-batches, then an AdamW step and a second step after it.

Declared before any measurement, and asserted as declared:
- **Seeds:** the model 0; the adapters' initialisation 1234, the same on both sides (asserted byte-equal); a seeded perturbation 77 of
  every LoRA B (std 1e-2, identical on both sides), because LoRA B starts at zero and would leave every A gradient exactly zero; the
  batch 3.
- **Loss:** |stock - chunked| <= LOSS_ULPS x fp32 eps x max(|loss|, 1), per micro-batch, at both steps.
- **Gradients:** every trainable tensor's max |stock - chunked| <= GRAD_REL x max(its max |grad|, GRAD_FLOOR x the model's largest
  gradient). These are ``test_chunked_lm_loss.py``'s values: only the loss's fp32 summation order differs.
- **The update:** AdamW's first step is lr x g / (|g| + eps), about lr x sign(g). So it is compared where the reference gradient
  clears GRAD_FLOOR of its tensor's largest: same sign, and per element |stock - chunked| <= UPD_REL x lr + 2 ulp(|t|). The second
  term is there because the update is read as a difference of fp32 parameters: each side's stored parameter carries up to one ulp of
  rounding at its own magnitude, which at |t| ~ 0.2 is about 100x UPD_REL x lr. Elements under the floor are counted and reported in
  the assertion message, never asserted, because their sign is rounding.
- **Semantics:** identical inputs and labels (ignore_index -100), the chunked side materialises no full logits, and both sides train
  exactly the same parameter set, attention and expert adapters both present.
"""
import pytest

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora.engines import chunked_lm_loss as C  # noqa: E402
from experts4bit_qlora.recipe import QLoRASetup, prepare_qlora_training  # noqa: E402

V = 97
EPS32 = torch.finfo(torch.float32).eps
LOSS_ULPS = 8
GRAD_REL = 1e-5
GRAD_FLOOR = 1e-3
UPD_REL = 1e-6
LR, WD = 2e-4, 0.001                       # TC1's field recipe's learning rate and weight decay
SEED_MODEL, SEED_INIT, SEED_B, SEED_BATCH = 0, 1234, 77, 3
SETUP = QLoRASetup(r=16, alpha=16, adapter_dtype="fp32", expert_kernel="reference", expert_residency="device", pin=False)


@pytest.fixture(scope="module")
def tiny_qwen3_moe(tmp_path_factory):
    cfg = tr.Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=64, num_experts=8, num_experts_per_tok=2,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=V,
                            max_position_embeddings=64, decoder_sparse_step=1, norm_topk_prob=True, tie_word_embeddings=False,
                            output_router_logits=True)
    torch.manual_seed(SEED_MODEL)
    path = tmp_path_factory.mktemp("golden_qwen3_moe")
    tr.Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).save_pretrained(path)
    return path


def _prepared(path):
    torch.manual_seed(SEED_INIT)
    p = prepare_qlora_training(str(path), SETUP, device="cpu")
    g = torch.Generator().manual_seed(SEED_B)
    with torch.no_grad():
        for n, t in sorted(p.model.named_parameters()):
            if t.requires_grad and "lora_B" in n:
                t.copy_(torch.randn(t.shape, generator=g, dtype=torch.float32).to(t.dtype) * 1e-2)
    p.model.train()
    return p


def _batches():
    """Two micro-batches of two rows: row 0 has a 5-token masked prompt, row 1 is right-padded by 4 (labels -100, attention 0)."""
    g = torch.Generator().manual_seed(SEED_BATCH)
    out = []
    for w in (24, 17):
        ids = torch.randint(1, V, (2, w), generator=g)
        att = torch.ones(2, w, dtype=torch.long)
        att[1, w - 4:] = 0
        lbl = ids.masked_fill(att == 0, -100)
        lbl[0, :5] = -100
        out.append((ids, lbl, att))
    return out


def _accumulate(model, batches):
    model.zero_grad(set_to_none=True)
    losses = []
    for ids, lbl, att in batches:
        out = model(input_ids=ids, labels=lbl, attention_mask=att)
        (out.loss / len(batches)).backward()
        losses.append((out.loss.detach(), out.logits))
    return losses, {n: t.grad.detach().clone() for n, t in model.named_parameters() if t.grad is not None}


def _assert_loss_close(a, b, what):
    a, b = float(a), float(b)
    assert abs(a - b) <= LOSS_ULPS * EPS32 * max(abs(a), 1.0), f"{what}: stock {a!r} vs chunked {b!r}"


def _assert_grads_close(ref, got, what):
    assert ref.keys() == got.keys(), f"{what}: different parameters received gradients"
    top = max(float(g.abs().max()) for g in ref.values())
    for n in ref:
        x, y = ref[n].float(), got[n].float()
        err, scale = float((x - y).abs().max()), float(x.abs().max())
        assert err <= GRAD_REL * max(scale, GRAD_FLOOR * top) + 1e-30, f"{what} {n}: max |diff| {err:.3e} vs max |grad| {scale:.3e}"


def test_chunked_loss_matches_stock_through_qlora_and_an_adamw_step(tiny_qwen3_moe):
    ref, chk = _prepared(tiny_qwen3_moe), _prepared(tiny_qwen3_moe)
    names = sorted(n for n, t in ref.model.named_parameters() if t.requires_grad)
    assert names == sorted(n for n, t in chk.model.named_parameters() if t.requires_grad)
    assert any("self_attn" in n and "lora" in n for n in names) and any("experts" in n and "lora" in n for n in names), names
    for (n, a), (_, b) in zip(sorted(ref.model.named_parameters()), sorted(chk.model.named_parameters())):
        assert torch.equal(a, b), f"{n}: the two sides do not start from the same bytes"
    assert C.enable_chunked_lm_loss(chk.model, 7) == 1                          # 7-token chunks leave ragged tails
    batches = _batches()
    opt = {k: torch.optim.AdamW([t for _, t in sorted(p.model.named_parameters()) if t.requires_grad], lr=LR, weight_decay=WD)
           for k, p in (("ref", ref), ("chk", chk))}
    for step in (1, 2):
        l_ref, g_ref = _accumulate(ref.model, batches)
        l_chk, g_chk = _accumulate(chk.model, batches)
        for i, ((a, la), (b, lb)) in enumerate(zip(l_ref, l_chk)):
            _assert_loss_close(a, b, f"step {step} micro-batch {i}")
            assert la is not None and lb is None, "the chunked side must not materialise the full logits"
        assert set(g_ref) == set(names), "a trainable parameter received no gradient"
        _assert_grads_close(g_ref, g_chk, f"step {step}")
        before = {n: t.detach().clone() for n, t in ref.model.named_parameters() if t.requires_grad}
        opt["ref"].step()
        opt["chk"].step()
        if step == 1:
            p_chk = dict(chk.model.named_parameters())
            under = 0
            for n, t in ref.model.named_parameters():
                if not t.requires_grad:
                    continue
                g = g_ref[n].float()
                mask = g.abs() >= GRAD_FLOOR * float(g.abs().max())
                under += int((~mask).sum())
                du_ref, du_chk = (t.detach() - before[n])[mask], (p_chk[n].detach() - before[n])[mask]
                assert torch.equal(torch.sign(du_ref), torch.sign(du_chk)), f"{n}: an update changed sign above the floor"
                mag = before[n][mask].abs()
                ulp = torch.nextafter(mag, torch.full_like(mag, float("inf"))) - mag          # one fp32 ulp at each element's magnitude
                over = (du_ref - du_chk).abs() > UPD_REL * LR + 2 * ulp
                assert not bool(over.any()), f"{n}: {int(over.sum())} updates differ beyond UPD_REL x lr + 2 ulp ({under} elements under the floor so far)"
