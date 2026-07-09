"""Weekly upstream canary — does the fused-MoE expert surface still hold under
transformers@main + bitsandbytes@main?

Three tripwires, cheapest first, no downloads (tiny random model, no tokenizer):

1. **Layout probe** — transformers still stores OLMoE experts as fused 3-D
   ``gate_up_proj``/``down_proj`` parameters on a ``*.mlp.experts`` module. This is the
   assumption everything downstream (quantizer matching, loader, offload) stands on;
   a rename/reshape upstream must surface here as a red run, not as a user bug report.
2. **Quantize probe** (the #1849 regression, in miniature) — ``Experts4bit.from_float``
   actually 4-bit-packs those stacks (uint8, >3x smaller than fp16) using
   bitsandbytes@main's ``quantize_4bit``/``QuantState`` APIs.
3. **Train probe** — two LoRA steps over the quantized base: loss finite, base frozen
   bit-exact, and (CUDA only) peak VRAM under a pinned ceiling.

Exit 0 = all green. Any exception/assert = red run -> the workflow opens/updates an
issue with the versions manifest.
"""

import sys

import torch

CEILING_VRAM_GB = 1.0  # tiny model; anything near this means the memory story regressed


def main() -> int:
    import bitsandbytes
    import transformers

    import experts4bit_qlora
    from experts4bit_qlora import Experts4bit

    print(
        f"versions: torch={torch.__version__} transformers={transformers.__version__} "
        f"bitsandbytes={bitsandbytes.__version__} experts4bit_qlora={experts4bit_qlora.__version__} "
        f"cuda={torch.cuda.is_available()}"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # -- 1. layout probe ---------------------------------------------------------------
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    cfg = OlmoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = OlmoeForCausalLM(cfg).to(device)

    experts = [(n, m) for n, m in model.named_modules() if n.endswith("mlp.experts")]
    assert experts, "LAYOUT: no '*.mlp.experts' modules — transformers moved the MoE layout"
    for name, mod in experts:
        gu, dn = getattr(mod, "gate_up_proj", None), getattr(mod, "down_proj", None)
        assert isinstance(gu, torch.nn.Parameter) and gu.dim() == 3, (
            f"LAYOUT: {name}.gate_up_proj is not a fused 3-D Parameter (got {type(gu)}/{getattr(gu, 'shape', None)})"
        )
        assert isinstance(dn, torch.nn.Parameter) and dn.dim() == 3, (
            f"LAYOUT: {name}.down_proj is not a fused 3-D Parameter"
        )
    print(f"layout probe: {len(experts)} fused expert modules, 3-D gate_up/down present")

    # -- 2. quantize probe (#1849 in miniature) -----------------------------------------
    src = experts[0][1]
    fp16_bytes = (src.gate_up_proj.numel() + src.down_proj.numel()) * 2
    q = Experts4bit.from_float(src.gate_up_proj.data.float(), src.down_proj.data.float(), compute_dtype=torch.float32)
    assert q.gate_up_proj.dtype == torch.uint8 and q.down_proj.dtype == torch.uint8
    quantized_bytes = (
        q.gate_up_proj.numel() + q.down_proj.numel() + (q.gate_up_absmax.numel() + q.down_absmax.numel()) * 4
    )
    assert quantized_bytes < fp16_bytes / 3, f"QUANTIZE: {quantized_bytes}B vs fp16 {fp16_bytes}B — packing regressed"
    print(f"quantize probe: {fp16_bytes}B fp16 -> {quantized_bytes}B packed")

    # -- 3. train probe ------------------------------------------------------------------
    from experts4bit_qlora.lora import ExpertsLoRA

    lora = ExpertsLoRA(q, r=4, alpha=8, dtype=torch.float32).to(device)
    packed_before = q.gate_up_proj.detach().clone()
    opt = torch.optim.Adam((p for p in lora.parameters() if p.requires_grad), lr=1e-3)

    torch.manual_seed(1)
    hs = torch.randn(16, cfg.hidden_size, device=device)
    idx = torch.randint(0, cfg.num_experts, (16, cfg.num_experts_per_tok), device=device)
    wts = torch.softmax(torch.randn(16, cfg.num_experts_per_tok, device=device), dim=-1)
    target = torch.randn(16, cfg.hidden_size, device=device)

    losses = []
    for _ in range(2):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(lora(hs, idx, wts), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses))), f"TRAIN: non-finite losses {losses}"
    assert torch.equal(q.gate_up_proj, packed_before), "TRAIN: frozen base mutated"
    print(f"train probe: 2 steps, losses {losses}, base bit-exact")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        assert peak < CEILING_VRAM_GB, f"VRAM: peak {peak:.3f} GB over {CEILING_VRAM_GB} GB ceiling"
        print(f"vram probe: peak {peak:.3f} GB < {CEILING_VRAM_GB} GB")
    else:
        print("vram probe: skipped (no CUDA on this runner)")

    print("CANARY GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
