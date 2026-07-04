"""Inference for a QLoRA-tuned fused-MoE: stream-load the NF4 base, attach the trained adapters,
generate — on the same small GPU the fine-tune ran on, with no re-quantization.

This is the serving side of the package's story: the adapters were trained against *this exact*
NF4 base (same codebook, same per-expert absmax), so serving them over the same base keeps the
quantization error identical to what training saw — no NF4->GGUF/AWQ round trip. Decode uses the
single-token fast-path and (when supported) bnb's fused 4-bit GEMV (:mod:`.lora`); with
``OFFLOAD_EXPERTS=1`` the frozen experts stay in pinned CPU RAM and stream in per layer, with the
next layer prefetched on a side stream (:mod:`.offload`) — the way to *generate* with a fused-MoE
whose 4-bit experts exceed VRAM.

Env-configured like :mod:`.train`::

    ADAPTER=./out/adapter_best.pt python -m experts4bit_qlora.infer
    OFFLOAD_EXPERTS=1 BENCH_TOKENS=128 python -m experts4bit_qlora.infer   # timed decode benchmark

Variables: ``MODEL`` (default OLMoE-1B-7B), ``ADAPTER`` (path to a ``train.py`` adapter ``.pt``;
empty = base model), ``R``/``ALPHA`` (must match the adapter), ``OFFLOAD_EXPERTS``, ``OFFLOAD_PIN``,
``PREFETCH`` (default 1; only meaningful with offload), ``PROMPT``, ``MAX_NEW`` (default 64),
``BENCH_TOKENS`` (>0 switches to a timed prefill+decode measurement of exactly that many greedy
tokens). Kill-switches for A/B: ``E4B_DECODE_FASTPATH=0``, ``E4B_INFER_GEMV=0`` (see :mod:`.lora`).
"""

import os
import time

import torch

from .loader import load_moe_4bit_streaming
from .lora import _gemv_4bit_matches_dequant, add_attention_lora
from .util import log

MODEL = os.environ.get("MODEL", "allenai/OLMoE-1B-7B-0924")
DEVICE = "cuda"
DTYPE = torch.bfloat16
R, ALPHA = int(os.environ.get("R", "8")), int(os.environ.get("ALPHA", "16"))
ADAPTER = os.environ.get("ADAPTER", "")
OFFLOAD_EXPERTS = os.environ.get("OFFLOAD_EXPERTS", "0") == "1"
OFFLOAD_PIN = os.environ.get("OFFLOAD_PIN", "1") == "1"
PREFETCH = os.environ.get("PREFETCH", "1") == "1"
QUANT_TYPE = os.environ.get("QUANT_TYPE", "nf4")  # must match the training run
MAX_NEW = int(os.environ.get("MAX_NEW", "64"))
BENCH_TOKENS = int(os.environ.get("BENCH_TOKENS", "0"))
PROMPT = os.environ.get(
    "PROMPT", "### Instruction:\nExplain what a mixture-of-experts language model is.\n\n### Response:\n"
)


def load_for_inference():
    """Streaming NF4 load + adapter attach + freeze; returns ``(tokenizer, model)`` in eval mode."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model, _ = load_moe_4bit_streaming(
        MODEL,
        DEVICE,
        DTYPE,
        R,
        ALPHA,
        offload=OFFLOAD_EXPERTS,
        pin=OFFLOAD_PIN,
        prefetch=PREFETCH and OFFLOAD_EXPERTS,
        quant_type=QUANT_TYPE,
    )
    # Under offload the experts deliberately live in pinned CPU RAM; a blanket .to() would undo that.
    if not OFFLOAD_EXPERTS:
        model.to(DEVICE)

    if ADAPTER:
        sd = torch.load(ADAPTER, map_location="cpu", weights_only=True)
        # train.py saves every trainable "lora" tensor; attention adapters are present iff the run
        # trained them — wrap the projections first so those keys have somewhere to land.
        if any("self_attn" in k for k in sd):
            add_attention_lora(model, R, ALPHA, DTYPE)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(
                f"adapter {ADAPTER}: {len(unexpected)} tensors did not match the model "
                f"(first: {unexpected[0]}) — check MODEL/R/ALPHA against the training run"
            )
        log(f"  adapter: {len(sd)} tensors loaded from {ADAPTER}")

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.config.use_cache = True
    return tok, model


@torch.no_grad()
def timed_decode(model, tok, prompt, n_tokens):
    """Greedy-decode exactly ``n_tokens`` with a manual KV-cache loop, timing prefill and decode
    separately (CUDA-event-free version: sync + wall clock, adequate at per-token milliseconds)."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

    torch.cuda.synchronize()
    t0 = time.time()
    out = model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    t_prefill = time.time() - t0

    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    pieces = []
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_tokens):
        pieces.append(nxt)
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    t_decode = time.time() - t0

    text = tok.decode(torch.cat(pieces, dim=1)[0], skip_special_tokens=True)
    return text, t_prefill, n_tokens / t_decode


def main():
    torch.manual_seed(0)
    gemv = _gemv_4bit_matches_dequant() if os.environ.get("E4B_INFER_GEMV", "1") != "0" else False
    log(
        f"inference: {MODEL} | offload={'on' if OFFLOAD_EXPERTS else 'off'}"
        f" prefetch={'on' if (PREFETCH and OFFLOAD_EXPERTS) else 'off'}"
        f" gemv={'on' if gemv else 'off'}"
        f" fastpath={'on' if os.environ.get('E4B_DECODE_FASTPATH', '1') != '0' else 'off'}"
        f" | adapter={'yes' if ADAPTER else 'no'}"
    )
    tok, model = load_for_inference()
    torch.cuda.synchronize()
    log(f"loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()

    if BENCH_TOKENS > 0:
        # Warmup: probes (gemv), allocator pools, prefetch steady state — outside the timed region.
        timed_decode(model, tok, PROMPT, min(8, BENCH_TOKENS))
        text, t_prefill, tps = timed_decode(model, tok, PROMPT, BENCH_TOKENS)
        peak = torch.cuda.max_memory_allocated() / 1e9
        log(f"  prompt tokens: {len(tok(PROMPT).input_ids)} | prefill {t_prefill:.3f}s")
        log(f"  decode: {BENCH_TOKENS} tokens @ {tps:.2f} tok/s | peak GPU {peak:.2f} GB")
        log(f"  text: {text[:200]!r}")
        # Machine-readable summary line (the bench script greps this).
        print(
            f"BENCH offload={int(OFFLOAD_EXPERTS)} prefetch={int(PREFETCH and OFFLOAD_EXPERTS)} "
            f"gemv={int(gemv)} fastpath={int(os.environ.get('E4B_DECODE_FASTPATH', '1') != '0')} "
            f"tok_s={tps:.3f} prefill_s={t_prefill:.3f} peak_gb={peak:.3f}",
            flush=True,
        )
    else:
        t0 = time.time()
        out = model.generate(
            **tok(PROMPT, return_tensors="pt").to(DEVICE),
            max_new_tokens=MAX_NEW,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tok.eos_token_id,
        )
        dt = time.time() - t0
        n_prompt = len(tok(PROMPT).input_ids)
        text = tok.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
        log(
            f"  {out.shape[1] - n_prompt} new tokens in {dt:.1f}s | peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )
        log(f"---\n{text}")


if __name__ == "__main__":
    main()
