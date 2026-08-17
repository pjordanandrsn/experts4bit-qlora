# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""G7 quality harness: what FP8 KV costs a model, measured against FP16.

The gate is non-negotiable and its stop condition is explicit — a failed
quality clause means the format does not ship, whatever it does for the
batch ceiling. So this harness runs before the fused kernel exists, and it
is built to be falsifiable rather than reassuring:

* **Null control** — the cache object with quantization OFF, against the
  model's own stock cache. Any delta here is the harness's plumbing, not
  the format, and it must be ~0 or every later number is unreadable.
* **Positive control** — a deliberately destructive 2-bit store. The
  harness MUST see a large degradation. A harness that reports "no change"
  for `crush` is measuring nothing, and would happily certify FP8 for the
  same reason. This check gates the verdict: if the positive control does
  not trip, the run reports FAILED-TO-MEASURE rather than a pass.

Metrics: perplexity on held-out wikitext-2 (chunked, so the cache is read
across chunk boundaries rather than only within one forward), and LAMBADA
last-word accuracy — chosen over a same-length classification task because
it cannot be answered without the long-range context the KV cache holds,
which is precisely what a KV format can damage.

Arms share one loaded model and identical token streams, so the deltas are
paired: the absolute perplexity of a small model is uninteresting, the
difference between two storage formats over the same tokens is the result.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch


def _wikitext(tok, n_chunks, chunk):
    from datasets import load_dataset
    # namespaced id: current hub clients reject bare 'wikitext'
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = n_chunks * chunk
    if ids.numel() < need:
        raise RuntimeError(f"corpus has {ids.numel()} tokens, need {need}")
    return ids[:need].view(n_chunks, chunk)


def _lambada(tok, n):
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        text = row["text"].strip()
        ctx, _, last = text.rpartition(" ")
        c = tok(ctx, return_tensors="pt").input_ids[0]
        # add_special_tokens=False is load-bearing: tokenizers that prepend
        # BOS (Llama's does, Qwen's does not) would put an unpredictable
        # sentinel at target position 0, so EVERY item scores a miss and the
        # task reports a clean, plausible 0.0 for all arms — a metric that
        # measures nothing while looking like a result.
        t = tok(" " + last, return_tensors="pt",
                add_special_tokens=False).input_ids[0]
        if c.numel() > 8 and t.numel() >= 1:
            out.append((c, t))
    return out


# arm -> (value_mode, key_mode). "fp8_vonly" spends bytes asymmetrically:
# keys stay bf16 because they are the outlier-heavy side (amax/rms ~10 vs
# ~3), which this repo's NF4-KV work also found the expensive side.
# arm -> (value_mode, key_mode, key_group, value_group)
_ARMS = {
    "stock": None,
    "off": ("off", "off", None, None),
    "fp8": ("fp8", "fp8", None, None),
    "fp8_kg64": ("fp8", "fp8", 64, None),
    "fp8_kg32": ("fp8", "fp8", 32, None),
    "fp8_vonly": ("fp8", "off", None, None),
    "int4": ("int4", "int4", None, None),
    "crush": ("crush", "crush", None, None),
}


def _make_cache(arm):
    spec = _ARMS[arm]
    if spec is None:
        return None
    from experts4bit_qlora.engines.fp8_kv_cache import Fp8KVCache
    return Fp8KVCache(mode=spec[0], key_mode=spec[1],
                      key_group=spec[2], value_group=spec[3])


def ppl_arm(model, chunks, mode, seq_chunk):
    """Chunked teacher-forced NLL. Each row of `chunks` is one independent
    sequence fed in pieces, so the cache is written in one call and READ in
    the next — a single whole-sequence forward would never exercise a
    cross-call KV read."""
    total_nll, total_tok = 0.0, 0
    for row in chunks:
        cache = _make_cache(mode)
        ids = row.unsqueeze(0).cuda()
        past = cache
        prev_logits = None
        for i in range(0, ids.shape[1], seq_chunk):
            piece = ids[:, i:i + seq_chunk]
            with torch.no_grad():
                kw = {"past_key_values": past} if past is not None else {}
                out = model(input_ids=piece, use_cache=True, **kw)
            past = out.past_key_values
            logits = out.logits.float()
            # score piece[1:] from this chunk's logits, and piece[0] from
            # the previous chunk's last logit — no token is skipped or
            # double-counted at a boundary
            if prev_logits is not None:
                lp = torch.log_softmax(prev_logits, dim=-1)
                total_nll -= lp[0, piece[0, 0]].item()
                total_tok += 1
            if piece.shape[1] > 1:
                lp = torch.log_softmax(logits[:, :-1], dim=-1)
                tgt = piece[:, 1:]
                total_nll -= lp.gather(2, tgt.unsqueeze(-1)).sum().item()
                total_tok += tgt.numel()
            prev_logits = logits[:, -1]
    return math.exp(total_nll / total_tok), total_tok


def lambada_arm(model, items, mode):
    """Greedy last-word accuracy: every target token must be the argmax,
    teacher-forced through the target."""
    hit = 0
    for ctx, tgt in items:
        cache = _make_cache(mode)
        with torch.no_grad():
            kw = {"past_key_values": cache} if cache is not None else {}
            out = model(input_ids=ctx.unsqueeze(0).cuda(), use_cache=True,
                        **kw)
            past = out.past_key_values
            ok = True
            nxt = out.logits[:, -1].argmax(-1)
            for j, want in enumerate(tgt.tolist()):
                if int(nxt) != want:
                    ok = False
                    break
                if j + 1 < tgt.numel():
                    out = model(input_ids=torch.tensor([[want]]).cuda(),
                                past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    nxt = out.logits[:, -1].argmax(-1)
        hit += int(ok)
    return hit / len(items)


def main(model_id, n_chunks, chunk, seq_chunk, n_lambada, out_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(1689)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).cuda().eval()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads)

    chunks = _wikitext(tok, n_chunks, chunk)
    lam = _lambada(tok, n_lambada)
    print(f"corpus {chunks.numel()} tokens · lambada {len(lam)} items",
          flush=True)

    arms = {}
    for mode in _ARMS:
        t0 = time.time()
        ppl, ntok = ppl_arm(model, chunks, mode, seq_chunk)
        acc = lambada_arm(model, lam, mode)
        arms[mode] = {"ppl": ppl, "lambada_acc": acc, "tokens": ntok,
                      "secs": time.time() - t0}
        print(f"ARM {mode:6s} ppl={ppl:.5f} lambada={acc:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    base = arms["off"]["ppl"]
    def d(m):
        return (arms[m]["ppl"] - base) / base

    null_delta = abs(arms["stock"]["ppl"] - base) / base
    crush_delta = d("crush")
    # the harness certifies nothing unless it demonstrably SEES damage
    measured = crush_delta > 0.05
    fp8_delta = d("fp8")
    # ...and the downstream task certifies nothing unless the BASELINE can
    # do it. A task the unquantized model scores 0 on is not a hard task,
    # it is a broken metric (a tokenizer prepending BOS to the target does
    # exactly this), and reporting "FP8 changed nothing" from it would be
    # the same lie as reporting it from a KV that was never read.
    lam_base = arms["off"]["lambada_acc"]
    lam_usable = lam_base > 0.05

    from experts4bit_qlora.engines.fp8_kv_cache import Fp8KVCache
    bpt = {m: Fp8KVCache(mode=_ARMS[m][0], key_mode=_ARMS[m][1],
                         key_group=_ARMS[m][2], value_group=_ARMS[m][3]
                         ).bytes_per_token(cfg.num_key_value_heads, head_dim,
                                           cfg.num_hidden_layers)
           for m in ("off", "fp8", "fp8_kg64", "fp8_kg32", "fp8_vonly",
                     "int4")}

    others = [m for m in _ARMS if m != "off"]
    rep = {"model": model_id, "arms": arms,
           "ppl_delta_vs_off": {m: d(m) for m in others},
           "lambada_delta_vs_off": {
               m: arms[m]["lambada_acc"] - arms["off"]["lambada_acc"]
               for m in others},
           "bytes_per_token": bpt,
           "controls": {"null_delta": null_delta,
                        "null_ok": null_delta < 0.002,
                        "positive_control_delta": crush_delta,
                        "harness_can_measure": measured,
                        "lambada_baseline": lam_base,
                        "lambada_usable": lam_usable},
           "gate_g7_quality": {
               "fp8_ppl_delta": fp8_delta,
               "fp8_lambada_delta": (arms["fp8"]["lambada_acc"] - lam_base
                                     if lam_usable else None),
               "bar": 0.005,
               "quality_ok": bool(measured and lam_usable
                                  and fp8_delta <= 0.005),
               "verdict": (
                   "FAILED_TO_MEASURE" if not (measured and lam_usable)
                   else ("PASS" if fp8_delta <= 0.005 else "MISS"))}}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    name = model_id.split("/")[-1]
    (Path(out_dir) / f"g7_quality_{name}.json").write_text(
        json.dumps(rep, indent=2))
    print("G7_QUALITY " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--chunks", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seq-chunk", type=int, default=256)
    ap.add_argument("--lambada", type=int, default=200)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    main(a.model, a.chunks, a.chunk, a.seq_chunk, a.lambada, a.out)
