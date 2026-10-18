#!/usr/bin/env python3
"""p137_box.py -- lane P137's BOX script (bench/p137/PREREG-p137.md). One fresh process per arm.

The subject is TC3's ``fused_attn4_m_offload`` configuration on a 24 GB card: Qwen3-30B-A3B's NF4 experts in pinned host
RAM (``offload=True``), ``enable_fast_train(dgrad=True)`` (the grouped kernels, its reentrant checkpoint), the chunked
loss at its default, r 16 / alpha 16, AdamW 8-bit at lr 2e-4, alpaca at seq 2048, micro-batch 2 x accumulation 4.
The two arms differ only in the accumulation loop:
  A0  the stock loop: per micro-batch, ``model(**b)`` then ``(loss * w).backward()`` (train.py's loop);
  L1  ``layer_major_step(model, batches, weights)`` (engines/layer_major.py), with the same weights.
Every arm runs the same steps on the same micro-batches from the same seed. Recorded per step: the wall (CUDA-synced),
the loss, the host-to-device stagings (``E4B_OFFLOAD_STATS``'s copy count), and the peak device memory; per arm: the
step-0 per-micro-batch losses, a held-out loss after the last step (REPORTING), and the versions.

    python p137_box.py --mode arm --loop stock|layer_major --model DIR --out arm_A0a.json --tag A0a
    python p137_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

SEQ, MB_ROWS, ACCUM = 2048, 2, 4
WARM, TIMED = 2, 8              # step 0 (the equality step) is the first warm step
HELD_ROWS = 16
SEED, LR, R, ALPHA = 3407, 2e-4, 16, 16
LOOPS = {"A0": "stock", "L1": "layer_major"}


def encode(tok, ex, seq=SEQ):
    """train.py's encode_alpaca, at seq 2048: the prompt masked, loss on the response."""
    head = f"### Instruction:\n{ex['instruction']}\n\n"
    if ex.get("input"):
        head += f"### Input:\n{ex['input']}\n\n"
    prompt = head + "### Response:\n"
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    full = tok(prompt + ex["output"] + tok.eos_token, add_special_tokens=True)["input_ids"][:seq]
    labels = list(full)
    for i in range(min(len(p_ids), len(labels))):
        labels[i] = -100
    return full, labels


def micro_batches(rows, n_steps, accum=ACCUM, mb_rows=MB_ROWS):
    """Consecutive rows -> n_steps x accum micro-batches of mb_rows rows; each a list of (ids, labels). Rows with no
    supervised token are dropped first (train.py's filter)."""
    rows = [r for r in rows if any(t != -100 for t in r[1])]
    need = n_steps * accum * mb_rows
    if len(rows) < need:
        raise ValueError(f"{len(rows)} usable rows < {need}")
    out = [rows[i:i + mb_rows] for i in range(0, need, mb_rows)]
    return [out[s * accum:(s + 1) * accum] for s in range(n_steps)]


def pad(mb, pad_id, device):
    """Right-pad one micro-batch: labels -100 and attention 0 on the padding (train.py's _pad_batch)."""
    import torch
    w = max(len(ids) for ids, _ in mb)
    ids = torch.full((len(mb), w), pad_id, dtype=torch.long)
    lbl = torch.full((len(mb), w), -100, dtype=torch.long)
    att = torch.zeros((len(mb), w), dtype=torch.long)
    for i, (a, b) in enumerate(mb):
        ids[i, :len(a)] = torch.tensor(a)
        lbl[i, :len(b)] = torch.tensor(b)
        att[i, :len(a)] = 1
    return {"input_ids": ids.to(device), "labels": lbl.to(device), "attention_mask": att.to(device)}


def weights_of(batches):
    """The stock loop's weights: 1 / accum each (train.py's ``(out.loss / GRAD_ACCUM).backward()``)."""
    return [1.0 / len(batches)] * len(batches)


def summarize(steps, warm=WARM):
    """Per-arm summary from the per-step records: the median timed wall, and the steady stagings (all timed steps)."""
    timed = [s["wall_s"] for s in steps[warm:]]
    srt = sorted(timed)
    mid = len(srt) // 2
    med = srt[mid] if len(srt) % 2 else (srt[mid - 1] + srt[mid]) / 2
    # peak_alloc (the STRICT memory term) is over the timed steps: step 0 also carries layer-major's one-time drift-guard
    # forward, a cost no steady step pays. The all-steps peak is kept for the record.
    return {"median_s": med, "timed_walls": timed, "stagings_steady": sorted({s["stagings"] for s in steps[1:]}),
            "stagings_step0": steps[0]["stagings"], "peak_alloc": max(s["peak_alloc"] for s in steps[warm:]),
            "peak_alloc_all": max(s["peak_alloc"] for s in steps)}


def _train(model, plan, pad_id, dev, loop, handles, opt, steps):
    """The timed loop. Appends one record per step to ``steps`` (so an OOM keeps the steps before it)."""
    import torch

    from experts4bit_qlora.engines import offload as off
    from experts4bit_qlora.engines.layer_major import layer_major_step

    for s, mbs in enumerate(plan):
        batches = [pad(mb, pad_id, dev) for mb in mbs]
        w = weights_of(batches)
        off.reset_offload_stats()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if loop == "stock":
            losses = []
            for b, wk in zip(batches, w):
                out = model(**b)
                (out.loss * wk).backward()
                losses.append(out.loss.detach())
        else:
            losses = layer_major_step(model, batches, w, handles=handles)
        opt.step()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        st = off._stats()
        rec = {"step": s, "wall_s": wall, "loss": sum(float(x) * wk for x, wk in zip(losses, w)),
               "stagings": len(st.copies), "peak_alloc": torch.cuda.max_memory_allocated()}
        print("P137_STEP " + json.dumps(rec), flush=True)
        rec["losses"] = [float(x) for x in losses]
        steps.append(rec)


def run_arm(a):
    import torch
    from transformers import AutoTokenizer

    from experts4bit_qlora.engines import offload as off
    from experts4bit_qlora.engines.fast import enable_fast_train
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    loop = LOOPS[a.tag[:2]]
    assert loop == a.loop, (a.tag, a.loop)
    os.environ["E4B_OFFLOAD_STATS"] = "1"
    dev = "cuda"
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model, _ = load_moe_4bit_streaming(a.model, dev, torch.bfloat16, r=R, alpha=ALPHA, offload=True, pin=True,
                                       quant_type="nf4")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    n_fast = enable_fast_train(model, dgrad=True)
    if n_fast <= 0:
        raise SystemExit(f"P137 REFUSED: enable_fast_train patched {n_fast} modules")
    from experts4bit_qlora.engines import ckpt_offload
    handles = off.offload_handles(model)
    params = [p for p in model.parameters() if p.requires_grad]
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=LR)

    rows = [encode(tok, ex) for ex in json.load(open(a.data))]
    n_steps = WARM + TIMED
    plan = micro_batches(rows[: n_steps * ACCUM * MB_ROWS + 4 * HELD_ROWS], n_steps)
    held = [r for r in rows[n_steps * ACCUM * MB_ROWS:] if any(t != -100 for t in r[1])][:HELD_ROWS]

    steps = []
    try:
        _train(model, plan, pad_id, dev, loop, handles, opt, steps)
    except torch.cuda.OutOfMemoryError as e:
        rec = {"tag": a.tag, "loop": loop, "model": a.model, "oom": f"step {len(steps)}: {str(e)[:200]}", "steps": steps}
        json.dump(rec, open(a.out, "w"), indent=1)
        print("P137_ARM " + json.dumps({"tag": a.tag, "oom": rec["oom"]}), flush=True)
        return 0
    step0 = steps[0].pop("losses")
    step0_hex = [x.hex() for x in step0]
    for st in steps[1:]:
        st.pop("losses")
    model.eval()
    with torch.no_grad():
        hl = []
        for i in range(0, len(held), MB_ROWS):
            hl.append(float(model(**pad(held[i:i + MB_ROWS], pad_id, dev)).loss))
    import importlib.metadata as md
    import transformers

    out = {"tag": a.tag, "loop": loop, "model": a.model, "steps": steps, "step0_losses": step0, "step0_hex": step0_hex,
           "summary": summarize(steps), "heldout": sum(hl) / len(hl), "n_fast": n_fast, "n_offloaded": len(handles),
           "ckpt": dict(ckpt_offload.CKPT_OFFLOAD_STATS),
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "bitsandbytes": md.version("bitsandbytes"), "gnf4": md.version("grouped-nf4-gemm"),
                        "e4b_sha": os.environ.get("E4B_SHA"), "gpu": torch.cuda.get_device_name(0)},
           "recipe": {"seq": SEQ, "mb_rows": MB_ROWS, "accum": ACCUM, "warm": WARM, "timed": TIMED, "seed": SEED,
                      "lr": LR, "r": R, "alpha": ALPHA}}
    json.dump(out, open(a.out, "w"), indent=1)
    print("P137_ARM " + json.dumps({k: out[k] for k in ("tag", "loop", "summary", "heldout", "step0_losses")}), flush=True)
    return 0


def self_test():
    n = 0

    class Tok:
        eos_token = "</s>"

        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [ord(c) % 97 for c in text]}

    ids, lbl = encode(Tok(), {"instruction": "hi", "input": "", "output": "yo"})
    plen = len("### Instruction:\nhi\n\n### Response:\n")
    assert lbl[:plen] == [-100] * plen and lbl[plen:] == ids[plen:]
    n += 1
    ids2, _ = encode(Tok(), {"instruction": "x" * 5000, "input": "", "output": "y"})
    assert len(ids2) == SEQ
    n += 1
    rows = [([1, 2, 3], [-100, 2, 3])] * 9 + [([1], [-100])]
    plan = micro_batches(rows, 2, accum=2, mb_rows=2)
    assert len(plan) == 2 and all(len(s) == 2 and all(len(mb) == 2 for mb in s) for s in plan)
    n += 1
    try:
        micro_batches(rows, 3, accum=2, mb_rows=2)
        raise AssertionError("short rows must refuse")
    except ValueError:
        n += 1
    assert weights_of([0, 0, 0, 0]) == [0.25] * 4
    n += 1
    steps = [{"wall_s": w, "stagings": st, "peak_alloc": p} for w, st, p in
             [(9.0, 381, 1), (5.0, 380, 2), (4.0, 380, 3), (6.0, 380, 2), (5.5, 380, 1)]]
    sm = summarize(steps, warm=2)
    assert sm["median_s"] == 5.5 and sm["stagings_steady"] == [380] and sm["stagings_step0"] == 381
    n += 1
    assert sm["peak_alloc"] == 3 and sm["peak_alloc_all"] == 3
    n += 1
    steps[0]["peak_alloc"] = 9
    assert summarize(steps, warm=2)["peak_alloc"] == 3 and summarize(steps, warm=2)["peak_alloc_all"] == 9
    n += 1
    sm2 = summarize(steps[:4], warm=2)
    assert sm2["median_s"] == 5.0
    n += 1
    assert set(LOOPS) == {"A0", "L1"} and LOOPS["L1"] == "layer_major"
    n += 1
    print(f"p137_box self-test OK ({n}/10 cases)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--mode", choices=["arm"])
    p.add_argument("--loop", choices=["stock", "layer_major"])
    p.add_argument("--model")
    p.add_argument("--data", help="alpaca rows as a JSON list (the runner's prepared slice)")
    p.add_argument("--out")
    p.add_argument("--tag", choices=["A0a", "L1a", "L1b", "A0b"])
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    return run_arm(a)


if __name__ == "__main__":
    sys.exit(main())
