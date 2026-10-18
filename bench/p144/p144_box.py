#!/usr/bin/env python3
"""p144_box.py -- lane P144's BOX script (bench/p144/PREREG-p144.md). One fresh process per arm.

The subject is P137's configuration, here with Mixtral-8x7B-Instruct on an RTX 4090: its NF4 experts in pinned host
RAM (``offload=True``), ``enable_fast_train(dgrad=True)`` (the grouped kernels, its reentrant checkpoint), the chunked
loss at its default, r 16 / alpha 16, AdamW 8-bit at lr 2e-4, alpaca at seq 2048, micro-batch 2 x accumulation 4.
The two arms differ only in the accumulation loop:
  A0  the stock loop: per micro-batch, ``model(**b)`` then ``(loss * w).backward()`` (train.py's loop);
  L1  ``layer_major_step(model, batches, weights)`` (engines/layer_major.py), with the same weights.
Every arm runs the same steps on the same micro-batches from the same seed. Recorded per step: the wall (CUDA-synced),
the loss, the host-to-device stagings (``E4B_OFFLOAD_STATS``'s copy count), and the peak device memory; per arm: the
step-0 per-micro-batch losses, a held-out loss after the last step (REPORTING), and the versions.

    python p144_box.py --mode abba --model DIR --data data.json --outdir DIR   (P137 Amendment 2: one load, four arms)
    python p144_box.py --self-test

P137 Amendment 2 (after p137-4090-2): one process loads the model once and runs the four arms A0a L1a L1b A0b in turn.
Before each arm it restores the adapters from the snapshot taken right after the load (asserted by hash), builds a new
optimizer, reseeds, evicts every offloaded layer, empties the allocator cache and resets the peak counter. The fresh-process
design re-quantized the experts in every arm (the loader fuses offloaded expert rows on the host CPU), about 38 min per arm
on a contended host.
"""
import argparse
import json
import os
import sys
import time

SEQ, MB_ROWS, ACCUM = 2048, 2, 4
WARM, TIMED = 2, 32             # step 0 (the equality step) is the first warm step; P137 Amendment 3: 32 timed (was 8)
HELD_ROWS = 16
SEED, LR, R, ALPHA = 3407, 2e-4, 16, 16
LOOPS = {"A0": "stock", "L1": "layer_major"}
TAGS = ("A0a", "L1a", "L1b", "A0b")             # ABBA, in one process (P137 Amendment 2)


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
        print("P144_STEP " + json.dumps(rec), flush=True)
        rec["losses"] = [float(x) for x in losses]
        steps.append(rec)


def _load(model_dir):
    import torch
    from transformers import AutoTokenizer

    from experts4bit_qlora.engines import offload as off
    from experts4bit_qlora.engines.fast import enable_fast_train
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    os.environ["E4B_OFFLOAD_STATS"] = "1"
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_dir)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model, _ = load_moe_4bit_streaming(model_dir, "cuda", torch.bfloat16, r=R, alpha=ALPHA, offload=True, pin=True,
                                       quant_type="nf4")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    n_fast = enable_fast_train(model, dgrad=True)
    if n_fast <= 0:
        raise SystemExit(f"P144 REFUSED: enable_fast_train patched {n_fast} modules")
    return tok, pad_id, model, n_fast, off.offload_handles(model)


def adapter_hash(params):
    """sha256 over every trainable tensor's bytes, in order: every arm must start from the same adapters."""
    import hashlib

    import torch

    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().to("cpu").contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _reset(model, params, snapshot, handles):
    """Between arms: adapters back to the snapshot, no grads, every offloaded layer evicted, caches and peaks reset."""
    import torch

    from experts4bit_qlora.engines import offload as off

    import gc
    import random

    import numpy as np

    with torch.no_grad():
        for p, s in zip(params, snapshot):
            p.copy_(s)
            p.grad = None
    for h in handles:
        h.evict()
    off._ExpertOffload._resident = None
    off.reset_offload_stats()
    gc.collect()                        # the previous arm's optimizer state goes here, before the memory check
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(SEED)             # CPU and CUDA generators
    random.seed(SEED)
    np.random.seed(SEED)


def _arm(tag, model, params, snapshot, snap_hash, handles, plan, held, pad_id, n_fast, model_dir, out_path, mem_baseline,
         warmup):
    import torch

    import bitsandbytes as bnb
    from experts4bit_qlora.engines import ckpt_offload

    loop = LOOPS[tag[:2]]
    _reset(model, params, snapshot, handles)
    start_hash = adapter_hash(params)
    mem_at_start = torch.cuda.memory_allocated()
    model.train()
    opt = bnb.optim.AdamW8bit(params, lr=LR)
    steps = []
    try:
        _train(model, plan, pad_id, "cuda", loop, handles, opt, steps)
    except torch.cuda.OutOfMemoryError as e:
        rec = {"tag": tag, "loop": loop, "model": model_dir, "oom": f"step {len(steps)}: {str(e)[:200]}", "steps": steps,
               "adapter_sha_at_start": start_hash, "adapter_sha_snapshot": snap_hash, "mem_at_start": mem_at_start,
               "mem_baseline": mem_baseline, "warmup": warmup}
        json.dump(rec, open(out_path, "w"), indent=1)
        print("P144_ARM " + json.dumps({"tag": tag, "oom": rec["oom"]}), flush=True)
        return
    step0 = steps[0].pop("losses")
    for st in steps[1:]:
        st.pop("losses")
    model.eval()
    with torch.no_grad():
        hl = [float(model(**pad(held[i:i + MB_ROWS], pad_id, "cuda")).loss) for i in range(0, len(held), MB_ROWS)]
    import importlib.metadata as md

    import transformers

    out = {"tag": tag, "loop": loop, "model": model_dir, "steps": steps, "step0_losses": step0,
           "step0_hex": [x.hex() for x in step0], "summary": summarize(steps), "heldout": sum(hl) / len(hl),
           "n_fast": n_fast, "n_offloaded": len(handles), "ckpt": dict(ckpt_offload.CKPT_OFFLOAD_STATS),
           "adapter_sha_at_start": start_hash, "adapter_sha_snapshot": snap_hash, "process": "one-process ABBA (A2)",
           "mem_at_start": mem_at_start, "mem_baseline": mem_baseline, "warmup": warmup,
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "bitsandbytes": md.version("bitsandbytes"), "gnf4": md.version("grouped-nf4-gemm"),
                        "e4b_sha": os.environ.get("E4B_SHA"), "gpu": torch.cuda.get_device_name(0)},
           "recipe": {"seq": SEQ, "mb_rows": MB_ROWS, "accum": ACCUM, "warm": WARM, "timed": TIMED, "seed": SEED,
                      "lr": LR, "r": R, "alpha": ALPHA}}
    json.dump(out, open(out_path, "w"), indent=1)
    print("P144_ARM " + json.dumps({k: out[k] for k in ("tag", "loop", "summary", "heldout", "step0_losses")}), flush=True)


def run_abba(a):
    """P137 Amendment 2: one load, then A0a L1a L1b A0b in this process, each from the same adapter snapshot."""
    t0 = time.perf_counter()
    tok, pad_id, model, n_fast, handles = _load(a.model)
    print("P144_LOAD " + json.dumps({"seconds": round(time.perf_counter() - t0, 1), "n_fast": n_fast,
                                     "n_offloaded": len(handles)}), flush=True)
    rows = [encode(tok, ex) for ex in json.load(open(a.data))]
    n_steps = WARM + TIMED
    plan = micro_batches(rows[: n_steps * ACCUM * MB_ROWS + 4 * HELD_ROWS], n_steps)
    held = [r for r in rows[n_steps * ACCUM * MB_ROWS:] if any(t != -100 for t in r[1])][:HELD_ROWS]
    params = [p for p in model.parameters() if p.requires_grad]
    snapshot = [p.detach().to("cpu").clone() for p in params]   # on the host: no arm's device peak carries it (review)
    snap_hash = adapter_hash(params)
    import torch

    # One untimed warm-up forward + backward (no optimizer step; the reset restores the adapters) BEFORE the baseline, so
    # the one-time lazy device workspaces of the first training step (p144-dry-9: +16.26 MiB from the second arm on, e.g.
    # the cuBLAS workspace) are inside the baseline rather than read as an arm-to-arm leak.
    # P137 Amendment 3: then one untimed layer-major step on step 0's micro-batches, so the layer-major path's first-use costs
    # (and the drift guard's one-time stock forwards, once per model) are paid here and not inside L1a's timed steps.
    from experts4bit_qlora.engines import offload as off
    from experts4bit_qlora.engines.layer_major import layer_major_step

    model.train()
    off.reset_offload_stats()
    warm = pad(plan[0][0], pad_id, "cuda")
    model(**warm).loss.backward()
    torch.cuda.synchronize()
    warmup = {"stock_stagings": len(off._stats().copies)}
    _reset(model, params, snapshot, handles)
    batches = [pad(mb, pad_id, "cuda") for mb in plan[0]]
    layer_major_step(model, batches, weights_of(batches), handles=handles)
    torch.cuda.synchronize()
    warmup["layer_major_stagings"] = len(off._stats().copies)
    print("P144_WARMUP " + json.dumps(warmup), flush=True)
    _reset(model, params, snapshot, handles)
    mem_baseline = torch.cuda.memory_allocated()       # post-load, post-warm-up, post-snapshot, everything evicted
    for tag in TAGS:
        _arm(tag, model, params, snapshot, snap_hash, handles, plan, held, pad_id, n_fast, a.model,
             os.path.join(a.outdir, f"arm_{tag}.json"), mem_baseline, warmup)
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
    assert set(LOOPS) == {"A0", "L1"} and LOOPS["L1"] == "layer_major" and TAGS == ("A0a", "L1a", "L1b", "A0b")
    n += 1
    assert (WARM, TIMED) == (2, 32)                                    # P137 Amendment 3
    n += 1
    print(f"p144_box self-test OK ({n}/11 cases)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--mode", choices=["abba"])
    p.add_argument("--model")
    p.add_argument("--data", help="alpaca rows as a JSON list (the staged slice)")
    p.add_argument("--outdir")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    return run_abba(a)


if __name__ == "__main__":
    sys.exit(main())
