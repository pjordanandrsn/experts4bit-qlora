"""bench/dq4/dq4_cap.py -- lane DQ4, BOX side: ONE capacity measurement per process (bench/dq4/DQ4-PREREG.md).

The subject is DQ3's (``bench/dq3/dq3_arm.py``, staged beside this file): Qwen3-32B architecture, random NF4, PEFT LoRA,
non-reentrant checkpointing. The LM loss is ``--loss``: ``chunked`` (``enable_chunked_lm_loss(chunk=512)``, the graded pairs:
the full-vocabulary logits would otherwise be a large term identical in R and S) or ``stock`` (Hugging Face's, descriptive).
The arm is R (resident) or S (``enable_dense_offload(train_prefetch=True)``, with #1183's
late-bound backward). The allocator configuration is whatever ``PYTORCH_CUDA_ALLOC_CONF`` the runner set before CUDA
initialised; it is recorded, not chosen here.

Modes:
  ladder   sequence lengths --start, --start+--step, ... up to --top, ``--steps`` AdamW steps each (default SDPA, as a
           real run trains), ascending; stops at the FIRST CUDA OOM. Records every rung.
  confirm  one rung at --seq in this fresh process (the ladder's boundary re-measured without the ladder's history).
Writes one JSON receipt (``--out``); the verdict is dq4_reduce.py's.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "dq3")]     # staged flat on the box; bench/dq3/ in the repo
import dq3_arm as A  # noqa: E402


def late_bound_count(handles) -> int:
    return sum(1 for h in handles for mod, _a, _p, _hm in h.slots if getattr(mod, "_dense_offload_late_bound", False))


def rung(pm, opt, seq: int, steps: int, vocab: int) -> dict:
    """``steps`` training steps at ``seq`` tokens; ok=False with the error on a CUDA OOM."""
    g = torch.Generator(device="cpu").manual_seed(1234 + seq)
    ids = torch.randint(0, vocab, (1, seq), generator=g).to("cuda")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = {"seq": seq, "ok": True, "steps": []}
    try:
        for _ in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = pm(input_ids=ids, labels=ids).loss
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            lv = float(loss.detach())
            out["steps"].append({"s": time.perf_counter() - t0, "loss": lv, "finite": math.isfinite(lv)})
            del loss
    except torch.OutOfMemoryError as exc:
        out["ok"] = False
        out["error"] = str(exc)[:300]
    out["peak_alloc"] = torch.cuda.max_memory_allocated()
    out["peak_reserved"] = torch.cuda.max_memory_reserved()
    if not out["ok"]:
        opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=("R", "S"))
    p.add_argument("--mode", required=True, choices=("ladder", "confirm"))
    p.add_argument("--out", required=True)
    p.add_argument("--seq", type=int, default=0, help="confirm: the rung")
    p.add_argument("--loss", required=True, choices=("chunked", "stock"))
    p.add_argument("--start", type=int, default=2048)
    p.add_argument("--step", type=int, default=1024)
    p.add_argument("--top", type=int, default=32768)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--layers", type=int, default=64, help="rehearsal only: fewer layers")
    p.add_argument("--rehearsal", action="store_true")
    args = p.parse_args()
    if args.mode == "confirm" and args.seq <= 0:
        p.error("--mode confirm needs --seq")

    from experts4bit_qlora.engines.chunked_lm_loss import CHUNKED_LM_LOSS_STATS, enable_chunked_lm_loss
    from experts4bit_qlora.engines.dense_offload import dense_offload_report, enable_dense_offload

    over = {"num_hidden_layers": args.layers}
    rec = {"schema": "dq4-cap/1", "arm": args.arm, "mode": args.mode, "loss": args.loss, "rehearsal": bool(args.rehearsal),
           "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or "default",
           "started_at": time.strftime("%FT%TZ", time.gmtime()), "device": torch.cuda.get_device_name(),
           "total_bytes": torch.cuda.get_device_properties(0).total_memory,
           "config_overrides": over, "env": {k: os.environ.get(k) for k in ("E4B_SHA",)},
           "ladder": {"start": args.start, "step": args.step, "top": args.top, "steps": args.steps}}

    def flush():
        with open(args.out + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(args.out + ".tmp", args.out)

    t0 = time.time()
    model, cfg = A.build_model(over)
    pm = A.lora_wrap(model)
    rec["build_s"] = round(time.time() - t0, 1)
    rec["engagement"] = A.engagement(pm)
    if args.loss == "chunked":
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            patched = enable_chunked_lm_loss(pm, chunk=512)
        rec["chunked_loss"] = {"patched": patched == 1, "chunk": 512,
                               "refused": dict(CHUNKED_LM_LOSS_STATS.get("refused", {})),
                               "warnings": [str(w.message)[:200] for w in ws]}
    handles = []
    if args.arm == "S":
        handles = enable_dense_offload(pm, pin=True, prefetch=False, train_prefetch=True)
        rep = dense_offload_report(handles)
        rec["offload"] = {k: rep[k] for k in ("layers", "tensors", "host_bytes", "per_layer_bytes", "all_pinned")}
        rec["offload"]["late_bound_4bit"] = late_bound_count(handles)
    gc.collect()
    torch.cuda.empty_cache()
    rec["allocated_after_setup"] = torch.cuda.memory_allocated()
    rec["rungs"] = []
    flush()

    opt = torch.optim.AdamW([q for q in pm.parameters() if q.requires_grad], lr=2e-4)
    seqs = [args.seq] if args.mode == "confirm" else list(range(args.start, args.top + 1, args.step))
    for seq in seqs:
        r = rung(pm, opt, seq, args.steps, cfg.vocab_size)
        rec["rungs"].append(r)
        flush()
        print(f"DQ4 {args.arm} {rec['alloc_conf']} seq {seq}: {'ok' if r['ok'] else 'OOM'} "
              f"peak {r['peak_alloc'] / 2**30:.2f} / reserved {r['peak_reserved'] / 2**30:.2f} GiB", flush=True)
        if not r["ok"]:
            break
    oks = [r["seq"] for r in rec["rungs"] if r["ok"]]
    rec["max_ok"] = max(oks) if oks else None
    rec["first_oom"] = next((r["seq"] for r in rec["rungs"] if not r["ok"]), None)
    rec["finished_at"] = time.strftime("%FT%TZ", time.gmtime())
    flush()
    print(f"DQ4 {args.arm} {args.mode} done: max_ok {rec['max_ok']} first_oom {rec['first_oom']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
