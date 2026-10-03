#!/usr/bin/env python3
"""Lane P98's measurement (bench/p98/PREREG-p98.md; e4b#564): one ARM of hybrid decode through the production serving
stack (``serve_paged.build_engine``: the NF4 arena, the hybrid expert tier, all-VRAM placement, the fp8 paged KV on a
compact pool, the per-slot Gated DeltaNet state), on a fresh engine, in its own process.

Arms (each a separate process: ``build_engine`` sets ``hot_residency``'s grouping switches, which are module globals):
- ``g``  bucketed decode graphs, captured (``E4B_PAGED_GRAPHS=1``);
- ``e``  the same buckets, the same grouping, every step run EAGERLY on the same padded layout (``capture=False``):
         the bitwise oracle for ``g``'s replays;
- ``p``  plain eager decode, no graphs (the serving default).

Workloads, through the engine's own continuous scheduler, greedy, no stop set:
- ``W16`` 16 requests at once, ``--prompt``-token prompts (wikitext-2 test windows k * 4096, k = 0..15) with
  ``--new-base + --new-step * i`` new tokens, so sequences finish one by one and the active set walks 16 -> 1 through
  every bucket;
- ``W1``  one request (window 16), ``--b1-new`` new tokens: the batch-1 decode.

Every ``run_decode`` call is timed (CUDA-synchronised before and after) with its row count. Decode throughput is
sum(rows) / sum(time) after the first ``--warm-steps`` calls of each workload. The record also carries the engine's
census (graph status per bucket, replays and eager steps per bucket, the grouping switches, the KV and linear-state
layout) and the loaded commit, read from the hub snapshot path (a composite checkpoint's model config carries none).

REHEARSAL knobs (the A2000: no native e4m3, 12 GB), recorded under ``rehearsal``; the reducer voids any record that set
one: ``--stand-in-attention`` (SDPA over the pool's fp8 bytes instead of the kernel, arm ``p`` only),
``--placement solver --vram-gb N`` (experts outside VRAM).

    python p98_box.py --arm g --model ID --revision SHA --arena PATH --calib PATH --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

WIN = 4096


def wikitext_prompts(tok, ks, length):
    """Window k starts at token k * 4096 of wikitext-2-raw test, joined as the K8 corpus joins it."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids[0]
    out = []
    for k in ks:
        w = ids[k * WIN:k * WIN + length]
        assert w.numel() == length, f"window {k} has {w.numel()} tokens"
        out.append(w.tolist())
    return out


def new_tokens(n, base, step):
    """W16's per-request lengths: staggered, so the active set shrinks one sequence at a time."""
    return [base + step * i for i in range(n)]


class DecodeTimer:
    """Wraps ``runner.run_decode``: one record per call, (rows, milliseconds), CUDA-synchronised."""

    def __init__(self, runner, sync=True):
        self.inner, self.sync, self.calls = runner.run_decode, sync, []
        runner.run_decode = self

    def __call__(self, rids):
        if self.sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        got = self.inner(rids)
        if self.sync:
            torch.cuda.synchronize()
        self.calls.append((len(rids), (time.perf_counter() - t) * 1e3))
        return got


def decode_summary(calls, warm):
    """Throughput over the calls after the first ``warm``; and per-row-count mean step time."""
    use = calls[warm:]
    rows = sum(r for r, _ in use)
    ms = sum(m for _, m in use)
    per = {}
    for r, m in use:
        per.setdefault(r, []).append(m)
    return {"steps": len(calls), "timed_steps": len(use), "rows": rows, "ms": round(ms, 3),
            "tok_per_s": round(rows / ms * 1e3, 3) if ms else None,
            "ms_per_step_by_rows": {str(r): round(sum(v) / len(v), 3) for r, v in sorted(per.items())}}


def run_workload(sched, prompts, max_new):
    before = len(sched.done)
    rids = [sched.add_request(p, m) for p, m in zip(prompts, max_new)]
    sched.run_until_idle(max_steps=1_000_000)
    done = {r.rid: list(r.out) for r in sched.done[before:]}
    return [done[r] for r in rids]


def loaded_commit(model_id, revision):
    """The commit the hub cache resolved this checkpoint to (a local directory has none)."""
    if os.path.isdir(model_id):
        return None
    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id, revision=revision, local_files_only=True, allow_patterns=["config.json"])
    return os.path.basename(os.path.normpath(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("g", "e", "p"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buckets", default="1,2,4,8,16")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--new-base", type=int, default=48)
    ap.add_argument("--new-step", type=int, default=4)
    ap.add_argument("--b1-new", type=int, default=96)
    ap.add_argument("--warm-steps", type=int, default=3)
    ap.add_argument("--stand-in-attention", action="store_true", help="REHEARSAL: SDPA over the pool's fp8 bytes")
    ap.add_argument("--placement", default="all-vram", help="REHEARSAL when not all-vram")
    ap.add_argument("--vram-gb", default="1.2")
    a = ap.parse_args()
    buckets = tuple(int(x) for x in a.buckets.split(","))
    max_new = new_tokens(a.n, a.new_base, a.new_step)

    os.environ.update({
        "E4B_PAGED_MODEL": a.model, "E4B_PAGED_REVISION": a.revision, "E4B_PAGED_ARENA": a.arena,
        "E4B_PAGED_CALIB": a.calib, "E4B_PAGED_MAX_SEQS": str(a.n), "E4B_PAGED_BUCKETS": a.buckets,
        "E4B_PAGED_GRAPHS": "0" if a.arm == "p" else "1", "E4B_PAGED_PLACEMENT": a.placement,
        "E4B_PAGED_VRAM_GB": a.vram_gb, "E4B_PAGED_MAX_TOKENS_PER_SEQ": "512", "E4B_PAGED_CHUNK_TOKENS": "512"})

    import transformers

    from experts4bit_qlora.engines import linear_state, paged_runner
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    if a.arm == "e":
        # the padded-eager oracle: build_engine's own enable_decode_graphs call, with capture off
        orig = paged_runner.PagedModelRunner.enable_decode_graphs

        def no_capture(self, buckets=paged_runner.DEFAULT_BUCKETS, **kw):
            kw["capture"] = False
            return orig(self, buckets, **kw)

        paged_runner.PagedModelRunner.enable_decode_graphs = no_capture

    t0 = time.time()
    parts = build_engine(PagedServeConfig.from_env())
    build_s = time.time() - t0
    runner, sched, tok = parts.runner, parts.scheduler, parts.tokenizer
    kv = runner.kv
    if a.stand_in_attention:
        if a.arm != "p":
            raise SystemExit("--stand-in-attention is a rehearsal knob for arm p only")

        def stand_in(layer, q, slots=None, sm_scale=None, window=None, sinks=None, **_):
            outs = []
            for b, slot in enumerate(slots):
                kr, vr = kv.reference_kv(layer, slot)
                outs.append(torch.nn.functional.scaled_dot_product_attention(
                    q[b][None, :, None].float(), kr.permute(1, 0, 2)[None].float(), vr.permute(1, 0, 2)[None].float(),
                    scale=sm_scale, enable_gqa=True)[0, :, 0].to(q.dtype))
            return torch.stack(outs)

        kv.attention = stand_in
    prompts = wikitext_prompts(tok, list(range(a.n)) + [a.n], a.prompt)
    timer = DecodeTimer(runner)
    t1 = time.time()
    w16 = run_workload(sched, prompts[:a.n], max_new)
    w16_calls, timer.calls = timer.calls, []
    w1 = run_workload(sched, prompts[a.n:], [a.b1_new])
    w1_calls = timer.calls
    run_s = time.time() - t1
    pool = runner.linear_state
    lin = list(linear_state.linear_layers(runner.model.config))
    stats = getattr(runner, "graph_stats", None)
    rec = {
        "arm": a.arm, "model": a.model, "revision": a.revision, "loaded_commit": loaded_commit(a.model, a.revision),
        "transformers": transformers.__version__, "buckets": list(buckets), "n": a.n, "prompt": a.prompt,
        "max_new": max_new, "b1_new": a.b1_new, "warm_steps": a.warm_steps,
        "rehearsal": {"stand_in_attention": a.stand_in_attention, "placement": a.placement},
        "build_s": round(build_s, 1), "run_s": round(run_s, 1),
        "engine": {k: v for k, v in parts.info.items() if k in ("moe_layers", "experts", "top_k", "model_type", "kv",
                                                               "graph_status", "grouping")},
        "graph_stats": {str(k): dict(v) for k, v in stats.items()} if isinstance(stats, dict) else None,
        "layout": {"kv_pool_layers": kv.L, "attn_layers": list(runner.attn_layers), "pool_layers": list(runner.pool_layers),
                   "linear_layers": lin, "linear_state": pool is not None,
                   "linear_state_frozen": bool(pool is not None and pool.frozen),
                   "linear_state_allocated": bool(pool is not None and pool.allocated(lin))},
        "w16": {"tokens": w16, "decode": decode_summary(w16_calls, a.warm_steps),
                "rows_per_call": [r for r, _ in w16_calls]},
        "w1": {"tokens": w1, "decode": decode_summary(w1_calls, a.warm_steps)},
        "gpu": torch.cuda.get_device_name(0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    open(a.out, "w").write(json.dumps(rec, indent=1))
    d16, d1 = rec["w16"]["decode"], rec["w1"]["decode"]
    print(f"P98_ARM {a.arm}: build {build_s:.0f}s | graph_status {parts.info.get('graph_status')} | W16 {d16['tok_per_s']} "
          f"tok/s over {d16['timed_steps']} steps | W1 {d1['tok_per_s']} tok/s | mem {rec['max_mem_gb']} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
