#!/usr/bin/env python
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Stall census, part 1: the KV bookkeeping one request costs ``serve_paged`` OUTSIDE the model forward.

Exploratory, unregistered, $0 (the NAS RTX A2000). No claim is drawn from it. It sizes one candidate SC2b named
("the prompt's K/V staging and flush into the FP8 pool, and host bookkeeping") before any box is rented.

Per request, ``serve_paged`` (e4b ``51639bf``) runs the following on the engine thread, in this order:

* ``admit``: ``PagedModelRunner.bind`` -> ``Fp8PagedKV.reset(slot)``.
* ``flush``: in ``run_prefill`` once the prompt completes, per pool layer ``ctx.flush`` + ``Fp8PagedKV.append``. That
  quantizes, claims the prompt's blocks with one ``fill_`` each, and writes each block with ``narrow().copy_()``.
* ``ready``: in ``_run_decode_bucketed`` at the slot's first decode, ``_ensure_graph_ready`` claims every remaining
  block with one ``fill_`` each.
* ``free``: ``free_slot`` -> ``Fp8PagedKV.reset(slot)``.

This script calls those same library methods on an ``Fp8PagedKV`` built at the served model's geometry. There is no
model. For each phase it reports host issue time, GPU time (CUDA events) and launch counts (``torch.profiler``).

It then runs bench-local bulk versions of the same work, ``--impl bulk``:
* one claim per request for every layer, written with one async H2D copy;
* one quantize per side over all layers;
* one scatter per region per side.

They are checked bitwise against the library path: every valid token's pool bytes through the block tables, the
block-table mirror and ``seq_lens``. A fix's upper bound is therefore measured, not assumed.

    python kv_bookkeeping_bench.py --out kv_bench.json           # Qwen3-30B-A3B geometry: 48 layers, 4 KV heads x 128
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

import torch

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
from experts4bit_qlora.engines.paged_attention import PagedAttentionContext

PHASES = ("admit", "flush", "ready", "free")
RUNTIME = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cudaMemcpyAsync", "cudaMemsetAsync",
           "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaStreamWaitEvent")


# ---------------------------------------------------------------------------------------------------------------
# the library path, as serve_paged calls it (paged_runner.bind / run_prefill / _ensure_graph_ready / free_slot)
# ---------------------------------------------------------------------------------------------------------------
class Library:
    name = "library"

    def __init__(self, kv, layers):
        self.kv, self.layers = kv, layers

    def admit(self, ctx, slot):
        self.kv.reset(slot)

    def flush(self, ctx, slot):
        for layer in self.layers:
            k, v = ctx.flush(layer, slot)
            self.kv.append(layer, slot, k.contiguous(), v.contiguous())

    def ready(self, ctx, slot):
        last = self.kv.blocks_per_seq - 1
        for layer in self.layers:
            self.kv._ensure_blocks(layer, slot, last)

    def free(self, ctx, slot):
        ctx.drop(slot)
        self.kv.reset(slot)


# ---------------------------------------------------------------------------------------------------------------
# bulk prototypes (bench-local; the library is untouched)
# ---------------------------------------------------------------------------------------------------------------
class Bulk:
    """The same bookkeeping with launch counts independent of layers and blocks.

    * ``reset``: the host mirrors as the library keeps them, then ONE ``fill_`` of the slot's lengths and ONE
      ``zero_`` of its table rows, over every layer.
    * ``claim``: every layer's rows are popped from the host free lists in the library's order (lowest first), so a
      slot gets the rows the library would have given it. One pinned buffer then lands them in the device table with
      one async H2D copy, guarded by an event before the buffer is reused.
    * ``flush``: the layers' staged K/V are stacked, quantized once per side (the scales are per token and head, or per
      group, so this is bitwise the per-layer quantize), and written with one ``index_copy_`` per region per side,
      through row ids the device reads back from the table it was just given.
    """
    name = "bulk"

    def __init__(self, kv, layers):
        self.kv, self.layers = kv, list(layers)
        self.all_layers = self.layers == list(range(kv.L))
        dev = kv.device
        self._lidx = torch.tensor(self.layers, dtype=torch.long, device=dev)
        self._pin = torch.empty(kv.L * kv.blocks_per_seq, dtype=torch.int32).pin_memory()
        self._ev = None
        geo = {(kv.Hs[l], kv.Ds[l], kv.kgs[l]) for l in self.layers}
        if len(geo) != 1:
            raise NotImplementedError(f"bench prototype covers one KV geometry, got {sorted(geo)}")

    def _tbl(self):
        kv = self.kv
        return kv._bt_all if self.all_layers else kv._bt_all.index_select(0, self._lidx)

    def reset(self, slot):
        kv = self.kv
        for layer in range(kv.L):
            rows = kv._rows.pop((layer, slot), [])
            if rows:
                kv._free[layer].extend(rows)
                kv._free[layer].sort()
            kv._seen[layer][slot] = 0
        kv.seq_lens[:, slot].fill_(0)
        kv._bt_all[:, slot].zero_()

    def claim(self, slot, upto_blk):
        kv = self.kv
        have = {len(kv._rows.get((layer, slot), ())) for layer in self.layers}
        if len(have) != 1:
            for layer in self.layers:                    # layers out of step: the library's own path
                kv._ensure_blocks(layer, slot, upto_blk)
            return
        lo = have.pop()
        n = upto_blk + 1 - lo
        if n <= 0:
            return
        got = []
        for layer in self.layers:
            free = kv._free[layer]
            if len(free) < n:
                raise RuntimeError(f"layer {layer} is out of KV blocks -- the scheduler admitted past capacity")
            take = free[:n]
            del free[:n]
            kv._rows.setdefault((layer, slot), []).extend(take)
            got.extend(take)
        if self._ev is not None:
            self._ev.synchronize()                       # the previous copy has left the pinned buffer
        G = len(self.layers)
        src = self._pin[:G * n].view(G, n)
        src.copy_(torch.tensor(got, dtype=torch.int32).view(G, n))
        if self.all_layers:
            kv._bt_all[:, slot, lo:lo + n].copy_(src, non_blocking=True)
        else:
            dst = torch.empty(G, n, dtype=torch.int32, device=kv.device)
            dst.copy_(src, non_blocking=True)
            kv._bt_all[self._lidx, slot, lo:lo + n] = dst
        self._ev = torch.cuda.Event()
        self._ev.record()

    def admit(self, ctx, slot):
        self.reset(slot)

    def flush(self, ctx, slot):
        kv = self.kv
        staged = []
        for layer in self.layers:                        # ctx.flush without its cat when one chunk was staged
            ks, vs = ctx.staging.pop((layer, slot))
            staged.append((ks[0], vs[0]) if len(ks) == 1 else (torch.cat(ks), torch.cat(vs)))
        T = int(staged[0][0].shape[0])
        if any(kv._seen[layer][slot] for layer in self.layers) or any(int(k.shape[0]) != T for k, _ in staged):
            raise NotImplementedError("bench prototype flushes a fresh slot's whole prompt")
        l0 = self.layers[0]
        H, D, kg, bt = kv.Hs[l0], kv.Ds[l0], kv.kgs[l0], kv.bt
        G = len(self.layers)
        K = torch.stack([k for k, _ in staged]).reshape(G * T, H, D)
        V = torch.stack([v for _, v in staged]).reshape(G * T, H, D)
        vq = kv._quant_bytes(V, 1)
        kq = kv._quant_bytes(K, kg)
        nblk = -(-T // bt)
        self.claim(slot, nblk - 1)
        rows = self._tbl()[:, slot, :nblk].to(torch.long)                       # [G, nblk], device
        gidx = (self._lidx[:, None] * kv._n_rows + rows)                         # flat pool row ids
        nfull, tail = T // bt, T % bt
        # V first, K last: append()'s publish-last order
        for pool, pay, groups, (qb, sb) in ((kv.vp, kv._v_pays[l0], 1, vq), (kv.kp, kv._k_pays[l0], kg, kq)):
            srow = H * groups * 4
            flat2d = pool.dev.view(-1, pool.row_bytes)
            qb3, sb3 = qb.view(G, T, H * D), sb.view(G, T, srow)
            if nfull:
                idx = gidx[:, :nfull].reshape(-1)
                flat2d[:, :bt * H * D].index_copy_(0, idx, qb3[:, :nfull * bt].reshape(G * nfull, bt * H * D))
                flat2d[:, pay:pay + bt * srow].index_copy_(0, idx, sb3[:, :nfull * bt].reshape(G * nfull, bt * srow))
            if tail:
                idx = gidx[:, nfull].contiguous()
                flat2d[:, :tail * H * D].index_copy_(0, idx, qb3[:, nfull * bt:].reshape(G, tail * H * D))
                flat2d[:, pay:pay + tail * srow].index_copy_(0, idx, sb3[:, nfull * bt:].reshape(G, tail * srow))
        for layer in self.layers:
            kv._seen[layer][slot] = T
        if self.all_layers:
            kv.seq_lens[:, slot].add_(T)
        else:
            kv.seq_lens[self._lidx, slot] += T

    def ready(self, ctx, slot):
        self.claim(slot, self.kv.blocks_per_seq - 1)

    def free(self, ctx, slot):
        ctx.drop(slot)
        self.reset(slot)


class LibraryBulk:
    """The library's own bulk forms (E4B_PAGED_BULK_KV), called in PagedModelRunner's order: reset_all_layers at
    admit and free; at the flush, claim every reachable block (graphs on) and then append_prompt; ready is the
    runner's _ensure_graph_ready, a no-op for a slot the flush already claimed."""
    name = "library_bulk"

    def __init__(self, kv, layers):
        self.kv, self.layers = kv, layers
        self.ready_slots = set()

    def admit(self, ctx, slot):
        self.kv.reset_all_layers(slot)

    def flush(self, ctx, slot):
        ks, vs = [], []
        for layer in self.layers:
            k_, v_ = ctx.staging.pop((layer, slot))
            ks.append(k_[0] if len(k_) == 1 else torch.cat(k_))
            vs.append(v_[0] if len(v_) == 1 else torch.cat(v_))
        self.kv.claim_blocks(slot, self.kv.blocks_per_seq - 1, self.layers)
        self.ready_slots.add(slot)
        self.kv.append_prompt(slot, self.layers, ks, vs)

    def ready(self, ctx, slot):
        if slot not in self.ready_slots:
            self.kv.claim_blocks(slot, self.kv.blocks_per_seq - 1, self.layers)
            self.ready_slots.add(slot)

    def free(self, ctx, slot):
        ctx.drop(slot)
        self.ready_slots.discard(slot)
        self.kv.reset_all_layers(slot)


# ---------------------------------------------------------------------------------------------------------------
def stage(ctx, layers, slot, kvs):
    """What a prefill-graph replay leaves behind for a prompt that completes: one staged chunk per layer."""
    for layer, (k, v) in zip(layers, kvs):
        ctx.staging[(layer, slot)] = ([k], [v])


def timed(fn, dev):
    torch.cuda.synchronize(dev)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    e0.record()
    fn()
    e1.record()
    t1 = time.perf_counter()
    torch.cuda.synchronize(dev)
    t2 = time.perf_counter()
    return {"host_ms": (t1 - t0) * 1e3, "wall_ms": (t2 - t0) * 1e3, "gpu_ms": e0.elapsed_time(e1)}


def launches(fn, dev):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize(dev)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize(dev)
    out = {}
    for e in prof.events():
        if e.name in RUNTIME:
            out[e.name] = out.get(e.name, 0) + 1
    out.pop("cudaDeviceSynchronize", None)           # the census's own sync above
    out["kernels"] = sum(out.get(k, 0) for k in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel"))
    return out


def lifecycle(impl, ctx, layers, slot, kvs, dev, measure):
    """One request's bookkeeping, phase by phase; ``measure`` is ``timed`` or ``launches``."""
    out = {}
    out["admit"] = measure(lambda: impl.admit(ctx, slot), dev)
    stage(ctx, layers, slot, kvs)
    out["flush"] = measure(lambda: impl.flush(ctx, slot), dev)
    out["ready"] = measure(lambda: impl.ready(ctx, slot), dev)
    return out


def snapshot(kv, layers, slot):
    """What a slot holds, logically: per layer its seen/seq_lens, its rows' table coverage, and every valid token's
    pool bytes (payload and scales), gathered through the block table in logical block order."""
    snap = {"seen": [kv._seen[l][slot] for l in layers],
            "lens": kv.seq_lens[:, slot].tolist(),
            "nrows": [len(kv._rows.get((l, slot), ())) for l in layers]}
    tables_ok = all(kv._bt_all[l, slot, :len(kv._rows[(l, slot)])].tolist() == kv._rows[(l, slot)] for l in layers)
    snap["table_matches_mirror"] = tables_ok
    blobs = []
    for l in layers:
        t = kv._seen[l][slot]
        H, D, kg, bt = kv.Hs[l], kv.Ds[l], kv.kgs[l], kv.bt
        parts = []
        for pool, pay, g in ((kv.kp, kv._k_pays[l], kg), (kv.vp, kv._v_pays[l], 1)):
            srow = H * g * 4
            for blk in range(-(-t // bt)):
                row = pool.dev[l, kv._rows[(l, slot)][blk]]
                take = min(bt, t - blk * bt)
                parts.append(row[:take * H * D])
                parts.append(row[pay:pay + take * srow])
        blobs.append(torch.cat(parts) if parts else torch.zeros(0, dtype=torch.uint8, device=kv.device))
    snap["bytes"] = blobs
    return snap


def same(a, b, rows=True):
    """``rows=False`` skips the per-layer block COUNT only (the library's bulk arm claims every block at the flush, as the
    runner does with decode graphs; the per-layer arm claims the rest at the first decode). Bytes, lengths and the
    table-to-mirror check are always compared."""
    if a["seen"] != b["seen"] or a["lens"] != b["lens"] or (rows and a["nrows"] != b["nrows"]):
        return False, "seen/lens/nrows differ"
    if not (a["table_matches_mirror"] and b["table_matches_mirror"]):
        return False, "a device table row does not match its host mirror"
    bad = [i for i, (x, y) in enumerate(zip(a["bytes"], b["bytes"])) if not torch.equal(x, y)]
    return (not bad), (f"pool bytes differ on layers {bad[:8]}" if bad else "bitwise")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=48)        # Qwen3-30B-A3B
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--slots", type=int, default=16)          # serve_paged's max_seqs
    ap.add_argument("--max-tokens", type=int, default=4096)   # E4B_PAGED_MAX_TOKENS_PER_SEQ
    ap.add_argument("--prompts", default="512,500")            # SC2's 512, and one with a partial tail block
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--min-free-gb", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.manual_seed(1689)

    def build(slots):
        return Fp8PagedKV(a.layers, a.kv_heads, a.head_dim, batch=slots, max_tokens_per_seq=a.max_tokens,
                          k_groups=None, batched_append=True, device=str(dev), scratch_slots=slots)

    # the pool is the bench's whole footprint; shrink the slot count rather than squeeze a shared card
    free0 = torch.cuda.mem_get_info(dev)[0]
    probe = build(1)
    per_slot = (probe.k_row + probe.v_row) * probe.blocks_per_seq * a.layers
    del probe
    torch.cuda.empty_cache()
    slots = a.slots
    while slots > 2 and per_slot * slots * 1.02 > free0 - a.min_free_gb * 2**30:
        slots //= 2
    kv = build(slots)
    layers = list(range(a.layers))
    info = {"argv": sys.argv, "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__,
            "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
            "loadavg_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
            "slots_requested": a.slots, "slots": slots, "free_before_gib": round(free0 / 2**30, 2),
            "pool_gib": round(per_slot * slots / 2**30, 2), "blocks_per_seq": kv.blocks_per_seq,
            "free_list_len": len(kv._free[0]), "k_groups": kv.kgs[0]}
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    print("INFO", json.dumps(info), flush=True)

    ctx = PagedAttentionContext(kv=kv, slots=[], mode="prefill")
    results = {"info": info, "prompts": {}}
    for T in [int(x) for x in a.prompts.split(",") if x.strip()]:
        kvs = [(torch.randn(T, a.kv_heads, a.head_dim, device=dev, dtype=torch.bfloat16),
                torch.randn(T, a.kv_heads, a.head_dim, device=dev, dtype=torch.bfloat16)) for _ in layers]
        res = {}
        # parity first: the library on slot 0, bulk on slot 1, the same staged K/V
        lib, blk, lbk = Library(kv, layers), Bulk(kv, layers), LibraryBulk(kv, layers)
        res["parity"] = {}
        for other, oslot in ((blk, 1), (lbk, 2)):
            for impl, slot in ((lib, 0), (other, oslot)):
                impl.admit(ctx, slot)
                stage(ctx, layers, slot, kvs)
                impl.flush(ctx, slot)
            torch.cuda.synchronize(dev)
            ok_flush, why_flush = same(snapshot(kv, layers, 0), snapshot(kv, layers, oslot),
                                       rows=other.name != "library_bulk")
            for impl, slot in ((lib, 0), (other, oslot)):
                impl.ready(ctx, slot)
            torch.cuda.synchronize(dev)
            a0, a1 = snapshot(kv, layers, 0), snapshot(kv, layers, oslot)
            ok_ready, why_ready = same(a0, a1)
            full = all(n == kv.blocks_per_seq for n in a1["nrows"])
            for impl, slot in ((lib, 0), (other, oslot)):
                impl.free(ctx, slot)
            torch.cuda.synchronize(dev)
            cleared = (kv.seq_lens[:, :3].abs().sum().item() == 0 and kv._bt_all[:, :3].abs().sum().item() == 0
                       and not any((l, s) in kv._rows for l in layers for s in (0, 1, 2)))
            res["parity"][other.name] = {"after_flush": why_flush, "after_ready": why_ready,
                                         "ready_claims_every_block": full, "free_clears_both": cleared,
                                         "pass": bool(ok_flush and ok_ready and full and cleared)}
        print(f"PARITY T={T}", json.dumps(res["parity"]), flush=True)
        for impl in (lib, blk, lbk):
            # warm, then launches once, then timed reps over rotating slots
            for i in range(3):
                lifecycle(impl, ctx, layers, i % slots, kvs, dev, timed)
                impl.free(ctx, i % slots)
            cnt = lifecycle(impl, ctx, layers, 0, kvs, dev, launches)
            cnt["free"] = launches(lambda: impl.free(ctx, 0), dev)
            reps = []
            for i in range(a.reps):
                slot = i % slots
                r = lifecycle(impl, ctx, layers, slot, kvs, dev, timed)
                r["free"] = timed(lambda: impl.free(ctx, slot), dev)
                reps.append(r)
            med = {ph: {m: round(statistics.median(r[ph][m] for r in reps), 3) for m in ("host_ms", "gpu_ms", "wall_ms")}
                   for ph in PHASES}
            tot = {m: round(sum(med[ph][m] for ph in PHASES), 3) for m in ("host_ms", "gpu_ms", "wall_ms")}
            res[impl.name] = {"median": med, "total": tot, "launches": cnt,
                              "reps": [{ph: r[ph] for ph in PHASES} for r in reps]}
            print(f"{impl.name.upper()} T={T}", json.dumps({"median": med, "total": tot,
                                                           "kernels": {ph: cnt[ph]["kernels"] for ph in PHASES}}),
                  flush=True)
        results["prompts"][str(T)] = res
    results["info"]["loadavg_end"] = os.getloadavg() if hasattr(os, "getloadavg") else None
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
