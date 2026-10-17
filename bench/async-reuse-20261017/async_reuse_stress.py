"""Async-reuse stability, correctness only (no speed claims). RTX A2000, $0.

The FlashInfer grouped-WGRAD race (a workspace reused while an earlier asynchronous use was still in flight) and the
TensileLite StreamK partial-store bug (distinct partial results overwriting one slot) have no direct analogue in our code, but
three of our paths reuse device workspaces across asynchronous launches. Each is exercised under poisoned workspaces, repeated
launches and concurrent host-to-device copy traffic, and checked BITWISE against a path that does not reuse:

A. grouped-nf4-gemm int4_b32 FUSED_REDUCE (lane K17): one preallocated part/cnt/out reused across launches, part and out
   filled with NaN before every launch; each result must equal the two-launch path, and cnt must read all-zero after each.
   Also two streams racing, each with its own workspace.
B. grouped-nf4-gemm fp8 split-KV paged attention with the in-kernel combine (per-stream arrival counters): the caching
   allocator is pre-poisoned with NaN before every launch, so internal partials land on poisoned memory; each result must
   equal the two-kernel combine. Also two streams racing.
C. experts4bit-qlora's expert offload (pinned host homes, single-slot staging, record_stream): a reduced Qwen3-30B-A3B,
   repeated forward-backward passes with the allocator poisoned between them; loss and every adapter gradient must equal
   the first pass's (and the resident model's, when the resident model is itself deterministic).

    python async_reuse_stress.py --model Q3RED_L4 --out OUT.json [--iters 300 --passes 30]
"""
import argparse
import json
import threading
import time
import traceback

import torch

POISON = 0xFF               # every fp32 / bf16 / fp16 element reads NaN


def poison_allocator(sizes=(64 << 20, 8 << 20, 1 << 20, 256 << 10, 64 << 10, 16 << 10, 4 << 10)):
    """Fill freshly freed caching-allocator blocks with NaN bytes, so the next allocations of these sizes start poisoned."""
    ts = [torch.empty(s, dtype=torch.uint8, device="cuda") for s in sizes for _ in range(4)]
    for t in ts:
        t.fill_(POISON)
    del ts


class Pressure:
    """Concurrent pinned host-to-device copies on a side stream, from a Python thread."""

    def __init__(self, mb=256):
        self.src = torch.empty(mb << 20, dtype=torch.uint8).pin_memory()
        self.dst = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
        self.stream, self.stop, self.copies = torch.cuda.Stream(), threading.Event(), 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with torch.cuda.stream(self.stream):
            while not self.stop.is_set():
                self.dst.copy_(self.src, non_blocking=True)
                self.copies += 1
                if self.copies % 4 == 0:
                    self.stream.synchronize()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        torch.cuda.synchronize()


def part_a(iters):
    from int4_b32 import _plan, _sm_count, gemv_counter_len, gemv_int4_b32, quant_x_rows
    from int4_pack_ref import pack_int4_b32
    shapes = [(1536, 2048), (2048, 768), (4096, 2048), (512, 2048), (2048, 4096), (5120, 2048), (1984, 2048)]  # last: N % 128 != 0
    E, out, fails = 8, [], []
    sms = _sm_count(torch.device("cuda"))
    for (N, K) in shapes:
        torch.manual_seed(N + K)
        w = torch.randn(E, N, K) / (K ** 0.5)
        pk, sc = zip(*(pack_int4_b32(w[e].float()) for e in range(E)))
        packed = torch.stack([p.reshape(N, K // 2) for p in pk]).contiguous().cuda()
        scales = torch.stack([s.reshape(N, K // 32) for s in sc]).contiguous().cuda()
        for R in (1, 8, 16, 128):
            x = (torch.randn(R, K) / 8).to(torch.bfloat16).cuda()
            eids = (torch.arange(R) % E).to(torch.int32).cuda()
            xq, xs = quant_x_rows(x)
            bn, _wp, sk, _ku = _plan(N, K, R, sms)
            ref = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, fused_reduce=False).clone()
            cnt = torch.zeros(gemv_counter_len(N, R, bn), dtype=torch.int32, device="cuda")
            o = torch.empty(R, N, dtype=torch.bfloat16, device="cuda")
            part = torch.empty(sk * R, N, dtype=torch.float32, device="cuda")
            bad = 0
            for i in range(iters):
                part.fill_(float("nan"))
                o.fill_(float("nan"))
                y = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, part=part, cnt=cnt, out=o, fused_reduce=True)
                if not torch.equal(y, ref) or int(cnt.abs().sum()) != 0:
                    bad += 1
                    if len(fails) < 5:
                        fails.append({"N": N, "K": K, "R": R, "sk": sk, "iter": i, "nan": int(torch.isnan(y.float()).sum()),
                                      "max_abs_diff": float((y.float() - ref.float()).abs().nan_to_num(1e30).max()),
                                      "cnt_sum": int(cnt.abs().sum())})
            out.append({"N": N, "K": K, "R": R, "sk": sk, "launches": iters, "mismatches": bad})
    # two streams racing, each with its own workspace, on the largest split case
    N, K, R = 4096, 2048, 16
    torch.manual_seed(7)
    w = torch.randn(E, N, K) / (K ** 0.5)
    pk, sc = zip(*(pack_int4_b32(w[e].float()) for e in range(E)))
    packed = torch.stack([p.reshape(N, K // 2) for p in pk]).contiguous().cuda()
    scales = torch.stack([s.reshape(N, K // 32) for s in sc]).contiguous().cuda()
    bn, _wp, sk, _ku = _plan(N, K, R, sms)
    jobs = []
    for s in range(2):
        x = (torch.randn(R, K) / 8).to(torch.bfloat16).cuda()
        eids = (torch.arange(R) % E).to(torch.int32).cuda()
        xq, xs = quant_x_rows(x)
        ref = gemv_int4_b32(xq, xs, packed, scales, eids, N, K, fused_reduce=False).clone()
        jobs.append((torch.cuda.Stream(), xq, xs, eids, ref,
                     torch.zeros(gemv_counter_len(N, R, bn), dtype=torch.int32, device="cuda"),
                     torch.empty(R, N, dtype=torch.bfloat16, device="cuda"),
                     torch.empty(sk * R, N, dtype=torch.float32, device="cuda")))
    torch.cuda.synchronize()
    race_bad = 0
    for i in range(iters):
        ys = []
        for st, xq, xs, eids, ref, cnt, o, part in jobs:
            with torch.cuda.stream(st):
                part.fill_(float("nan"))
                ys.append(gemv_int4_b32(xq, xs, packed, scales, eids, N, K, part=part, cnt=cnt, out=o, fused_reduce=True))
        torch.cuda.synchronize()
        for (st, xq, xs, eids, ref, cnt, o, part), y in zip(jobs, ys):
            if not torch.equal(y, ref) or int(cnt.abs().sum()) != 0:
                race_bad += 1
    return {"cases": out, "split_cases": sum(1 for c in out if c["sk"] > 1),
            "launches": sum(c["launches"] for c in out), "mismatches": sum(c["mismatches"] for c in out),
            "two_stream_race": {"N": N, "K": K, "R": R, "sk": sk, "rounds": iters, "mismatches": race_bad},
            "first_failures": fails}


def _fp8_build(B, hq, hkv, d, seq_lens, seed, BT=16):
    """Random sequences packed into fp8 pools through the real pack path, with a permuted block table: gnf4's
    kernel/test_fp8_paged_attn.py ``_build`` (k_groups = v_groups = 1, layout "tokens"), copied so no pytest is needed."""
    from fp8_kv import pack_kv_block, quantize_kv_fp8
    g = torch.Generator().manual_seed(seed)
    max_blocks = max((t + BT - 1) // BT for t in seq_lens)
    total_rows = sum((t + BT - 1) // BT for t in seq_lens)
    k_row = v_row = BT * hkv * d + BT * hkv * 4
    k_pool = torch.zeros(total_rows * k_row, dtype=torch.uint8)
    v_pool = torch.zeros(total_rows * v_row, dtype=torch.uint8)
    perm = torch.randperm(total_rows, generator=g)
    table = torch.zeros(B, max_blocks, dtype=torch.int32)
    row_i = 0
    for b, t in enumerate(seq_lens):
        n_blk = (t + BT - 1) // BT
        kt = torch.randn(n_blk * BT, hkv, d, generator=g) * 1.5
        vt = torch.randn(n_blk * BT, hkv, d, generator=g)
        for i in range(n_blk):
            row = int(perm[row_i])
            table[b, i] = row
            qk, sk = quantize_kv_fp8(kt[i * BT:(i + 1) * BT], group=d)
            pack_kv_block(qk, sk, k_pool[row * k_row:(row + 1) * k_row], layout="tokens")
            qv, sv = quantize_kv_fp8(vt[i * BT:(i + 1) * BT], group=d)
            pack_kv_block(qv, sv, v_pool[row * v_row:(row + 1) * v_row], layout="tokens")
            row_i += 1
    q = torch.randn(B, hq, d, generator=g, dtype=torch.float32).to(torch.bfloat16)
    return q, k_pool, v_pool, table, torch.tensor(seq_lens, dtype=torch.int32)


def part_b(iters):
    from fp8_paged_attn import fp8_paged_decode_attention
    out, fails = [], []
    for trial, (b, hq, hkv, lens) in enumerate([(1, 16, 4, [513]), (2, 16, 4, [96, 511]), (3, 8, 2, [17, 250, 384]),
                                                 (4, 32, 8, [1024, 33, 700, 2048])]):
        q, kp, vp, tab, sl = (t.cuda() for t in _fp8_build(b, hq, hkv, 64, lens, 100 + trial))
        for n_split in (2, 5, 8):
            kw = dict(n_kv_heads=hkv, head_dim=64, n_split=n_split, compute="f32")
            ref = fp8_paged_decode_attention(q, kp, vp, tab, sl, fuse_combine=False, **kw).clone()
            bad = 0
            for i in range(iters):
                poison_allocator()
                y = fp8_paged_decode_attention(q, kp, vp, tab, sl, fuse_combine=True, **kw)
                if not torch.equal(y, ref):
                    bad += 1
                    if len(fails) < 5:
                        fails.append({"trial": trial, "n_split": n_split, "iter": i,
                                      "nan": int(torch.isnan(y.float()).sum())})
            out.append({"trial": trial, "lens": lens, "n_split": n_split, "launches": iters, "mismatches": bad})
    # two streams racing, fused combine, each its own inputs (the per-stream counters)
    builds = [tuple(t.cuda() for t in _fp8_build(2, 16, 4, 64, [600, 900], 300 + s)) for s in range(2)]
    refs = [fp8_paged_decode_attention(*bld, n_kv_heads=4, head_dim=64, n_split=5, compute="f32", fuse_combine=False).clone()
            for bld in builds]
    streams = [torch.cuda.Stream() for _ in builds]
    torch.cuda.synchronize()
    race_bad = 0
    for i in range(iters):
        poison_allocator()
        ys = []
        for bld, st in zip(builds, streams):
            with torch.cuda.stream(st):
                ys.append(fp8_paged_decode_attention(*bld, n_kv_heads=4, head_dim=64, n_split=5, compute="f32", fuse_combine=True))
        torch.cuda.synchronize()
        race_bad += sum(not torch.equal(y, r) for y, r in zip(ys, refs))
    return {"cases": out, "launches": sum(c["launches"] for c in out), "mismatches": sum(c["mismatches"] for c in out),
            "two_stream_race": {"rounds": iters, "mismatches": race_bad}, "first_failures": fails}


def _build_model(path, offload):
    import experts4bit_qlora as e4b
    from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
    torch.manual_seed(0)                            # the loader draws the expert LoRA init: same seed, same adapters
    model, cfg = e4b.load_moe_4bit_streaming(path, "cuda", torch.bfloat16, 16, 16, offload=offload, pin=offload,
                                             prefetch=False, quant_type="nf4")
    if not offload:
        model.to("cuda")
    quantize_attention_projections_4bit(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    torch.manual_seed(5)
    add_attention_lora(model, 16, 16, torch.bfloat16)
    torch.manual_seed(6)
    for n, p in model.named_parameters():          # B is zero-initialized; give it values so A's gradients are nonzero too
        if n.endswith("lora_B"):
            with torch.no_grad():
                p.normal_(0, 0.02)
    n = e4b.enable_fast_train(model)
    assert n > 0, "enable_fast_train patched nothing"
    return model, cfg


def _pass(model, ids):
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    torch.cuda.synchronize()
    return float(loss), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _maxrel(ga, gb):
    return max(((ga[n].float() - gb[n].float()).abs().max() / ga[n].float().abs().max().clamp_min(1e-30)).item() for n in ga)


def part_c(path, passes, resident_passes=6):
    """Same model built resident and offloaded (same seeds, so the same adapters). The training kernels are not bitwise
    run-to-run deterministic, so the test is the SPREAD: the offloaded passes, with the allocator poisoned before each and copy
    traffic running, must stay within the resident model's own pass-to-pass spread, with no NaN, and match it in loss."""
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(0, 151936, (2, 512), generator=g).cuda()
    res, cfg = _build_model(path, offload=False)
    r_runs = [_pass(res, ids) for _ in range(resident_passes)]
    r_loss = {lv for lv, _ in r_runs}
    r_spread = max(_maxrel(r_runs[0][1], gr) for _, gr in r_runs[1:])
    del res
    torch.cuda.empty_cache()
    off, _ = _build_model(path, offload=True)
    o_spread, nan_runs, o_loss, vs_res = [], 0, set(), []
    first = None
    for i in range(passes):
        poison_allocator()
        loss, grads = _pass(off, ids)
        nan_runs += int((loss != loss) or any(torch.isnan(v.float()).any().item() for v in grads.values()))
        o_loss.add(loss)
        vs_res.append(_maxrel(r_runs[0][1], grads))
        if first is None:
            first = grads
        else:
            o_spread.append(_maxrel(first, grads))
    return {"model": path, "tokens": list(ids.shape), "layers": cfg.num_hidden_layers,
            "resident": {"passes": resident_passes, "distinct_losses": sorted(r_loss),
                         "grad_spread_max_rel_vs_its_first_pass": r_spread},
            "offload": {"passes": passes, "nan_passes": nan_runs, "distinct_losses": sorted(o_loss),
                        "grad_spread_max_rel_vs_its_first_pass": max(o_spread) if o_spread else 0.0,
                        "grad_max_rel_vs_resident_first_pass": {"max": max(vs_res), "min": min(vs_res)}},
            "losses_equal_resident_vs_offload": sorted(r_loss) == sorted(o_loss)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--only", default="", help="run only these parts, e.g. C")
    a = ap.parse_args()
    rec = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with Pressure() as pr:
        parts = (("A_int4_fused_reduce", lambda: part_a(a.iters)), ("B_fp8_split_kv_fused_combine", lambda: part_b(a.iters)),
                 ("C_offload_staging_reuse", lambda: part_c(a.model, a.passes)))
        for name, fn in (p for p in parts if not a.only or p[0][0] in a.only):
            t0 = time.time()
            try:
                rec[name] = fn()
            except Exception as e:  # noqa: BLE001 -- a crash is a finding, recorded with its trace
                rec[name] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
            rec[name + "_seconds"] = round(time.time() - t0, 1)
            json.dump(rec, open(a.out, "w"), indent=1, default=str)
        rec["pressure_copies_256MB"] = pr.copies
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rec.items() if not k.endswith("_seconds")}, default=str)[:3000])


if __name__ == "__main__":
    main()
