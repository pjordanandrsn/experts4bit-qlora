"""Pin the perf/memory effect of commit 97fa09f — "route Experts4bit forward through bnb.matmul_4bit".

Both projection paths the commit swapped still coexist in the current source, so we A/B them in a
SINGLE process — no git checkout (which would corrupt the live ablation's PYTHONPATH=. import),
no worktree, no rebuild:

  BEFORE (pre-commit):  w = m._dequantize_expert(...); F.linear(x, w)   # materialize [out,in] bf16 weight
  AFTER  (post-commit): m._expert_matmul(...) -> bnb.matmul_4bit(...)   # fused dequant-in-GEMM

Measures, per tokens-per-expert M: latency (fwd, fwd+bwd) and peak CUDA memory, plus a numerical-
equivalence check. Run on an IDLE GPU for trustworthy latency.

    cd /home/node/work/bitsandbytes && . .venv-cuda/bin/activate && PYTHONPATH=. \
      python /home/node/work/ablation/bench_matmul4bit.py --mode correctness
"""

import argparse
import statistics

import torch
import torch.nn.functional as F_nn

from experts4bit_qlora import Experts4bit

HIDDEN, INTER, N_EXP = 2048, 1024, 8  # OLMoE-1B-7B expert geometry (few experts: we bench one)
DTYPE = torch.bfloat16
DEV = "cuda"
EXPERT = 0


def build_layer():
    torch.manual_seed(0)
    gate_up = torch.randn(N_EXP, 2 * INTER, HIDDEN, dtype=DTYPE)  # [n, 2*inter, hidden]
    down = torch.randn(N_EXP, HIDDEN, INTER, dtype=DTYPE)  # [n, hidden, inter]
    return Experts4bit.from_float(
        gate_up, down, has_gate=True, activation=torch.nn.SiLU(), quant_type="nf4", compute_dtype=DTYPE
    ).to(DEV)


# --- the two paths, isolated to the exact lines 97fa09f swapped (gate_up projection) ---
def after_proj(m, x):  # post-commit: fused 4-bit matmul
    return m._expert_matmul(m.gate_up_proj, m.gate_up_absmax, m._gate_up_shape, EXPERT, x, DTYPE)


def before_proj(m, x):  # pre-commit: dequantize full weight, then linear
    w = m._dequantize_expert(m.gate_up_proj, m.gate_up_absmax, m._gate_up_shape, EXPERT, DTYPE)
    return F_nn.linear(x, w)


def make_x(m_tokens, requires_grad=False):
    x = torch.randn(m_tokens, HIDDEN, dtype=DTYPE, device=DEV)
    return x.requires_grad_(requires_grad)


def time_ms(fn, x, trials, warmup, backward):
    for _ in range(warmup):
        if x.grad is not None:
            x.grad = None
        out = fn(x)
        if backward:
            out.float().sum().backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        if x.grad is not None:
            x.grad = None
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(x)
        if backward:
            out.float().sum().backward()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def peak_mem_mb(fn, x, backward):
    if x.grad is not None:
        x.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn(x)
    if backward:
        out.float().sum().backward()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e6


def pctl(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p / 100 * len(xs)))]


def correctness(m):
    print("=== numerical equivalence (commit claims 'numerically identical') ===")
    ok = True
    for name in ("gate_up",):
        x = make_x(32)
        with torch.no_grad():
            a, b = after_proj(m, x), before_proj(m, x)
        max_abs = (a - b).abs().max().item()
        allclose = torch.allclose(a, b, rtol=1e-2, atol=1e-2)
        ok = ok and allclose
        print(f"  {name}: max|after-before| = {max_abs:.3e}  allclose(1e-2) = {allclose}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def bench(m, trials, warmup, m_values):
    print(f"=== latency + peak-mem A/B (trials={trials}, warmup={warmup}) ===")
    print(
        f"{'M tok':>6} | {'fwd BEFORE':>11} {'fwd AFTER':>10} {'x':>6} | "
        f"{'f+b BEFORE':>11} {'f+b AFTER':>10} {'x':>6} | {'mem BEF':>8} {'mem AFT':>8} {'save':>6}"
    )

    def before(x):
        return before_proj(m, x)  # bind the layer; timed fns take only x

    def after(x):
        return after_proj(m, x)

    for M in m_values:
        # fwd-only latency
        xf = make_x(M)
        tb_f = time_ms(before, xf, trials, warmup, backward=False)
        ta_f = time_ms(after, xf, trials, warmup, backward=False)
        # fwd+bwd latency + peak mem (x requires grad so gradient flows through the matmul)
        xg = make_x(M, requires_grad=True)
        tb_fb = time_ms(before, xg, trials, warmup, backward=True)
        ta_fb = time_ms(after, xg, trials, warmup, backward=True)
        mb = peak_mem_mb(before, make_x(M, requires_grad=True), backward=True)
        ma = peak_mem_mb(after, make_x(M, requires_grad=True), backward=True)
        b_f, a_f = statistics.median(tb_f), statistics.median(ta_f)
        b_fb, a_fb = statistics.median(tb_fb), statistics.median(ta_fb)
        print(
            f"{M:>6} | {b_f:>9.3f}ms {a_f:>8.3f}ms {b_f / a_f:>5.2f}x | "
            f"{b_fb:>9.3f}ms {a_fb:>8.3f}ms {b_fb / a_fb:>5.2f}x | "
            f"{mb:>6.1f}MB {ma:>6.1f}MB {(1 - ma / mb) * 100:>4.0f}%"
        )


def _layer_forward(m, hidden, top_k_index, top_k_weights, proj):
    """Replicate Experts4bit.forward but with a swappable per-projection op `proj(packed,absmax,shape,idx,x)`.

    This is the right granularity for the memory claim: with all hit experts' weights live across the
    fwd->bwd boundary (no gradient checkpointing), the BEFORE path saves every dequantized [out,in]
    weight as an activation; the AFTER path saves only the 4-bit packed weight and re-dequantizes.
    """
    compute_dtype = m.compute_dtype if m.compute_dtype is not None else hidden.dtype
    hidden = hidden.to(compute_dtype)
    final = torch.zeros_like(hidden, dtype=torch.float32)
    with torch.no_grad():
        mask = F_nn.one_hot(top_k_index, num_classes=m.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)
    for e in hit:
        pos, tok = torch.where(mask[e])
        cur = hidden[tok]
        p = proj(m.gate_up_proj, m.gate_up_absmax, m._gate_up_shape, e, cur, compute_dtype)
        gate, up = p.chunk(2, dim=-1)
        cur = m.act_fn(gate) * up
        cur = proj(m.down_proj, m.down_absmax, m._down_shape, e, cur, compute_dtype)
        cur = cur * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, cur.to(final.dtype))
    return final.to(hidden.dtype)


def layer_mem(m, n_tokens=256, top_k=8):
    """Full-layer fwd+bwd peak-memory A/B, no gradient checkpointing — where the commit's claim lives."""
    print(f"=== full-layer fwd+bwd peak memory (tokens={n_tokens}, top_k={top_k}, experts={N_EXP}, no ckpt) ===")
    torch.manual_seed(1)
    top_k_index = torch.stack([torch.randperm(N_EXP, device=DEV)[:top_k] for _ in range(n_tokens)])
    top_k_weights = torch.rand(n_tokens, top_k, dtype=DTYPE, device=DEV)

    def after(packed, absmax, shape, idx, x, dt):
        return m._expert_matmul(packed, absmax, shape, idx, x, dt)

    def before(packed, absmax, shape, idx, x, dt):
        return F_nn.linear(x, m._dequantize_expert(packed, absmax, shape, idx, dt))

    def run(proj):
        x = torch.randn(n_tokens, HIDDEN, dtype=DTYPE, device=DEV, requires_grad=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = _layer_forward(m, x, top_k_index, top_k_weights, proj)
        out.float().sum().backward()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 1e6

    for _ in range(2):  # warmup
        run(before)
        run(after)
    mb, ma = run(before), run(after)
    print(
        f"  peak fwd+bwd:  BEFORE {mb:7.1f} MB   AFTER {ma:7.1f} MB   saved {mb - ma:6.1f} MB ({(1 - ma / mb) * 100:4.0f}%)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["correctness", "bench", "layer", "both"], default="both")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--m-values", type=int, nargs="+", default=[8, 32, 128, 512, 2048])
    a = ap.parse_args()
    assert torch.cuda.is_available(), "need CUDA"
    print(f"device: {torch.cuda.get_device_name(0)} | dtype {DTYPE} | dims h={HIDDEN} i={INTER}")
    m = build_layer()
    if a.mode in ("correctness", "both"):
        correctness(m)
    if a.mode in ("bench", "both"):
        bench(m, a.trials, a.warmup, a.m_values)
    if a.mode in ("layer", "both"):
        layer_mem(m)


if __name__ == "__main__":
    main()
