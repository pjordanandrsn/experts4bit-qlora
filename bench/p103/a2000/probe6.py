"""Indicative per-call GPU time of Qwen3.6's Gated DeltaNet functions (one layer; 30 such layers per token), torch path
against whatever transformers resolves (fla / causal_conv1d when installed). Prefill: the chunk rule and conv fn at
T = 256 / 512 / 2048 (B = 1). Decode: the recurrent rule and conv update at T = 1, B = 1 / 4 / 16, eager and as a
CUDA-graph replay (serving decode runs under graphs). A shared A2000: a go / no-go indication, never a claim."""
import inspect
import statistics
import torch
import torch.nn.functional as F
import transformers.models.qwen3_5.modeling_qwen3_5 as m

H, HK, K, V, CONV_K = 32, 16, 128, 128, 4
CONV_DIM = 2 * HK * K + H * V
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)


def impl_of(name):
    fn = getattr(m, name)
    return inspect.getclosurevars(fn).nonlocals.get("implementation"), getattr(fn, "__wrapped__", None)


def timed(fn, n=50, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def graphed(fn, n=50):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    return timed(gr.replay, n=n)


def r(*s, dt=torch.bfloat16, scale=1.0):
    return (torch.randn(*s, generator=g, device=dev) * scale).to(dt)


for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
    impl, ref = impl_of(name)
    print("RESOLVED", name, "->", getattr(impl, "__module__", None))
chunk_impl, chunk_ref = impl_of("torch_chunk_gated_delta_rule")
rec_impl, rec_ref = impl_of("torch_recurrent_gated_delta_rule")
cfn_impl, cfn_ref = impl_of("causal_conv1d_fn")
cup_impl, cup_ref = impl_of("causal_conv1d_update")
kw = dict(use_qk_l2norm_in_kernel=True, output_final_state=True)
w = r(CONV_DIM, CONV_K, scale=0.1); bias = None
for T in (256, 512, 2048):
    q, k, v = r(1, T, H, K), r(1, T, H, K), r(1, T, H, V)
    gg = (-torch.rand(1, T, H, generator=g, device=dev) * 2).float(); beta = torch.rand(1, T, H, generator=g, device=dev).to(torch.bfloat16)
    x = r(1, CONV_DIM, T)
    row = []
    for label, fn in (("torch", chunk_ref), ("resolved", chunk_impl)):
        if fn is None or (label == "resolved" and fn is chunk_ref):
            continue
        row.append(f"{label} {timed(lambda: fn(q, k, v, g=gg, beta=beta, initial_state=None, **kw)):.3f} ms")
    for label, fn in (("torch", cfn_ref), ("resolved", cfn_impl)):
        if fn is None or (label == "resolved" and fn is cfn_ref):
            continue
        row.append(f"conv {label} {timed(lambda: fn(x, w, bias, activation='silu')):.3f} ms")
    print(f"PREFILL T={T}: chunk {' | '.join(row)}")
for B in (1, 4, 16):
    q, k, v = r(B, 1, H, K), r(B, 1, H, K), r(B, 1, H, V)
    gg = (-torch.rand(B, 1, H, generator=g, device=dev) * 2).float(); beta = torch.rand(B, 1, H, generator=g, device=dev).to(torch.bfloat16)
    s0 = r(B, H, K, V, dt=torch.float32, scale=0.1)
    x = r(B, CONV_DIM, 1); cs = r(B, CONV_DIM, CONV_K)
    row = []
    for label, fn in (("torch", rec_ref), ("resolved", rec_impl)):
        if fn is None or (label == "resolved" and fn is rec_ref):
            continue
        call = lambda fn=fn: fn(q, k, v, g=gg, beta=beta, initial_state=s0, **kw)
        row.append(f"rec {label} eager {timed(call):.3f} graph {graphed(call):.3f} ms")
    for label, fn in (("torch", cup_ref), ("resolved", cup_impl)):
        if fn is None or (label == "resolved" and fn is cup_ref):
            continue
        call = lambda fn=fn: fn(x, cs, w, bias, "silu")
        row.append(f"convupd {label} eager {timed(call):.3f} graph {graphed(call):.3f} ms")
    print(f"DECODE B={B}: {' | '.join(row)}")
