"""Benchmark the fused Triton NF4 double-dequant (Task A) on the current GPU.

For each shape: assert the kernel is bit-exact vs ``bitsandbytes.dequantize_4bit`` (correctness gate),
then time our kernel against bnb — and, if it imports cleanly, against Unsloth's ``fast_dequantize``
(the exact rubric baseline). Prints per-shape median latency + speedup and a geomean.

The rubric measures speedup vs Unsloth's ``fast_dequantize`` on a **Tesla T4**. Unsloth is a thin wrap
over bnb (~1.05x), so ``ours/bnb`` is the portable number and ``ours/unsloth`` (when present) is the
literal rubric figure. Run it on a T4:  python bench_triton_nf4.py
"""

import torch
import bitsandbytes.functional as F
from bitsandbytes.nn import Linear4bit
from triton.testing import do_bench

try:  # in-repo
    from experts4bit_qlora.triton_nf4 import your_dequantize_nf4
except ImportError:  # standalone (Kaggle: triton_nf4.py fetched next to this file)
    from triton_nf4 import your_dequantize_nf4

# Unsloth's fast_dequantize is the literal rubric baseline; optional (best-effort import, never fatal).
_unsloth_fast = None
for _imp in ("unsloth.kernels", "unsloth.kernels.utils"):
    try:
        _mod = __import__(_imp, fromlist=["fast_dequantize"])
        _unsloth_fast = getattr(_mod, "fast_dequantize", None)
        if _unsloth_fast is not None:
            break
    except Exception:
        continue

# out_features x in_features — a spread from small attn projections to large MLP matrices.
SHAPES = [(1024, 4096), (2048, 8192), (4096, 4096), (4096, 14336), (14336, 4096), (8192, 8192)]


def _make(out_features, in_features, dtype):
    lin = Linear4bit(
        in_features, out_features, bias=None, compute_dtype=dtype,
        compress_statistics=True, quant_type="nf4",
    ).to("cuda")
    lin.weight.quant_state.dtype = dtype
    return lin


def _geomean(xs):
    import math
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    assert torch.cuda.is_available(), "CUDA required"
    dtype = torch.float16  # the rubric dtype; native on the T4 (Turing)
    dev = torch.cuda.get_device_name(0)
    print(f"device={dev} | torch={torch.__version__} | dtype={dtype} | "
          f"unsloth.fast_dequantize={'present' if _unsloth_fast else 'absent'}")
    print(f"{'shape':>13} {'bnb µs':>9} {'ours µs':>9} {'ours/bnb':>9} "
          f"{'unsloth µs':>11} {'ours/uns':>9}")

    sp_bnb, sp_uns = [], []
    for out_f, in_f in SHAPES:
        lin = _make(out_f, in_f, dtype)
        w, qs = lin.weight.data, lin.weight.quant_state

        ref = F.dequantize_4bit(w, qs)
        got = your_dequantize_nf4(lin)
        torch.testing.assert_close(got, ref)  # correctness gate before timing

        t_bnb = do_bench(lambda: F.dequantize_4bit(w, qs))          # median ms
        t_ours = do_bench(lambda: your_dequantize_nf4(lin))
        sp_bnb.append(t_bnb / t_ours)
        row = (f"{out_f}x{in_f:<7} {t_bnb*1e3:9.1f} {t_ours*1e3:9.1f} {t_bnb/t_ours:8.2f}x")

        if _unsloth_fast is not None:
            try:
                t_uns = do_bench(lambda: _unsloth_fast(lin.weight, qs))
                sp_uns.append(t_uns / t_ours)
                row += f" {t_uns*1e3:11.1f} {t_uns/t_ours:8.2f}x"
            except Exception as e:  # signature drift across unsloth versions — don't kill the run
                row += f"  (unsloth call failed: {type(e).__name__})"
        print(row)

    print(f"\ngeomean  ours/bnb = {_geomean(sp_bnb):.2f}x"
          + (f" | ours/unsloth = {_geomean(sp_uns):.2f}x" if sp_uns else ""))


if __name__ == "__main__":
    main()
