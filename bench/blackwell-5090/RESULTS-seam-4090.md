# The seam: `enable_fast` was silently unreachable in train mode — 0 kernel calls, 2.95x left on the floor
### 2026-07-29 · RTX 4090 (24.5 GB, **sm_89**) · torch 2.8.0+cu128 · e4b 0.6.4 · grouped-nf4-gemm 0.2.3

Companion to `RESULTS-blackwell-5090.md` (sm_120). Run on a 4090 because 5090
stock had gone by then — so this adds Ada alongside the Blackwell evidence.

## How the two packages fit together

They meet at exactly one function and one layout contract.

- **`experts4bit-qlora` owns storage + surgery.** An `ExpertsNbit`/`Experts4bit`
  module holds nibble-packed `gate_up_proj` / `down_proj` and blocksize-64
  `absmax`. `enable_fast(model)` rebinds each such module's `forward` to
  `fused_experts_forward`.
- **`grouped-nf4-gemm` owns the kernel.** `gemm_4bit_grouped(a, w, absmax,
  sizes, expert_ids)` — one Triton launch across all active experts, directly on
  the 4-bit bytes.
- **The seam**: `fused_experts_forward` sorts tokens by expert, calls the kernel
  **twice per layer** (gate_up, then down), applies the activation, and
  scatter-combines. The coupling is a *layout contract* expressed at the call
  site — `gate_up_proj.view(E, n1, k1 // 2)` and
  `gate_up_absmax.view(E, n1, k1 // 64).float()`. `//2` is nibble packing, `//64`
  is the blocksize. Declared as `grouped-nf4-gemm>=0.2.1` under the `[fast]`
  extra.

Namespaces do not collide: e4b ships one package (`experts4bit_qlora`), gnf4
ships 16 flat top-level modules. Verified across 7 install permutations.

## The defect: patched ≠ called

`enable_fast` returned **16** and every expert module's `forward` really was the
patched function. Yet during `timed_decode`:

| | kernel invocations | tok/s |
|---|---|---|
| `model.train()` — the state loaders return | **0** | 8.34 |
| `model.eval()` | **288** | **33.60** |

**Root cause.** `ExpertsLoRA` wraps `Experts4bit` (`…experts` → `…experts.base`)
and re-implements the expert math inline so it can inject the low-rank delta
*before* the SwiGLU. It hands off to the patched base only via
`_delegate_to_base()`, which requires **`not self.training`**. The streaming
loaders return a model in `nn.Module`'s default **train** mode, and
`timed_decode` never called `.eval()` — so the delegation never fired and the
fused kernel was never reached. No error, no warning: `enable_fast` reported 16
successful patches on a path nothing invoked.

The existing guard in `enable_fast` catches the *other* unreachability cause (a
non-zero adapter) and correctly stayed silent here, because the adapter **was**
zero. Training mode was the uncovered case.

## What it costs, measured

Interleaved A/B, fast toggled in place, 6 pairs, both paths warmed, eval mode:

| | median tok/s | samples |
|---|---|---|
| eager | 10.59 | 10.15 – 11.40 |
| fused | **31.28** | 30.32 – 31.99 |
| **ratio** | **2.95x** | ranges do not overlap |

**This is the number my earlier runs missed entirely.** Two prior measurements
reported ~1.00x; both were timing the un-accelerated path because the model sat
in train mode. Consistent in magnitude with the published 3.65x (different
model, shapes and card).

The kernel in isolation, on a real module's tensors (gate_up, E=64, sm_89),
against a per-expert dequantize-then-matmul reference loop:

| rows/expert | active experts | fused (ms) | dequant-loop ref (ms) | ratio |
|---|---|---|---|---|
| 1 | 8 | 0.162 | 1.570 | 9.7x |
| 1 | 64 | 0.586 | 12.271 | 20.9x |
| 8 | 64 | 0.777 | 12.272 | 15.8x |
| 64 | 64 | 0.948 | 13.816 | 14.6x |

**Scope that honestly:** the reference is a naive Python loop doing
`dequant_ref` + dense matmul per expert, *not* bitsandbytes' optimised CUDA
dequant. These ratios are against a deliberately simple baseline and are **not**
comparable to the published 3.65x, which uses a different reference. They
establish that the kernel is genuinely doing work, not that it is 20x faster
than a tuned dequant path.

## Fixes

1. **`enable_fast` now warns on train mode** — the silent case. It already
   warned about non-zero adapters; a model in train mode with a zero adapter
   produced no diagnostic at all while every patch was dead.
2. **`timed_decode` now calls `model.eval()`** — load-bearing, not hygiene. The
   package's own inference benchmark was measuring the un-accelerated path.

Verified on the pod against the fixed source: the warning fires, and
`timed_decode` goes from **0 to 288** kernel invocations.

## Method notes

- **`enable_fast`'s return value is a patch count, not a call count.** Trusting
  it is what hid this. The counting wrapper around `gemm_4bit_grouped` is what
  made it visible, and is worth keeping as a test.
- The first diagnosis attempt was wrong: I suspected the documented
  `requires_grad` fallback, but `timed_decode` is `@torch.no_grad()` and the
  adapter's `requires_grad` was `False`. Tracing `fused_experts_forward` entry
  (0) rather than reasoning about its branches is what localised it to "never
  entered" instead of "entered and fell back".
- Two pods were wasted getting here: the launcher's fallback produced a wedged
  pod (RUNNING, no IP), and a stock-probe pod created via a minimal API body had
  no SSH key injected. Both terminated and 404-verified; the lane's pod-id file
  briefly pointed at a dead pod while a live one ran unguarded, which is the
  leak class this lane already knows about.
