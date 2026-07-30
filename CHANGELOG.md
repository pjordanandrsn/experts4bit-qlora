# Changelog

## 0.7.0 — 2026-07-30

**If you train Gemma-4-class models with `offload=True`, 0.6.x could not do it
at all** — the first backward raised `backward re-dequantization read an
offload-evicted expert`. The evict *post*-hook fired during the
gradient-checkpoint recompute and un-staged the layer before its own backward
re-dequantized it. `offload.py` documented the opposite as an invariant
("PyTorch stops that recompute early, so the evict post-hook does **not** fire");
whether the recompute reaches the post-hook depends on where the checkpointed
region's last needed tensor is produced, which is an architecture detail. OLMoE,
Qwen3-MoE and GraniteMoe stop early, which is why the wrong premise survived.
The post-hook is now a no-op inside a backward; residency is unchanged, because
the single-resident-slot policy already evicts there.

**Two new ways to fit a model that does not fit.** `nvme_experts` serves the
cold expert tail from NVMe, and `dense_offload` + `dense_disk` serve the
*dense* side — the 114.4 GB of non-expert weights that pinned host RAM cannot
hold for a K3-class model — straight from the checkpoint's own safetensors,
byte for byte. Nothing is transformed: the alternative way to fit a 114 GB dense
side on a small card is to quantize it, which changes the model.

Also: the flagship matrix's **B1 bit-exactness gate is withdrawn** — it hashed
`getattr(module, "gate_up_proj")`, which under `offload=True` is a 0-element
placeholder, so it compared `sha256(b"")` with itself and could not fail. The
performance numbers are independent measurements and stand; the assurance that
the frozen stack was untouched during those ten cells does not, and is
separately evidenced by the fused-train gate (16.31 GB hashed, byte-flip control
fires). Its B2 table is corrected too: it reported "Δ eval" where the protocol
registers `|Δ final-**train**-loss|` *and* median step-wise `|Δ|`. Both still
pass; the worst cell is 3.4× inside the band, not the 7× the eval column implied.

The README and `METHODOLOGY.md` §11 no longer end on "a memory optimization, not
a speedup" without saying what the fused path measured: **1.75–1.81× faster per
step at 0.754–0.755× peak VRAM and 0.797–0.846× energy**, both arms offloaded.

## 0.6.7 — 2026-07-30

**`e4b serve` could not load Kimi K3 at all.** Four blockers, each hidden behind
the last: no `trust_remote_code` anywhere in the package (so `AutoConfig` raised
before the architecture gate, with an opaque message), `kimi_k3` missing from
`SUPPORTED_ARCHITECTURES`, a multimodal prefix that is per-family rather than
universal, and per-expert MXFP4. `trust_remote_code` is a new argument plus
`E4B_TRUST_REMOTE_CODE=1` and **defaults to OFF** — executing
checkpoint-supplied code is the caller's decision, never a default.

## 0.6.6 — 2026-07-29

**Per-expert MXFP4 layouts load.** `dequantize_mxfp4` ended in
`out.transpose(1, 2)`, hardcoding gpt-oss's rank-4 `[E, rows, G, B]` blocks, so a
single expert projection `[rows, G, B]` raised `IndexError: Dimension out of
range` instead of returning `[K, rows]`. That is the layout every
DeepSeek-V3-lineage checkpoint ships per expert. `transpose(-2, -1)` is
equivalent for the rank-4 case and correct for both.

## 0.6.5 — 2026-07-29

**Training could never reach the fused kernel, and `enable_fast` reported
success anyway.** Two distinct problems, both fixed here:

- `enable_fast()` patched all expert modules and returned a non-zero count while
  the kernel was invoked **zero** times in train mode. `ExpertsLoRA` hands off to
  the patched base only via `_delegate_to_base()`, which requires
  `not self.training` — and the streaming loaders return a model in `nn.Module`'s
  default train mode. Measured on an RTX 4090: 0 kernel calls / 8.34 tok/s
  against 288 calls / 33.6 tok/s. **A patch count is not a call count.**
- `enable_fast_train()` is new, and is the differentiable path: it patches the
  `ExpertsLoRA` **wrapper** — the module the model actually calls — and composes
  the frozen projection with the trainable `B(Ax)` delta at the pre-activation
  point, the only correct place, since `act(Wx + BAx) != act(Wx) + d`. Opt-in on
  purpose: it changes the expert summation order (group-sorted vs ascending
  expert id), which should be a deliberate choice in a training run.

Requires `grouped-nf4-gemm>=0.2.4`. `--help` no longer loads a model.

## 0.6.4 — 2026-07-28

**If you installed `[fast]` on 0.6.3 or earlier, the fused kernel was not
running.** `enable_fast()` patched `ExpertsNbit.forward`, but `ExpertsLoRA`
inlines the expert math and never calls `self.base(...)` — and
`load_moe_4bit_streaming` always wraps in `ExpertsLoRA`. The advertised speedup
was a silent no-op on the loader this package tells you to use. Upgrade to get
it; nothing about your code changes.

**This is a behaviour change, not only a fix.** With delegation live, the fused
path actually executes, and it is a different computation from the reference
loop — priced at **+0.023% perplexity** (see `docs/METHODOLOGY.md`). If you were
unknowingly running the reference path, your numbers will move slightly.

- **`enable_fast()` now reaches the streaming-loader path (PR #36, `c2bf990`).**
  `ExpertsLoRA` previously inlined the expert math and never called
  `self.base(...)`, so the `[fast]` fused kernel was patched onto a method that
  was never invoked — a silent no-op for every model loaded with
  `load_moe_4bit_streaming`. `ExpertsLoRA` now delegates to its base when the
  adapter provably contributes nothing (B is zero-init, so an untrained adapter
  is *identically* zero), guarded so a trained adapter is never silently dropped.
- **Docs corrections (2026-07-28).** The informed-hot-set decode gain is scoped
  to the (bandwidth-limited) hosts it was measured on — it does not replicate on
  a fat-PCIe box. The `memlock` deployment note no longer claims `cudaHostAlloc`
  is gated by `RLIMIT_MEMLOCK`; that cause is false and the observed slowdown is
  now marked unattributed. README states that both residency engines require
  standalone expert modules and refuse/skip `ExpertsLoRA`-wrapped bases.

## 0.6.3 — 2026-07-21
- **Behavior change — serve binds to `127.0.0.1` by default** (was `0.0.0.0`).
  LAN exposure is now opt-in: set `E4B_HOST=0.0.0.0` to restore the old
  default. Migration: one env var. Rationale: a localhost tool for the
  machine's owner should not be reachable from the network unless asked.
- Optional bearer auth: set `E4B_TOKEN` and the generation routes require
  `Authorization: Bearer <token>` (off by default; `/health` stays open).
- README: first-screen "It dials" bullet (informed hot sets +57-120% at
  identical VRAM); engine-tier tags on the v0 offload-path decode figures;
  the serving posture paragraph. Length pass — the storage-modes matrix,
  serving/Docker, benchmarks, and the bitsandbytes essay moved to `docs/`
  with anchor-preserving stubs.
- `[fast]` pins `grouped-nf4-gemm>=0.2.1`.

## 0.6.2 — 2026-07-21
- `enable_hot_residency` deprecated at call (superseded by
  `enable_pipelined_residency` — same capability, K is config; kept through
  0.6 so the stamped v0 receipts stay reproducible; removal in 0.7).
- README: "Which door?" decision tree covering all six execution modes with
  honest status tiers (`enable_cold_engine` labeled performance-experimental —
  the host decode is a correctness path until the AVX2 kernel lands); all
  relative links absolutized (they rendered as pypi.org 404s in the PyPI
  long_description); CPU-only bitsandbytes first-import notice documented.
- `py.typed` marker ships (the public API already carries annotations).
- Permanent built-artifact smoke in CI and the release gate: wheel installed
  into a clean venv, README-surface import battery + deprecation-warning
  check; README link check blocks publish.

## 0.6.1 — 2026-07-20
- Cold engine (`enable_cold_engine`): hot partition GPU-resident, cold tail
  computed on the host from CPU-resident NF4 (activation-sized bus traffic).
  Host decode bit-exact vs bitsandbytes' CPU `dequantize_4bit`;
  `dequant="auto"` gates bnb behind `avx512f` (on AVX2-only hosts bnb falls
  below naive torch — grouped-nf4-gemm `bench/cold-engine/` receipts).
  All-cold + `device="cpu"` is a pure-host MoE (no CUDA, no `[fast]`).
  gpt-oss epilogue supported. 0.6.0 shipped from a pre-merge tree without
  the engine; 0.6.1 is the real release.

## 0.6.0 — 2026-07-20
- Hot-expert residency (`enable_hot_residency`, #26/#27): expert-granular
  partial residency — hot experts VRAM-resident on the fused kernel, cold
  tail streamed from pinned host RAM; gpt-oss (clamped-GLU + per-expert
  biases) supported; requires `[fast]`, fails at enable time with an install
  hint.
- Routing-informed hot sets (#28): calibrate-then-pin reference driver;
  decode gain tracks routing coverage on thin-link hosts (gpt-oss +56/+120%, Gemma-4 +44%,
  OLMoE +19%); multi-socket affinity law documented (pin `taskset` before
  any cold-path number).
- Hybrid-vs-llama same-box A/B receipts + Gemma-4 gated-weights serving gate
  (`bench/RESULTS-gptoss-hybrid-ab.md`, `bench/RESULTS-informed-hotsets.md`).
- README package-family section (the `[fast]` seam with grouped-nf4-gemm).

## 0.5.0 — 2026-07-18
- `[fast]` extra: fused grouped-GEMM inference via grouped-nf4-gemm —
  `enable_fast()` routes frozen-expert inference through the single-launch
  kernel (measured 3.65× at bs=1 decode, OLMoE geometry, A2000; #25).
