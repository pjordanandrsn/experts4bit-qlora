# Changelog

## Unreleased

**Untied output heads on multimodal checkpoints were silently tied to `embed_tokens`**
(#37). `load_moe_4bit_streaming` builds the text tower by keeping only keys under the
multimodal prefix (`model.language_model.` for Gemma-4, `language_model.model.` for Kimi
K3). `lm_head.weight` sits *outside* that prefix, so the filter dropped it — and the
meta-tie fallback then assigned the embedding matrix as the output head unconditionally.
Correct for a genuinely tied checkpoint (Gemma-4 ships no head on disk); for a
`tie_word_embeddings: false` checkpoint carrying a real head it meant every logit was
computed through the wrong matrix, with nothing raised: plausibly-shaped generations,
initial train loss at `ln(vocab)`, and a LoRA that "converges" by learning to steer hidden
states into `embed_tokens` — then collapses when the adapter is served on a stack that
maps `lm_head` correctly. The symptom is quant-invariant, so it reads as a quantization
fidelity problem, and an A/B against a tied model passes because the fallback is right
there.

The loader now recovers the head from outside the prefix (logging where it found it) and
gates the tie on `tie_word_embeddings`: untied config with no head that reached the model
raises instead of tying. The multimodal test previously covered only the tied path; the
untied load and the refusal are now both tested.

## 0.10.0 — 2026-08-05

**The residency engine was unreachable for every model the streaming loader produces.**
`enable_pipelined_residency` raised `NotImplementedError` the moment every `ExpertsNbit`
under the model was an `ExpertsLoRA.base` — which is every model
`load_moe_4bit_streaming` returns, i.e. the path most callers take. The composition it
needed already existed and went unused: `ExpertsLoRA._delegate_to_base` hands the whole
forward to the base when an engine is attached and the adapter provably contributes
nothing (`B` is zero-initialised, so an untrained adapter is *identically* zero), and it
already checked for this engine's own `_e4b_pipe_ref` marker. Only the patch site was
missing.

`target_modules()` now includes `ExpertsLoRA` bases. Membership means "targetable and
index-bearing", not "reachable by every engine" — the deprecated v0 `enable_hot_residency`
is not delegated to and still skips them, consuming its `hot_sets` entry. `ExpertsLoRA(r=0)`
raises a `ValueError` naming the supported way to get a zero delta, instead of dying on
`alpha / r` with a bare `ZeroDivisionError`.

Two silent-failure modes were fixed alongside it. `enable_mxfp4_nvme_residency` refused
wrapped bases via `isinstance(m, ExpertsLoRA)` over a list that only ever held
`ExpertsNbit`, so the check could never fire. And `enable_pipelined_residency` now WARNS
when a patch installs but cannot run (train mode, or a non-zero adapter) rather than
returning a count that implies work — a residency split that never executes reproduces the
unsplit reference exactly, so a dead patch scores a perfect zero and reads as a pass.

**`dispatched_modules()`** (new, exported) closes the footgun the above created. Hook what
is CALLED, not what is patched: a wrapped base is not called until an engine is attached,
so a `register_forward_pre_hook` on one fires zero times. The usual reason to hook these
modules is to build a routing histogram for an informed hot set — and that calibration pass
runs before the engine exists, by construction. Zero counts make `topk` return `0..K-1`, so
"informed" silently becomes the by-index set it exists to beat. It fails as a plausible
null, not as an error; it did exactly that once before the helper existed.

**Hot experts are now read in place.** The hot stack and the k-slot store were separate
allocations, so a hot hit still paid a device-to-device row copy before the GEMM could read
it. One shared `[n_hot + k, row_bytes]` store lets the GEMM address a resident row directly.
`sizes` stays the host constant `[1]*k`, still one GEMM launch, and the hot/cold decision
stays device-side — the fixed-shape, zero-host-sync decode loop is untouched. OLMoE-1B-7B on
an A2000, 7 interleaved reps x 96 tokens:

| hot set | before | after | delta | p | gather MB/tok |
|---|---:|---:|---:|---:|---|
| none (pure stream) | 11.77 | 11.78 | +0.1% | 0.225 | 418.4 -> 418.4 |
| by index | 13.29 | 13.37 | +0.7% | 0.025 | 418.4 -> 356.9 |
| informed | 16.51 | **16.83** | **+1.9%** | 0.025 | 418.4 -> **263.5** |

**The byte count that motivated that change overstated it, and the correction is the more
useful result.** The re-copy was 48.5% of all gather traffic on granite, which read as a
large lever. It is not: that copy runs at HBM bandwidth (~5 us/expert), while the PCIe cold
reads are what bind — so removing 37% of gather BYTES bought 1.9% of TIME. They were the
cheap bytes. A first version also cost -0.7% (p=0.013) on pure streaming, where an empty hot
set has nothing to gain but the new per-fetch row dispatch ran anyway; guarded on `n_hot`,
after which the change is non-negative on every config measured.

**Informed hot sets are a property of the model, not just the host.**
`docs/RESIDENCY-ENGINES.md` attributed the size of the gain to the host. Holding the host
fixed at one A2000: OLMoE-1B-7B gains **+24.2% (p=0.002)** from an informed hot set over a
by-index one, and granite-3.0-1b-a400m gains **nothing** (+0.7%/+1.4%/-1.0% across three
runs, never significant) — despite coverage working exactly as designed there (49.7% of
routed slots vs 24.5%, a real 2.0x skew, and a 46% cut in cold traffic). Reads have to bind
before coverage converts, and on a 1.3B model at ~4.2 GB/s they do not.
`bench/bench_hotsets_ab.py` measures this per (model, host) instead of assuming it.

**`quantize_layers`** (loader, #63) restricts 4-bit quantization to a subset of MoE layers,
reusing the loop's existing skip semantics so no new code path appears; `None` preserves
current behaviour bit-for-bit. Motivation is measurement rather than serving: the
KL-vs-knowledge work bounded the churn-to-destruction transition to somewhere in
2.2e-02 .. 1.41e-01 KL but could not locate it, because no quantization scheme lands in that
gap.

**KL-from-reference fidelity instrument** (`bench/kl_fidelity.py`, `bench/kl_paths.py`,
`bench/kl_ikp.py`, `bench/kl_sweep.py`) with K0 control receipts gating every measurement,
a committed 200-prompt set, and a path table where every row names its reference. Its
tier-transition row — 544 GB streamed from host DRAM against an all-resident reference,
**KL exactly 0.000 over 6,813 tokens, top-1 1.000000** — is the row this release's residency
fix unblocked. That row now carries a mandatory execution witness: its expectation is 0.000,
and an engine that never ran satisfies it perfectly, so the test side must stream nonzero
cold bytes and the reference side must stream none.

## 0.9.0 — 2026-08-01

**The trainer ran at batch size 1.** Its inner loop put one variable-length row through each
forward. A fused-MoE step's cost is largely *fixed per active expert* — the reference path
dequantizes each routed expert once, the fused path launches one grouped GEMM per expert
group — so a forward carrying 100 tokens paid nearly what one carrying 2000 does. On OLMoE
(16 layers x 64 experts, top-8) a single ~100-token row was dequantizing ~128 experts.

Rows are now packed until a **token budget** is reached. Measured on an RTX A2000
(OLMoE-1B-7B, SEQ=192, alpaca, 15 steps x grad_accum 4):

| `TOKEN_BUDGET` | s/step | tok/s | peak GPU |
|---|---:|---:|---:|
| 0 (one row per forward) | 17.8 | **22** | 5.23 GB |
| 1024 | 22.2 | **144** | 5.88 GB |
| 2048 | 22.8 | **248** | 6.67 GB |

**11.3x throughput for +1.4 GB.** Steps get 28% *slower* — each carries ~15x more data — so
tok/s is the metric this moves and s/step reads like a regression. `TOKEN_BUDGET=0` restores
the one-row path the v0.2.0 convergence receipts were measured on.

**The ceiling is VRAM, and it is not knowable in advance.** 4096 OOMs on a 12 GB card — but
only sometimes, on an unlucky batch of long rows; it sustained 353 tok/s for six steps first.
A static default cannot be right for both a 12 GB card running OLMoE and a 30B model on the
offload path, so an OOM now **halves the budget and retries the step** rather than killing
the run, down to a floor of 256. Verified: a run at 4096 that previously died now backs off
to 2048 at step 7 and finishes.

Padded batching, not sequence packing: pad positions carry label `-100` and attention `0`,
so no row can see another's tokens and no padding contributes loss — correct without
touching the model. Rows are drawn length-sorted within a bucket to bound padding waste, and
the budget counts the *padded* cost (`rows * width`), which is the work the GPU actually does.

Not claimed: better optimization. The batched arms reach a lower eval loss at equal `STEPS`
(-0.351 vs -0.181) purely because they see ~15x more tokens per step. Loss-per-token parity
is unmeasured.

## 0.8.0 — 2026-08-01

**DeepSeek-V4 (Flash / Pro) loads, serves and trains.** Full V4-Flash — 43 layers
x 256 experts, 284B params — loads in ~10 s at **8.74 GiB peak VRAM** and
generates, with 147 GB of experts served from an on-disk arena. The dense side
measured 8.28 GiB against 8.40 predicted from the shard headers alone. See
`docs/DEEPSEEK-V4.md`.

V4 needed three things the package did not have. Its experts are per-expert
MXFP4 with an epilogue that is gpt-oss's *clamps* over SwiGLU's *combination*,
so neither existing class was correct. Its dense half is block-scaled FP8 rather
than bf16 — `fp8_blocks` serves it at ~1 byte/param instead of 2, which is 8.4
GiB resident against ~14, i.e. whether it fits a 12 GB card. And the published
checkpoint ships in DeepSeek's own `inference/` spelling; transformers converts
that via its central `conversion_mapping.py`, but only inside `from_pretrained`,
which the streaming loader never enters.

**Two fixes to existing code that V4 exposed.**

`mxfp4` was *value-casting* scale bytes rather than reinterpreting them. That was
right by accident for gpt-oss, which ships both blocks and scales as `U8`, and
silently catastrophic for any checkpoint labelling scales `F8_E8M0`:
`.to(torch.int32)` yields the value (`2**-5` -> 0), not the exponent byte, so
every block would be scaled by `2**-127`. torch < 2.7 fails loudly at the read;
torch >= 2.7 materializes the dtype and the error goes silent.

`ExpertsLoRA` hardcoded `act_fn(gate) * up`. Since the adapter re-implements the
expert math inline — to inject the delta before the nonlinearity — it also owns
the choice of nonlinearity, so wrapping **any** clamped-expert architecture
trained a function the frozen base does not compute, with the loss still falling.
The base now supplies its epilogue via `_apply_gate`. This is why gpt-oss and V4
were built bare; V4 is now trainable.

**Hot sets are worth choosing properly.** `expert_profile` only probed
`ExpertsLoRA`, so it found *zero* layers on gpt-oss and V4 — exactly the models
worth profiling. It now probes whichever module is dispatched, and
`hot_sets_from_profile` / `coverage_from_profile` turn a routing histogram into a
hot set. Measured on full V4-Flash: frequency-ranked hot sets are **+37.1%** over
index-ordered at identical VRAM, and index-ordered is statistically
indistinguishable from pure streaming — 4.4 GiB spent for nothing.

**Also:** `scripts/energy_probe.py` (CPU RAPL via powercap or MSR, plus GPU) for
honest J/token, which needs bare metal — containers block both interfaces.
`tools/make_v4_fixtures.py` regenerates the real-bytes test fixtures, so those
tests are coverage instead of permanent skips. README consolidated 476 -> 359
lines (30.6 -> 23.9 KB) on top of 0.7.1's rewrite, keeping every measured number and
receipt link but stating each once — the fused-train figures appeared three times. The
"which door" decision procedure is now a ten-row table on the landing page with the
reasoning in `docs/CHOOSING.md`; residency, decode and V4 long-form moved to
`docs/RESIDENCY-ENGINES.md`, `docs/INFERENCE.md` and `docs/DEEPSEEK-V4.md`. All 19 repo
links are absolute and pinned — relative links 404 on PyPI.

## 0.7.1 — 2026-07-30

**0.7.0's headline features were not importable from the top level.**
`enable_fast_train` — the differentiable fused training path that both flagship
matrices measure, twenty cells of evidence, the thing the front page leads with —
was absent from `__init__.py` entirely, as were `enable_dense_offload`,
`DenseDiskSource` / `DiskHome` / `disk_homes_for`, and `enable_nvme_residency`.
The modules shipped; the names did not. `from experts4bit_qlora import
enable_fast_train` raised `ImportError` on 0.7.0. `__all__` goes 33 → 43, and a
check now asserts every symbol the README tells you to call is actually exported.

**The README described a package two releases old.** Its "Which door? (all six,
one line each)" table listed six execution modes when there were ten, and told a
*training* reader to call nothing — which stopped being the whole answer in
0.6.5. It is now a decision procedure keyed on what ran out (VRAM, host RAM, or
disk), with every mode's entry point, when to pick it, and what it requires.

Three factual corrections in the same pass: a cost figure still quoted the *eval*
delta against a band the protocol registers on **train** loss and median
step-wise; the "measured on an RTX A2000" section header sat above numbers from
two different hosts; and a note promising `enable_hot_residency` would be removed
*in* 0.7 was falsified by 0.7.0 shipping with it still exported — withdrawn
explicitly rather than quietly edited, since someone may have planned around it.

Also in this release: the second flagship-matrix model completed all ten
registered cells under a pre-stamped protocol
(`bench/flagship-matrix-model2/`), with C1 hashing 12.85 GB per cell against the
first matrix's withdrawn gate that hashed zero — and a C4 winner that **flips
sign** when the same cell is re-run on a second host, reported beside the
registered verdict rather than in place of it.

No code behaviour changed beyond the exports.

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
