# Changelog

## 0.17.2 — 2026-08-13

**Docs-only. The published page still lacked the one number the feature is for.**

0.17.0/0.17.1 describe `enable_nvme_train_residency` as lifting the **host-RAM**
ceiling and never said by how much. The 0.17.0 entry below now carries it —
**~2.5–3.5×**, as pinned expert bytes (3.83 GB → ~0.2 GB) and steady RSS after
load (4.94–5.9 GB → 1.42–2.37 GB) — together with the reason the obvious
instruments give the wrong answer.

`ru_maxrss` is not usable here: it reports 18.6 GB for the host arm on a roomy box
and 10.8 GB for the *same arm* on a constrained one, because the 13.84 GB bf16
checkpoint is mmap'd and read in full to fuse and quantize, and those pages are
clean, file-backed and reclaimable. Peak *anonymous* is not the fix either —
~1.5 GB for **both** arms, since `smaps_rollup` counts pinned CUDA memory as
file-backed.

Established by five reproductions across two independently built stacks on one
pod whose unpinned install resolved to identical ML package versions, plus an
A/B/A memory-balloon test (18.57 → 10.80 → 18.56, bit-identical losses) that rules
out drift. No code changed; the wheel is byte-identical to 0.17.1 apart from the
version.

## 0.17.1 — 2026-08-12

**Docs-only. The published 0.17.0 page carried a timing number that was never a
measurement.**

0.17.0's release notes reported `s/step` as one measurement per arm, and from
those single samples claimed the fully-pinned arena cost **1.03×** the host-RAM
reference. Re-measured under this repo's paired protocol — 5 scored rounds, every
arm timed once per round in fixed order, warmup dropped, plus a `host_self`
control that times the *same* host-RAM model twice per round — the control came
back at **0.986 with a 0.898–1.080 spread**. That spread is the harness's
resolution limit, and `hot_rows=1024`'s 0.957 sits *inside* it.

So the honest statement is **"indistinguishable from host RAM"**, not 1.03×. The
disk-bound arms are unaffected and remain real: **1.679×** at the `hot_rows`
floor, **1.479×** at 256 — both far outside the noise, with the ladder moving as
the tier's additive law predicts.

The warmup round is the argument for the protocol: `host` and `host_self` read
3.084 vs 1.901 in round 0 — the identical model, 62% apart. No one-shot-per-arm
run can see that.

The full table, the control, and what still is **not** established (a model whose
experts genuinely exceed host RAM) are in the 0.17.0 entry below, now corrected.
No code changed; the wheel is byte-identical to 0.17.0 apart from the version.

## 0.17.0 — 2026-08-12

**Training whose frozen experts live on NVMe — plus the namespace split, and a CI
gap that had been reporting 42 tests as coverage while running none of them.**

- **`enable_nvme_train_residency(model, arena_path, hot_rows=…)`** — QLoRA on a
  MoE whose frozen experts exceed **host RAM**. The arena served inference and
  refused training in as many words (*"Load without `arena=` to train, or drop the
  adapters to serve"*); that refusal was right, since the serving engines replace
  the module's forward and would discard the adapter's delta. This moves the
  **home** instead: `_ArenaExpertOffload` is an offload handle whose homes are
  `meta` — shape and dtype, no storage — and whose rows come off the arena:
  disk row → ColdTier pinned slot → `[E, …]` device stack → kernel.

  `enable_fast_train` is untouched. `nf4_qlora`'s `weights_fn` closure already
  re-reads whatever is staged when backward runs, which is the seam that makes
  this work at all.

  **Gradient checkpointing is required, and enforced.** The evict hook fires when
  a forward returns, so the checkpoint recompute is what re-stages a layer for its
  own backward. Routed staging fills only the routed rows of a full-shaped stack,
  so a recompute that routed differently would read uninitialized memory —
  `assert_rows_staged` runs inside the weights_fn closures (i.e. **at backward**)
  and refuses instead.

  **It does not bound VRAM.** The staged stack keeps its full `[E, …]` shape so
  every consumer still indexes by global expert id; one layer is device-resident,
  as with ordinary offload. This lifts the host-RAM ceiling, not the VRAM one.

  Needs `grouped-nf4-gemm >= 0.9.0` for `nvme_residency.segment_into`.

- **Fixes a latent bug in the shared single-resident-layer policy.** The slot was
  written through `type(self)`, so a subclass bound a *second* slot and left the
  base class pointing at a handle nothing would evict — two layers resident at
  once, the exact bound that policy exists to hold. Unreachable with one handle
  class; reachable the moment there are two.

- **`enable_nvme_residency` validates before allocating now.** It built its
  `ColdTier` first, which made every refusal unreachable on a host without an
  accelerator: the tier pins its landing buffer, so a CPU-only machine raised
  "Cannot access accelerator device" and the caller got an allocator error instead
  of the message naming their mistake. On a host that does pin, a refusal leaked
  the tier. The error path also no longer closes a tier that modules are already
  serving from, and counts what *this call* attached rather than trusting a sticky
  `_e4b_hot_ref` marker that survives an earlier enable.

- **CI actually runs the arena tests now.** `.[test]` did not install
  grouped-nf4-gemm, so on the runner `tests/test_nvme_train_residency.py` went
  from 29 tests to **one skip**, and `tests/test_nvme_residency_equivalence.py`
  did the same — both reporting as coverage while executing nothing. Adding the
  dependency took the runner from **577 to 661 passing**, and immediately
  surfaced the `enable_nvme_residency` bug above, in an engine that shipped
  months earlier.

- **The README link check stopped calling throttling a dead link**, and no longer
  forwards `Authorization` across origins. It opened a fresh TLS connection per
  link and GitHub's edge dropped some of that churn; one run reported 28 of 28
  links dead on a tree where every path existed. Connection reuse fixed it
  (measured: a URL that failed four `urlopen` attempts answered 200 three times
  running under `curl`). 404 and 403 are never retried into a pass.

Also shipping here, from the preceding commits: the `arch/` `formats/` `engines/`
namespace split (old submodule paths still resolve via aliases in `__init__`), the
architecture support matrix, transformers checkpoint-key renamings, and the
gap-heuristic fix.

- **`arena_train=True` on the loader — without it the above could not be reached
  at all.** The arena branch built bare `meta` experts (its serving shape) and
  silently ignored `r`/`alpha`, so `enable_nvme_train_residency` refused every
  module with *"not ExpertsLoRA-wrapped"* and its own documented usage failed.
  29 CPU tests missed it because each constructs `ExpertsLoRA` by hand in the
  fixture — they exercised the mechanism, never the route a caller takes. It is
  gated on an explicit flag rather than on `r`, because `r` is a required
  positional and the *serving* example passes `r=8`; keying off it would have
  fixed training by breaking serving.

**Verified on a GPU** (RTX A5000, sm_86; OLMoE-1B-7B; 12 steps on Alpaca;
identical data and bit-identical starting adapters; every arm through
`enable_fast_train`, so only residency differs):

| arm | s/step | peak GB | final loss | med \|ΔL\| |
|---|---|---|---|---|
| host RAM (reference) | 1.62 | 2.26 | 0.5839 | — |
| arena, `hot_rows=64` | 2.65 | 2.03 | 0.6143 | **0.0059** |
| arena, `hot_rows=256` | 2.41 | 2.04 | 0.6426 | **0.0086** |
| arena, `hot_rows=1024` | 1.67 | 2.04 | 0.5944 | **0.0099** |

All three pass `bench/fused-train-gate`'s registered 0.05 median-|ΔL| band, 5–8×
inside it. A precondition run first established the arena's bytes are **bitwise
identical** to loader-quantized bytes, so the arms differ only in residency.

**Timing re-measured under the paired protocol** (5 scored rounds, every arm timed
once per round in fixed order so drift hits all arms equally; warmup round
dropped; optimizer not stepped so routing is identical every round). The
`s/step` column above was one measurement per arm and is superseded by this:

| arm | s/step (med) | ratio med | ratio min–max |
|---|---|---|---|
| host RAM (reference) | 1.910 | 1.000 | — |
| **`host_self` (control)** | 1.906 | **0.986** | 0.898–1.080 |
| arena, `hot_rows=64` | 3.207 | **1.679** | 1.490–1.724 |
| arena, `hot_rows=256` | 2.919 | **1.479** | 1.351–1.542 |
| arena, `hot_rows=1024` | 1.867 | **0.957** | 0.890–1.051 |

`host_self` is the *same* host-RAM model timed a second time in each round. At
0.986 the instrument is unbiased, and its 0.898–1.080 spread is the **resolution
limit** — which is the number that makes the rest readable:

- The two disk-bound arms sit far outside it, so **1.68× at the `hot_rows` floor
  and 1.48× at 256 are real**, and the ladder moves the way the tier's additive
  law predicts: the cost is disk traffic.
- **`hot_rows=1024` is indistinguishable from host RAM.** Its 0.957 sits inside
  the control's own spread, so the honest claim is *smaller than this harness can
  resolve* — not the "1.03×" a single measurement suggested.

The warmup round is why this matters: `host` and `host_self` read 3.084 vs 1.901
in round 0 — the identical model, 62% apart. A one-shot-per-arm run cannot see
that.

**How much host RAM this actually saves: ~2.5–3.5×.** The point of the feature is
the host-RAM ceiling, so here is the measurement, with the caveat that makes it
readable. Same OLMoE run, RTX A5000:

| | host-RAM offload | arena, `hot_rows=64` |
|---|---|---|
| **pinned expert bytes** | **3.83 GB** (all 1024 experts) | **~0.2 GB** (64 hot rows) |
| steady RSS after load | 4.94–5.9 GB | 1.42–2.37 GB |

**Do not measure this with `ru_maxrss`.** It reports 18.6 GB for the host arm on a
roomy box and 10.8 GB for the same arm on a constrained one — an A/B/A with a
memory balloon moved it 18.57 → 10.80 → 18.56 GB with bit-identical losses and an
unchanged unreclaimable footprint. The checkpoint is 13.84 GB of bf16
safetensors, mmap'd and read in full to fuse and quantize; those pages are clean,
file-backed and reclaimable, so the "peak" is page cache the process happened to
have mapped, not memory it needed. An earlier 8× figure came from comparing that
inflated host peak against an arena arm that never reads the expert bytes at all.

Peak *anonymous* memory is not the fix either — it is ~1.5 GB for **both** arms,
because `smaps_rollup` counts pinned CUDA memory as file-backed. Use steady-state
RSS after load, or peak of anon + `/dev/zero` mappings.

**What this still does NOT establish.** OLMoE's arena is 3.6 GB and fits
everywhere, so this shows the mechanism is correct and how the cost scales with
residency — **not** a model whose experts exceed host RAM, which is the case the
tier exists for. A demonstration of that needs a machine capped near the host
arm's true ~5–6 GB working set; attempts at 11–16 GB could not fail the host arm,
because it never needed 18 GB. Rented-instance NVMe varies ~7× between pods, so
these ratios characterise this box and do not travel.

## 0.16.3 — 2026-08-12

**Makes 0.16.2's headline feature actually importable, and turns one opaque load
failure into an accurate refusal.**

- **`capture_decode`, `CapturedDecoder` and `probe_capture` are exported from the
  package root.** 0.16.2 shipped `capture.py` with no top-level export, no in-repo
  caller and no README mention, so the only route to it was knowing the private module
  path — the feature was published unreachable. Documented under Inference with the
  measured numbers (4.4–5.6x on 2-layer fixtures, **1.11x** on OLMoE-1B-7B, **1.04x**
  on Qwen3-30B-A3B) and both costs: `StaticCache` is allocated to `max_length` up
  front, and `step()` is greedy argmax with no logits processors, stopping criteria or
  streamer.

  Deliberately **not** used by the HTTP server: `_generate_once` needs sampling,
  repetition penalty, stop signals and streaming, and reimplementing those on a
  captured step to gain the measured ~4% at 30B is a bad trade against the risk.

- **Identity ("zero-computation") experts are refused with the counts named.**
  longcat_flash previously died with `AttributeError: ExpertsLoRA has no attribute
  '10'`, which names neither the architecture nor the limitation. LongCat-Flash
  allocates `gate_up_proj` over `n_routed_experts + zero_expert_num` (512 + 256 by
  default) but `down_proj` over the routed count only — its forward sends
  `expert_idx >= num_routed_experts` through `nn.Identity` scaled by the router weight
  and never reads those `gate_up` rows, so the surplus experts are ragged on disk by
  construction. The per-expert reader consumed `0..n_routed-1`, orphaned the rest, and
  the generic weight walk then called `get_submodule(".../experts.10")` on a path whose
  leaf was already the fused module.

  Loading only the routed experts is not a fix: the router keeps selecting over the full
  space, so it would address experts that do not exist. An identity slot belongs in the
  expert primitive, not the loader.

## 0.16.2 — 2026-08-12

**CUDA-graph decode capture, and one less allocation per decode step.**

- **`capture_decode()` / `probe_capture()`** (`experts4bit_qlora.capture`). Wraps one decode
  step in a `torch.cuda.CUDAGraph` backed by a `StaticCache`, so every step has identical
  shapes and ONE graph serves the whole generation — a growing KV cache otherwise gives a
  distinct shape, and a distinct graph, per token. The cost is explicit: the cache is
  allocated to `max_length` up front. `torch.compile` cannot be used instead; inductor dies
  on `aot_autograd() does not yet handle input mutations on views with different dtypes`,
  which is exactly the engine's one-uint8-store-viewed-as-int64-and-float32 row block.
  Capture also throws on a host sync inside the region, so a successful capture doubles as
  a check on the zero-sync decode contract.

  Measured, 16 new tokens greedy: **4.4–5.6x** on 2-layer fixtures (qwen2_moe, qwen3_moe,
  granitemoe, hunyuan_v1_moe, glm4_moe, dots1, olmoe), **1.11x** on OLMoE-1B-7B-0924-Instruct
  (3090) and **1.04x** on Qwen3-30B-A3B (A5000). The speedup is inversely proportional to
  real GPU work per step, which is what a fixed per-step launch cost predicts — so this is
  worth having for small models and for the sync contract, not as a throughput claim at
  scale. Both real-weight models replay **bit-identical** to eager decode.

  `probe_capture()` reports support rather than assuming it, and distinguishes a bf16 argmax
  tie from a real defect by measurement: it teacher-forces the same tokens down both paths
  and compares logits. The reference is eager INCREMENTAL decode against a cache — comparing
  against one full-sequence forward charges a few ulp of kernel/reduction-order difference to
  capture. `qwen3_next` is not capturable: `StaticCache` does not cover LinearAttention.

- **Persistent `row_idx` buffer in the pipelined engine** — the per-step device allocation
  and H2D copy are gone. **-16.4%** host time per decode step.

## 0.16.1 — 2026-08-12

**Correctness fix for the segmented cold source, plus per-round overhead removed.**

- **Prime each segment from its own offset.** `seg_addr` pointed every hot lane at the
  resident row START for all four segments instead of `hot_row + off[j]`. `_prime` seeds
  every slot from expert 0 with `have = -1`, so it does NOT take the hot skip — with
  expert 0 hot on the segmented (offload-homes) path it primed the absmax and down
  regions with gate_up bytes. Latent in 0.16.0: `_fetch` forces hot lanes to skip and
  they read the resident row in place, so nothing read those bytes — but that was an
  undocumented invariant holding up a wrong address table, and nothing tested it.
- **Traffic counting is now opt-in** (`E4B_PIPELINED_TRAFFIC=1`, or `count_traffic = True`
  on the engine before the first fetch). The two device reductions cost ~8.9% of the
  decode step on an A5000 to produce numbers nothing reads in production.
  `traffic()` RAISES when counting was off rather than reporting zeros, because
  `hot_d2d_bytes == 0` is a regression witness and several tests use the counters to
  prove the engine ran at all — silent zeros would let those pass while measuring nothing.

## 0.16.0 — 2026-08-12

- **Residency reads the offload homes in place** (#104, closes #86 with #87).
  Under offload the homes already hold every expert in pinned host RAM and the
  pipelined engine baked a SECOND full-size arena from them — 0.316 GiB per layer
  twice over on Qwen3-30B-A3B geometry, ~15 GiB duplicated for the one
  configuration that exists to fit a big model on a small card. The homes cannot
  be freed (prefill, grad and odd-dtype forwards still fall back to the reference
  path and need staging), so the engine stops making the copy instead.

  The homes group by tensor and the row layout groups by expert, so an expert is
  four contiguous runs and the gather issues one launch per segment. The kernel
  gained a destination offset, a length, and a separate IDENTITY vector — four
  launches read four addresses but must skip or copy together as one expert.
  Measured: RSS delta on enable 0.579 → 0.080 GiB per layer, output bit-identical
  to the copied-arena control.

  Guarded rather than assumed: offload packs homes one buffer per DTYPE with the
  offset advancing in ELEMENTS, so an odd-numel predecessor leaves the next tensor
  misaligned — undefined behaviour where the gather casts to `int64*`. Each
  segment is checked for pinned + contiguous + 8-byte-aligned base and
  8-byte-divisible length, falling back to the copied arena otherwise.
  Non-offloaded modules are unaffected.

## 0.15.0 — 2026-08-11

**Five more quantized checkpoint formats, and the DFlash drafter load path.**

- **AWQ** (`awq.py`) — the first ASYMMETRIC format, using autoawq's exact
  `[0,4,1,5,2,6,3,7]` nibble order. Packed along OUT.
- **GPTQ** (`gptq.py`) — packed along IN, sequential order, `+1` zero offset.
  Told apart from AWQ by its `g_idx` sibling; AWQ had been silently claiming all
  18624 GPTQ tensors, which is a wrong-answer bug, not a load failure. `g_idx` is
  now range- and length-validated (a negative index would wrap to the last group).
- **compressed-tensors int4** (`compressed_int.py`) — llm-compressor / vLLM.
  `num_bits`/`group_size` are DERIVED from shapes because the config often omits
  them. Vectorized unpack.
- **NVFP4** (`nvfp4.py`) — E2M1 codebook with two-level scaling; also serves
  **NVIDIA ModelOpt FP4**, verified against modelopt itself.
- **DFlash drafter** (`glimmer_draft.py`, `speculative.py`) — drafter load for both
  released spellings with coverage reconciled, plus a greedy speculative loop that
  is token-identical to plain greedy.

The dispatch matrix is pinned in BOTH directions, so a format is claimed by exactly
one decoder.

## 0.14.0 — 2026-08-10

**MoE breadth: the convention system, and every execution config measured.**

- **12 adjudicated conventions** covering 45 `model_type`s, each checked against
  transformers' own converter table so coverage DRIFT fails a test rather than
  silently going stale. Gate/up are shape-identical, so orientation can never be
  inferred — every entry is adjudicated, not guessed.
- New families: `gpt_oss` (pre-fused MXFP4 through the generic planner),
  `qwen3_vl_moe` (pre-fused + load-time transpose), `dbrx` and `jetmoe` (flat
  native stacks, bit-identical passthrough), `qwen3_5_moe` (native passthrough),
  `nemotron_h` (NON-gated: stack up/down, no gate to fuse), `minimax_m3_vl`
  (VL-prefixed mixtral), `axk1` (hybrid dense/MoE, layer-conditional keymap).
- **block-FP8 routing**, and MTP heads are never dropped silently.
- Tied heads are tied even when the checkpoint also ships the head.
- Rotary dim/theta buffers are materialized for VL vision towers.
- **Every execution config measured**, not just dtype: the decode/prefill ranking
  inverts, gains shrink as experts widen, and `dgrad=True` is the fastest training
  lane.


## 0.13.0 — 2026-08-10

**Muse Glimmer (Meta) and GLM-5 (Zhipu) checkpoint support.**

- **`glimmer.py` + `glimmer_load.py`** — serve Muse-Glimmer-30B from a released
  GGUF text tower. Glimmer is DENSE (Gemma-3 lineage), so it uses the dense lanes,
  not the expert path. Weights are decoded through grouped-nf4-gemm's k-quant lane
  (**needs gnf4 >= 0.8.0**; the import is capability-gated, so older installs get an
  actionable message rather than an AttributeError). The load streams each tensor to
  the target device as it decodes — peak host RAM is one tensor, not the ~60 GB a
  dequantized 30B would need — and ends with a coverage reconciliation: every
  text-tower parameter must be materialized or it raises.
- **`glm5.py`** — GLM-5 (`glm_moe_dsa`) checkpoint keymap and expert fusion.
  DeepSeek-V3 lineage, so it reuses the existing MLA/per-expert machinery; the new
  surface is DSA's lightning indexer.

Every mapping in both was adjudicated against the real released checkpoint AND the
instantiated transformers module tree, then reverse-armed (every model parameter must
be claimed by some checkpoint key — the direction that catches a silently dropped
weight). The traps that arithmetic alone gets wrong, now asserted:

- Glimmer's head is **untied** and must never be aliased to the embedding.
- Its four per-layer norms are centered (`x*(1+w)`) with the `+1` baked into the GGUF
  bytes, so the parameter is `gguf - 1.0` — while the FINAL norm is used as-is.
- Its `attn_q/k_norm` are uniform vectors equal to `config.qk_scale_factor` and 1.0,
  absorbed by a parameter-free norm; dropped only after asserting that identity, so a
  genuinely learned qk-norm fails loudly.
- GLM-5's checkpoint carries **one more layer than the model builds** (an MTP head);
  it is skipped explicitly, and MTP markers on a built layer raise.
- GLM-5's experts are per-expert on disk and fused in the tree, with gate/up
  concatenated as **blocks** — the interleave convention would mis-activate with every
  shape agreeing.
- Rotary `inv_freq` is computed, never shipped; it is rebuilt through the module's own
  rope initializer instead of being left on `meta`.

Validated end-to-end on an A100 80GB: the 30B loaded from Meta's released
`kquant-dynamic` GGUF (19.65 GB) in 205 s — 627 tensors assigned, 104 dropped,
0 unfilled — and generated correct text at 13.8 tok/s, 55.8 GB VRAM.

## 0.12.0 — 2026-08-06

**`serve` grows a residency dial** — the missing piece between 0.10.0's
residency-reachability fix and the deployment that needed it: `python -m
experts4bit_qlora.serve` could only stream every expert, which is why the judge
deployment decodes at 0.38 tok/s with VRAM sitting idle.

`E4B_RESIDENCY=pipelined` + `E4B_HOT_PROFILE=<jsonl>` + `E4B_HOT_PER_LAYER=<K>` attaches
`enable_pipelined_residency` after load (post-`eval()`, pre-warmup — the ordering the
wrapper's delegation preconditions require). Hot sets are **frequency-ranked from a
profile, never by index** — an index-ordered set on a 256-expert top-6 layer serves ~6%
of routed slots, so there is deliberately no by-index fallback; without a profile it
raises. `E4B_K_SLOTS` overrides the routed top-k when the config lacks
`num_experts_per_tok`. `/health` gains a `residency` block (mode, patched-module count,
profile-predicted coverage).

`E4B_EXPERT_PROFILE` now works under serve: the routing profiler was only ever attached
by `train.py`, so profiling a *serving* workload — the input the residency dial consumes —
silently wrote nothing. serve attaches it at load (no-op unless set); the JSONL lands once
at clean shutdown.

Every quiet failure mode refuses or warns instead: unknown mode, missing/most-wrong
profile (a set-count/module-count mismatch would silently shift every hot set one layer),
an engine that patches 0 modules ("residency on" in the logs, streaming in reality), and
— the subtle one — **activating a trained adapter turns residency off silently** (the
wrapper only delegates while the adapter is provably zero), so `_swap_adapter` warns when
that happens. Serving `base` (the judge/eval case) is what the dial is for.

Also: `test_reenable_with_a_different_dgrad_setting_says_so` gains a capability skip —
against a pre-0.7.0 grouped-nf4-gemm the dgrad mismatch it tests cannot be constructed
(the flag is coerced off first, with its own warning), which surfaced as a spurious
failure on any box with an old wheel installed.

## 0.11.1 — 2026-08-06

Docs-and-tests patch. The PyPI page for 0.11.0 froze training-path guidance that
same-day measurement falsified; this ships the corrected long_description plus the
receipts and one new test. No functional code changes.

- **The 24x was a toy-shape artifact, and the guidance is corrected.** 0.11.0's README
  recommended `enable_batched_train` for "VRAM to spare" on an A2000 microbench (hidden
  512) where it measures 24x. At Qwen3-30B-A3B/48-layer width it is **1.05x at the
  highest peak memory of any lane**, while `enable_fast_train(dgrad=True)` is fastest at
  **2.52x**. The README now defaults to the latter and positions `enable_batched_train`
  as the no-extras fallback it actually is. `batched.py`'s docstring — which predicted a
  bigger card "should narrow this" — carries the measured reversal.
- **The dgrad fidelity caveat is retired by measurement** (`bench/dgrad-gate/`, published
  wheels, 16 + 48 layers): dgrad adds nothing to composed gradient error; an fp32-truth
  arm shows every lane — the reference loop included — on the composed bf16 noise floor
  (vs-reference divergence is rounding *similarity*, not accuracy); a 20-step real-data
  trajectory gate passes at a third of its band, with dgrad at **2.87x** the reference's
  real-data step rate; sm_120 (RTX PRO 4500 Blackwell) runs all 95 release-tag tests
  clean with the sm_86-tuned tile default holding.
- **New test:** DeepSeek-V4's clamped SwiGLU pinned through the batched path
  (`test_batched_train.py`) — the one epilogue composition nothing covered.
- **Credit** for the `enable_batched_train` approach (@jiwoon-ahn, #38) now appears in
  the README, not only the docstring/CHANGELOG.

## 0.11.0 — 2026-08-06

**`enable_batched_train` — a kernel-free batched training path.** Training
without `grouped-nf4-gemm` fell back to `ExpertsLoRA.forward`'s per-expert Python
loop: ~10k sync-gated iterations per forward at 256 experts over 40 layers, with
the GPU idle through most of it. That extra has to build and is arch-gated, so it
is not a rare configuration.

Experts are frozen, so the decoded stack is a constant w.r.t. autograd — and it
comes out of ONE `dequantize_4bit` call, because `_quantize_stack` uses
`compress_statistics=False` and the constructor refuses straddling shapes, making
the flattened absmax an exact concatenation. Verified **bit-identical** to the
per-expert loop and pinned as a test, since a future double-quant would break it
silently. Measured 32x against the per-expert decode at E=256.

One training step, E=256, 512 tokens, top_k 8, hidden 512, RTX A2000:

| path | step | vs loop | peak |
|---|---|---|---|
| reference per-expert loop | 601.2 ms | 1.00x | 59 MB |
| `enable_fast_train` | 132.6 ms | 4.53x | 108 MB |
| `enable_batched_train` | 25.0 ms | 24.01x | 417 MB |

Faster than the kernel lane it was written to fall back *from* — and it spends
peak memory to get there, materializing a stack where the kernel lane holds one
expert. At production width that trade is ~1.6 GB per layer against a few MB, so
**the kernel lane stays the answer under offload or VRAM pressure**. The two are
mutually exclusive and each refuses to patch over the other.

The approach is [@jiwoon-ahn](https://github.com/jiwoon-ahn)'s, from #38. Two
differences from the design proposed there: the backward re-decodes rather than
letting autograd save the stack, so gradient checkpointing is an option rather
than a precondition; and the LoRA delta is a padded double-`bmm`, so expert-LoRA
trains and the package default `TRAIN_EXPERTS=1` works.

**`enable_fast_train(..., dgrad=True)`** routes the fused lane's *backward*
through `grouped-nf4-gemm >= 0.7.0`'s single-launch dgrad kernel instead of its
per-expert decode loop, which measured 78-84% of a training step. A second opt-in
rather than part of the first, because it is a second numerics change: the loop is
exact, the kernel lands near 2.9e-3. Requested against an older kernel package it
turns off with a warning rather than raising from inside a forward.

**A parity contract for training paths** (`tests/test_fused_train_parity.py`).
Gradient *values* through the `ExpertsLoRA` composition were unverified —
`enable_fast_train` was covered by a forward comparison plus
`grad is not None`. A backward wrong by a constant factor still trains and still
descends, and nothing raises. Both lanes now satisfy one contract: forward,
`dL/dx`, and `dL/d` every LoRA parameter against the reference. Tolerances are
measured, not fitted, and a control proves the contract rejects a 1% scaling
error that forward parity alone passes.

**Fixed: `fast.py`'s module header described the whole module as inference-only.**
The paragraph predated `enable_fast_train`, which lives in the same file. It was
quoted back at us in #38 as evidence the package had no training accelerator.

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
