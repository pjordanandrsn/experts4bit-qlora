# How the ablation numbers are established

*Placement ablation for the bitsandbytes `Experts4bit` QLoRA path. Written 2026-07-01.*

> **Packaging note.** §9–§10 measure `matmul_4bit` on the fork (bnb 0.50-dev) — the routing proposed
> for the upstream PR; those benches live under `bench/_upstream/`. On released bnb (≤ 0.49.x)
> `matmul_4bit` is **incorrect** for this `[packed, 1]` weight layout in training shapes, and as of
> v0.2.0 the packaged library doesn't need it: `ExpertsNbit._project` runs dequantize-then-`linear`
> through a **recompute-in-backward** autograd Function, which delivers §9c's activation-memory
> property (backward holds only the packed bytes, never the dequantized expert) on **any** released
> bitsandbytes and every storage scheme, at §9b's small backward re-dequant cost. The only
> `matmul_4bit` use left in the package is the probe-gated `no_grad` decode GEMV (§12a). The
> ablation eval numbers (§1–§8) are unaffected — those runs went through `ExpertsLoRA`'s dequantize
> forward, and the recompute Function computes the *identical* forward (it **is**
> dequantize-then-`linear`; recomputation changes what is saved for backward, never what is
> computed).

## 0. What "the numbers" are

Each config produces three scalars on a **fixed held-out set**:

| Number | Meaning |
|--------|---------|
| `BEFORE` | held-out loss of the frozen 4-bit model **before any training** |
| `AFTER` | held-out loss after `STEPS` optimizer steps, at the final adapter |
| `best`  | lowest held-out loss seen at any periodic eval (the checkpoint kept) |

The headline result is the **delta** `AFTER − BEFORE`. A negative delta is the claim we are trying to substantiate: *LoRA adapters trained on top of the frozen NF4 experts actually learn.* (These runs used `ExpertsLoRA`'s dequantize path; the v0.2.0 recompute-in-backward Function computes the identical forward — see the packaging note — so the eval numbers carry over.) The ablation isolates **which adapter placement** is responsible for the gain.

## 1. What is under test

The bitsandbytes PR adds `Experts4bit`: 4-bit NF4 storage for the **fused 3-D expert stacks** of a sparse-MoE (`gate_up_proj` / `down_proj`), which the ordinary `Linear4bit` walker skips. The training script exercises the full path end-to-end:

- a **streaming loader** that reads the checkpoint tensor-by-tensor straight onto the GPU, quantizes each `16×64` expert stack to NF4 on the way, and frees the bf16 source immediately — the full bf16 model is never materialized (fits a 12 GB card with a 3 GB container RAM cap);
- **per-expert LoRA** adapters over the frozen `Experts4bit` base (`ExpertsLoRA`);
- **per-projection LoRA** over the frozen attention q/k/v/o (`LoRALinear`);
- the frozen-base projections run dequantize-then-`linear` with **re-dequantize-in-backward** — activation memory stays flat in the number of experts, on any released bitsandbytes (§9 measures the upstream `matmul_4bit` equivalent of this trade on the fork).

Model: `allenai/OLMoE-1B-7B-0924` (hidden 2048, intermediate 1024, 16 layers, 64 experts, top-8). Hardware: RTX A2000 12 GB. bitsandbytes `0.50.0.dev0`.

## 2. The single controlled procedure

Every number comes from **one deterministic procedure** (`examples/olmoe_experts4bit_qlora.py`), identical across configs except for the three train-placement switches:

```
seed = 0                                   # torch.manual_seed(0), fixed
1. stream-load OLMoE, quantize experts -> NF4 (frozen base)
2. attach LoRA (B init = 0  =>  delta is exactly 0 at step 0)
3. eval_before = mean loss over the held-out set          # <- BEFORE
4. train STEPS steps (AdamW, cosine+warmup, grad-clip 1.0)
     every EVAL_EVERY steps: eval; if new low, save adapter_best  # <- best
5. eval_after  = mean loss over the same held-out set      # <- AFTER
```

Because `B = 0` at init (standard LoRA), the adapted model is **bit-identical to the frozen base** at step 0 — so `BEFORE` measures the pure NF4-quantized model, and every config starts from the *same* `BEFORE` value (1.4813). That shared baseline is the control that makes deltas comparable across configs.

## 3. Exactly how each loss number is computed

`eval_loss()` — the source of every reported scalar:

- **Held-out set:** Alpaca rows `[N_TRAIN : N_TRAIN+64]` (64 examples), **disjoint** from the training slice `[:N_TRAIN]` (`N_TRAIN=10000`). Fixed rows, fixed order.
- **Response-only loss:** the prompt tokens are masked to `-100`; loss is cross-entropy on the response continuation only. Same masking in train and eval.
- **Aggregation:** plain mean of per-example `model(...).loss` over the 64 examples, batch size 1, `@torch.no_grad()`, `use_cache=False`.
- **Deterministic:** no sampling, no dropout in the loss path; greedy. Re-running the same config reproduces the number.

The three switches select what gets `requires_grad_(True)` (everything else is frozen):

| Config | experts | attn | router | trainable params |
|--------|:---:|:---:|:---:|---|
| `experts_only`        | ✓ | · | · | 58,720,256 (per-expert LoRA) |
| `attention_only`      | · | ✓ | · | 2,097,152 (q/k/v/o LoRA) |
| `experts_attn`        | ✓ | ✓ | · | ≈ 60.8 M |
| `experts_attn_router` | ✓ | ✓ | ✓ | ≈ 60.8 M + router (`gate.weight`, LR×0.1) |

Param counts are a consequence of the geometry, not tuned: e.g. `experts_only` = 16 layers × 64 experts × (gate_up `8·2048 + 2048·8` + down `8·1024 + 2048·8`) = 16·64·57344 = 58,720,256.

## 4. Everything held constant across configs

Set once in `bench/run-ablation.sh` and never varied, so the only independent variable is *placement*:

```
STEPS=150  GRAD_ACCUM=4  LR=1e-4  SEQ=256  N_TRAIN=10000
R=8  ALPHA=16 (scaling=2.0)  EVAL_EVERY=50  seed=0
optimizer=AdamW  schedule=cosine + warmup(max(5, STEPS/10))
grad clip=1.0  gradient_checkpointing=on (use_reentrant=False)
router LR = 0.1 × LR (only when router trains)
```

Same data slices, same eval set, same hyperparameters, same rank. A difference in the delta between two configs is therefore attributable to *what trained*, not to any confound.

## 5. Why the numbers are trustworthy (controls)

- **Shared, reproducible baseline** — identical `BEFORE=1.4813` across all configs (zero-init LoRA + fixed seed) means deltas are directly comparable.
- **Held-out, disjoint eval** — the eval rows are never trained on, so the delta measures generalization, not memorization of the 64 examples.
- **Frozen base everywhere** — the 6.4 B expert weights and all non-adapter weights stay in NF4/frozen; only the named adapters (and optionally the router) receive gradients. Verified in-log by the `trainable:` line each run prints.
- **Gradient checkpointing is disabled before the AFTER eval** (`gradient_checkpointing_disable()`), so the reported loss is a clean forward, not a recompute artifact.
- **Resumable & isolated** — each config writes to its own `ablation/<name>/` dir and drops a `done.flag` on success; re-launching skips finished configs and never overwrites another config's adapter. Logs are persistent (`<name>/run.log`), not `/tmp`, so numbers survive a container recycle.
- **Toolkit-free CUDA** — runtime libs come from torch's bundled `nvidia/*` wheels on the persistent volume, not an ephemeral system CUDA toolkit, so a rebuild can't silently change the numeric path.

## 6. How to read a result line

```
held-out eval loss: BEFORE 1.4813 -> AFTER 1.0452 (delta -0.4361) | best 1.0339
```

- `delta` is the primary signal (lower = the adapters learned).
- `best ≤ AFTER` is normal: cosine decay can leave the very last step slightly above the mid-training minimum; `best` is the checkpoint actually kept.
- Compare **deltas across configs** to attribute the gain to placement. (E.g. if `attention_only` — 28× fewer params — captures most of `experts_only`'s delta, the expert adapters are not where the headroom is; if `experts_attn` beats both, the two placements are complementary.)

## 7. Results (updated as configs land)

| Config | BEFORE | AFTER | delta | best | trainable | status |
|--------|:---:|:---:|:---:|:---:|---|---|
| `experts_only`        | 1.4813 | 1.0452 | **−0.4361** | 1.0339 | 58.7 M | ✅ done |
| `attention_only`      | 1.4813 | 1.0667 | **−0.4146** | 1.0535 | 2.10 M | ✅ done |
| `experts_attn`        | 1.4813 | 1.0290 | **−0.4522** | **1.0268** | ≈60.8 M | ✅ done |
| `experts_attn_router` | 1.4813 | 1.0384 | **−0.4429** | 1.0323 | ≈60.8 M+ | ✅ done |

**Conclusion (all 4 done, ranked by best held-out loss):**
1. `experts_attn` **1.0268** — quality winner.
2. `experts_attn_router` 1.0323 — **adding router training regresses it** (both AFTER and best worse than `experts_attn`); the router is sensitive even at 0.1× LR. Don't train it — validates the script's `TRAIN_ROUTER=0` default.
3. `experts_only` 1.0339.
4. `attention_only` 1.0535 — **efficiency winner**: 92% of the achievable reduction at **2.1 M params (3.5%)**.

Two clean findings: (a) the per-placement deltas do **not** add (experts −0.436 + attn −0.415 ≫ combined −0.452) ⇒ the placements are **largely redundant**, each recovering most of the ceiling alone; combining only nudges to best 1.0268. (b) **Router training hurts** on this task. Practical guidance: `attention_only` for efficiency, `experts_attn` for max quality, never the router.

## 8. Reproduce

```bash
pip install -e ".[train]"
# one config:
STEPS=150 GRAD_ACCUM=4 LR=1e-4 SEQ=256 N_TRAIN=10000 R=8 ALPHA=16 EVAL_EVERY=50 DO_GEN=0 \
  TRAIN_EXPERTS=1 TRAIN_ATTENTION=0 TRAIN_ROUTER=0 OUT=./ablation-out/experts_only \
  python -m experts4bit_qlora.train
# or the whole 4-config sweep (resumable):
bash bench/run-ablation.sh
```

## 9. Pinning claim #4 — the `matmul_4bit` routing (`97fa09f`), measured

Harness: [bench/_upstream/bench_matmul4bit.py](../bench/_upstream/bench_matmul4bit.py) (requires bitsandbytes ≥ 0.50). Both paths (`_dequantize_expert`→`F.linear` vs `bnb.matmul_4bit`) coexist in the primitive, so they're A/B'd in one process. RTX A2000, bf16, OLMoE dims.

*(Status as of v0.2.0: the packaged library achieves this commit's memory property portably via the recompute-in-backward Function — see the packaging note; the `matmul_4bit` training routing measured here remains the upstream PR's approach, benched on the fork.)*

**a. Numerically identical — confirmed bit-exact.** `max|after−before| = 0.000e+00` on CUDA (commit only claimed CPU bit-exactness). So any Δ below is purely the path swap.

**b. Throughput — NOT a training speedup (memory-for-compute trade).** Per-projection latency, median of 100:

| tokens/expert | fwd (before→after) | fwd+bwd (before→after) |
|---:|:---:|:---:|
| 8 | 0.179→0.155 ms (1.16×) | 0.554→0.699 ms (**0.79×**) |
| 32 | 0.181→0.166 ms (1.09×) | 0.441→0.420 ms (1.05×) |
| 128 | 0.198→0.205 ms (0.96×) | 0.422→0.587 ms (**0.72×**) |
| 512 | 0.285→0.285 ms (1.00×) | 0.510→0.647 ms (0.79×) |
| 2048 | 0.666→0.696 ms (0.96×) | 1.507→1.597 ms (0.94×) |

Forward has a small edge only at tiny token counts; **fwd+bwd is neutral-to-slightly-slower** because the fused path re-dequantizes the weight in backward. Despite the `perf:` prefix, this commit does not speed up training.

**c. Memory — the real win, confirmed large.** Op-level isolation shows **0%** (the dequantized weight is transient at peak either way with nothing else live) — the wrong granularity. At **full-layer fwd+bwd, no gradient checkpointing**:

```
peak fwd+bwd:  BEFORE 135.3 MB   AFTER 39.4 MB   saved 95.9 MB (71%)   [8 experts, 256 tok]
```

The AFTER path saves only the 4-bit packed weight, not each expert's dequantized `[out,in]` activation. Saving ≈ (experts hit) × ~12 MB. Extrapolated to real OLMoE (≈32 experts hit/layer × 16 layers) this is **multiple GB without checkpointing** — the difference between fitting a 12 GB card or not. In the actual training run gradient checkpointing keeps ~one layer live at a time, so the realized saving is ~one layer's worth (~hundreds of MB) at the checkpoint boundary, not the full multi-GB.

**Verdict on #4:** it's a **memory optimization mislabeled `perf:`** — bit-identical results, ~71% lower layer-level training memory, at a small (≤~1.4×) backward compute cost. The commit *body* ("keeping memory low") is accurate; the throughput framing (including my own earlier gloss) was not.

## 10. Energy — measured ([bench/_upstream/bench_energy.py](../bench/_upstream/bench_energy.py), [bench/bench_energy_excluded.py](../bench/bench_energy_excluded.py))

Actual GPU energy on the idle A2000 (70 W cap): background `nvidia-smi` power sampling against a tight op-loop; energy/op = mean-power ÷ throughput. Three expert-projection paths, incl. an unquantized **native bf16** reference. *Caveat: the idle baseline read high and unstable (clocks slow to drop), so idle-subtracted "dynamic" energy is unreliable — only **total** J is reported. One card, microbench.*

**a. On a GPU that already fits the model → no win, mostly penalty.**

| Workload | native bf16 | before (dequant) | after (matmul_4bit) |
|----------|:---:|:---:|:---:|
| decode M=1 (mem-bound) | 2 705 µJ (1.00×, 68 W) | 7 237 µJ (2.68×) | 3 203 µJ (1.18×, **36 W**) |
| prefill M=512 (compute-bound) | 12 559 µJ (1.00×) | — | 16 244 µJ (1.29×) |
| train fwd+bwd M=32 | 12 216 µJ (1.00×) | 19 164 µJ (1.57×) | 27 541 µJ (**2.25×**) |

4-bit costs **1.2–2.3× the energy** of native bf16. NF4 is a *storage* format — the GEMM runs in bf16 either way (no 4-bit tensor cores; none on Ampere regardless), so there's no compute saving, only added dequant. The one real physical effect: at decode the fused path draws **half the power** (36 vs 68 W) from reading 4× less weight data — but it runs ~2.25× slower, so total energy lands at ~break-even (1.18×). The naive dequant path (2.68×) is worst — it round-trips the bf16 weight through HBM.

**b. The excluded case (model doesn't fit / freed memory → batch) → the real energy win.**

*Memory wall* — full OLMoE-1B-7B on the 12.5 GB card: bf16 experts (12.9 GB) → **OOM**; 4-bit experts (3.2 GB) → fits. bf16 model ~13.9 GB vs 4-bit ~4.6 GB. **On this card bf16 simply won't run** — its energy/token is undefined; you'd need a bigger/second GPU (≈an A100-40 G at ~250–300 W, *estimate — not measured*).

*Utilization* — tokens-per-joule of the fused 4-bit MoE forward as freed memory buys batch:

| batch (tok) | tok/s | power | J/tok (total) | vs batch 64 |
|---:|---:|---:|---:|---:|
| 64 | 12 659 | 29 W | 2 311 µJ | 1.00× |
| 256 | 44 490 | 65 W | 1 455 µJ | 0.63× |
| 1024 | 92 341 | 68 W | 732 µJ | 0.32× |
| 4096 | 130 679 | 68 W | 520 µJ | **0.23×** |

Raising batch 64→4096 cuts energy/token **4.4×** (GPU goes from 29 W underutilized to 68 W saturated). 4-bit's memory savings are what let you reach those batches; bf16 hits the wall far sooner.

**Verdict on energy:** same technology, opposite sign depending on whether memory is the binding constraint. **Memory-not-binding (your excluded premise): a 1.2–2.3× energy penalty.** **Memory-binding (the real case for MoE on small cards): the difference between running and not, plus up to 4.4× lower energy/token via freed-memory batch.** Quantization is a memory-capacity technology; its energy benefit is entirely downstream of that.

## 11. Expert CPU-offload (`OFFLOAD_EXPERTS`) — correctness proven, OLMoE A/B specified

**Test host.** All offload numbers in §11–§12 were measured on an RTX A2000 12 GB installed in a
QNAP TVS-h1688X NAS (QuTS hero, Intel Xeon W-1250) — a Comet Lake platform, PCIe 3.0 only, with the
card in the chassis's x8 electrical slot (the widest it wires; the A2000 is x16-capable and
negotiates Gen3 x8 under load — at idle ASPM reads gen 1 and upshifts). Measured pinned H2D ceiling:
**6.16–6.18 GB/s** (256 MB × 20, event-bracketed; pageable 4.5–5.5). Every per-layer transfer figure
below runs at ~100 % of that ceiling, so these numbers are a **floor** for the code: the identical
build on a Gen3 x16 host has ~2× the H2D ceiling, a Gen4 x16 host ~3.5× — untested here. The NAS was
**not quiesced**: it ran its full production stack during the runs (media automation, two game
servers, Home Assistant, DNS, seven dev-agent containers *sharing this same A2000*, and CPU fuzzers;
load average ~45), so the ceiling and per-layer figures are a floor under realistic contention, not a
quiet-rig best case.

Even in 4-bit, the experts are the bulk of a fused-MoE's weights, and for the real targets they alone exceed a 12 GB card: Qwen3-30B-A3B ≈ **15 GB** of 4-bit experts, Gemma-4-26B-A4B ≈ **13 GB**. `OFFLOAD_EXPERTS=1` keeps each `Experts4bit` base's packed weights + absmax in **pinned CPU RAM** and streams one layer's experts to the GPU just-in-time (forward pre-hook on `ExpertsLoRA`), evicting after. Gradient checkpointing (`use_reentrant=False`) recomputes each layer's forward in backward, so the pre-hook re-stages for the recompute; PyTorch stops that recompute *early* (the evict post-hook does **not** fire on it), so a **single-resident-slot** — staging a layer first evicts the previously-staged one — is what keeps **only one layer's experts GPU-resident at a time, in forward and backward alike.** Mechanism and correctness argument: [`experts4bit_qlora/offload.py`](../experts4bit_qlora/offload.py).

### a. Correctness — offload changes tensor *location*, not math (the load-bearing claim)

Offload never alters the computation: staging restores the exact bytes, and eviction is safe because `ExpertsLoRA` uses the dequantize path (`_dequantize_expert` → `F.linear`), so autograd saves the *dequantized* weight for `grad_x` and never the packed base — the packed weight is only read during the forward to produce it. Two independent checks:

- **Unit tests** ([`tests/test_offload.py`](../tests/test_offload.py) — run on CPU-only torch *and* on an **RTX A2000 12 GB, 8/8 pass**): an `ExpertsLoRA` forward is **bit-identical** with vs. without offload; backward still runs after the base is evicted (the frozen base gets no grad); under `use_reentrant=False` gradient checkpointing the pre-hook **re-stages the experts on the backward recompute** (asserted by a pre-hook counter) with gradients matching a non-offloaded reference, and a **single-resident-slot keeps ≤ 1 layer's experts staged through backward** (a 3-layer residency test — without it every recomputed layer would stay staged and accumulate to the full footprint); and the evicted base serializes as a 0-element placeholder so `save_adapter`'s `"lora"` filter is unaffected. This is the location-not-math analogue of §9a's "numerically identical."
- **OLMoE A/B on an RTX A2000 12 GB** ([`bench/run-offload-ab.sh`](../bench/run-offload-ab.sh), measured): two runs identical in seed/data/hyperparameters (`STEPS=15 GRAD_ACCUM=4 SEQ=192 LR=1e-4`), flipping only `OFFLOAD_EXPERTS`:

| config | BEFORE | AFTER | delta | loaded GPU | peak GPU | s/step |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| `offload_off` | 1.3975 | 1.2204 | −0.1771 | 4.70 GB | **5.97 GB** | 18.7 |
| `offload_on`  | 1.3975 | 1.2164 | −0.1810 | **1.08 GB** | **2.57 GB** | 20.8 |

**`BEFORE` is bit-identical (1.3975 == 1.3975)** — the frozen NF4 forward is unchanged, so offload is location, not math (the analogue of §9a's "numerically identical"). The `AFTER`/delta match to within training non-determinism (`index_add_` atomics + async H2D → run-to-run jitter at the 1e-3 level; the deltas are −0.177 vs −0.181). **Peak GPU falls 5.97 → 2.57 GB (−3.40 GB, −57 %)** — the ~3.4 GB of experts leave the card (load-time footprint 4.70 → 1.08 GB), matching the §11b arithmetic — at an **+11 % s/step** PCIe cost. Exactly the memory-for-compute trade of §9b.

### b. The trade — peak GPU down, throughput down (memory-for-compute, not a speedup)

Like the `matmul_4bit` result in §9, this is a **memory optimization, not a throughput one**. Offload moves the resident expert footprint from *all layers* to *one*: for OLMoE (hidden 2048, inter 1024, 16 layers, 64 experts) the packed experts + absmax are ≈ **216 MB/layer** — 128 MB `gate_up` + 64 MB `down` packed, + 16 MB + 8 MB absmax — so ≈ **3.4 GB across 16 layers** collapses to one layer resident. The cost is a per-layer host→device copy on the forward *and* the recompute (so ~2× the transfer of a checkpointing-free forward); at ~0.2 GB/layer over PCIe this is why s/step rises. The §11a A/B measured exactly this — peak GPU down **3.40 GB** (vs the 3.4 GB predicted here) at **+11 %** s/step — a memory optimization, not a speedup (§9b).

### c. The real targets — Qwen3-30B & Gemma-4-26B fit 12 GB *only* with offload (measured)

The benefit scales with layer count: one layer's resident experts ≈ *(total 4-bit experts) / n_layers*. Both real targets were run on the same A2000 with `OFFLOAD_EXPERTS=1` (`STEPS=5 SEQ=128`):

| model | MoE layers × experts | 4-bit experts | loaded GPU | peak GPU (train) | BEFORE → AFTER |
|---|:---:|:---:|:---:|:---:|:---:|
| `Qwen/Qwen3-30B-A3B` | 48 × 128 | ~15 GB | 3.77 GB | **7.16 GB** | 4.0085 → 2.1850 |
| `google/gemma-4-26B-A4B` | 30 × 128 | ~13 GB | 5.32 GB | **8.47 GB** | 1.6829 → 1.2125 |

Both **QLoRA-train on a 12 GB card** — ≈ 0.3–0.4 GB of experts resident per layer, the other 13–15 GB held in pinned CPU RAM — and the adapters learn (delta −1.82 / −0.47). Without offload they don't fit: `Qwen3-30B OFFLOAD_EXPERTS=0` raises `torch.OutOfMemoryError` **during load** (15 GB of experts onto an 11.6 GB card, before a single step), and Gemma-4's 13 GB experts + its heavier non-expert footprint (≈256 K-token embeddings + a parallel dense MLP) likewise exceed the card. Offload is *what makes them fit* — the §11a OLMoE result at the scale it was built for.

*(Loading the real Gemma-4 needed two small loader touch-ups on top of the merged Gemma-4 support, since the released checkpoint is the **multimodal** `gemma4`: build the LM from `text_config`, strip the `model.language_model.` prefix and skip the vision tower, and reconstruct rotary from `text_config`. OLMoE / Qwen3 paths are unchanged — the 15 CPU+GPU tests still pass.)*

**Verdict on offload:** a capacity feature — it decides *what fits*, at a PCIe throughput cost (the honest framing this repo already applies to 4-bit itself). Measured on an RTX A2000 12 GB: correctness is **location, not math** (OLMoE `BEFORE` bit-identical off-vs-on, §11a) *through the gradient-checkpoint recompute* (unit tests, incl. the single-slot residency guard); the trade is **peak GPU ↓ 57 % at +11 % s/step** on OLMoE (§11a–b); and the headline holds — **Qwen3-30B-A3B (peak 7.16 GB) and Gemma-4-26B-A4B (peak 8.47 GB) both QLoRA-train on a 12 GB card with offload, and OOM without it** (§11c).

## 12. Inference mode — decode paths + prefetched offload, measured

Serving is the adapter-preservation story: the LoRA weights were trained against *this exact* NF4
base (same codebook, same per-expert absmax), so `python -m experts4bit_qlora.infer` serves them
over that same base with no re-quantization round trip — the quantization error at serving time is
the one training was regularized against. Three additions make that practical, all `no_grad`- and
eval-mode-only (training paths byte-identical), each with an env kill-switch for A/B:

- **decode fast-path** (`E4B_DECODE_FASTPATH=0` disables): a 1-token forward skips the one-hot
  expert-mask machinery and loops the token's `top_k` experts with 0-d device indices;
- **fused 4-bit GEMV** (`E4B_INFER_GEMV=0` disables): single-row base projections dispatch to
  `bnb.matmul_4bit`'s GEMV kernel (packed weight read directly, no dequantized-expert
  materialization);
- **prefetched offload** (`OFFLOAD_EXPERTS=1`, `PREFETCH=1` default): layer `L+1`'s experts copy on
  a side CUDA stream while layer `L` computes.

### a. Correctness gates (the load-bearing claims)

- **Fast-path = same math, different summation order.** It runs the identical projections with the
  identical fp32 accumulation; only the order differs (routing order vs ascending expert id).
  Pinned by [`tests/test_inference_decode.py`](../tests/test_inference_decode.py): equivalence vs
  the mask path (including deliberately duplicated expert indices — semantics preserved, not
  "fixed"), kill-switch respected, never taken grad-enabled / multi-token / in train() mode, and
  output returned in the *caller's* dtype (§the-dtype-contract, PR #4).
- **GEMV route is probe-gated per configuration.** `bnb.matmul_4bit` dispatches 1-row,
  `requires_grad=False` inputs to `gemv_4bit` — a kernel the multi-row training probe never
  exercises. A separate probe validates exactly that decode shape for this module's
  (quant_type, blocksize, compute dtype) on a deliberately **non-square** weight (an
  orientation bug cannot pass by symmetry). Measured finding: the probe **passes on stock
  bitsandbytes 0.49.2**, where the multi-row `[packed, 1]` route is broken and correctly refused —
  so decode GEMV works on a stock install. The gate also requires `requires_grad=False` (bnb
  dispatches on the tensor attribute, not grad mode) and eval mode (a reentrant-checkpoint initial
  forward runs under `no_grad` but must keep training kernels). Under offload the route is safe at
  inference because the eviction hazard is a *backward* construct (§11a) — there is no backward.
- **Prefetch = location/timing, not math.** A prefetched no_grad chain is **bit-identical** to a
  non-offloaded one ([`tests/test_offload_prefetch.py`](../tests/test_offload_prefetch.py)),
  residency is bounded at two layers (computing + in flight, asserted mid-forward), the circular
  wrap-around leaves layer 0 pre-warmed for the next token, and the first grad-enabled staging
  sweeps prefetch leftovers back to the single-slot training invariant (asserted mid-forward — the
  end state alone would pass without the sweep). Stream safety: consumption waits on the copy's
  recorded event and `record_stream`-marks the tensors for the compute stream; frees are ordered on
  the allocation (side) stream, so evicting an in-flight prefetch cannot recycle memory early.

### b. The decode grid — measured ([bench/run-decode-bench.sh](../bench/run-decode-bench.sh))

RTX A2000 12 GB, OLMoE-1B-7B + the §7 r16 experts+attention adapter (192 tensors), greedy decode of
128 tokens, manual KV-cache loop, warmup pass excluded, `cuda.synchronize` + wall clock:

| config | offload | prefetch | gemv | tok/s | prefill (s) | peak GPU |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| resident, dequantize | – | – | – | **3.08** | 1.42 | 4.86 GB |
| resident, GEMV | – | – | ✓ | 2.98 | 3.37 | 4.85 GB |
| resident, mask-path control | – | – | ✓ | 2.99 | 1.31 | 4.85 GB |
| offload, serial | ✓ | – | ✓ | 0.40 | 2.41 | **1.45 GB** |
| **offload, prefetched** | ✓ | ✓ | ✓ | **1.44** | 1.36 | **1.68 GB** |
| offload, prefetched, dequantize | ✓ | ✓ | – | 1.43 | 2.14 | 1.69 GB |

Readings, in the order they matter:

- **Prefetch is the real result: 0.40 → 1.44 tok/s (3.65×) over serial offload.** The schedule is
  deterministic because staging is layer-granular — the whole expert stack moves together, so no
  expert-choice prediction is needed (unlike expert-granular prefetch systems) — and the side-stream
  copy overlaps the *entire* next-layer compute (attention + dense + the current experts), not just
  the MoE block. Steady state pays `max(transfer, compute)` per layer instead of their sum, and
  one-layer-ahead already saturates the bus: each pre-hook kicks the next copy immediately, so
  deeper pipelining cannot beat the per-layer transfer floor.
- **Decode in 1.68 GB at 47 % of resident speed.** The capability framing of §11 carries over
  unchanged: offload decides what *generates* on the card, at a PCIe cost — decode is
  transfer-bound (~216 MB/layer, §11b), which is also why the GEMV column barely moves under
  offload (1.44 vs 1.43).
- **GEMV and the fast-path are neutral at OLMoE scale (−3 % / ±0 % resident).** Per-expert
  dequantize traffic (~12 MB/expert) is small enough that materialize+cuBLAS matches the fused
  GEMV, and the mask machinery is microseconds against millisecond kernels. Their value is
  **correctness and portability** — a validated decode route on a stock `pip install bitsandbytes`
  — not speed. Do not claim speed for them; the kill-switch columns exist so anyone can re-check.

### c. Big models — experts exceed VRAM (Gemma-4-26B-A4B and Qwen3-30B-A3B measured)

The §11c targets are where offload decode actually matters: their 4-bit experts (~13 GB Gemma-4,
~15 GB Qwen3-30B) don't fit a 12 GB card, so the *resident* configs are supposed to OOM — and do.
Same A2000, base model (no adapter), 96 greedy tokens, `run-bigmoe-decode.sh`:

**Gemma-4-26B-A4B** (30 layers × 128 experts; loaded non-expert footprint 5.27 GB, matching §11c):

| config | offload | prefetch | gemv | tok/s | peak GPU |
|---|:---:|:---:|:---:|:---:|:---:|
| resident | – | – | ✓ | **OOM** | — |
| offload, serial | ✓ | – | ✓ | 0.315 | 5.73 GB |
| **offload, prefetched** | ✓ | ✓ | ✓ | **0.429** | 6.16 GB |
| offload, prefetched, dequantize | ✓ | ✓ | – | 0.293 | 6.17 GB |

Two findings that **correct expectations set by the OLMoE grid** — measure, don't extrapolate:

- **Prefetch's *relative* win shrinks with expert size: 1.36× here vs 3.65× on OLMoE.** I had
  predicted a *larger* win for big models; the measurement says the opposite, and the mechanism is
  clear in hindsight. Prefetch overlaps a layer's transfer with the *next layer's compute*, so its
  ceiling is the per-layer compute time. Gemma-4 moves ~433 MB/layer (2× OLMoE's ~216 MB) against
  a per-layer compute that hasn't grown proportionally — so decode is more deeply PCIe-bound and
  less of each transfer can hide behind compute. The absolute speedup holds (0.315 → 0.429); the
  *ratio* falls as experts/layer grow. Offload decode of a big MoE is transfer-bound, full stop.
- **GEMV flips from neutral (OLMoE) to a 1.46× win here (0.429 vs 0.293 with it off).** At 128
  experts/layer the per-expert dequantize traffic — materializing a full bf16 expert just to matmul
  one token — finally dominates, so the fused GEMV reading the packed 4-bit weight directly is a
  real decode win, not just a portability feature. This is the scale at which the probed GEMV route
  earns its place; at OLMoE scale (64 experts, smaller stacks) it was correctly reported neutral.

**Qwen3-30B-A3B** (48 layers × 128 experts; loaded non-expert footprint 3.72 GB; ~15 GB of 4-bit
experts ≈ 313 MB/layer):

| config | offload | prefetch | gemv | tok/s | peak GPU |
|---|:---:|:---:|:---:|:---:|:---:|
| resident | – | – | ✓ | **OOM** | — |
| offload, serial | ✓ | – | ✓ | 0.203 | 4.07 GB |
| offload, prefetched | ✓ | ✓ | ✓ | 0.219 | 4.41 GB |
| **offload, prefetched, dequantize** | ✓ | ✓ | – | **0.238** | 4.42 GB |

Scoring the prediction stated above (recorded for exactly this purpose): resident OOM — ✓;
offload sub-0.5 tok/s — ✓; prefetch a modest ratio — ✓ (**1.08×**, even more transfer-bound than
Gemma-4's 1.36×: per-layer compute keeps shrinking relative to the ~313 MB/layer copy); **GEMV a
clear win — ✗, falsified.** GEMV *loses* 8 % here (0.219 vs 0.238 with it off), and the best
measured Qwen3 decode config is prefetch **+ dequantize**. Qwen3's per-expert stacks are markedly
smaller than Gemma-4's (same 128 experts/layer spread over a thinner intermediate), so the
dequantize traffic GEMV avoids no longer dominates its per-call overhead — the win was
shape-dependent, not expert-count-dependent as predicted. Same lesson this section already taught
once with prefetch: **measure, don't extrapolate** — the kill-switches exist so each deployment
can A/B its own model.

The takeaway for the whole feature: on the models offload is *for*, the decode story is "it runs
at all, in 4.4–6.2 GB, where resident OOMs" (0.24–0.43 tok/s) — capability. Prefetch still helps
but is nowhere near its OLMoE-scale multiplier, and the GEMV route swings from +46 % (Gemma-4) to
−8 % (Qwen3-30B) with expert shape — A/B it per model. One more measured anchor for "transfer-
bound": the measurement host's pinned H2D ceiling is **6.16–6.18 GB/s** (the §11 *Test host* note —
an A2000 in a NAS's Gen3 x8 slot), and stats-instrumented decode shows every per-layer copy running
at ~100 % of it (see [`docs/OFFLOAD-TRANSFER-NOTES.md`](OFFLOAD-TRANSFER-NOTES.md)), so on this host
there is essentially nothing left for any schedule to hide.

### d. Reproduce + limits

```bash
ADAPTER=./out/adapter_best.pt python -m experts4bit_qlora.infer                 # generate
OFFLOAD_EXPERTS=1 BENCH_TOKENS=128 python -m experts4bit_qlora.infer            # one timed config
OUT_DIR=./decode-bench ADAPTER=./out/adapter_best.pt bash bench/run-decode-bench.sh  # OLMoE grid
MODEL=google/gemma-4-26B-A4B bash bench/run-bigmoe-decode.sh                    # big-model grid
```

Limits, stated plainly: batch-1 greedy decode on a single GPU (no batching, no CUDA graphs, no
`torch.compile`); grids are OLMoE (resident-capable) plus Gemma-4-26B and Qwen3-30B
(offload-only); and per-expert GEMV vs a true grouped-GEMM kernel (the deferred future work)
remains unexplored territory for throughput.

**Verdict on inference mode:** the same honest shape as §9–§11 — **capability, not throughput**.
It serves the QLoRA fine-tune over the exact base it was trained against, on the card it was
trained on; prefetch turns offloaded decode from unusable (0.40 tok/s) to usable (1.44 tok/s) in
1.68 GB; and the decode-route additions are correctness-gated conveniences, not speedups.
