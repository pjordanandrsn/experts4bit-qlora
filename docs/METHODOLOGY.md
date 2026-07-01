# How the ablation numbers are established

*Placement ablation for the bitsandbytes `Experts4bit` QLoRA path. Written 2026-07-01.*

> **Packaging note.** §9–§10 measure `matmul_4bit` on the fork (bnb 0.50-dev). On released bnb (≤ 0.49.x)
> `matmul_4bit` is **incorrect** for this `[packed, 1]` weight layout, so the packaged `Experts4bit`
> auto-gates to the portable dequantize forward (`_matmul_4bit_matches_dequant`); the matmul_4bit
> benches live under `bench/_upstream/`. The ablation eval numbers (§1–§8) are unaffected — training
> runs through `ExpertsLoRA`, which already uses the dequantize path.

## 0. What "the numbers" are

Each config produces three scalars on a **fixed held-out set**:

| Number | Meaning |
|--------|---------|
| `BEFORE` | held-out loss of the frozen 4-bit model **before any training** |
| `AFTER` | held-out loss after `STEPS` optimizer steps, at the final adapter |
| `best`  | lowest held-out loss seen at any periodic eval (the checkpoint kept) |

The headline result is the **delta** `AFTER − BEFORE`. A negative delta is the claim we are trying to substantiate: *LoRA adapters trained on top of the frozen NF4 experts (routed through `bnb.matmul_4bit`) actually learn.* The ablation isolates **which adapter placement** is responsible for the gain.

## 1. What is under test

The bitsandbytes PR adds `Experts4bit`: 4-bit NF4 storage for the **fused 3-D expert stacks** of a sparse-MoE (`gate_up_proj` / `down_proj`), which the ordinary `Linear4bit` walker skips. The training script exercises the full path end-to-end:

- a **streaming loader** that reads the checkpoint tensor-by-tensor straight onto the GPU, quantizes each `16×64` expert stack to NF4 on the way, and frees the bf16 source immediately — the full bf16 model is never materialized (fits a 12 GB card with a 3 GB container RAM cap);
- **per-expert LoRA** adapters over the frozen `Experts4bit` base (`ExpertsLoRA`);
- **per-projection LoRA** over the frozen attention q/k/v/o (`LoRALinear`);
- the forward routed through `bnb.matmul_4bit` — a memory optimization auto-engaged on bitsandbytes ≥ 0.50, else the portable dequantize path (§9).

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
