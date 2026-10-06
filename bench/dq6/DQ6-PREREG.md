# DQ6 — pre-registration: DQ4's capacity read on a 24 GB RTX 4090

Work item: #1083. Owner, 2026-10-06: "your calls" (the decision to run this read, and to skip the micro-batch and 70B
reads, is the CTO session's under that delegation).

**Why.** DQ4 (#1206) read CAP_REAL on a 32 GB RTX 5090. Streaming the frozen NF4 weights doubled the longest trainable
sequence on DQ3's Qwen3-32B-architecture subject: G = 2.00 (bracket [1.75, 2.14]), and 2.375 with `expandable_segments`.

On a 24 GB card the same 14.53 GiB of weights is a far larger share of memory. The resident model's static footprint
alone (about 19.5 GiB) leaves almost nothing for activations, so the ratio should grow sharply. The 24 GB card is also the
one most people training a 32B QLoRA actually have. DQ6 measures it.

## What runs

**DQ4's lane, unchanged:**
- the subject: Qwen3-32B architecture, 64 layers, random NF4, PEFT LoRA r16 on all seven projections, non-reentrant
  checkpointing;
- the per-process harness (`bench/dq4/dq4_cap.py`), with micro-batch 1, two AdamW steps per rung, and token ids seeded by L;
- the configurations, in order: **c_def** (chunked loss, default allocator: **graded**), **c_exp** (chunked loss,
  `expandable_segments:True`: secondary), **s_def** (stock loss: descriptive);
- the procedure: one process per ladder, then fresh-process confirmations at L\* and at the first OOM for c_def and c_exp;
- the software pins (torch 2.8.0+cu128, bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2, grouped-nf4-gemm
  `a5edec87`);
- DQ3's egress probe (rc 14).

The S arm calls `enable_dense_offload(train_prefetch=True)` explicitly. #1269's default change does not alter it.

**The registered differences** (`bench/dq6/dq6_run.sh` is `bench/dq4/dq4_run.sh` with exactly these;
`tests/test_dq6_lane.py` pins them):

1. **The card gate.** `nvidia-smi` must read the name `NVIDIA GeForce RTX 4090` with `memory.total` between 23000 and
   24999 MiB. Anything else exits **rc 19** before any install. That code is not one adertha admits as machine evidence
   (#1216).
   - The memory check exists because a 48 GB-modified 4090 reports the same name.
   - "RTX 4090 D" is a different card and is out of band.
   - **The link is recorded, not gated.** Capacity is bytes: DQ5 showed the link changes step time, not the saving. The
     offer search still asks for gen ≥ 4 / x16, so the descriptive step ratio is read on a normal link.
2. **The host floor.** DQ3's VRAM probe allocates to 28 GiB, which is above a 24 GB card's whole memory and would refuse
   every 4090. `bench/dq6/dq6_vram_probe.py` is the same probe with a **21.5 GiB** floor (refusal rc 18, as before).
   21.5 GiB is above R's predicted peak at its boundary, and below what a healthy 4090 hands out.
3. **The ladder: start 512, step 512** (DQ4: 2048 / 1024). R is predicted near 1.5k tokens, so DQ4's ladder would start
   at or above R's boundary (VOID), and a 1024 step could not resolve it.
4. **The reducer:** `bench/dq6/dq6_reduce.py` is DQ4's rule, imported unchanged, with two differences:
   - the registered device is `NVIDIA GeForce RTX 4090`;
   - every receipt's `total_bytes` must lie in [23, 24.5] GiB, else VOID.

   Its self-test runs DQ4's 25 cases on the 4090, plus 7 of its own, which include: a 5090 receipt is VOID; a 48 GB card
   is VOID; R passing only the first rung still grades; R failing the first rung is VOID.

## The rule

**DQ4's, as registered** (`bench/dq4/DQ4-PREREG.md`, "The rule"), with the device and memory checks above. Per
configuration:
- **VOID** for any of DQ4's VOID causes, or a receipt not from a 24 GB RTX 4090. This includes an OOM at the ladder's
  first rung (512).
- **FUNCTION_FAIL** for a non-finite loss.
- **NOISY** if a fresh confirmation disagrees with its ladder.
- Otherwise **G = L\*_S / L\*_R**: **CAP_REAL** if G ≥ 1.5, **CAP_MARGINAL** if 1.1 ≤ G < 1.5, **CAP_NONE** below that.

The lane verdict is c_def's.

## Predictions (from DQ4's rented-5090 receipts only; no A2000 number seeds them)

**What DQ4's ladders measured** (`bench/dq4/receipts/dq4-5090-2/`):
- Peak allocated grows linearly at **1.2557 GiB per 1024 tokens** in both arms with the chunked loss: R 21.96 → 28.23
  GiB over 2048 → 7168 tokens, and S 7.88 → 22.94 over 2048 → 14336.
- The intercepts are **19.45 GiB (R)** and **5.37 GiB (S)**. Both are memory-size independent: weights, adapters,
  optimizer state and the chunk.
- Reserved minus allocated at the binding rung was:
  - R: 1.46 GiB at 2048 tokens, rising to 2.50 GiB at L\*;
  - S, default allocator: 6.22 GiB at L\* (9.77 GiB at 2048);
  - expandable: R 1.28, S 1.45.
- The 5090's reserved memory never exceeded total − 0.73 GiB. The same overhead is assumed here: about **23.25 GiB
  usable** of a 24 GB card's ~23.99 GiB.

**L\* ≈ (23.25 − intercept − gap) / 1.2557 × 1024**, rounded down to the 512 ladder:

| c_def (graded) | gap used | predicted L\* | band |
|---|---|---|---|
| R | 1.46–2.50 GiB | **~1536** | [512, 2048] |
| S | 6.2 GiB (2–9.8) | **~9216** | [6144, 12800] |
| **G** | | **~6** | [3.0, 25]: **CAP_REAL** |

- **c_exp** (secondary): R ~2048 [1536, 2560], S ~13312 [12288, 14336], G ~6.5.
- **s_def** (descriptive). The stock loss's slope on the 5090 was 2.38 GiB per 1024 tokens. R is predicted at **512 or
  an OOM at the first rung** (descriptive VOID), and S at ~4608.
- **Step time** (descriptive): T(S)/T(R) at the largest common rung (~1.5k tokens). At that length a layer's forward
  compute is close to its gen 4 copy time, so the copy may not hide fully. Predicted **[1.0, 1.3]**; not graded.

**The stated risk** is S's fragmentation, as in DQ4. On a smaller card the allocator's free-and-retry reclaims cached
blocks before it raises, so the binding gap may be smaller than the 5090's 6.2 GiB (S toward 12k) or no smaller (S toward
6–9k). c_exp says which.

## Consequences

- **CAP_REAL:** the 24 GB number goes beside DQ4's in `docs/CHOOSING.md`, as the capacity a 4090 user gets.
- **CAP_MARGINAL or CAP_NONE:** the byte accounting failed on the smaller card. Write up why from the reserved and
  allocated readings and the c_exp pair; there is no re-read without a new registration.
- **FUNCTION_FAIL:** a defect independent of the card size. Stop until it is found.
- **NOISY or VOID:** no reading. One re-draw on another 24 GB 4090 under this registration (excluding the first host
  only by an admitted refusal that names it); a second NOISY or VOID closes DQ6 UNTESTED. An R OOM at 512 in c_def is
  VOID by DQ4's rule; it is not re-drawn, because it would repeat on any 24 GB card. It is written up as "resident does
  not train 512 tokens" beside S's L\*.

## Budget, guard, rehearsal

- **Run:** one RTX 4090 on `vast:verified-secure`, gen ≥ 4 / x16, no bandwidth floor, `preflight_bandwidth: none` (no
  download). Declared at **$0.60/h**, SV6's 4090 rate (#1268). Guard **2.0 h**, estimated **≤ $1.20**, inside the
  standing no-ask tier for a single run under $15 (#564).
- **Time estimate:**
  - 14 builds at the 84–240 s DQ4 and DQ5 measured;
  - ladders at about 1.4× the 5090's ~1.75 ms per token-step;
  - about 70–95 min in all.
- **Pre-launch gate** (A2000, correctness only, never timing), at the launch commit:
  1. the runner refuses the A2000 at **rc 19**;
  2. `dq6_vram_probe.py` refuses at rc 3 (12 GB < 21.5);
  3. DQ4's reduced-subject rehearsal (8 layers at Qwen3-32B width, chunked loss) through `dq4_cap.py` at start 512 /
     step 512 passes `bench/dq4/dq4_rehearsal_check.py` (L\*_S > L\*_R, `late_bound_4bit` 56, chunked loss engaged);
  4. `dq6_reduce.py --self-test` passes.

## Not claimed

- Real weights. Values don't change bytes.
- A micro-batch above 1.
- A model other than Qwen3-32B's architecture.
- Any 4090 variant other than the 24 GB RTX 4090, or a different allocator than the two named.
- Step time as anything but descriptive.
