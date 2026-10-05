# DQ4 — results

Pre-registration: [DQ4-PREREG.md](DQ4-PREREG.md). Work item: #1083.

**Scope:** one RTX 5090 (PCIe gen 5 x16, machine 37087), e4b `3f5a8bc9`, and the registered model and recipe (DQ3's
Qwen3-32B-architecture subject, PEFT + bnb QLoRA, micro-batch 1, two AdamW steps per rung).

## Verdict — `dq4-5090-2`, 2026-10-05: **CAP_REAL**, G = **2.00** (graded configuration c_def)

With the frozen NF4 weights streamed (`train_prefetch` plus #1183's late-bound backward), the longest sequence that
trains on one RTX 5090 doubles:

| c_def: chunked loss, default allocator (**graded**) | R (resident) | S (streamed) |
|---|---|---|
| L\*: the largest passing rung, confirmed in a fresh process | **7168** | **14336** |
| first OOM rung, confirmed in a fresh process | 8192 | 15360 |
| allocated after setup, before the first rung | 18.42 GiB | 3.89 GiB |
| peak allocated / reserved at L\* (fresh) | 28.23 / 30.71 GiB | 22.94 / 29.16 GiB |

- **G = L\*_S / L\*_R = 14336 / 7168 = 2.00. CAP_REAL** (≥ 1.5).
- Each L\* is a rung on a 1024-token ladder, so G is bracketed by [L\*_S / first-OOM_R, first-OOM_S / L\*_R] =
  [14336 / 8192, 15360 / 7168] = **[1.75, 2.14]**. The bracket sits wholly above 1.5, so the verdict does not depend on
  the rung step.
- Every ladder stopped at its first OOM, and all four fresh confirmations agreed with their ladders. Nothing was VOID,
  FUNCTION_FAIL or NOISY, and no configuration was skipped.
- **Step time is unchanged:** T(S)/T(R) = **1.0005** at the largest common rung, 7168 tokens. This is descriptive.
- **The static saving is the weights, exactly:** R − S allocated after setup = 14.53 GiB = 64 × 243,793,920 B.

**Against the registered predictions** (DQ3 5090 numbers only):

| | predicted | read |
|---|---|---|
| L\*_R | ~8.6k [6144, 11264] | 7168, inside the band |
| L\*_S | ~20k [14336, 26624] | 14336, at the band's lower edge |
| G | ~2.3 [1.6, 3.2] | 2.00, inside the band |

## The measured cause of S's shortfall: allocator fragmentation

L\*_S fell to the lower edge of its band for the reason the pre-registration named as the risk. Under the default
allocator, S reaches its OOM with memory reserved but not allocated:
- at L\* (14336): **22.94 GiB allocated against 29.16 GiB reserved**;
- at the failing rung (15360): 19.0 GiB allocated against 30.62 GiB reserved.

That leaves about 6.2 GiB stranded. R, whose weights never move, strands about 2.5 GiB at its L\* (28.23 / 30.71).

The secondary configuration recovers it. Under `expandable_segments:True`, S's peak at L\* is 29.18 / 30.63 GiB, and
L\*_S rises to 19456, close to the ~20k the byte accounting predicted. This is a measured cause. It is **not** a
recommendation to change e4b's allocator default; that would need its own registration.

## Secondary and descriptive configurations

| config | role | L\*_R | L\*_S | G (bracket) | verdict | T(S)/T(R) |
|---|---|---|---|---|---|---|
| c_exp: chunked loss, `expandable_segments:True` | **secondary** | 8192 | 19456 | 2.375 [2.11, 2.50] | CAP_REAL (secondary) | 0.9985 |
| s_def: stock loss, default allocator | descriptive | 3072 | 8192 | 2.67 [2.00, 3.00] | (descriptive) | 1.0024 |

- **c_exp** is reported, as registered, as a secondary verdict, never the headline. At L\*: R 29.45 / 30.73 GiB,
  S 29.18 / 30.63 GiB.
- **s_def** shows why the graded pair runs the chunked loss. With Hugging Face's stock loss, the full-vocabulary logits
  hold R to 3072 tokens, and S to 8192, against 7168 and 14336 with the chunked loss. Without the chunked loss (#1193),
  the logits term would set both boundaries well short.

| | |
|---|---|
| box | RTX 5090 (driver 595.84, 450 W), **PCIe gen 5 x16**, AMD Ryzen 9 9950X, machine 37087 |
| software | torch 2.8.0+cu128, bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2 (tripwire: e4b `3f5a8bc9`, `_bnb_mirror_mismatches() == []`, `Qwen3ForCausalLM` in the chunked-loss table) |
| on-box checks | link gen 5 x16; VRAM probe 29.5 GiB; egress probe passed |
| engagement | every receipt: 448 `peft.tuners.lora.bnb.Linear4bit`, fp32 adapters, checkpointing; chunked loss patched (c_*); S arms offload 64 layers with `late_bound_4bit` 448 |
| cost | $0.689; teardown proven |
| evidence | [`receipts/dq4-5090-2/`](receipts/dq4-5090-2/) (`SHA256SUMS`); launcher receipt and ledger row in the store, commit `38e40582` |

## Launch attempts ($0.703 in all)

| run | machine | outcome | cost |
|---|---|---|---|
| `dq4-5090-1` | 147454 (Shanghai) | NOT_RUN: stuck `loading` 600 s. This was the machine's third failed draw today (also `dq3-5090-2` egress and `dq3-5090-4` stuck loading). A stuck load is not machine evidence (adertha#164). | $0.014 |
| `dq4-5090-2` | 37087 | **OK: CAP_REAL** | $0.689 |

**The pre-launch gate** was an A2000 rehearsal, correctness and memory only (`dq4_rehearsal_check.py`: L\*_S > L\*_R,
`late_bound_4bit` 56, chunked loss engaged).
- It passed at the launch commit under the **default** allocator (`dq4-a2000-gate-default-3f5a8bc9`).
- An earlier gate, `dq4-a2000-gate-3f5a8bc9`, had silently inherited the A2000 container's
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and is marked superseded.
- The box runner sets the variable per process, which is why the 5090 receipts record `default` and
  `expandable_segments:True` correctly.
- No A2000 number is used here as a G or a speed figure.

## Independent review (before the PR)

**Who reviewed, and what they found.** An agent separate from the run re-checked everything from the raw receipts and
found **no defect that changes any verdict**. It confirmed:
- the staged box files are byte-identical to `main`, and `3f5a8bc9` is an ancestor;
- the reducer re-run is byte-identical, and the self-test gives 25 OK;
- all 14 receipts match the registered device, subject, engagement, loss mode, allocator, offload and `late_bound_4bit`;
- the E4B_SHA is the launch commit everywhere;
- every ladder is contiguous and stops at its first OOM;
- every confirmation matches its ladder;
- every passing rung has two finite steps;
- R and S losses agree to about 1e-4 at each L.

Every OOM is a `torch.OutOfMemoryError` whose failed allocation scales with L: 800 MiB at 8192, 1.46 GiB at 15360,
1.95 GiB at 20480, which is L × 25600 × 4 B, the MLP intermediate. The s_def OOMs are the full logits (2.32 / 5.22 GiB).
The OOM messages themselves confirm the allocator split: the default-allocator S had 12.73 GiB reserved but unallocated
at its OOM, against 0.96 GiB under `expandable_segments`.

**Its findings, recorded here:**

1. **One marginal boundary.** c_exp R's fresh confirmation at 9216 completed one step (16.8 s, finite) and ran out of
   memory on the second; its ladder ran out on the first. Both count as an OOM at 9216, so L\*_R = 8192 stands, but that
   boundary is marginal. c_exp is the secondary configuration.
2. **The reducer is weaker than the prereg's wording in four places,** none triggered by these receipts:
   - E4B_SHA is compared within a configuration, not across all receipts or against the launch commit;
   - the ladder parameters (2048 / 1024 / 32768, two steps) and stopping at the first OOM are not checked;
   - a confirmation receipt's arm and mode are not checked;
   - the link gen and width are checked only by the runner's rc-13 gate.

   The reviewer checked all four by hand, and they hold.
3. **Cost bookkeeping.**
   - The launcher's receipt carries an estimate of $2.375, which includes a 100 GB download allowance; the prereg stated
     ≤ $1.28. Actual cost was $0.689: GPU $0.53, storage $0.157, download $0.002, 53 minutes.
   - The receipt's `commit_sha` (1064e1ea) is the control-repo commit, not e4b's.
   - Its cpu/ram fields read "unknown", although `forensics.txt` has them.
   - The recorded approval is the owner's relayed go, inside #564's no-ask tier.

## Not shown

- Gen 4, or any card other than the 5090.
- A micro-batch above 1.
- Real weights. Values do not change bytes.
- A model larger than Qwen3-32B.
- Any change to e4b's defaults.

## Reproduce

The verdict re-derives byte for byte from the committed receipts:

```
cd bench/dq4/receipts/dq4-5090-2
python3 ../../dq4_reduce.py . | cmp - dq4_read.json
python3 ../../dq4_reduce.py --self-test
```

`SHA256SUMS` covers every committed file.
