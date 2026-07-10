# sm_120 validation + at-scale expert-offload A/B — RTX PRO 6000 Blackwell (rented)

First Blackwell (sm_120) run of the experts-4bit (e4b) stack, plus a seed-matched
Qwen3-30B-A3B expert-offload A/B at the scale of the headline claim — which the local
RTX A2000 12 GB cannot run (its resident arm OOMs). Session run 2026-07-10 over SSH on a
rented RunPod pod; artifacts pushed from the owner's Mac (no GitHub credential on the pod).

## Preflight (Gates)

| item | value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell **Workstation Edition** (600 W) |
| **Gate 1 — compute capability** | **(12, 0) = sm_120 — PASS** |
| GPU memory | 97 887 MiB (96 GB) |
| driver / VBIOS | 580.126.16 / 98.02.81.00.07 |
| torch (image) → after axolotl pins | 2.8.0+cu128 → **2.13.0+cu130** |
| **Gate 2 — power-limit writable** | **NO (host-locked, no SYS_ADMIN)** → Workstream C cannot be done by `-pl` simulation on this pod; needs a real Max-Q pod |
| **Gate 3 — disk** | `/workspace` 921 TB free (MooseFS network vol); `/` overlay 107 GB |
| pod | `x6opj6u8imwv3y`, SECURE, on-demand, **$1.89/hr** |
| network volume | none — US-IL-1 PRO6000-WS out of stock; fell back to any-DC + fresh 200 GB ephemeral vol (cold HF cache) |

Receipt: `receipts/preflight.json`.

## Environment

Built per the Unsloth Blackwell path (`TORCH_CUDA_ARCH_LIST=12.0`), from source for sm_120.

| component | pin |
|---|---|
| bitsandbytes | `feature/experts-4bit` @ `e7f4d86`, built `CMAKE_CUDA_ARCHITECTURES=120` → **`libbitsandbytes_cuda130.so`** |
| axolotl | PR #3797 `expert-offload-integration` @ `fd12f92` (public branch; cloned unauthenticated) |
| kit | `experts4bit-qlora` @ `ed796e5` |
| transformers / peft / accelerate / trl | 5.12.1 / 0.19.1 / 1.13.0 / 1.5.1 |
| attention backend | **PyTorch SDPA** (`flash_attention: false`; xformers not built — not required) |

Deviation from handoff (non-obvious, load-bearing): axolotl's dependency pins upgrade torch
to **2.13.0+cu130**, and bitsandbytes derives its native-library name from torch's CUDA
version — so a from-source build needs the **CUDA 13.0** toolkit (installed `cuda-nvcc-13-0`
+ `cuda-cudart-dev-13-0` + `libcublas-dev-13-0`), not the image's 12.8. Building against 12.8
produces `libbitsandbytes_cuda128.so`, which the torch-cu130 bnb refuses to load. Full
sequence in `receipts/environment.json` + `receipts/pip-freeze.txt`.

## Workstream A — e4b test suite on sm_120  ✅

First green run of the experts-4bit suite on Blackwell (previously Ampere-only, RTX A2000 sm_86).

| suite | result | wall |
|---|---|---|
| bitsandbytes `tests/test_experts4bit.py` | **59 passed, 0 failed** | 7.28 s |
| bitsandbytes `tests/ -k experts` (full-tree sweep) | **59 passed**, 6957 deselected | 7.19 s |
| experts4bit-qlora kit `tests/` | **168 passed, 4 skipped, 0 failed** | 12.64 s |

- **PINNED TEST COUNT: 59/59** experts-4bit tests pass on sm_120 with sm_120-compiled kernels.
- `tests/test_experts_nbit.py` — **does not exist** on `e7f4d86` (handoff anticipated "if present post-split"). `-k experts` matches exactly `test_experts4bit.py`; no experts tests live elsewhere.
- One environmental fix (handoff-permitted): `pip install lion_pytorch` — `tests/test_optim.py` failed to **import** at collection time (unrelated optional optimizer), not a test failure. **No library code was patched.**

Artifacts: `results/A-testsuite-sm120.log`, `results/A-sweep-k-experts.log`, `results/A-kit-tests-sm120.log`, `results/A-summary.json`.

## Workstream B — Qwen3-30B-A3B seed-matched expert-offload A/B  ✅

Both arms NF4 (`quantize_moe_experts: true`), QLoRA over frozen experts, **seed 42, identical
data order, 150 steps, eval every 50**. Configs differ by exactly three lines
(`expert_offload`, `output_dir`, `wandb_name`) — the single-variable discipline of the prior
OLMoE A/B, extended to the 30B model. `eval_on_start` gives a step-0 BEFORE baseline.

Shared config: seq_len 2048, sample_packing, grad_accum 4, micro_batch 1, lr 1e-4 cosine,
r8/α16, attn-only LoRA (q,k,v,o), SDPA. Per-arm `nvidia-smi` VRAM sampler.

| arm | expert_offload | BEFORE (step 0) | AFTER (step 150) | eval@50 | eval@100 | s/step | train mem (max_active) | peak VRAM (smi) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| resident | false | 5.30 | **1.512** | 1.874 | 1.536 | 3.15 s | 73.61 GiB | 76 992 MiB |
| offloaded | true | 5.30 | **1.510** | 1.868 | 1.534 | 7.12 s | 60.39 GiB | 63 948 MiB |

Offload homed **96 expert layers across 48 MoE blocks (14.50 GB) to pinned CPU RAM; one block
resident at a time**.

Load-bearing findings:
- **BEFORE is bit-identical off-vs-on (5.30 == 5.30)** and AFTER matches to 2e-3 (1.512 vs 1.510) — the frozen NF4 forward is unchanged; offload is **location, not math**, at 30B scale (the A2000 could only show this at OLMoE 1B-7B; its Qwen3-30B resident arm OOMs on 12 GB). First training losses match step-for-step (5.547, 5.107, 5.037…); the seed-matched curves overlay.
- **Offload s/step is 2.26× resident** (7.12 vs 3.15) — the per-block PCIe re-stage cost (forward + gradient-checkpoint recompute), consistent with the memory-for-compute framing. `train_mem_max_active` drops **73.61 → 60.39 GiB** (the ~14.5 GB of experts leave the resident set).
- **Both arms peak ~64–77 GB** because axolotl's loader materializes the full bf16 model before quantizing → this A/B is **only runnable on a ≥~80 GB card**, which is the at-scale point. (The kit's own streaming loader is what fits 12 GB; axolotl's does not stream — see Open questions.)

Prior baselines extended (RTX A2000 12 GB, sm_86 — cited, not overwritten):
OLMoE-1B-7B resident 6.00 GB peak / offload 2.60 GB; Qwen3-30B **resident OOM**, offload fine-tune 7.16 GB peak.

Artifacts: `results/B-ab-summary.json`, `results/B-qwen30b_resident.jsonl`, `results/B-qwen30b_offload.jsonl`, `results/B-loss-overlay.png`.

### B4 — decode serving slice (sm_120, greedy batch-1, 96 tokens)  ✅

| config | offload | prefetch | gemv | tok/s | peak GPU |
|---|:---:|:---:|:---:|:---:|:---:|
| **resident** | – | – | ✓ | **11.559** | 20.06 GB |
| offload, serial | ✓ | – | ✓ | 1.514 | 4.10 GB |
| offload, prefetched | ✓ | ✓ | ✓ | 1.752 | 4.44 GB |
| offload, prefetched, dequant | ✓ | ✓ | – | 1.751 | 4.44 GB |

**Headline contrast: Qwen3-30B-A3B decodes resident at 11.56 tok/s in 20 GB on the 96 GB card — where the A2000 OOMs resident and manages ~0.22 tok/s offloaded (a ~52× resident-vs-offloaded-A2000 gap).** Prefetch recovers 1.514 → 1.752 tok/s (1.16×) over serial offload; GEMV is neutral at this model shape (1.752 vs 1.751), matching the METHODOLOGY's Qwen3-30B finding. Artifact: `results/B4-decode.json`.

## Workstream C — real Max-Q (300 W) vs Workstation (600 W)  ◑ 600 W half done; 300 W pending stock

Gate 2 failed (`-pl` host-locked), so the handoff's power-limit *simulation* is impossible on this
pod — vindicating the owner's "rent both SKUs, measure real" choice (also better silicon-truth than
a power-capped 600 W card). **The real Max-Q SKU was out of stock at every attempt this session**
(6 tries, "no instances currently available"). Per the de-risk plan, the **600 W half is captured**
so only the 300 W half remains against the same pinned seed/config/commit whenever a Max-Q pod frees:

| SKU | watts | s/step (200-step slice, seed 42) | tok/s/gpu | actual draw (median / max) |
|---|:---:|:---:|:---:|:---:|
| Workstation Edition | 600 | **3.14** | 1434 | 483.9 W / 509.9 W |
| Max-Q Workstation | 300 | ⟨pending stock⟩ | — | — |

The 600 W card draws only ~484 W median under this training slice — already below the 600 W cap,
which is a useful prior for the Max-Q comparison (a 300 W cap may bite less than the nameplate
suggests here). Artifact: `results/C-powerlimit.json`. **Cross-SKU caveat:** the two SKUs are
different physical hosts, so the eventual delta carries host variance, not just the power binning —
recorded in the artifact.

## Deviations from handoff

1. **torch cu130, not cu128** — axolotl pins force it; bnb must be built with the CUDA 13.0 toolkit (see Environment). The handoff's `-DCMAKE_CUDA_ARCHITECTURES=120` is correct; the toolkit version is the added step.
2. **No persistent volume** — US-IL-1 had no PRO 6000 WS stock; fell back to any-DC ephemeral. Cost: cold HF download (~61 GB Qwen3-30B). No effect on results.
3. **Gate 2 host-locked** → Workstream C via `-pl` impossible; pivoted to real-SKU rental (owner-approved).
4. **Pod is on-demand, not spot** — SECURE on-demand at $1.89/hr (the SKU wasn't offered interruptible at launch).
5. **`test_experts_nbit.py` absent** on the branch — pinned count is from `test_experts4bit.py` (59).
6. **Push from Mac, not pod** — no GitHub credential placed on the rented pod (tighter than the handoff's PAT-on-pod).
7. **Workstream C is half-complete** — 600 W measured; 300 W blocked on Max-Q stock all session. Not skipped: pinned for resumption.

## Cost ledger

Single SECURE on-demand pod `x6opj6u8imwv3y` @ $1.89/hr. Session wall ~2 h at report time
(preflight + env build + A + B + B4 + C-600W) ≈ **$3.8** on this pod; still inside the $14–17
budget with headroom for the Max-Q pod when stock returns.

## Open questions

- Does a Max-Q (300 W) pod become available within budget for the real C delta?
- Would the kit's own streaming loader (not axolotl's) let this 30B A/B run on a 24 GB card? (axolotl loads full bf16 first.)
