# HBD: does the build, rather than the framework, change the exported base? (pre-registration)

**Status: REGISTERED, before any data.** A $0-design, at-most-$1.25 correctness read on one rented card. It reports which
variable moves the exported base, and at which stage (quantized bytes, or their dequantization). It makes **no parity claim**
in any outcome, and it does not touch HP1's, HP2's or HPN's records.

## Why

HP2 (`bench/hp2/RESULTS-HP2.md`) read a base term between e4b's and Unsloth's exported bases. HPN's reads on the RTX A2000
found the two frameworks' bases identical. The receipts already split part of the question:

| read | framework | torch build | Unsloth MoE backend | layers | `hp1.train_base.digest` |
|---|---|---|---|---|---|
| HP2 (`hp2-5090-1`) | e4b | 2.8.0+cu128 | - | 48 | `65b6083d` |
| HP2 (`hp2-5090-1`) | Unsloth 2026.9.14 | 2.12.1+cu130 | grouped_mm | 48 | `e8e441c0` |
| HPN reads 4 and 5 (`hpn-a2000-read4`, `hpn-a2000-read5`) | e4b | 2.8.0+cu128 | - | 2 | `5e7564ea` |
| HPN reads 4 and 5 | Unsloth 2026.9.14 | 2.8.0+cu128 | native_torch | 2 | `5e7564ea` |
| HPN reads 4 and 5 | e4b | 2.8.0+cu128 | - | 4 | `4268a9e3` |
| HPN reads 4 and 5 | Unsloth 2026.9.14 | 2.8.0+cu128 | native_torch | 4 | `4268a9e3` |

bitsandbytes is 0.50.2 in every row. Every row also carries the same labelled regimes: e4b's q/k/v projections are `nf4/64`
and Unsloth's are `nf4/64+dq`. So under one build that regime label does not move the digest. On one build the frameworks
agree. Where they differ (HP2), several variables changed together:
- the torch build, and with it triton (3.4.0 vs 3.7.1);
- transformers (e4b's venv at 5.18.0, Unsloth's at 5.5.0);
- Unsloth's MoE backend;
- the depth.

**The mechanism under test.** bitsandbytes ships a separate CUDA library per CUDA major (a cuda12x and a cuda130 build of
`libbitsandbytes`). A base's NF4 bytes are written by that library's quantize kernel when the framework loads the bf16
weights. So "the build" most likely means **which compiled quantize kernel ran**. This read records that library in every
cell, and digests the quantized state itself as well as its dequantization, so it can tell the two apart.

**Why a rented card.** The A2000's driver (575) cannot run cu130 (it needs >= 580), and a CPU run cannot exercise a CUDA
kernel. Upgrading the A2000's driver would touch every GPU container on its host for one read, so that is not done.

## The cells (one box, one variable at a time)

All cells share these settings:
- Qwen/Qwen3-30B-A3B @ `ad44e777`, cut to 2 layers by `bench/hp1/hp1_trunc.py`.
- HPN's `hpncal` recipe: one step on the calibration tokens (held-out pool rows 97-396), seed 3407, order seed 3407.
- `--hp1-export DIR --hp1-export-base 1`.

e4b runs HPN's arm a (`--attn-4bit 1`, the K00 cell). Unsloth runs HPN's Unsloth arm (seven targets, native LoRA init and
dtype). The interpreters are TC1's venvs.

| cell | framework | interpreter | torch | transformers | Unsloth MoE backend |
|---|---|---|---|---|---|
| E0a | e4b | `venv-e4b` | 2.8.0+cu128 | 5.18.0 | - |
| E0b | e4b | `venv-e4b` (a repeat of E0a) | 2.8.0+cu128 | 5.18.0 | - |
| E2 | e4b | `venv-unsloth-t28` + e4b and grouped-nf4-gemm at the box's pins | 2.8.0+cu128 | 5.5.0 | - |
| E1 | e4b | `venv-unsloth` + e4b and grouped-nf4-gemm at the box's pins (TC1's t212 row) | 2.12.1+cu130 | 5.5.0 | - |
| U0a | Unsloth 2026.9.14 | `venv-unsloth-t28` | 2.8.0+cu128 | 5.5.0 | native_torch |
| U0b | Unsloth 2026.9.14 | `venv-unsloth-t28` (a repeat of U0a) | 2.8.0+cu128 | 5.5.0 | native_torch |
| U1 | Unsloth 2026.9.14 | `venv-unsloth` | 2.12.1+cu130 | 5.5.0 | native_torch |
| U2 | Unsloth 2026.9.14 | `venv-unsloth` | 2.12.1+cu130 | 5.5.0 | grouped_mm |

The transformers versions above are the venvs' current pins. Every cell records its own, and the reduction reads the recorded
values, not this table. Triton moves with the torch build (3.4.0 with 2.8.0, 3.7.1 with 2.12.1), so "the build" here means the
torch, CUDA and triton set together. Recording triton in every cell keeps that visible.

**The exporter is pinned.** Every cell runs the same `tc1_arm.py` and `hp1_lib.py`, staged from the kit's merge commit. The
receipt records both files' sha256 and e4b's commit. E1 and E2 install e4b at that same commit into their venvs.

## What every cell records

1. **The dequantized digest** (`hp1.train_base.digest`, as `tc1_arm.hp1_base_digest` computes it at that commit). For
   every slot that `hp1_base_slots` lists (expert gate_up and down per expert, attention q/k/v/o, router, norms, embeddings
   and lm_head):
   - dequantize THE FRAMEWORK'S OWN WAY to bf16 on the device;
   - fingerprint as (numel, the sum of the int16 bit patterns, the same sum weighted by (position mod 65521) + 1);
   - fold with sha256 over the sorted slot keys.
2. **The stored-state digest**, new in the kit. For every 4-bit slot, the quantized state exactly as the framework holds it:
   - the packed bytes;
   - the absmax as stored, with its dtype, so a double-quantized absmax is its quantized bytes;
   - and, where the absmax is nested, the nested state's absmax, code, offset and blocksize.

   Each slot hashes to sha256 over those fields' bytes in a fixed order, folded with sha256 over the sorted slot keys. Every
   bf16 slot contributes its bytes. Per-slot hashes are written beside the folded digest, for localization.
3. **The slot structure:** the slot count and the sha256 of the sorted slot-key list.
4. **The library, read inside the cell.**
   - The `libbitsandbytes*.so` files mapped into the cell's own process (`/proc/self/maps`), with path and sha256, read right
     after the base export.
   - The environment as the cell sees it: `BNB_CUDA_VERSION` and `LD_LIBRARY_PATH`.
   - A sibling probe repeats the read in a fresh process with the cell's interpreter and environment, as a cross-check. For
     Unsloth cells it imports `unsloth` first, as the cell does.
5. **Versions and device:** torch (`torch.__version__`, `torch.version.cuda`), triton, transformers and bitsandbytes;
   the device's name, compute capability and driver version.

## Outcomes (decided in this order)

1. **VOID** if any of these holds:
   - E0a differs from E0b, or U0a from U0b, in either digest (the determinism controls; no build is compared then);
   - any cell is not `ok`, or carries `hp1.train_base_error`;
   - any cell's in-process read names no `libbitsandbytes`, or names more than one, or disagrees with its sibling probe;
   - a build pair (E2 vs E1, U0a vs U1) loaded the same `libbitsandbytes` file (sha256). The comparison then did not change
     the kernel it is meant to test.
2. **STRUCTURE** for any compared pair whose slot count or slot-key hash differs: the pair is not compared by digest, and
   the differing keys are listed. A naming change is not a value change.
3. **Per framework, across builds at fixed transformers** (E2 vs E1 for e4b; U0a vs U1 for Unsloth). Each pair reads:
   - **SAME** if both digests are equal: on this model at 2 layers, the build does not change that framework's exported base;
   - **DIFFERENT_QUANTIZED** if the stored-state digests differ: the build's quantize writes different bytes;
   - **DIFFERENT_DEQUANT** if the stored-state digests are equal but the dequantized digests differ: same bytes, decoded
     differently by the two builds.

   Either DIFFERENT means HP2's base term is, at least in part, a build effect.
4. **e4b across transformers at fixed build** (E0a vs E2): read the same way, as the load path's own effect.
5. **Reported:**
   - E0a vs E1, the two changes together (HP2's e4b side);
   - U1 vs U2, the backend under cu130. The backend changes compute, not stored weights, so SAME is expected and a DIFFERENT
     is a finding to name;
   - E0a vs U0a and E1 vs U1, the frameworks within one build. E0a vs U0a repeats HPN's finding on another card.
6. **Localization.** For any DIFFERENT, the per-slot stored-state hashes (and the dequantized fingerprints) show which slot
   kinds differ: experts, attention, or other.

**What SAME at 2 layers does not say:** anything about depth. HP2's gap is at 48 layers, so after SAME at both frameworks
the question stays open, and depth (or the backend at depth) is the next variable. That would be its own registration.

**No directional prediction.** Both outcomes are plausible: one kernel source compiled against two CUDA versions may or may
not round identically.

## The box

- **The card.** One rented card with driver >= 580 and >= 24 GB (an RTX 5090, HP2's class).
- **The host** is pinned by inclusion: one machine id, read and checked. The pre-flight refuses a driver below 580 before any
  install.
- **The model fetch** takes only what a 2-layer cut reads:
  - `model-00001-of-00016.safetensors` and `model-00016-of-00016.safetensors` (5.09 GB, layers 0-1 and the global
    tensors);
  - the index, the config and the tokenizer files.
- **The cost:**
  - rate cap $0.70/h;
  - guard 1.25 h;
  - `download_gb` 30 (the two shards plus three venvs' wheels).
  - Worst = 1.25 h × $0.70 + 30 GB × $0.011 = **$1.21**, stated as $1.25. It sits inside the lane's standing authority.
- **Strikes:** two, and the lane stops and simplifies, per the house rule.
- **The receipt** goes to the private store and is re-derived before anything is quoted.
- **The kit:** the cells, the in-cell stored-state digest and library read, the sibling probe, and a reducer that applies
  the table above. It comes in its own PR after this one, from this registration's merge commit.

## Limits

- One model (Qwen3-30B-A3B), one depth (2 layers), one card class, one bitsandbytes version (0.50.2).
- The build is the torch, CUDA and triton set together. It is not split further.
- A correctness read: it measures no speed and no loss.
