# HBD: does the build, rather than the framework, change the exported base? (pre-registration)

**Status: REGISTERED, before any data.** A $0-design, at-most-$1 correctness read on one rented card. It reports which
variable moves the exported-base digest. It makes **no parity claim** in any outcome, and it does not touch HP1's, HP2's or
HPN's records.

## Why

HP2 (`bench/hp2/RESULTS-HP2.md`) read a base term between e4b's and Unsloth's exported bases. HPN's reads on the RTX A2000
found the two frameworks' bases identical. The receipts already split part of the question:

| read | framework | torch build | Unsloth MoE backend | layers | `hp1.train_base.digest` |
|---|---|---|---|---|---|
| HP2 (`hp2-5090-1`) | e4b | 2.8.0+cu128 | - | 48 | `65b6083d` |
| HP2 (`hp2-5090-1`) | Unsloth 2026.9.14 | 2.12.1+cu130 | grouped_mm | 48 | `e8e441c0` |
| HPN read 5 | e4b | 2.8.0+cu128 | - | 2 | `5e7564ea` |
| HPN read 5 | Unsloth 2026.9.14 | 2.8.0+cu128 | native_torch | 2 | `5e7564ea` |
| HPN read 5 | e4b | 2.8.0+cu128 | - | 4 | `4268a9e3` |
| HPN read 5 | Unsloth 2026.9.14 | 2.8.0+cu128 | native_torch | 4 | `4268a9e3` |

bitsandbytes is 0.50.2 in every row. Every row also carries the same labelled regimes: e4b's q/k/v projections are `nf4/64`
and Unsloth's are `nf4/64+dq`. So under one build that regime label does not move the digest. On one build the frameworks
agree. Where they differ (HP2), three variables changed together: the build, Unsloth's MoE backend and the depth.

**The mechanism under test.** bitsandbytes ships a separate CUDA library per CUDA major (a cuda12x and a cuda130 build of
`libbitsandbytes`). A base's NF4 bytes are written by that library's quantize kernel when the framework loads the bf16
weights. So "the build" most likely means **which compiled quantize kernel ran**. This read records that library in every
cell.

**Why a rented card.** The A2000's driver (575) cannot run cu130 (it needs >= 580), and a CPU run cannot exercise a CUDA
kernel. Upgrading the A2000's driver would touch every GPU container on its host for one read, so that is not done.

## The cells (one box, one variable at a time)

All cells share these settings:
- Qwen/Qwen3-30B-A3B @ `ad44e777`, cut to 2 layers by `bench/hp1/hp1_trunc.py`.
- HPN's `hpncal` recipe: one step on the calibration tokens (held-out pool rows 97-396), seed 3407, order seed 3407.
- `--hp1-export DIR --hp1-export-base 1`.

e4b runs HPN's arm a (`--attn-4bit 1`, the K00 cell). Unsloth runs HPN's Unsloth arm (seven targets, native LoRA init and
dtype).

| cell | framework | interpreter (TC1's venvs) | torch | Unsloth MoE backend |
|---|---|---|---|---|
| E0a | e4b | `venv-e4b` | 2.8.0+cu128 | - |
| E0b | e4b | `venv-e4b` (a repeat of E0a) | 2.8.0+cu128 | - |
| E1 | e4b | `venv-unsloth` (`E4B_VENV=t212`, TC1's t212 row: e4b and grouped-nf4-gemm at the box's pins) | 2.12.1+cu130 | - |
| U0 | Unsloth 2026.9.14 | `venv-unsloth-t28` (`UNS_VENV=t28`) | 2.8.0+cu128 | native_torch |
| U1 | Unsloth 2026.9.14 | `venv-unsloth` | 2.12.1+cu130 | native_torch |
| U2 | Unsloth 2026.9.14 | `venv-unsloth` | 2.12.1+cu130 | grouped_mm |

**The exporter is pinned.** Every cell runs the same `tc1_arm.py` and `hp1_lib.py`, staged from the kit's merge commit. The
receipt records both files' sha256 and e4b's commit. E1 installs e4b at that same commit into `venv-unsloth`.

**The digest.** It is `hp1.train_base.digest` as `tc1_arm.hp1_base_digest` computes it at that commit. For every slot that
`hp1_base_slots` lists (expert gate_up and down per expert, attention q/k/v/o, router, norms, embeddings and lm_head):
1. dequantize THE FRAMEWORK'S OWN WAY to bf16 on the device;
2. fingerprint as (numel, the sum of the int16 bit patterns, the same sum weighted by (position mod 65521) + 1);
3. fold with sha256 over the sorted slot keys.

**Recorded in every cell** (condition 2):
- the `libbitsandbytes*.so` that loaded, by its path and sha256, read from the process's own memory maps after a CUDA
  quantize;
- `torch.__version__` and `torch.version.cuda`;
- the device's name, compute capability and driver version;
- bitsandbytes' version.

A probe records these in a fresh process with the cell's own interpreter and environment, just before the cell runs, so
the receipt states that the probe stands in for the cell's process.

## Outcomes (decided in this order)

1. **VOID** if:
   - E0a's digest differs from E0b's (the determinism control: the build is not compared in that case);
   - any cell is not `ok` or carries `hp1.train_base_error`;
   - any cell's probe is missing or names no `libbitsandbytes` library;
   - E0a and E1 loaded the same `libbitsandbytes` file (sha256), or U0 and U1 did. Then the build comparison did not
     change the kernel it is meant to test.
2. **Per framework, across builds** (E0a vs E1 for e4b; U0 vs U1 for Unsloth):
   - **SAME**: on this model at 2 layers, the build does not change that framework's exported base.
   - **DIFFERENT**: a build-dependent base. HP2's base term is then, at least in part, a build effect.
3. **Backend under cu130** (U1 vs U2): reported. The backend changes compute, not stored weights, so SAME is expected and a
   DIFFERENT is a finding to name.
4. **Across frameworks within one build** (E0a vs U0, E1 vs U1): reported. E0a vs U0 repeats HPN's finding on another card.
5. **Localization.** For any DIFFERENT, the exported bases of the two cells are compared slot by slot, by tensor content
   (`.safetensors` bytes are not stable across saves: the header's metadata order varies). The slot kinds that differ are
   reported: experts, attention, or other.

**What SAME at 2 layers does not say:** anything about depth. HP2's gap is at 48 layers, so after SAME at both frameworks
the question stays open, and depth (or the backend at depth) is the next variable. That would be its own registration.

**No directional prediction.** Both outcomes are plausible: one kernel source compiled against two CUDA versions may or may
not round identically.

## The box

- One rented card with driver >= 580 and >= 24 GB (an RTX 5090, HP2's class).
- The host is pinned by inclusion: one machine id, read and checked. The pre-flight refuses a driver below 580 before any
  install.
- Worst-case cost $1.00 (the installs plus six one-step cells at 2 layers). It sits inside the lane's standing authority.
- Two strikes and the lane stops and simplifies, per the house rule.
- The receipt goes to the private store and is re-derived before anything is quoted. The kit (the cells, the probe and a
  reducer that applies the table above) comes in its own PR after this one, from this registration's merge commit.

## Limits

- One model (Qwen3-30B-A3B), one depth (2 layers), one card class, one bitsandbytes version (0.50.2).
- The probe reads the library in a sibling process, not inside the cell.
- A correctness read: it measures no speed and no loss.
