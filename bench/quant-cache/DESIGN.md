# A sha256-keyed cache of the quantized experts — design packet (before any code)

**Goal.** Every start of `load_moe_4bit_streaming(..., offload=True)` rebuilds the same NF4 experts from the bf16
checkpoint. On Qwen3-30B-A3B that is 48 layers × 128 experts, and on `p137-4090-7` the loader's "fusing + quantizing
experts" phase took 33 min (08:57 → 09:29, about 40 s a layer, on an AMD EPYC 7742). Every trainer user pays it at every
start, and so does every paid box. A cache keyed by what the bytes depend on would pay it once per model and stack.

## What the loader does today (e4b main)

The Qwen3 per-expert branch, with offload on CUDA:
1. Read each expert's rows from the safetensors shards: lazy CPU `safe_open`, 128 × 3 reads a layer
   (`loader.py:1773-1796`).
2. Fuse them on the CPU with `fuse_experts` (`arch/moe_conventions.py:724-750`): a stack and cat of about 1.2 GB of
   bf16 per layer. The fused tensors then go to the GPU.
3. Quantize on the GPU: `Experts4bit.from_float` → `_quantize_stack`, one `bnb.functional.quantize_4bit` per expert
   matrix (`_vendor/experts.py:399-427`; `compress_statistics=False`).
4. Make the offload homes: copy GPU → CPU, then `pin_memory()`, a second host copy (`engines/offload.py:507-543`).
   The GPU tensors are then swapped for placeholders.

**One phase receipt exists, on another host.** Q235's `LoadProfile` (`bench/q235/instrument/q235_box.py:325`) wraps the
reads (`get_tensor`, with bytes), the assembly (`fuse_experts` / `stack_experts`), the quantize (`from_float`) and the
offload (`enable_expert_offload`), and synchronizes CUDA at each boundary. On the A2000 legs (`bench/q235/gate2/`,
Qwen3-235B's 4-layer slice) it reads about 17.2 s reading, 25.1 s assembling, 0.72 s quantizing and 3.35 s offloading.
That is an A2000 timing, so it sets no number for another host. It is structural support only: the GPU quantize is
small, and the host-side reads and assembly are the cost. The cache skips steps 1–3 and the device round trip of
step 4. The later read reuses this instrument.

**The per-layer state that results** (`Experts4bit`, `_vendor/experts.py:124, 253-281`):

| tensor | dtype | shape |
|---|---|---|
| `gate_up_proj` | uint8 | [E, 2I·H/2] |
| `down_proj` | uint8 | [E, H·I/2] |
| `gate_up_absmax` | fp32 | [E, numel/blocksize] |
| `down_absmax` | fp32 | [E, numel/blocksize] |

The extra state is `quant_type, blocksize, has_gate, num_experts, hidden_dim, intermediate_dim`. The offload engine
reads exactly these four names (`offload.py:377-378`). Qwen3-30B-A3B is 0.3397 GB a layer, 16.3 GB a model.

**No save or load of quantized experts exists in the package.** `save_adapter` saves LoRA tensors only. The NVMe arena
(grouped-nf4-gemm's `nvme_bake_nf4`) holds the same bytes, but its handles are `_routed_in_train`, which
`hold_forward_residency` refuses. So layer-major, P137's licensed path, cannot train from it. The cache is a host-RAM
offload input, not an arena. Prior art for the format is `bench/hp1/hp1_lib.py:219-300` (`export_base` / `BaseStore`:
per-layer safetensors, packed bytes, absmax, an index sha256).

## The change

1. **`E4B_EXPERT_CACHE=DIR`, opt-in** (unset: today's path, byte for byte). It applies to `load_moe_4bit_streaming`
   for 4-bit storage (`Experts4bit`, NF4 or FP4) with `offload=True`. The arena and 8-bit storage are refused by
   structure.
2. **The key** is the sha256 of a canonical JSON of everything the bytes depend on:
   - the checkpoint identity: `config._commit_hash`, or for a local directory the sha256 of every shard and the index;
   - `model_type`, and the set of quantized layers with each one's `(quant_type, blocksize)`;
   - the load `dtype`, `has_gate` and the fuse order;
   - `E, H, I`, and the extra-state schema version;
   - `compress_statistics=False`;
   - the bitsandbytes version (its kernel quantizes) and the torch version;
   - `QUANT_FORMAT_VERSION`, not the e4b version. It is a constant bumped when `fuse_experts`, `_quantize_stack` or the
     extra-state schema changes, and a CPU test pins those functions' source hashes against it. A release that touches
     none of them keeps every user's cache.
   - a cache-format version (the file layout).

   The GPU name and capability are **recorded, not keyed**. Cross-device equality is unproven
   (`docs/claims.json`: same-box re-bakes are bit-identical, and nothing is said across boxes), so a hit on another card
   is verified, never assumed (point 4). If verification fails, the device joins the key and the run is a miss.
3. **The layout:** `DIR/<key>/manifest.json`, plus one safetensors file per MoE layer with the four tensors.
   - The manifest holds the key's fields, each file's sha256 and size, and the extra state.
   - A miss builds the model as today, then writes the cache from the pinned homes: no extra GPU work, and the write
     is atomic (temporary directory, then rename).
   - A write failure warns and never fails the load.
   - **Concurrent misses.** Two processes may write the same key. Each writes its own temporary directory. The one whose
     rename loses (renaming onto an existing non-empty directory fails on Linux) verifies the winner's manifest and
     files, uses them, and discards its own temporary directory. A CPU test runs that race.
4. **A hit:**
   - Every file's sha256 must match the manifest. A mismatch **refuses** with the path, unless
     `E4B_EXPERT_CACHE_REBUILD=1`; a corrupt cache is never trained on.
   - The `Experts4bit` modules are built from the packed tensors through a from-packed constructor, which runs the same
     `set_extra_state` validation, and the safetensors expert rows are not read.
   - The homes go through the **same builder** as today (`offload._build_homes` / `_to_home`, or the slab or arena
     builders under `E4B_OFFLOAD_PIN_SLAB` / `E4B_OFFLOAD_ARENA`). Only the source changes: each tensor is read from the
     file into a CPU tensor, then homed exactly as a fresh load's would be. That keeps the pinned footprint identical,
     including torch's caching host allocator rounding each pinned request up to a power of two (1.333× on Qwen3 today;
     a per-file slab would round differently, about 1.5×). It also keeps `is_pinned()` as it is today, False for
     registered slab memory, which the pipelined engine's zero-copy gather checks. There is no device round trip.
   - On a device whose name differs from the one recorded, one layer is re-quantized fresh and compared byte for byte
     before the hit is used. A difference is **a miss, not a refusal**: the model is rebuilt and written under a key
     with the device folded in, with a warning. Only a sha256 mismatch, which means corruption, refuses.
5. **Unchanged:** the non-expert weights, `verify_moe_4bit(strict=True)`, the loader's `n_moe` guard, the offload
   engine's structural check, the absmax-DQ refusal under offload (the cache always stores fp32 absmax), and every
   training path downstream.
6. **No code runs from the cache.** Reads are safetensors and the JSON manifest only, never a pickle (`torch.load`)
   path. A cache directory copied from elsewhere can supply bytes, which the sha256 and the key check, but it cannot
   execute code.
7. **The disk budget** at the recorded NF4 scheme (fp32 absmax, blocksize 64):
   - Qwen3-30B-A3B: 0.3397 GB a layer × 48 = **16.3 GB**;
   - Qwen3-235B-A22B: 1.359 GB a layer × 94 = **127.7 GB**.

   Before writing, a miss checks free space against the manifest's total plus 5 %, and skips the write with a
   warning if it would not fit. A partial write (a full disk or a crash) leaves only the temporary directory, which is
   never read as a cache and is removed on the next miss. A hit whose files do not match the manifest refuses (point
   4). The cache is never trimmed automatically; removing `DIR/<key>` is the user's call.
8. **The trainer** inherits it through the environment. One log line says hit, miss-and-wrote, or rebuilt, with the
   key and the seconds spent.

## The $0 correctness legs (no timing is read)

**A2000, on the tiny Qwen3-MoE fixture first, then on the real Qwen3-30B-A3B.** Each check is STRICT:
- **Bytes.** On a miss followed by a hit in a second process, every expert tensor of every layer is equal byte for byte
  (a sha256 over all four tensors per layer) to a fresh load without the cache.
- **Training.** Under `FAST_TRAIN=1 LAYER_MAJOR=1 OFFLOAD_EXPERTS=1`, the trainer's step-0 losses are bitwise equal with
  and without a cache hit, and so are the stagings per step.
- **Refusals:**
  - a flipped byte in one layer file refuses on the hit;
  - a key field changed (`blocksize`, or the bitsandbytes version spoofed in the manifest) is a miss, never a hit;
  - a different device name runs the re-quantize check, and a forced mismatch there is a miss that rebuilds, never a
    refusal;
  - `E4B_EXPERT_CACHE` with 8-bit storage or an arena refuses.
- **Host memory, per pin mode** (default, `E4B_OFFLOAD_PIN_SLAB=1`, `E4B_OFFLOAD_ARENA=1`). On a hit, host RSS
  (`RssShmem` / `RssAnon`) and the homes' byte count equal a cache-less load's, and so does `is_pinned()` per home.
- **Default off.** With the variable unset, the loader is byte-for-byte today's: same tensors, same log lines.

## The gain (a reading, not part of this packet)

How much of the 33 minutes the cache removes is unread on a training host: the only phase split is Q235's A2000
legs, above. A registered read would put a
miss and a hit on one rented box, record the phases with Q235's `LoadProfile`, and be REPORTING-only, or registered against a
bar if a number is to be quoted. Until then the docs say only what the cache skips. Disk cost: 16.3 GB for
Qwen3-30B-A3B.

## If deferred

Every offloaded start keeps paying the full rebuild: about 33 minutes on `p137-4090-7`'s host, which is the larger part
of each P137/P140 box's paid time.
