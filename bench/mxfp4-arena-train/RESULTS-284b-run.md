# The 284B run — P1 confirmed, P2–P4 ungraded, blocked upstream (2026-08-14)

Graded against `PREREG.md` as amended twice. **The claim is not established.** The
ARENA arm stopped before its first training step, on a dependency gap, and that is
the result — not a reason to retry.

## Verdicts

| prediction | verdict |
|---|---|
| **P1 — STOCK fails** | **CONFIRMED**, and on the mechanism amendment 1 respecified. `torch.OutOfMemoryError: CUDA out of memory … GPU 0 has a total capacity of 23.56 GiB of which 2.38 MiB is free`. Attempt 1's control died on a quantizer-class rejection (`FineGrainedFP8Config`); this one is a genuine memory ceiling. |
| **P2 — VRAM 4–8 GB** | **UNGRADED.** No training step ran. One adjacent datum did land: the model *loads* at **10.63 GiB peak** on a 23.6 GiB card, dense FP8 resident, experts deferred to the arena. That is a load figure, not P2's staged-stack figure, and it is reported as such. |
| **P3 — bytes unchanged** | **UNGRADED.** Nothing trained, so nothing could move. |
| **P4 — it learns** | **UNGRADED.** |
| G1 (>0 modules patched) | **not reached** — `enable_nvme_train_residency` raised inside its own geometry check. |
| G2 (overlay in the imported module) | **PASSED**, per file, functionally. |
| G4 (on the lane it claims) | **not reached.** |

## What did work, and is worth recording

The whole pipeline up to the geometry check ran clean on one 24 GB card:

- **download** — 149 GB in **128 s** (~1.16 GB/s) to local NVMe.
- **bake** — the MXFP4 **relocation** arena in **76.8 s**: 43 layers × 256 experts,
  `row_stride` **13,369,344 B**, six per-projection segments. That stride is
  exactly the figure computed from the checkpoint's own shard header before
  renting (12.75 MiB/expert → 147.2 GB total, against the 147 GB in the docs).
- **load** — `load_moe_4bit_streaming(..., arena=…, arena_train=True)` in **20.7 s**:
  365 block-scaled FP8 linears installed, "experts deferred to the arena: 43
  module(s), 215 unmaterialized buffer(s)". No re-quantization: the loader takes
  `build_meta_experts` on this path (the "fusing + quantizing" line it prints is
  unconditional, emitted before the branch).

## The blocker

```
enable_nvme_train_residency -> check_arena_geometry -> nvme_residency.segment_geometry
KeyError: 'F8_E8M0'                                    (nvme_residency.py:287)
```

`grouped-nf4-gemm`'s `nvme_residency._ST_TO_TORCH` maps
`U8/I8/F16/BF16/F32/F64/I16/I32/I64/U16/U32/U64` — and **not `F8_E8M0`**, which is
what a real DeepSeek-V4 checkpoint labels its expert scales
(`w1.scale F8_E8M0 [2048, 128]`, verified from the shard header).

The gap is narrow and internally inconsistent: **two other gnf4 modules already
know the dtype** — `mxfp4_residency._PACKED_BYTE_DTYPES` and
`nvme_bake_nf4._MXFP4_BYTE_DTYPES`, both with comments saying in as many words
that "V4 says I8/F8_E8M0, K3 says U8/U8 — same bytes, different label". So gnf4
can **bake** a V4 arena and **serve** from it (the documented 147 GB A5000 serving
run went through `mxfp4_residency`), but the **training** tier's geometry check
goes through `segment_geometry` and cannot read it.

This is a dependency gap, not a defect in the overlay under test. Fixing it is a
mapping entry upstream. It is deliberately **not** patched here: the stop rules
say the code changes only under a new stamped prereg, and a run patched mid-flight
would not be the run that was registered.

## Why the CPU gate did not catch this, and what changed

The gate's fixture baked **every** tensor as `U8`. Real V4 ships honest labels.
Same bytes, different string — so every staging test passed against a label no
DeepSeek-V4 checkpoint uses, and the gap surfaced only after ~3.5 minutes of I/O
on a rented card.

That is *test the ROUTE, not the mechanism* in its purest form: a hand-built
fixture agreed with the code instead of with the world.

`_bake_mxfp4` now takes a `labels` argument and
`test_a_real_v4_arena_labels_its_scales_f8_e8m0` bakes `V4_LABELS`
(`I8` blocks, `F8_E8M0` scales). It reproduces the pod's exact failure —
`KeyError: 'F8_E8M0'` at the same line — **in 3.6 s on CPU**. It is
`xfail(strict=True)`, so the day gnf4 ships the mapping it fails for passing,
which is the signal to drop the marker and re-run this leg.

## Two harness bugs, both mine, both fixed

1. **`load_moe_4bit_streaming(MODEL_ID, …)` instead of the local path.** The
   loader does `snap = model_id if os.path.isdir(model_id) else snapshot_download(...)`,
   so passing the hub id started a SECOND 160 GB download onto a disk already
   holding the first copy plus the 138 GiB arena. It surfaced as
   `RuntimeError: Internal Writer Error`, not as an honest ENOSPC — the disk was
   at 99%.
2. **`capacity_for_bytes(stride)`** — the signature is
   `capacity_for_bytes(usable_bytes, row_stride, *, pinned=True)`. The one-arg
   call raised and fell back to a hardcoded 4000 rows = **53 GB** of pinned DRAM
   off a 12.75 MiB stride. Now sized from the **cgroup** (`memory.max` minus
   `memory.current`, 60 GB measured → 3072 rows), because `free`/psutil report the
   *host's* memory on a pod.

A third, earlier: the G2 gate grepped `_eligible.__doc__` for a marker that lives
in the function's **code**, so it failed a correctly-overlaid pod. It is now a
behavioural check — call `_eligible` on an MXFP4-flagged stub, require a refusal
naming MXFP4 — verified in the CPU container both with and without the overlay.

## Cost and teardown

Four pods this leg (two A5000 wedges, one G2 abort, one full run), **~$0.28**;
about **$0.41** across the whole session against the $35/job cap. Every pod
verified gone by re-query (`404`), reconciled by id against the account list, and
the single backstop stood down on its own. The two other lanes' pods
(`k3-postfix`, `gnf4-style-lora`) were never touched.

`HOLD` did its job: the ARENA failure landed *after* the download and bake, so the
pod was kept rather than torn down, which is what made the diagnosis above
possible without paying for the 3.5 minutes of I/O a second time.

## Next step

One mapping entry in `grouped-nf4-gemm`'s `nvme_residency._ST_TO_TORCH`
(`F8_E8M0` → `uint8`, and `F8_E4M3` alongside it), with a test that bakes V4's
labels. Then a pre-data amendment 3 and a re-run. **No number in this file should
be read as evidence that the re-run will succeed** — the arms past G1 have never
executed.
