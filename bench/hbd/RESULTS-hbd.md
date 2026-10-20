# HBD results: READ. Across builds, each framework's exported base differs in one attention slot; within a build, the two frameworks' bases are identical

**Status: READ, 2026-10-20**, under `PREREG-hbd.md` (#1902) with the kit of #1911, on one RTX 5090. The run is `hbd-5090-1`
(e4b `248c35ae`). It cost $0.416, against a worst case of $1.58. The maintainer re-derived the reading from the private
receipt store with `hbd_reduce.py` at `248c35ae`, and it came out byte-identical.

**No parity claim** is made in any outcome, and none is made here. Nothing in HP1's, HP2's or HPN's records is changed.

**The reading.** At 2 layers, the build changes the exported base, in one attention slot, and the framework does not. HP2's
base term (e4b on torch 2.8.0+cu128, Unsloth on 2.12.1+cu130) is therefore **consistent with a build effect**, not a
framework effect. Depth is untested here: HP2 read its term at 48 layers, and the size of the build effect at depth stays
open.

## The cells

The 2-layer cut of Qwen3-30B-A3B @ `ad44e777` was used, with HPN's one-step recipe on held-out pool rows 97-396 and the export
plus `--hbd-record` in every cell.

| cell | framework | torch | transformers | libbitsandbytes mapped in the cell (= its probe) | dequantized digest | stored-state digest |
|---|---|---|---|---|---|---|
| E0a | e4b | 2.8.0+cu128 | 5.18.0 | `libbitsandbytes_cuda128.so` | `5e7564ea` | `91160aab` |
| E0b | e4b | 2.8.0+cu128 | 5.18.0 | `libbitsandbytes_cuda128.so` | `5e7564ea` | `91160aab` |
| E2 | e4b | 2.8.0+cu128 | 5.5.0 | `libbitsandbytes_cuda128.so` | `5e7564ea` | `91160aab` |
| E1 | e4b | 2.12.1+cu130 | 5.5.0 | `libbitsandbytes_cuda130.so` | `19363fa6` | `c97cd201` |
| U0a | Unsloth 2026.9.14 | 2.8.0+cu128 | 5.5.0 | `libbitsandbytes_cuda128.so` | `5e7564ea` | `c8451191` |
| U0b | Unsloth 2026.9.14 | 2.8.0+cu128 | 5.5.0 | `libbitsandbytes_cuda128.so` | `5e7564ea` | `c8451191` |
| U1 | Unsloth 2026.9.14 (native_torch) | 2.12.1+cu130 | 5.5.0 | `libbitsandbytes_cuda130.so` | `19363fa6` | `f867c76d` |
| U2 | Unsloth 2026.9.14 (grouped_mm) | 2.12.1+cu130 | 5.5.0 | `libbitsandbytes_cuda130.so` | `19363fa6` | `f867c76d` |

All eight cells are `ok`, with 533 slots each and the same slot-key set. The device was an RTX 5090, compute capability 12.0,
driver 595.84. bitsandbytes is 0.50.2 in every cell.

The stored-state digests of e4b and Unsloth differ within each build by construction. e4b holds its q/k/v projections as
`nf4/64` and Unsloth as `nf4/64+dq`, so the cross-framework rows are read on the dequantized digest only.

## The registered outcomes, in order

1. **No VOID.**
   - The determinism controls hold: E0a = E0b and U0a = U0b in both digests.
   - Every cell mapped exactly one `libbitsandbytes`, and its sibling probe agrees.
   - Each build pair loaded a different library file.
2. **No STRUCTURE.** Every compared pair has the same slot count and key set.
3. **Across builds, at fixed transformers:**
   - **e4b (E2 vs E1): DIFFERENT_QUANTIZED.**
   - **Unsloth (U0a vs U1): DIFFERENT_QUANTIZED.**
4. **e4b across venvs at a fixed build (E0a vs E2): SAME.** It moves transformers from 5.18.0 to 5.5.0 (and adds Unsloth to the
   venv) and does not move the base.
5. **Reported rows:**
   - E0a vs E1, HP2's e4b side with both changes together: DIFFERENT_QUANTIZED, in the same slot.
   - U1 vs U2, the backend under cu130: **SAME**. The backend changes compute, not stored weights.
   - E0a vs U0a, the frameworks under cu128, on the dequantized digest: **SAME_DEQUANT** (`5e7564ea`). This repeats HPN's
     finding on another card.
   - E1 vs U1, the frameworks under cu130, on the dequantized digest: **SAME_DEQUANT** (`19363fa6`).
6. **Localization.**
   - In both frameworks, exactly one of 533 slots differs across builds, in its stored state and in its dequantized
     fingerprint: `1|q|-1`, layer 1's q_proj (8,388,608 elements).
   - Its dequantized fingerprint is (int16 sum, weighted sum) = (−5,606,734,617, −184,948,031,200,168) under cu128, and
     (−5,606,734,063, −184,948,019,457,595) under cu130.
   - The dequantized fingerprint's int16 sum differs by 554.
   - The values move identically in e4b and in Unsloth under each build. Every other slot is bit-identical across builds.
   - The receipt does not count how many codes differ within the slot.

## With HPN

Under one build, the two frameworks export identical bases:
- HBD: SAME_DEQUANT under cu128 and under cu130;
- HPN's reads on the RTX A2000: equal base digests at L2, L4 and L8 under torch 2.8.0+cu128.

So no evaluator can move HPN's δ0 between them, which is HPN's stage-1b MOOT. That is the mechanism. HPN's registered call is
completed by its own record.

## The box

- One RTX 5090, driver ≥ 580, pinned by inclusion after a 6-read stability screen and a 40 s hold. The pin was posted before
  the create.
- The model fetch took only the two shards the cut reads.
- The rate cap was **$1.00/h** (maintainer, 2026-10-20), replacing the PREREG's $0.70. The card class is unchanged. Worst case
  = 1.25 h × $1.00 + 30 GB × $0.011 = $1.58; actual $0.416.
- A $0 real-record smoke on the RTX A2000 (E0a and U0a with `--hbd-record`) ran before the box, as the maintainer's launch
  condition required.

## Limits

- One model (Qwen3-30B-A3B), one depth (2 layers), one card (RTX 5090, sm_120) and one bitsandbytes version (0.50.2).
- "The build" is the torch, CUDA and triton set together. Which of them moves the q_proj slot is not split further, though
  each build loaded its own `libbitsandbytes` CUDA library.
- The count of differing codes in the slot is not measured.
- A correctness read: it measures no speed and no loss.

## Receipts

[`receipts/hbd-5090-1/`](receipts/hbd-5090-1/README.md) holds the box's own outputs, byte-identical to the private receipt
store, with `SHA256SUMS`: the eight cell receipts, the in-cell records and probes, the spec and `result.json`, the logs and
`summary.txt`. Six logs carry carriage returns from the box's progress output. They stay only in that store, byte-exact, and
`STORE-ONLY.sha256` lists them.
