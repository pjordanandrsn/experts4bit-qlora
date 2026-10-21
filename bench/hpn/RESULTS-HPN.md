# HPN results: MOOT. Under one build, e4b's and Unsloth's exported bases are identical, so no evaluator can move δ0

**Status: READ, 2026-10-21. The registered call is MOOT**, terminal under `HPN-DESIGN.md`: stage 2 cannot lift G1, so the
lane stops with the answer in, at $0 on the RTX A2000. The call comes from `hpn-a2000-read8`: one climb at L4 and L8 at e4b
`8dc66652`. The maintainer re-derived it from the private receipt store, and it came out byte-identical.

**No parity claim** is made, about either framework. HP1's and HP2's records stand as written.

**Scope label:** identical bases under one build (torch 2.8.0+cu128); not comparable to HP2's 1b.

## The question

HP2's calibration read a per-row gap g = (evaluator zero-adapter loss) − (in-loop step-0 loss), about 0.03 nats/row at 48
layers. HPN asks where that gap comes from. It reads three readers on the same rows: the in-loop step 0 (I), the bf16 evaluator
(E) and an fp32 reference (F). Then g = (E − F) − (I − F).

Stage 1b asks first whether the base itself moves δ0, the mean e4b-minus-Unsloth gap. If the fp32 reader leaves δ0 where the
evaluator put it (|δ0_F| / |δ0_E| > 0.93), no evaluator can lift HP2's G1, and the read is MOOT.

## The registered read (`hpn-a2000-read8`)

The cut of Qwen3-30B-A3B @ `ad44e777` at L4 and L8, on HPN's one-step recipe on held-out pool rows 97-396 (300 rows). e4b
ran K00 (two arms), and Unsloth 2026.9.14 ran on torch 2.8.0+cu128 with the native_torch MoE backend.

| depth | base digest (e4b / Unsloth) | δ0_E (SE) | \|δ0_F\| / \|δ0_E\| | 1b | sd(g) | share_e | corr(E, I) | call |
|---|---|---|---|---|---|---|---|---|
| L4 | `4268a9e3` / `4268a9e3` | 0.004872 (0.000335) | 1.0 | **MOOT** | 0.006142 | 0.5662 | 0.4633 | I |
| L8 | `0976b540` / `0976b540` | 0.0052 (0.000647) | 1.0 | **MOOT** | 0.010446 | 0.5879 | 0.4233 | I |

- **N0:** every reader repeats exactly (I, E and F).
- **N3** fails at both depths, and the two deepest calls are I and I (N2: not comparable).
- **The noise guard holds:** |δ0_E| > 2 SE at both depths, so the read is MOOT, not INDETERMINATE.
- The base term is 0 under each reader (base_term_E = base_term_F = 0). The two frameworks' bases have the same dequantized
  digest at each depth: the same bf16 weights, though their stored formats differ. So the fp32 reader cannot move δ0 at all.

**Reported only, labelled EXTRAPOLATION, never in the call:**
- A log-log fit of sd against depth, from L4 and L8, gives 0.041 (g), 0.039 (E) and 0.029 (I) nats/row at 48 layers.
- Read 7's L2 + L8 fit gave 0.033 (read 7, run hpn-a2000-read7-combined-2-8, the private receipt store; supporting only).
- HP2 measured about 0.03 at 48 layers.

## Supporting reads (same lane, earlier commits)

| read | what it read | why it is not the call |
|---|---|---|
| `hpn-a2000-read4` | L2 + L4 READ; base digests `5e7564ea` (L2) and `4268a9e3` (L4), identical across the frameworks | L8 skipped on a stale disk reading (fixed in #1867) |
| `hpn-a2000-read5` | L2 + L4 with the same reduced depth records as read 4 | L8's base export ran out of memory on the 12 GB card (classed as a depth limit in #1890) |
| `hpn-a2000-read6` | L8 only: the export OOM classed as the limit | its cause, `slot_fingerprint`'s int64 temporaries, fixed in #1913 (bit-identical) |
| `hpn-a2000-read7` | L2 (reproducing the committed `5e7564ea` for both frameworks, the end-to-end check of #1913) + L8 (`0976b540`) | two runs, and its second deciding depth was L2, below the registered L4 floor |

Read 8's L8 digest repeats read 7's, and its L4 depth record repeats reads 4 and 5. Reads 1 to 3 failed in the harness, not
on the question: a venv gate (#1850), a trainable count (#1854) and a shadowed variable (#1859).

## The mechanism, with HBD

`bench/hbd/RESULTS-hbd.md` reads the same question from the other side, on one RTX 5090:
- **Within each build,** e4b's and Unsloth's dequantized bases are identical. That holds under cu128 and under cu130.
- **Across builds,** each framework's base differs in one attention slot.

So the frameworks agree whenever they share a build, and an evaluator cannot move δ0 between bases that are identical. That
is why stage 1b reads MOOT here. HP2's base term, read with e4b on cu128 and Unsloth on cu130, is consistent with a build
effect; HBD's record states that reading and its limits (2 layers; depth untested).

## Limits

- One model (Qwen3-30B-A3B) and depths up to L8 of 48. L16 is a stated limit of the design.
- One card (the lab's RTX A2000, correctness only; nothing here is a timing) and one build per read (torch 2.8.0+cu128).
- Unsloth's native_torch MoE backend, not grouped_mm.
- The side call (I) and N0-N3 are reported. With 1b MOOT the design takes no decomposition verdict from them.

## Receipts

[`receipts/hpn-a2000-read8/`](receipts/hpn-a2000-read8/README.md) holds the run's public subset. `result.json`, the spec, the
pin proof, the bulk manifest and 12 logs are byte-identical to the private receipt store, with `SHA256SUMS`.

The registered reducer's inputs carry the host's identifiers or the run's own paths, so they stay in the store. That covers
the arm receipts, the evaluator outputs and the reference outputs. Their projections, holding only the fields
`hpn_reduce.py` reads, are under `proj/`, written by `bench/hpn/hpn_public_inloop.py`. With `hpn_spec.public.json`, the
reducer rebuilds `hpn/result.json` byte-identically from committed files, and `tests/test_hpn.py` checks it.
`STORE-ONLY.sha256` lists every other file of the run.
