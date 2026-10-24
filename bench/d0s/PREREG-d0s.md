# D0s: does D0's retained expert delta survive the paged server's merged arena? One box, both fixtures (registered before any run)

**Status:** registered before any data. One rented card, worst about USD 2.4 (see Budget). It is the served read that
#2078 and `bench/d0/ARENA-DESIGN.md` name: the gate for serving a trained expert delta through a merged NF4 arena
(`E4B_PAGED_MERGED_ARENA=1`, merged-arena writer #2079, server rules #2083). Until it reads, nothing about a merged arena
is claimed.

## The question

D0 (`bench/d0/RESULTS-d0.md`) found the expert delta RETAINED, by held-out loss, when re-quantized into NF4 offline,
on one OLMoE-1B-7B fixture in two forms. That was a CPU fp32 re-implementation of the forward. Does the PAGED SERVER,
serving a merged arena with the adapter it was merged from, keep at least 0.95 of the trained expert improvement on the
same 64 held-out rows, within the margin, at the cost of the base arena?

## Inputs (all fixed now; hashes in the store before any create)

- **Base:** `allenai/OLMoE-1B-7B-0924-Instruct` at `7f1c97f4`, the checkpoint D0 used.
- **Adapters:** D0's two fixtures as backend-adapter/2, derived by `bench/d0s/d0s_v2.py` (#2091) and committed to the
  store from the merged tool before any create: `fixture-nodq` (fp32 absmax) and `fixture-dq` (double-quantized).
  - **The provenance limitation, as registered:** the fingerprints were measured after training, over the verified
    stored base, not recorded by the trainer. 2 of 16 stacks are verified against the trainer digest; the other 14 rest
    on the deterministic rebuild.
  - The adapter weights are the trained ones, byte for byte (the same adapter id).
  - The scale is equal by the reader rule; the training code applies alpha/r.
- **Rows:** D0's 64 held-out rows per fixture (`bench/d0/receipts/d0-p224s3/{dq,nodq}/heldout.json`, 7,642 scored
  tokens), the same rows D0 scored.
- **REF and Q (attention only):** D0's registered per-row NLL (`.../pass/arms.csv`, arms `REF` and `Q`). Their
  per-position argmax, needed for the canary, is computed before any create, on CPU, by D0's own pinned forward
  (`bench/d0s/d0s_ref.py`, reusing `d0_forward.py` and `d0lib.py` from D0's receipt). It must reproduce `arms.csv`'s
  per-row NLL to a relative 1e-6, or the read is VOID before any rental. The argmax files' sha256 go to the store.

## Arms (per fixture, on one box, the paged server's default stack, all-VRAM placement)

| arm | arena | adapter |
|---|---|---|
| `BASE-ATTN-1`, `BASE-ATTN-2` (the self-pair) | the base arena: `bake_nf4` of the checkpoint on the box | the fixture with its expert `lora_B` zeroed: attention only, a derived file whose sha256 is recorded |
| `MERGED-Q` | the writer's merge, `source="arena"` (D0's "NF4 from Q") | the fixture |
| `MERGED-W` | the writer's merge, `source="checkpoint"` (D0's "NF4 from W") | the fixture |

Each arm builds the server through `serve_paged.build_engine` (the merged arms under `E4B_PAGED_MERGED_ARENA=1`, so
refusal 13 and the post-load `merged_qfp_v1` check run as served). It then makes one teacher-forced pass over the 64 rows
in fp32 log-probs. That gives per-row NLL sums and the per-position argmax.

If the writer refuses (its self-check, or its base verification), the cell is **WRITER_REFUSED**, with the writer's
reason. It is reported and not served. The read is still read for the other cells.

## The rule, per fixture f and merged cell X

`D_f = L_D0(Q) − L_D0(REF)` is D0's registered expert improvement: 0.32865370096757274 nats (nodq) and
0.3365517870601764 nats (dq), `bench/d0/receipts/d0-p224s3/{nodq,dq}/summary.json`, `step2.improvement`.

1. **Retention:** `R = [L(BASE-ATTN-1) − L(X)] / D_f`, with every L pooled per token over the 64 rows. The 90 % CI
   comes from a paired bootstrap over rows (2,000 resamples, seed 0), the numerator and the denominator resampled
   together from the served per-row NLL and D0's per-row REF and Q.
   - **Pass:** the CI's lower bound ≥ 0.95.
2. **Gap:** `gap = [L(X) − L_D0(REF)] − [L(BASE-ATTN-1) − L_D0(Q)]`, the served cell's distance to REF less the
   attention-only arm's served-path offset. It equals `D_f (1 − R)`.
   - **Pass:** `|gap| ≤ M_f`. M_f is fixed now from D0's record (0.05 x D_f): **M_nodq = 0.016432685048378637** and
     **M_dq = 0.01682758935300882** nats, never recomputed from this read.
3. **Canary**, pooled exact counts over the N_f scored positions (7,642 per fixture):
   - `a_X` = positions where X's argmax equals REF's;
   - `a_B` = positions where `BASE-ATTN-1`'s argmax equals Q's.
   - **Pass:** `100 a_X ≥ 100 a_B − N_f`, in integers, which means agreement at least BASE-ATTN's minus 1 percentage
     point.
4. **Cost:** TTFT, ITL and tokens/s at W1 (one request) and W16 (16 concurrent requests), each request 512 prompt
   tokens and 64 new tokens, two passes with the second one timed; plus peak VRAM.
   - **Pass:** for each metric, `|m_X − m_B1| ≤ |m_B2 − m_B1| + 0.03 m_B1` (the self-pair's band, floored at 3 %).
     Peak VRAM must be within 1 %.

**A cell PASSES iff 1-4 all pass.** Reported for every cell, never gated:
- R's point and CI;
- the bootstrap SE;
- the self-pair's NLL difference (expected 0: the server is deterministic);
- KL is not computed (REF's full distribution is not stored);
- the writer's retention receipt (ρ̄, c̄, f) beside the measured R;
- the served attention-only offset `L(BASE-ATTN-1) − L_D0(Q)`.

## Predictions (written before any data)

| # | prediction | confidence |
|---|---|---|
| P1 | All four cells PASS. D0's NF4 cells had R 0.99-1.01 and gap_to_ref up to 0.0036 nats, and the served path adds only bf16 compute | about 70 % |
| P2 | R stays within ±0.03 of D0's corresponding NF4 R (nodq Q 1.0110, W 0.9917; dq Q 0.9968, W 1.0006) | about 60 % |
| P3 | The served attention-only offset is under 0.02 nats in both fixtures | about 70 % |
| P4 | The self-pair is bitwise identical (NLL difference 0) | about 85 % |
| P5 | The cost metrics of every MERGED arm fall within the band (same arena format and size) | about 90 % |
| P6 | No cell is WRITER_REFUSED. A CUDA-baked base passes the writer's CPU self-check, as on D0's fp32 base (2,048 of 2,048) | about 75 % |

## What follows, registered now

- **All four PASS:** the merged NF4 arena is a gated cell PASS on D0's fixtures. `docs/ADAPTER_PORTABILITY.md`'s
  expert matrix and `docs/STATUS.md` cite this read, with its scope: one OLMoE-1B-7B fixture per absmax form, 200 steps.
  `E4B_PAGED_MERGED_ARENA` stays opt-in; a default is a separate decision.
- **A cell FAILS:** that cell (source x trained-base form) is refused. A follow-up PR makes the writer or the server
  refuse it by name, citing this read, and the reason enters the matrix, as the spec's gated-cell rule says.
- **WRITER_REFUSED:** reported with the writer's reason. A CUDA base the CPU self-check cannot reproduce is a finding
  for the writer (a device-matched self-check), not a strike.
- Nothing here moves a gate, a threshold or a registered claim beyond these.

## The shakedown, then the reading

1. **Shakedown** (one box, never data): installs and tripwires, then the WHOLE flow on the GPU serve smoke's tiny
   random Qwen3-MoE, with no checkpoint download:
   - its bake;
   - both writer sources;
   - the attention-only derivation;
   - the four arms' passes;
   - the cost measures;
   - the reducer.

   It prints SHAKE_OK or the failing step. The premises run here too: the default build carries no merge record, and
   `verified (merged)` holds for both merged arms.
2. **The reading** launches only after the shakedown is SHAKE_OK and after the maintainer's explicit Approved on this
   registration's merge PR. It goes through pod-launch with `RENT_OWNER_GATE=1` on a pinned host. The kit's sha256 is
   posted on the bus before the create.
3. **A CI test** (`tests/test_d0s_*.py`, CPU, in the kit PR that adds these tools) runs every phase that needs no GPU: the reducer's rule on fixed numbers
   (including each gate's boundary), the REF tool on D0's synthetic plumbing fixture, the attention-only derivation, and
   the writer on a tiny bake.

## Budget and STOP rules

- **Card and cost:** one RTX 4090 (24 GB; OLMoE-1B-7B in NF4 is about 4 GB), declared at USD 0.85/h. Guard 2.5 h.
  Worst `2.5 x 0.85 + 20 GB x 0.011` ≈ **USD 2.35** for the reading, and ≤ USD 1 for the shakedown.
- **Strikes:** a box that STARTS and does not read is a strike; two strikes stop the lane.
- A refused create, NOT_RUN, or a box that never starts is not a strike.
- REF's reproduction failing is VOID before any rental.

## What this read cannot say

- **Scope:** one fixture per absmax form, one model and one card. Nothing generalizes beyond them.
- **Formats:** NF4 merged arenas only; INT4 and MXFP4 are later items.
- **Quality instrument:** teacher-forced held-out loss and argmax agreement, not generation quality.
- **Placement:** all-VRAM. The solver placement's own quality read is a separate item.
- **Provenance:** the base fingerprints are derived, with the limitation stated above.

## Receipts

The receipt lands in the store and is committed within minutes of teardown:
- the kit's sha256;
- both derived adapters' manifests;
- the attention-only files' sha256;
- the arenas' merge blocks and retention receipts;
- per-arm per-row NLL and argmax;
- the cost measures;
- the reducer's output.

The public results PR copies the receipt without tensors, after the maintainer's re-derivation.
