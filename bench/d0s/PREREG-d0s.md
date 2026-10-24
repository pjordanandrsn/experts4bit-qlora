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
  - The scale is equal by the reader rule; the training code applies alpha/r: `lora.py` read at the training
    version's release commit, `v0.56.1` = `b6bf338eb2a378c3273d4a84fb42a5d45992be5b` (an annotated tag; the tool
    resolves it to the commit and records it, and the re-derivation checks it against the remote tag).
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
   attention-only arm's served-path offset. It equals `D_f (1 − R)`: every arm scores the same 7,642 positions, and D0's
   pooled per-row `Q − REF` equals `D_f` to within 2e-16.
   - **Pass:** `|gap| ≤ M_f`. M_f is fixed now from D0's record (0.05 x D_f): **M_nodq = 0.016432685048378637** and
     **M_dq = 0.01682758935300882** nats, never recomputed from this read. With gate 1, a cell needs the CI's lower
     bound ≥ 0.95 AND the point R within [0.95, 1.05].
   - **Two-sided on purpose.** The question is fidelity: does the served merged arena reproduce the trained adapter's
     effect (REF)? A cell whose R exceeds 1.05 (`gap < −M_f`) is not reproducing REF; it is a different model or a
     different eval path. It FAILS as **BEYOND_REF**, "beyond REF: out of band, an artifact to explain", and is never
     read as worse than REF. A cell with `gap > M_f` FAILS as **BELOW_REF**. The verdict and the reducer name the side.
     The CI's lower bound does not rescue a high R.
3. **Canary**, pooled exact counts over the N_f scored positions (7,642 per fixture):
   - `a_X` = positions where X's argmax equals REF's;
   - `a_B` = positions where `BASE-ATTN-1`'s argmax equals Q's.
   - **Pass:** `100 a_X ≥ 100 a_B − N_f`, in integers, which means agreement at least BASE-ATTN's minus 1 percentage
     point.
4. **Cost**, the overhead question, so ONE-SIDED (slower only): TTFT, ITL and tokens/s at W1 (one request) and W16
   (16 concurrent requests), each request 512 prompt tokens and 64 new tokens, two passes with the second one timed;
   plus peak VRAM.
   - **Pass:** for each metric, X is slower than `BASE-ATTN-1` by at most the self-pair's band, floored at 3 %:
     `m_X − m_B1 ≤ |m_B2 − m_B1| + 0.03 m_B1` for TTFT and ITL, and `m_B1 − m_X ≤ |m_B2 − m_B1| + 0.03 m_B1` for
     tokens/s. Peak VRAM must be within 1 % either way.
   - A faster merged arm passes cost, with the speedup reported. A speedup beyond 3 % is named in the text as
     unexplained, since the merged arena should run the same kernels.
5. **Engagement**, checked before any gate. Every arm's record carries its served path. A BASE-ATTN arm serves no
   merged arena. A MERGED arm serves the merged block its writer wrote (`source` `arena` for MERGED-Q, `checkpoint` for
   MERGED-W) with the expert base `verified (merged)`. All of a fixture's arms share the placement, the graph status and
   the device. An arm whose record shows a different path is **VOID**, never a cell result.
   A record that is incomplete (an argmax list not one per scored position, a missing NLL, a cost record without all
   three metrics at both widths) is also VOID.

**A cell PASSES iff 1-4 all pass** and 5 holds. Reported for every cell, never gated:
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
| P5 | Every MERGED arm passes cost, and none is faster than `BASE-ATTN-1` by more than 3 % (same arena format and size) | about 90 % |
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

- **One RTX 4090 on Vast** (24 GB; OLMoE-1B-7B in NF4 is about 4 GB), rented through pod-launch at a declared
  **USD 0.85/h** (the card has no policy class row), with a download rate of at most USD 0.04/GB at the pin. The
  runner refuses a disk under 60 GB and host RAM under 30 GiB before the install.
- **The work, every term conservative:** install 600 s; checkpoint fetch 350 s (14 GB at the 40 MB/s floor); the GPU
  premise 300 s; the base bake 300 s; the attention-only derivations and the four writer runs 2,400 s (600 s each, the
  writer's CPU self-check over every row); eight arms at 400 s each (load, the eager teacher-forced pass over 7,642
  positions, and the cost runs), 3,200 s; reserve 600 s. **FIT = 7,750 s, x 1.25 = 9,688 s.**
- **Guard 3 h** ((boot 300 + 7,750) x 1.25 = 10,063 s ≤ 10,800 s). **Worst case:** 3 h x USD 0.85 + 30 GB x USD 0.04
  = **USD 3.75** a started box. A normal box is expected near 1.5 hours, about USD 1.4. The shakedown reads the real
  phase times before the reading, and an amendment re-fits if they differ.
- **At most two reading boxes start.** A box STARTS when its runner passes the refusals before the install and begins
  the install, whatever its outcome after that. Two started boxes cost at most 2 x USD 3.75 = **USD 7.50**.
- **A box that does not start** (refused at create, or by the runner's card, disk or RAM refusals before the install)
  costs at most **USD 0.20**: about 10 minutes of a box at USD 0.85/h (USD 0.15), rounded up, with nothing downloaded.
  It counts toward no cap and no strike, but it uses an id. At most six such boxes are spent in this lane; the seventh
  stops it for a person.
- **The shakedown** is explore tier: guard 0.75 h, worst 0.75 h x USD 0.85 + 8 GB x USD 0.04 = **USD 0.96** a box,
  at most three shakedown boxes (**USD 2.88**).
- **The lane's worst case:** USD 7.50 + 6 x USD 0.20 + USD 2.88 = **USD 11.58**, inside the USD 15 rail.
- **Ids:** `d0s-4090-1` to `-3` for readings, plus `d0s-4090-r1` to `-r6` for any box after one that did not start.
  Ids are labels, not budget: the two-started-box cap and the six non-start cap govern. `d0s-shake-1` to `-3` are for
  the shakedown. Two consecutive started runs not read stop the lane, and a shakedown never counts.
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
