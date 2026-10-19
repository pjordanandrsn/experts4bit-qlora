# P150 (bench/p150): D2's quality ruling, the int8 lm_head on held-out windows (e4b#1313)

**Status:** registered, not run. A ruling on **quality only**. It times nothing and changes no default, gate or claim.

## Why

[P148](../p148/RESULTS-p148.md) read **D2**, an int8 lm_head (W8A16, a per-row fp32 scale), GO in a CUDA-graph
microbench: it projects 0.190 ms of the one-request step on an RTX 5090. That is a premise, never a gain. The int8 head
perturbs the logits directly, so before any kernel reaches the served path, a registered read must rule that the
rounding keeps P125's quality gate. The earlier reads of the int8 head were reported, never ruled:
- the W8A16 split premise ([bench/i8dense](../i8dense/PREMISE-w8a16.md), #1745) read I8HEAD on this model and
  instrument as reported only;
- P147 read it on Mixtral (a different model).

## Held-out windows: what makes this a confirmation

The split box already read I8HEAD on P125's own windows. Those are wikitext-2-raw-v1 **test**, windows 0-195 at t1
and 0-122 at k16, at stride 1024. A rule fixed after that read, re-run on the same deterministic windows, would be a
reproduction, not a confirmation. So **P150 rules on windows nobody has read**:
- **The source is the same corpus's validation split** (disjoint articles), with P125's join and tokenisation. P125's
  test split leaves only 96 windows disjoint from the ones already read, too few for 196. The validation split holds
  262,210 tokens with the pinned tokenizer, room for 256 windows of 640 at stride 1024, so the counts stay at **196
  (t1) and 123 (k16)** without shrinking. As in P125, k16's windows are t1's first 123.
- **Their sha256s are registered here, before any data.** Each is the sha256 of the JSON of the gate's token-id
  windows, with no spaces, measured with transformers 5.17.0 and Qwen3-30B-A3B's tokenizer at `ad44e777`. As a check on
  the pipeline, it reproduces P125's 298,938 test tokens exactly.
  - t1 (196 windows): `5c444982538d1aaba2b4f47c00509c0604a72b3845f5cb98c7b30d07b81a76ec`
  - k16 (123 windows): `7ab52a502ffc5292802db52e4325f025688deeed735a8c04ce33fba5789acded`
- **The box checks them before any arm.** `p150_box.py --windows-check` recomputes both, and a mismatch refuses the box
  (19). Every quality record carries the hashes of the windows it read, and the reducer VOIDs on any other.
- **Continuity, reported only:** the split box's I8HEAD read on P125's windows sits beside the ruling as the earlier
  read, cited, not re-read: t1 argmax 0.9892, upper bound 0.00103 against 0.00549; k16 argmax 0.9904, 0.00125 against
  0.00580.

## The instrument: the split premise's, plus one wrapper

- **The box and the round trip are staged at their merged bytes.** `i8_box.py` and `i8_fakequant.py` come from
  `bench/i8dense`, pinned in `staged-p150.sha256` to the hashes `staged-i8dense.sha256` pins. So are P125's instrument,
  P115's quality instrument and its imports, P39's bake, and P125's premise tests.
- **`p150_box.py` is the one new piece.** It points `i8_box.windows` at the held-out loader, which is P125's
  `wikitext_windows` with only the split changed (`tests/test_p150.py` holds the two equal line for line). It checks and
  records the hashes. c4 stays P125's (16 windows, reported only).
- **The subject is P125's:** e4b `4934aba9` with grouped-nf4-gemm 0.43.0 (`6ee2e104`), Qwen3-30B-A3B at `ad44e777`,
  and the default server built eager at all-VRAM. Every int4 lever and engine knob is refused.
- **Two arms, each its own process, in this order:**
  - **A:** the default. It writes R's log-probs per gate.
  - **I8HEAD:** A's build, then `i8_fakequant.apply(model, "head")`. The lm_head (151936 x 2048, untied) is rounded in
    place to its per-output-row int8 grid, `s = absmax / 127` (1.0 for a zero row) and `q = clamp(round(w / s), ±127)`,
    stored as `q x s` in bf16. Every attention projection, expert, the router, the embeddings and the norms stay bf16,
    untouched.
- **This is P148's lever.** The grid equals `p148_kernels.quantize_rows_int8`, bit for bit (`tests/test_p150.py`).
  P148's int8 kernel multiplies by the fp32 scale after an fp32 accumulation, where the round trip stores `q x s` in
  bf16. P148's fp64 contract already held the kernel at cuBLAS's own error on the bf16 head (1.660e-3 against 1.665e-3).
  What this read rules on is the rounding.

## The rule (`p150_reduce.py`, fixed before data)

I8HEAD alone decides, against A, on the held-out windows at **both** `t1` and `k16`, with the split premise's rule
unchanged:
- **GO** iff, at t1 **and** at k16:
  - mean argmax agreement ≥ 0.95, **and**
  - the one-sided 95 % upper bound d + 1.645 SE ≤ P125's bound, ln(1 + 0.05 / ppl_A) over that gate's own windows.
- **PARK** otherwise. The int8 head parks with the numbers.
- **VOID** if any of these holds:
  - A or I8HEAD is missing or not ok, or an arm is off its name;
  - I8HEAD's round trip is not exactly the lm_head (one module, no attention module);
  - an arm's recorded windows (counts and the 1024 stride) differ from the ruled ones;
  - an arm's held-out window hashes are not the registered ones;
  - a ruled gate's windows are off 196 / 123, or are not the same windows in both arms.

  A VOID is not a ruling (the runner exits 27).

**Reported, never ruling:** c4, P125's own gate outcome at each gate, the round trip's error, and the continuity row.

**Mutants** the reducer's self-test must kill (`tests/test_p150.py`):
- the point d instead of the upper bound;
- either gate instead of both;
- t1 alone;
- the argmax gate unread;
- the scope unchecked;
- the window counts unchecked;
- the recorded windows unchecked;
- the held-out hashes unchecked.

## Predictions (graded, never ruling)

**GO.** Each gate's upper bound lands below half its bound, and argmax is at least 0.985 at both. The continuity read
had upper bounds near a fifth of the bound; new windows add spread, so the prediction keeps a margin.

## What follows

- **A GO** licenses the int8 head on quality only. Its speed is a served read of its own, in P143's form:
  - a grouped-nf4-gemm int8 GEMV at M == 1 (every other M stays on the bf16 head);
  - its own tolerance contract;
  - an ABBA served read with a NOISY rung and a power term.

  Nothing flips before that read.
- **A PARK** ends D2. The int8 head does not reach the served path, and P148's D2 premise is closed with the numbers.

## The box

- **Run ids:** `p150-5090-1`, with `-2` and `-3` as pre-create spares.
- **Host terms:** the split premise's: one RTX 5090, at least 150 GB disk and 60 GiB host RAM, ≤ $0.85/h and
  ≤ $0.005/GB.
- **Guard:** 1.25 h, from the split box's measured stages: setup about 11 minutes (install, premise, an 8-minute fetch,
  the bake) and each arm about 15 minutes (t1 780-797 s, k16 63 s, c4 8 s, load 24 s). Two arms come to about 41
  minutes, × 1.3 is about 53, plus setup is about 64. The held-out split adds one small dataset read.
- **Worst case:** 1.25 h × $0.85/h + 70 GB × $0.005/GB = **$1.41**. That is within the standing rule. No power term:
  a quality read times nothing, though forensics records the card's limit.
- **Runner** (`p150_run.sh`): the split premise's runner by counted edits. The arms are A then I8HEAD, every box call
  goes through `p150_box.py`, and the reducer is this lane's. A deadline before I8HEAD is 40, never a skip.
- **Driver** (`p150_drive.sh`): P143 A3's driver by counted edits, on `bench/common/lane_liveness.sh` (launch PID and
  `/proc` identity), with the split premise's staging plus `p150_box.py`, and the hf token as A3 staged it.
- **Approval:** the maintainer's routing of D2 (P150, renumbered from P149) after P148's re-derivation. One review by a
  seat that does not consume this work, then the maintainer's final pass.
