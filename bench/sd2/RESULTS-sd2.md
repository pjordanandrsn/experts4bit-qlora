# SD2 results: an EAGLE-3 speculative decode path in `serve_paged`, measured end to end at B = 1 (#1313)

## The verdict (`sd2-5090-3`): **VERIFY_COST_REFUTES**

**The consequence, as registered.** The verify step's measured cost and stage E's direct check close EAGLE-3 at B = 1 for
`Qwen/Qwen3-30B-A3B` on this card and stack. The build stays off: `E4B_PAGED_SPEC` remains an opt-in, off by default,
with no speed claim. The SD2 lane closes.

**The re-derivation.** The maintainer re-derived the verdict from the receipt store (`a3a6e304`) with main's
`sd2_reduce.py`:
- main's copy is byte-identical to the staged copy (`9eff2720…`);
- the self-test passes 89 proof cases and 43 read cases;
- the run used the box's expect flags and rule a4, with torch present so the depth report was computed too;
- all 15,613 fields match the box's `verdict_read.json`, the depth report included.

**The registration:**
- SD2 itself: #1549, merged as `88c829b5`.
- Amendment 1 (#1561): the CUDA proof, its target and its harness.
- Amendment 1b (#1564): the proof's time budget.
- Amendment 2 (#1567): V0's addressing gate becomes a logit gate.
- Amendment 3 (#1585, with its code in #1576): the read, its target, its harness and its budget.
- Amendment 4 (#1589): a key-row position gate in V0.
- Amendment 5 (#1594): the key-row gate at layer 0, rule a4.
- Amendment 5b (#1601, merged as `af924eab`): the canary for E and Q, and fail-soft reporting computed offline.
- The fix #1599: the box's rotation quantiles on any device.

**Under test:**
- **The target:** e4b `88ae1cfd` (0.53.0, main at build 5's merge, #1559) and grouped-nf4-gemm `724ccc45` (0.45.0).
  The package-diff audit against the proven `539a2d26` found 7 changed files, none unlisted and none stale.
- **The server:** `serve_paged` as shipped (16 slots, CUDA graphs at every bucket) plus `E4B_PAGED_SPEC=eagle3`. The
  spec build has k up to 3, verify buckets 2 / 3 / 4 in graphs, and auxiliary layers 2 / 24 / 45.
- **The head:** `RedHatAI/Qwen3-30B-A3B-speculator.eagle3` at `6afc5aa2`, `model.safetensors` sha256 `d2d6e2e6…`,
  checked on the box.
- **The model:** `Qwen/Qwen3-30B-A3B` at `ad44e77`, NF4 arena baked on the box.
- **The card:** one RTX 5090 (vast machine 55913), on harness `af924eab`. The target's GPU tests passed 123 / 0 / 0.

### Stage V

**V0, the correctness gate, passed under rule a4** (`v0_fails = []`). It ran on rows outside calibration:
- **The capture check:** bitwise equal at verify widths 2, 3 and 4.
- **The real build's addressing:** 18 records (rows R1–R3 and C1–C3, k = 1, 2, 3):
  - the largest mean |Δ log p| is 0.083 nats, under the 0.25 bound;
  - the largest key-row pair-0 rotation at layer 0 is 0.0033 rad, under the derived 0.142 bound for the real build.
- **The six mutants all fail the gate:**

| mutant | max mean \|Δ log p\| (nats) | layer-0 rotation (rad) | caught by |
|---|---|---|---|
| (a) stagger one low, R1 | 2.591 | 0.0 | logit gate |
| (a) stagger one low, C1 | 6.430 | 0.0 | logit gate |
| (b) alias append one low, R1 | 5.373 | 3.061 | both |
| (b) alias append one low, C1 | 14.238 | 2.773 | both |
| (c) RoPE positions +1, R1 | 0.161 | 1.038 | key-row gate |
| (c) RoPE positions +1, C1 | 0.064 | 1.042 | key-row gate |

Mutant (c) passes the logit gate here, as it did on `sd2-5090-1`. The layer-0 key-row gate catches it, above the derived
0.858 bound for the shift. Amendment 5 named `sd2-5090-2` as rule a4's out-of-sample test. That run ended before a
verdict, so this read is the first out-of-sample test of rule a4.
- **The draft and the transition checks** also passed: 574 of 576 draft tokens agree, with 0 mirror mismatches and no
  state left over.

**V1, the verify-cost reading.** Each term is the median of 64 timed steps after 8 warm-ups. The verify and the
post-verify graph are timed by CUDA events and the full step by wall clock; the loop is the remainder. The plain decode
anchor ran twice, bracketing the k terms: 4.1845 and 4.1805 ms, a ratio of 0.999, so the read is not V_NOISY.

| k | full step c(k) | verify | post-verify graph | loop |
|---|---|---|---|---|
| 1 | 8.9166 ms | 8.3016 ms | 0.5130 ms | 0.1020 ms |
| 2 | 9.7852 ms | 8.7415 ms | 0.9421 ms | 0.1016 ms |
| 3 | 10.5617 ms | 9.0799 ms | 1.3824 ms | 0.0994 ms |

S_V(k, w) = τ_SD1(k, w) × anchor / c(k):

| workload | k = 1 | k = 2 | k = 3 | best |
|---|---|---|---|---|
| R | 0.720 | **0.773** | 0.771 | 0.773 |
| C-nothink | 0.772 | 0.869 | **0.895** | 0.895 |
| C-think | 0.805 | 0.937 | **0.988** | 0.988 |

The best S_V is below 1.00 on every workload: **VERIFY_COST_REFUTES**. The measured-cost speedup came in 34%, 24% and
15% below SD1's model at k = 1, 2 and 3 (`model_error` = S_V / S_SD1 − 1). One verify of a single draft token costs
about twice a plain decode step.

**The stop rule held.** Stage E ran as the direct check, reported and not ruled. Stage Q was skipped. The E canary
(ON2) passed before E.

### Stage E: end to end (reported, not ruled)

Six arms ran in palindrome order: OFF-a, ON1-a, ON2-a, ON2-b, ON1-b, OFF-b. Decode cost in ms per token:

| workload | OFF-a | ON1-a | ON2-a | ON2-b | ON1-b | OFF-b |
|---|---|---|---|---|---|---|
| R | 4.194 | 5.507 | 4.996 | 4.991 | 5.497 | 4.190 |
| C-nothink | 4.214 | 5.372 | 4.758 | 4.769 | 5.398 | 4.205 |
| C-think | 4.190 | 5.310 | 4.628 | 4.621 | 5.303 | 4.287 |

The speed ratio g (OFF / ON; above 1 would be faster), with its two palindrome pairs:

| workload | g, k = 1 | g, k = 2 |
|---|---|---|
| R | 0.762 (0.762, 0.762) | 0.839 (0.839, 0.839) |
| C-nothink | 0.782 (0.784, 0.779) | 0.884 (0.886, 0.882) |
| C-think | 0.799 (0.789, 0.808) | 0.916 (0.905, 0.928) |

- **The timings were stable.** The a/b self-pairs fall within 0.977–1.002, and the W1 tripwire matched on both OFF
  passes.
- **Drafting works as SD1 measured it.** Live τ tracks SD1's offline τ within 1.5%:

| workload | τ live, k = 1 | τ SD1, k = 1 | τ live, k = 2 | τ SD1, k = 2 |
|---|---|---|---|---|
| R | 1.5553 | 1.5353 | 1.8325 | 1.8094 |
| C-nothink | 1.6691 | 1.6465 | 2.0639 | 2.0339 |
| C-think | 1.7247 | 1.7157 | 2.2117 | 2.1912 |

What loses is the verify step's cost on this server, not the draft.

### Reporting only (not verdict inputs)

- **Identity:** in 47 of 48 prompt rows, the speculative greedy output departs from plain decode's at some token
  (positions 3–147). The departure points are the same at k = 1 and k = 2, and in both passes.
- **Memory:** speculation adds about 1.1 GB (allocated 25.34 GB at k = 1 against 24.23 GB off).
- **Quality:** stage Q did not run, so this read carries no quality result.
- **The depth report:** the per-layer rotation quantiles are in `verdict_read.json` under `depth`, computed offline from
  `v_angles.pt` (sha256 `aa193351…`, 1.4 MB). That file stays in the receipt store, since `*.pt` is not tracked here.

### The lane's rentals

| run | outcome | cost | what followed |
|---|---|---|---|
| `sd2-prove-1` | HARNESS_ERROR (rc 40) | $0.046 | Amendment 1b |
| `sd2-prove-2` | FAILED | $0.660 | Amendment 2 |
| `sd2-prove-3` | PROVED | $0.681 | the build merges, Amendment 3 |
| `sd2-5090-1` | VOID | $0.566 | Amendment 4 |
| `sd2-prove-4` | FAILED (rule a3) | $0.784 | Amendment 5 |
| `sd2-5090-2` | HARNESS_ERROR (rc 23) | $0.713 | #1599, Amendment 5b |
| `sd2-5090-3` | VERIFY_COST_REFUTES | $0.930 | this read |

The lane spent **$4.380**. The read itself took 40 minutes against a 3.75 h guard and a $3.87 ceiling.

### The records

`receipts/sd2-5090-3/` holds the run's records under `SHA256SUMS`, byte-identical to the receipt store's copy at
`a3a6e304`:
- the six E arms;
- V's record `v.json`;
- the E canary;
- the audit and the GPU tests;
- the box's prompts;
- `staged.sha256`, `summary.txt`, `forensics.txt` and `versions.txt`;
- the box's `verdict_read.json`.

The staged inputs (the scripts, `calib.json`, `chat_prompts.json`, `expect_w1.json`) are the files `staged.sha256` names
in `bench/sd2/`.
