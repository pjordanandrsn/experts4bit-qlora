# P68 — which part of the attention makes a verify or a prefill differ from T = 1 decode, and is it over the bar once enough positions are read? (registered 2026-09-24, before the run)

Record: [#725](https://github.com/pjordanandrsn/experts4bit-qlora/issues/725), filed under lane P63's decision rule.
Predecessor: P63 ([`../p63/RESULTS-p63.md`](../p63/RESULTS-p63.md), #708, merged in #727). Authorization: the work,
and a later rental within the standing caps, is the maintainer's standing assignment from the owner ("continue as
maintainer", 2026-09-23, relayed for P63–P66). The rental needs its own relayed owner comment on #725 before it
launches; that comment is not part of this change.

## Question

P63 read, on one RTX 5090 with Qwen3-30B-A3B, that no position of a 16- or 17-row verify, or of a 160-row prefill, is
bit-equal to incremental T = 1 decode. This held on every stack and route. Every first difference was at layer 0 in
attention: `attn_core` (the o_proj input), or `attn_in` for the int4 stack's prefill, where the norm fold leaves its
kernel above 64 rows. It was never in the experts, whose routes P63 registered as row-exact. KL mean was 0.008–0.033
nats/token, and 16 of 45 cells were under the shipped top-1 bar of 0.93. #725 asks:

1. **Which attention component makes the difference?** The candidates P63 named are:
   - the projections, whose kernel is picked by row count (int4: a GEMV at one row, K16 at 2–16, cuBLAS above; bf16:
     cuBLAS by M);
   - the attention core (one query at decode; a masked multi-query SDPA call at verify and prefill);
   - torch's bf16 reduced-precision cuBLAS reduction;
   - on the paged backend, the fp8 decode kernel's own verify path.
2. **Is the T > 1 path within the shipped bar relative to T = 1?** All 16 of P63's OVER-BAR cells were over on
   top-1, measured over 64–160 positions, where one argmax flip moves top-1 by 1.6 %. None was over on KL: the
   largest mean was 0.033 against 0.10. The bar needs enough positions to be read at all.

This is not a speed lane and it moves no default.

## Instrument

[`p68_probe.py`](p68_probe.py) reuses P63's instrument unchanged:
- `p63_probe`'s stack build (P44's `serve_stack`, the int4 lanes as served, fused q/k/v on Qwen3), its site capture
  at six sites per layer, and its HF and paged backends;
- `p63_compare`'s arithmetic (bit equality, KL in fp64 over the full vocabulary, the first differing layer and site).

It adds two things.

**1. Forcing arms (the ablation).** A forcing makes one component compute a multi-row call the way T = 1 decode
computes it: one row at a time, so every row takes the T = 1 kernel at the T = 1 shapes.
- **`proj`.** Each decoder layer's attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, or the fused
  `qkv_proj`) are wrapped so an n-row call becomes n one-row calls.
- **`core`.** Transformers' `sdpa` entry is replaced for the arm. Every attention forward looks it up at call time,
  the int4 folds included (`engines/glue_r2.py` reads `ALL_ATTENTION_FUNCTIONS` per call). Row i of a T-query call
  over S keys becomes a one-query call over the first S − T + i + 1 keys, contiguous and with no mask. That is
  exactly the prefix and the call its T = 1 decode made.
- **`router`, `lm_head`, `norms`.** The router module (`mlp.gate`), the LM head and every module whose class name ends
  in `RMSNorm`, one row per call.
- **`fp32red`.** `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` for the arm.

How a forcing is made safe and visible:
- **Inert at one row.** A forcing passes a one-row call straight through, so each arm's T = 1 control is the unforced
  control. Gate G0 checks this bit for bit.
- **Counted.** Every wrapped call is counted (`counts`), and the `core` forcing also counts decode calls that carried
  a mask. A forcing that did not engage, or a `core` forcing whose decode calls were masked, fails G0.
- **Tested on CPU.** `tests/test_p68_force.py` pins the forcings: rows returned in order and bit-equal to one-row
  calls, inert at one row, tuple outputs (the router), restoration of both class and instance forwards (the folds
  patch instances), and the core forcing's prefix and no-mask call against a reference attention.

**The arms** (`p68_probe.ARMS`):

| arm | backend | forced | experts route | fixture |
|---|---|---|---|---|
| `hf.base` | HF (SDPA, DynamicCache) | — | singleton (row-exact) | P63's, L = 160 |
| `hf.proj` | HF | proj | singleton | P63's |
| `hf.core` | HF | core | singleton | P63's |
| `hf.proj_core` | HF | proj + core | singleton | P63's |
| `hf.all` | HF | proj + core + router + lm_head + norms | singleton | P63's |
| `hf.fp32red` | HF | (fp32 cuBLAS reduction) | singleton | P63's |
| `paged.base` | paged fp8 (the serving path) | — | singleton | P63's |
| `paged.proj` | paged fp8 | proj | singleton | P63's |
| `size.hf` | HF | — | **default** (as served) | long rows |
| `size.paged` | paged fp8 | — | **default** | long rows |

- **Why the ablation arms use the singleton experts route:** P63 registered it row-exact. With it, any difference in
  an ablation arm is not the experts'. The size arms run the served default.
- **P63's modes, unchanged.** For the ablation arms: the control (incremental T = 1 decode of the 160-token
  fixture), verify widths 17 and 16 at p0 ∈ {128, 96, 64, 32}, and one 160-row prefill.
- **Reading per arm and mode:** exact positions, the first differing (layer, site) per position, argmax flips and KL.

**2. The size reading.** The served configuration, unforced, on long rows.
- **Rows:** the first 4 of P64's committed wikitext rows, `bench/p64/receipts/prompts_wikitext.json` (P59's window,
  digest `f67e7e4d…`), 512 tokens each. Nothing is fetched for it.
- **Per row:** incremental T = 1 decode of all 512 tokens. Then, windows descending, a verify of width 17 and of
  width 16 at every window p0 ∈ {64, 96, …} that fits (14 per width), and one 512-row prefill.
- **Logits only.** Per position: KL in fp64 over the full vocabulary, argmax agreement, and the control's top-1
  margin.
- **Positions per stack and backend:** 952 (verify17), 896 (verify16) and 2,048 (prefill).
- **Confidence.** [`p68_reduce.py`](p68_reduce.py) bootstraps the KL mean and top-1 agreement over (row, window)
  clusters (B = 2,000, seed 68), because positions inside one window are not independent.

**Stacks.** `int4`, the served lane: RTN int4 experts, int4 attention with K16 auto, the folds, fused q/k/v. `nf4`,
its bf16-attention control. Each stack runs in one process.

## Family and box

- **Family:** Qwen/Qwen3-30B-A3B @ `ad44e777…`, on one RTX 5090. That is P63's configuration, so the ablation starts
  from P63's reading.
- **Why not a bf16 reference:** it does not fit (61 GB). The size reading therefore compares T > 1 with T = 1 on the
  same quantised stack, as P63 did.

## Predictions (written before the lane's data; code-derived)

**G0 — validity, per stack.** All of the following, else the stack is read for nothing:
- the unforced control repeats bit for bit;
- every forced arm's control equals its backend's unforced control;
- every registered forcing engaged;
- no T = 1 decode attention call carried a mask.

**Q1 — `hf.base` reproduces P63** (`hf.singleton` there):
- no position exact;
- every first difference at layer 0: `attn_core` on nf4 (verify and prefill) and on int4 verify, `attn_in` on int4
  prefill.

**Q2 — projections alone (`hf.proj`, verify):** not exact at any position, and more than half of the positions'
first differences are at layer 0 in `attn_core` or `attn_out`. With q/k/v forced, the SDPA call still differs: one
masked multi-query call against one-query calls.

**Q3 — core alone (`hf.core`, verify):** as Q2. With the core forced, q/k/v and o still come from multi-row projection
calls.
- **Why a majority, not every position.** A row of a multi-row cuBLAS call can happen to match its M = 1 call bit for
  bit. When q/k/v match, the core's input agrees and the first difference shows at the o_proj output (`attn_out`).
  When o matches too, layer-0 attention is exact for that row and the difference appears later.
- **What the rehearsal read.** On sm_86, nf4 verify17 under this arm had 55 of 68 positions at `attn_core`, 12 at
  `attn_out` and 1 at `L0.mlp_out`.
- **What changed from the first draft.** It predicted `attn_core` for every position. The claim tested is unchanged:
  forcing one component alone does not make layer-0 attention row-invariant.

**Q4 — projections and core (`hf.proj_core`):**
- **Verify, both stacks:** layer-0 attention becomes exact. No position's first difference is at layer 0 in
  `attn_in`, `attn_core` or `attn_out`. It moves on: to the router's multi-row `F.linear` in layer 0's MLP, or later.
- **Prefill:**
  - int4: the first difference stays at layer 0 `attn_in`. The input-norm fold falls through to the torch chain above
    64 rows, and norms are not forced in this arm.
  - nf4: layer-0 attention exact, as for verify. P63's nf4 prefill already had `attn_in` exact, so the torch RMSNorm
    was row-invariant there.

**Q5 — everything forced (`hf.all`):**
- **verify17 and verify16: EXACT on both stacks.** Every site of every layer, the final norm and the logits are
  bit-equal to T = 1 decode. With the experts on their row-exact route and every other row-count-dependent component
  forced to its T = 1 call, nothing left depends on the row count below 64 rows, where every fold still takes its
  kernel. The rest is elementwise, or reduces only within a row:
  - rotary embedding's K = 1 product (one multiply);
  - residual adds;
  - int4's `rope_norm_heads`, one program per (row, head);
  - int4's `rmsnorm_resid_rows`.
- **nf4 prefill: EXACT,** for the same reason (no folds).
- **int4 prefill: NOT exact, first difference at layer 0 `attn_core`.** Above 64 rows the attention fold falls through
  to the fused-q/k/v forward. That forward norms q/k with the (forced, per-row) RMSNorm modules and rotates in torch,
  where T = 1 decode used the fused `rope_norm_heads` kernel: a different arithmetic, not a row-count effect a forcing
  can remove.

**Q6 — `hf.fp32red`:** no prediction. It reports whether turning off torch's bf16 reduced-precision cuBLAS reduction
alone changes any bit, and by how much KL, against `hf.base`.

**Q7 — `paged.proj`:**
- **Verify: no prediction.** The paged verify path appends the window's T rows of K/V in one call, then calls the fp8
  decode kernel with the T query rows as a batch (`slots=[slot] * T`, staggered lengths). Decode appends one row and
  calls it with a batch of one. The kernel's split plan and the append's quantisation can therefore differ by row
  count. The sm_86 rehearsal read it not exact with the projections forced, and the 5090 plans by its own SM count.
  Reported: whether layer-0 attention is exact.
- **Prefill: not exact.** The first difference is at layer 0 `attn_core` on nf4: the paged prefill reads staged bf16
  K/V, where decode reads fp8, a different function by design. On int4 it is at `attn_in`: the input-norm fold falls
  through above 64 rows before the attention runs, as in Q1. The first draft said `attn_core` for both; the rehearsal
  showed the int4 fold switch applies on this backend too, and the code says so.

**S — the size reading.** On every stack, backend and mode, the KL mean's 95 % CI upper bound is below 0.05 and top-1
agreement is ≥ 0.93. Basis: P63's KL means (≤ 0.033) and top-1 (0.88–0.98 over 64–160 positions). It is refuted on
any cell that misses either.

## Decision rule

- **¬G0** → the stack is an instrument fault. Nothing from it is read, and the fault is filed. No second box on a
  disappointing result.
- **Q5 holds on verify** → register a measured claim. On this stack a verify assembled from T = 1 calls (projections
  and attention core per row, router, LM head and norms per row, experts on the singleton route) is bit-identical to
  decode, and Q2–Q4 name which forcings are necessary. Recorded in `docs/STATUS.md`. Whether an exact verify is worth
  its cost is a separate speed lane; P68 makes no speed claim.
- **Q5 refuted** → the component still differing is named from the first-difference sites and filed before anything
  is claimed.
- **Attribution (Q2–Q4, Q7):** the arms' first differences and KL are reported as measured. A refuted prediction is
  reported as refuted, with the mechanism it disproves; none is re-derived after the data.
- **S, per cell**, against the shipped bar (KL ≤ 0.10, top-1 ≥ 0.93), from the bootstrap intervals:
  - **WITHIN-BAR** (KL upper ≤ 0.10 and top-1 lower ≥ 0.93) on every cell → the served T > 1 path is inside the bar
    relative to T = 1. P63's OVER-BAR cells were small-sample top-1. #725 closes on the ablation's registration.
  - **OVER-BAR** on any cell (KL lower > 0.10, or top-1 upper < 0.93) → that path's quality needs its own lane. A
    verify or prefill quality read against the NF4 anchor is filed with the cell's numbers. No default moves here.
  - **UNRESOLVED** → stated as such, with the interval.

## What this lane does not say

- **No bf16 reference.** The size reading is T > 1 against T = 1 on one quantised stack.
- **Scope.** One family, one box, P63's fixture for the ablation, and four wikitext rows for the size reading.
- **The paged core is not forced,** only its projections (`paged.proj`).
- **Speed.** An exact verify's cost is not measured.
- **B = 16.** Sixteen sequences of one row each is a different shape from a 16-row verify.

## Box, cost, receipts, exit codes

- **Box:** one RTX 5090 on Vast verified/secure, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`. The runner refuses
  another class (rc 15) and a dud box (rc 10).
- **Egress.** Measured before the install the way the fetch runs: four parallel 50 MB ranges, in Python, because the
  fetch uses `max_workers=4` and the image ships no `curl` (P65 Amendment 2). Below 80 MB/s the reading refuses (rc 13):
  61 GB at the floor is ~13 min. `tests/test_p68_staged_pin.py` holds the probe and the fetch to one worker count.
- **Proving rental first** (the compute rule for any guard over 1 h). One RTX 5090 of the same class and image runs the
  launcher's trivial command (`nvidia-smi` over ssh) at the SAME e4b commit on a clean tree. It proves attach,
  pre-flight, command handoff, receipt, ledger row, teardown proof and zero live instances afterwards.
  - **Guard 0.23 h at ≤ $0.65/h, so ≤ $0.15.** Launcher boot takes 3–5 minutes of a guard (P65 Amendment 1).
  - It has no reading-only floors to record: it runs no lane code.
  - The reading launches only after it returns OK. A failed proof is a row and a defect, not a retry.
  - Nothing merges to e4b `main` between the proof and the reading.
- **The reading:** a 2 h guard at ≤ $0.66/h, so ≤ $1.32. Expected ~75 minutes:
  - install ~8, egress 1, fetch ~10, bake ~5;
  - `int4` ~30: the pack ~2, eight ablation arms ~10, two size arms ~18;
  - `nf4` ~20;
  - reduce 1.
  - The runner starts a stack only if it can finish 10 minutes before teardown. `int4`, the served stack, runs first.
- **Lane ceiling:** $1.50 (proof ≤ $0.15 plus reading ≤ $1.32); hard stop $2.
- **Runner:** [`p68_drive.sh`](p68_drive.sh) (controller) and [`p68_run.sh`](p68_run.sh) (box), P63's pattern.
  - e4b is installed at the driver's own HEAD, refused if the tree is dirty.
  - grouped-nf4-gemm is installed at `GNF4_SHA`, from the launch manifest: `68a1250` (grouped-nf4-gemm#399's merge) or
    later.
  - The staged pieces, including P63's probe, comparison module and fixture and P64's rows, are pinned by
    [`staged.sha256`](staged.sha256), checked in CI by `tests/test_p68_staged_pin.py`.
- **Receipts:** `out/<stack>/p68_arm.json`, the reducer's `RESULTS-p68-generated.md` and `p68_rep.json`, `summary.txt`,
  `versions.txt`, `forensics.txt`, `logs/` and the teardown proof, into `bench/p68/receipts/`.
- **Exit codes:**
  - 78 refusal (configuration);
  - 9 stage, install or tripwire;
  - 10 dud box;
  - 11 fetch;
  - 12 bake;
  - 13 host-limited: egress below the floor;
  - 14 an instrument fault (G0);
  - 15 wrong class;
  - 3 a stack errored;
  - 30 no time for a stack;
  - 42 a stack wrote no receipt.

  The verdict is read from the reducer's JSON, never from the exit code.

## The rehearsal (NAS RTX A2000, free; NOT the lane's reading)

Receipts and the full account are in [`rehearsal-a2000/`](rehearsal-a2000/README.md).
- **Setup.** OLMoE-1B-7B-0924 on the NAS RTX A2000 (sm_86), both stacks, every arm; this branch's tree; grouped-nf4-gemm
  `68a1250`. The size rows are rehearsal-only (OLMoE-tokenized repository prose). OLMoE has no round-2 attention fold,
  and Qwen3 has 48.
- **Instrument.** G0 held on both stacks: every forcing engaged, no decode attention call carried a mask, and every
  control matched.
- **Ablation.**
  - Forcing the projections or the core alone left layer-0 attention differing.
  - Forcing both moved the first difference out of layer-0 attention.
  - **Forcing everything made every verify position, and the prefill, bit-exact on both stacks.**
- **Size.** Every cell WITHIN-BAR: KL ≤ 5e-3, top-1 ≥ 0.950.

**What it changed, before this registration merged:**
1. **Q2/Q3 became a majority.** A multi-row cuBLAS row can match its M = 1 call bit for bit. So "every first
   difference at layer-0 `attn_core`" became "no position exact, and most at layer-0 `attn_core` or `attn_out`".
   (nf4 `hf.core` verify17 read 55 / 12 / 1.)
2. **Q7 was revised.** The paged verify cell carries no prediction: its fp8 kernel runs T rows as a batch, and the
   rehearsal read it not exact. The paged int4 prefill's first difference is at `attn_in`, the fold switch, as the
   code says.
3. **One prediction was left as registered against the rehearsal.** The int4 prefill under `hf.all` was exact on
   OLMoE, and Q5 predicts not exact on Qwen3. The mechanism, Qwen3's attention fold above 64 rows, does not exist on
   OLMoE. If the 5090 reads exact, Q5's int4 prefill cell is refuted and reported as such.
4. **Fixed:** the probe's default fixture path in the repository layout.

Under the registered tables, the rehearsal reads 27 predictions held and 1 refuted, that same int4 prefill cell
(`rehearsal-a2000/read-registered.md`).

Amendments, dated, go below this line before any data is read.
