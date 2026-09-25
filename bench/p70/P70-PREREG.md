# P70 — what rounding the fused router's decode weights to the model's dtype costs: a decode-scored KL A/B (registered 2026-09-25, before the run)

Record: [#726](https://github.com/pjordanandrsn/experts4bit-qlora/issues/726), filed under lane P63's decision rule
("P7 held"). Owner authorization for the rental: "go ahead with the P67 rental (all rentals approved)" (Jordan,
2026-09-24, chat), relayed on #726 before any launch. The rental is not part of this change.

## Question

With `E4B_FUSE_ROUTER_EPI=1` (the int4 serve lane's configuration), a patched router runs grouped-nf4-gemm's fused
`router_epilogue` at **≤ 64 rows** — every decode step and every speculative verify — and the model's own router
**above 64 rows**, i.e. in a prefill (`experts4bit_qlora/engines/router_epilogue.py`, `_MAX_DECODE_ROWS`). The two
return different functions of the same logits (P63's P7; confirmed from the pinned transformers 5.16.1 / 5.17.0
source, `Qwen3MoeTopKRouter.forward`):

- the fused path returns the kernel's **fp32** routing weights (fp32 softmax over all experts, top-k, renormalise);
- the original router computes the same fp32 softmax, top-k and renormalisation, then
  **`router_top_value.to(router_logits.dtype)`** — **bf16** on a bf16 model.

So at the same token, a decode or verify step weights each expert's output by fp32 weights, and a prefill by
bf16-rounded ones. The experts selected are identical either way.

**On the served Qwen3-30B-A3B int4 stack at B = 1, what does making the fused path return the upstream function —
the same weights, rounded to bf16 — change at decode, in nats, against this instrument's own arithmetic-order floor?**

**A finding about P64's floor, made while designing this, and disclosed.** P64 scored with a 128-token prefill chunk
and took one floor sample at a **64-token** chunk (`a8_pc64`). A 64-row prefill forward is at the threshold, so it
runs the **fused** router (fp32 weights); the 128-row primary's prefill runs the **original** router (bf16). P64's
`a8 ‖ a8_pc64` floor sample therefore contains this very switch, in the 384 prefill positions whose KV cache every
decode step reads — not only an arithmetic reorder. P70's floor keeps the router function fixed (both samples prefill
in > 64-row forwards), and measures `pc64` beside it as an informational pass. P64's verdict is not re-derived here.

## Instrument

**The lever:** `E4B_ROUTER_EPI_CAST=1`, a new environment flag, **default off**, exposed to a harness as
`router_epilogue.CAST_WEIGHTS` (a one-element list read on every fused call, `hot_residency.DECODE_A16`'s pattern).
On, the fused `softmax_topk` and `topk_softmax` branches return `w.to(logits.dtype)`. On the `softmax_topk` kind
(Qwen3-MoE, OLMoE, Mixtral) that is the upstream function **to the bit**
(`tests/test_router_epilogue.py::test_cast_on_is_the_upstream_router_to_the_bit`; and at every row count,
`::test_cast_on_makes_every_row_count_one_function`). On `topk_softmax` (gpt-oss, GraniteMoe) the dtype matches but
not every bit: upstream takes the k-softmax in the logits' dtype. Gemma-4's branch already casts. Nothing here moves
a default.

**The scorer:** `bench/p70/kl_router.py` imports P64's `bench/p64/kl_a16.py` **unchanged** (staged beside it; its pins
equal P64's, `tests/test_p70_staged_pin.py`) and replaces four module-level names before P64's run loop starts: the
pass table, the toggle (P64's toggles held at the shipped default in every pass, its refusals kept), the expected
counts, and the counter installer. So the served stack, its build, the one-row scoring through P59's scorer (B = 1,
every decode forward T = 1), the logit storage and the census are P64's code byte for byte.

**The stack** is P64's: Qwen3-30B-A3B @ `ad44e777`, P55x's licensed calibrated int4 expert pack **rebuilt on the box
by P55x's recipe and checked by fingerprint** (`sha256:0c9955a9…`; `verify_artifact`, never the build's exit code),
calibrated int4 attention, folds r1/r2, the router epilogue, grouped-nf4-gemm at **v0.33.0** (`5ca1897`, P64's cut).
The rows are P64's: 16 rows × 512 tokens per text from `step_decomp._k8_window` (wikitext-2 test — P59's rows,
`f67e7e4d…` — and C4 validation shard 1), prefix 384, **2,048 decode positions per text**.

**The engagement census** (new here, on top of P64's decode counts): a forward hook on every patched router counts
each call by row class (≤ 64 / > 64) and by the dtype of the weights it returned, per phase. Every pass must show
exactly the registered counts (`kl_router.expected_counts`, restated in the reducer and held equal by
`tests/test_p70_reduce.py`), zeros included. That is the proof of what ran: under `ep32` every decode router call
returns fp32; under `ep16`, bf16; a 128-, 96- or 384-row prefill returns bf16 through the original router; a 64-row
prefill returns fp32 through the fused one. The runner also requires one hook per MoE layer.

## Passes

| pass | cast | prefill chunk | prefill router | role |
|---|---|---|---|---|
| `ep32` | off | 128 | original (bf16) | the served stack as shipped — the reference |
| `ep16` | **on** | 128 | original (bf16) | **the lever** |
| `ep32_rep` | off | 128 | original (bf16) | determinism control (must be bit-identical to `ep32`) |
| `ep32_pc96` | off | 96 | original (bf16) | floor sample 1 |
| `ep32_pc384` | off | 384 | original (bf16) | floor sample 2 |
| `ep32_pc64` | off | 64 | **fused (fp32)** | P64's floor sample — informational only |

One process, one build; the order above is the order the run spends its clock (primary pair on both texts first).
No NF4 anchor: this lane reads one stack against itself.

## Predictions (written before the data)

Every KL below is nats/token over 2,048 decode positions per text, kl_fidelity's (fp64, full vocabulary).
`F(t) = mean(KL(ep32 ‖ ep32_pc96), KL(ep32 ‖ ep32_pc384))`.

- **P1 — determinism.** `KL(ep32 ‖ ep32_rep)` is exactly 0 on every decode position of both texts.
- **P2 — the cast touches no > 64-row forward.** The prefill-last logits of `ep16` and `ep32_rep` are bit-identical
  to `ep32`'s on both texts.
- **P3 — engagement.** Every pass's counts equal the registered ones, and every router is hooked.
- **P4 — the primary.** `G(t) = KL(ep32 ‖ ep16) ≤ F(t)` on both texts (BELOW FLOOR). Basis: the cast rounds 8 weights
  per token per MoE layer to bf16 — a relative change ≤ 2⁻⁹ on each — where P64's larger perturbation (int8
  activations into every expert GEMV) read 0.61 and 0.77 of its floor.
- **P5 — NLL.** `|dNLL(t)| ≤ F_NLL(t)` on both texts, `dNLL = NLL(ep16) − NLL(ep32)`.
- **P6 — P64's floor sample (informational; moves no decision here).** `KL(ep32 ‖ ep32_pc64) / F(t)` lies in
  [0.5, 2] on both texts: the router switch in the prefill does not dominate a chunk-size floor sample.

## Decision rule (nats, against the measured floor — P64's rule)

Classes per text, exactly as `p64_reduce`: BELOW FLOOR (G ≤ F), NOT DISTINGUISHABLE (F < G ≤ 2F), DISTINGUISHABLE
(G > 2F); MATERIAL = distinguishable **and** dNLL > F_NLL **and** the paired row bootstrap's 95 % interval of dNLL
excludes 0. Lane verdict: MATERIAL if material on either text; else DISTINGUISHABLE, NOT MATERIAL if distinguishable on
either; else INDISTINGUISHABLE. A floor that reads exactly 0 is replaced by the on-record 0.0044 (P64's rule).

- **Validity not met** (P1, P2, P3 or the same rows): nothing is read. The failing gate is filed, and the run is
  repeated once under a dated amendment.
- **INDISTINGUISHABLE, or DISTINGUISHABLE, NOT MATERIAL:** the cast becomes the default **for the `softmax_topk`
  kind** — the kind this read measured and the one whose cast is the upstream function to the bit — in the next e4b
  release, with `E4B_ROUTER_EPI_CAST=0` restoring fp32 for one release. The `topk_softmax` kind keeps fp32 until it is
  read. A claim registers G, F, the class and dNLL per text; #726 closes. Reason: the fused path then computes the
  upstream function at every row count, which is what #726 asks for, at no measured cost.
- **MATERIAL** (the cast makes the model measurably worse): the default stays fp32 at ≤ 64 rows, and the
  discontinuity is closed the other way — a follow-up issue to lift the > 64-row path to fp32 weights, with its own
  read. A claim registers the measured cost; #726 stays open on the follow-up.
- **P6** outside [0.5, 2] on either text: a correction issue on P64's floor definition is filed with the numbers.
  P64's verdict is not re-derived by this lane (its `G` was measured on another box).

## What this lane cannot say

- Anything about **verify** (T = 16/17) or B = 16: every scored forward here is B = 1. The flip would also change
  what a verify computes; P68's machinery reads verify.
- Anything about the **`topk_softmax`** kind (gpt-oss, GraniteMoe) or Gemma-4.
- Anything about **speed**: a bf16 cast of 8 values per row is not timed.
- Anything about the model's quality against a bf16 reference: the read is the served stack against itself.

## Box and cost

**Two rentals, in order** (the compute rule: a guard over 1 h needs a proving rental first).

1. **`p70-prove-<n>`, the proving rental:** one RTX 5090, **0.2 h guard at ≤ $0.75/h (≤ $0.15)**. `p70_drive.sh` with
   `P70_PROVE=1`: the same staging, nonce handshake, class and disk refusals, pinned install, tripwire (which now also
   requires `router_epilogue.CAST_WEIGHTS`, off at import), the scorer's CPU self-test (P64's own self-test plus the
   router census for fused and original prefills, cast off and on) and K0 as the reading; then
   `kl_router.py --prove-cast` on the box's real `int4_b32.router_epilogue` at Qwen3's router shape (hidden 2048, 128
   experts, top-8, bf16): off → fp32 at ≤ 64 rows with upstream's experts; on → bf16 at ≤ 64 rows; above 64 rows the
   original forward in both states; the fraction of weights bit-equal to upstream's with the cast on is reported. It
   exits 0 with `PROVED`; no model, no pack, no KL. At most three attempts (≤ $0.45).
2. **`p70-5090-<n>`, the reading:** launched only after a proof returns rc 0 with its receipts fetched. One RTX 5090,
   Vast verified/secure, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`. **Guard 2.5 h at ≤ $0.75/h (≤ $1.875).**
   Estimate from P64's reading on the same stack: fetch + bake ~15 min, pack build 30–67 min (P64's box was
   host-limited at 67), the six passes on two texts ~50 min (every pass is P64's `a8` arithmetic, ~0.10 s per decode
   step; P64's twelve passes including the heavier `a16` took 57 min), reduce < 1 min: **~1.6–2.2 h, ≈ $1.2–1.65.**

**Lane ceiling $2.50; hard stop $3**; both under the $35 per-run cap.

## Rehearsal (NAS A2000, 2026-09-24) — NOT a reading

Run on the home NAS's RTX A2000 12GB (sm_86) before any rental, to prove the lane end to end. It is not a reading and
moves no decision: the knobs are off their registered defaults (`rows=4` → 512 decode positions per text, not 2,048;
`calib_nseq=8`, `hessian_budget_gb=2`, `build_ppl_steps=64`), the model is OLMoE-1B-7B-0924 @ `6d84c485`, and the pack
built on this box (`sha256:0fc7bc44…`) is **not** the licensed one. The runner marked it `REHEARSAL` itself. Tree:
this branch on e4b `abb198d` (0.37.4); gnf4 0.33.0 @ `5ca1897`; torch 2.8.0+cu128, transformers 5.16.1, bitsandbytes
0.50.1. Receipts: [`rehearsal-a2000/`](rehearsal-a2000/) (logs renamed `.txt`; the logits and pack payloads stay on the
NAS).

What it proved about the instrument:

- `p70_run.sh` ran to rc 0 under the NAS lock. Both self-tests passed (P64's scorer self-test, and the router census
  for fused and original prefills, cast off and on). K0 passed.
- The build dumped a verified pack and then returned rc 1 on the K8 cross-check. That is the expected sm_86 outcome.
  The runner took the dumped pack and labelled the read "not the licensed pack", as registered.
- All twelve pass-and-text runs engaged as registered. `ep32` decoded through fp32 router weights and `ep16` through
  bf16 at 8,192 router calls per text. Every registered prefill ran through the original router (192 calls); `ep32_pc64`
  ran through the fused one (384). Every MoE layer was hooked.
- Wall time was 21–43 s per pass and text at 512 decode steps.

What the rehearsal's numbers show, on a pack and row count the reading does not use:

| text | G | F | G / F | class | dNLL (95 % CI) | F_NLL | pc64 / F |
|---|---|---|---|---|---|---|---|
| wikitext | 0.001307 | 0.001819 | 0.72 | BELOW FLOOR | +0.00342 (+0.00170, +0.00578) | 0.00053 | 0.96 |
| c4val1 | 0.001772 | 0.001928 | 0.92 | BELOW FLOOR | −0.00145 (−0.00543, +0.00108) | 0.00055 | 0.95 |

- P1, P2, P3, P4 and P6 held.
- **P5 broke on wikitext.** `|dNLL|` is 6.5× `F_NLL`, and its interval excludes 0. It moves no class, because the
  MATERIAL test also needs G > 2F. P5 is left **unchanged**: rewriting a prediction after seeing a rehearsal would fit
  it to the data. The reading tests it as written, and a break there is reported as a broken prediction.
- The reducer returned **VALID, INDISTINGUISHABLE — REHEARSAL, NOT A READING**.

## Receipts and exit codes

Receipts: `receipts/experts4bit-qlora/<date>/p70-{prove,5090}-<n>/` in the adertha receipt store (the launcher's
`receipt.json`, `teardown-proof.json`, and the fetched `p70/`: every pass's census, the prompt rows, `pack.json`, the
pack's manifest, `build.json`, `summary.txt`, `versions.txt`, `forensics.txt`, `k0.json`, logs, and the reducer's
`RESULTS-p70-generated.md` + `p70_rep.json`). The logits (~0.6 GB per pass and text) stay on the box, as P64's did;
the reducer runs there.

Exit codes are P64's (`p70_run.sh` is derived from `p64_run.sh` by named substitutions only): 9 stage / install /
tripwire, 10 dud box, 13 disk, 15 card class, 16 K0, 19 prompt dump, 20 build or deadline, 21 scorer self-test, 27
the proving run's cast check, 30 host-limited build, 40 the stack or the counts did not engage as registered, 42 the
registered read cannot be made, 43 reducer, and the reducer's validity. Controller (`p70_drive.sh`, P64's): 78 pins,
20–25 stage / start / fetch / deadline / nonce / dead lane.

Amendments, dated, go below this line before any data is read.
