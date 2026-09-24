# Results — P64: what the int4 serve lane's int8 activation step costs at decode (RTX 5090, 2026-09-24)

Pre-registration: [`P64-PREREG.md`](P64-PREREG.md) (#724, merged before any rental). Record:
[#709](https://github.com/pjordanandrsn/experts4bit-qlora/issues/709). Receipts: [`receipts/`](receipts/).

The reducer ran on the box. Its output, [`receipts/RESULTS-p64-generated.md`](receipts/RESULTS-p64-generated.md) and
[`receipts/p64_rep.json`](receipts/p64_rep.json), is the record, and every number below is quoted from it. The KLs
are computed from per-pass logits (~0.6 GB per pass and text), which stayed on the box by registration. So unlike P63
the read cannot be recomputed from the committed files. What can be checked from them: every pass's census, the
prompt rows, the pack's identity and the build's K8.

## Runs

| run | what | outcome | cost | receipt (adertha-receipts) |
|---|---|---|---|---|
| `p64-prove-1` | the proving rental: install, tripwire, scorer self-test, K0, `kl_a16.py --prove-flag` on sm_120 | PASS rc 0 (`PROVEFLAG all_passed`) | $0.0201 | `0fd7f97` |
| **`p64-5090-1`** | **the reading**, same e4b commit `75c83bd` | **rc 0, VALID** | **$1.4193** | `33a7a0d` |

- **Lane total:** $1.4394 against a $2 ceiling, with teardown proven ([`receipts/teardown-proof.json`](receipts/teardown-proof.json)).
- **Box.** One RTX 5090 (driver 595.71.05, power limit 400 W; nothing here is timed) on an AMD EPYC 7C13 host. The host was shared: 808 of 1,007 GiB of RAM were in use at start (`forensics.txt`).
- **Software.** e4b 0.37.3 at `75c83bd`; grouped-nf4-gemm 0.33.0 at `5ca1897` (the registered pin); torch 2.8.0+cu128, Triton 3.4.0, transformers 5.16.1, bitsandbytes 0.50.1.

**The pack is the licensed one.** `pack.json` records fingerprint `sha256:0c9955a9…`, equal to P55x's licensed pack. It was verified by `verify_artifact`, not by the build's exit code. The build's wikitext K8 read ppl **6.36709**, P55x's figure to five decimals. The attention pack is `sha256:1824cbeb…`: 192 calibrated `Int4Linear`, 48 int4 expert layers, and the folds on.

**Where the time went.** The pack build took 67 min against the ~30 estimated. Its first calibration chunk took 760 s, which is host-limited. The served passes ran 04:01–04:58Z. The deadline-aware scorer then skipped, as registered and in the registered order, `a16_all` on both texts, `a8_pc384` on both texts, and the NF4 anchor. Nothing was shortened.

## Validity: VALID (read first)

- **Determinism:** `KL(a8 ‖ a8_rep)` is exactly 0 on all 2,048 decode positions of both texts.
- **The flag touches no T > 1 forward:** the prefill-last logits of every pass that ran are bit-identical to `a8`'s.
- **Engagement.** Every pass's decode counts equal the registered ones. `a8`: 196,608 expert quantises, 0 dequants, 393,216 attention quantises. `a16`: 0 / 1,572,864 / 393,216.
- **Rows.** The same 16 rows × 512 tokens were used in every pass. The wikitext rows are P59's (`f67e7e4d…`); the c4val1 rows are `17e68b42…`.

## The read

| text | G_exp = KL(a16 ‖ a8) | top-1 | floor F (a8 ‖ a8_pc64) | G / F | class | dNLL (95 % CI) |
|---|---|---|---|---|---|---|
| wikitext | **0.003392** | 0.9731 | 0.005578 | **0.61** | BELOW FLOOR | −0.00107 (−0.00707, +0.00519) |
| c4val1 | **0.002527** | 0.9717 | 0.003300 | **0.77** | BELOW FLOOR | −0.00180 (−0.00610, +0.00181) |

**VERDICT (the expert int8 step at T = 1 decode, G_exp): INDISTINGUISHABLE.** On both texts, routing the experts' T = 1 calls through bf16 activations changes the output distribution by less than this instrument's own arithmetic-order floor. dNLL's intervals span 0, and |dNLL| is under the family's 0.0095 K8 floor.

**G_all and G_attn: UNREAD.** The `a16_all` passes were skipped for time, so this lane makes **no** statement about the attention projections' int8 step.

### Predictions

| prediction | registered | read | verdict |
|---|---|---|---|
| P1 determinism | `KL(a8 ‖ a8_rep)` exactly 0 | 0 on 4,096 / 4,096 positions | **HELD** |
| P2 no T > 1 forward touched | prefill-last logits bit-identical | identical | **HELD** |
| P3 floor | F ∈ [0.001, 0.010], central ~0.004 | 0.00558 / 0.00330 (one floor sample, `pc64`; `pc384` skipped) | **HELD** |
| P4 expert int8 step | G_exp ≤ 2F, central 0.002–0.004, top-1 ≥ 0.97 | 0.00339 / 0.00253, ≤ 0.77 F, top-1 0.973 / 0.972 | **HELD** |
| P5 sign | \|dNLL\| < 0.0095 and CI includes 0 | −0.00107 / −0.00180, both CIs include 0 | **HELD** |
| P6 attention | G_attn ≤ 2F | not run | **UNREAD** |
| P7 anchor | KL(nf4 ‖ a8) ∈ [0.03, 0.10] | not run | **UNREAD** |
| P8 the flag's price | a16 s/step ≥ 1.5 × a8 | 0.3025 / 0.1187 = 2.55×; 0.2901 / 0.1175 = 2.47× | **HELD** (never a speed number) |

**The floor is one sample, not two.** F was registered as the mean of two floor pairs (`pc64` and `pc384`). Only `pc64` ran, so F here is that one pair, and the reducer labels it "in-lane". The verdict does not rest on the missing sample. With F the mean of the two pairs, G_exp > 2F (DISTINGUISHABLE) would need F below 0.0017 on wikitext (0.0013 on c4val1). The `pc64` pair alone already makes F ≥ 0.0028 (0.0017), whatever the second pair reads, because a KL is never negative. So no value of the missing sample could make the step distinguishable. A second sample below 0.0012 (0.0018) would move the class from BELOW FLOOR to NOT DISTINGUISHABLE, but the verdict for both classes is INDISTINGUISHABLE.

## What follows (the decision rule)

- **INDISTINGUISHABLE → a register row** records that the expert int8 step is below Qwen3's arithmetic-order floor on both texts: `e4b.serve.p64.qwen3.b1.expert-int8-step.5090.2026-09-24`. **W4A8 expert decode stays.** Its only cost on record is time: 0.153 ms/step at B = 1, from the K17 census. No A16 expert decode route is motivated on quality grounds.
- **#709 closes for the experts.** The attention half (G_attn) was not read. It stays open, stated in STATUS and filed with this read.
- **No default moves.** `E4B_INT4_DECODE_A16` stays an eager-only instrument, default off.
- **The registration's correction, restated.**
  - P59's B = 16 KL (`e4b.serve.p59b.qwen3.b16.fqkv-kl.5090.2026-09-22`) never ran the int8 activation step. Its scorer does not set `DEVICE_GROUPING`, so its decode experts took the host-grouped dequant + bf16 branch.
  - The rehearsal counted this: 0 expert quantises and 13,484 dequants.
  - The note is now on that register row. P59's licence of fused q/k/v is unaffected, because the expert route was the same in both of its arms. But nothing on record prices the int8 step at B = 16. This lane cannot, and neither could P59.

## What this read does not say

- **Batch size.** Nothing at B = 16, and nothing about the device-grouped GEMV (`:326`).
- **The attention projections.**
- **No bf16 reference.** The comparison is two arithmetic paths over the same int4 bytes.
- **Not the paged fp8 KV path.**
- **Scope.** One family, two texts, 2,048 decode positions each.
