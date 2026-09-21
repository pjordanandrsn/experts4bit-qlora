# P54 — qkv fusion on the int4 attention store: one launch per layer where the int4 lanes paid three

Written before any P54 data exists. Owner directive (Jordan, 2026-09-21): *"main thing needed is throughput work."* Record: [#652](https://github.com/pjordanandrsn/experts4bit-qlora/issues/652). Code under test: [#651](https://github.com/pjordanandrsn/experts4bit-qlora/pull/651) (`Int4Linear.fuse`; `fuse_qkv` takes the int4 store).

## The question

Every int4 serving census we quote (P42, K16 P5, bo7) ran `--no-fuse-qkv`, because the int4 swap happens at load and `fuse_qkv` — which read `mod.q_proj.weight` — ran after it and would have refused. So the int4 stack paid **q, k, v as three attention launches per layer** where the bf16 stack pays one. [#564](https://github.com/pjordanandrsn/experts4bit-qlora/issues/564) carried this as "qkv fusion ~0.44 ms, bounded, unmeasured".

#651 fuses the int4 stores by concatenating their packed rows and scales along N — byte-identical, the same function — so the fused projection is one GEMV at B=1 and one K16 small-M GEMM (N = 4096 + 512 + 512 = 5120) at B=16. **Does that pay at the model level, and how much?**

The kernel-level arithmetic, from receipts (5090, `kernel/RESULTS-k16-smallm-int4-gemm.md`): at M=16 `q_proj` 6.35 µs, `k_proj` 4.37, `v_proj` 4.49 against a 4.61 µs launch floor. A fused N=5120 launch reads 1.25× `q_proj`'s bytes; if it costs 1.25× `q_proj`'s time (7.9 µs — an upper bound, `q_proj` is not bandwidth-bound at this M) the saving is 15.21 − 7.9 = **7.3 µs/layer × 48 = 0.35 ms/step**. At B=1 the GEMV path pays three launches of a kernel whose small-N calls are launch-bound too; the same shape of saving, ~0.3–0.4 ms of a 4.20 ms step.

## Arms

All on `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, one RTX 5090, P42's protocol (`step_decomp.py`, P39's harness pieces staged byte-for-byte, the P42 hook), int4 experts (RTN) + uncalibrated int4 attention (`E4B_SERVE_ATTN_INT4=1`, the same `Int4Linear` the calibrated lane installs — P42 amendment 1's reasoning), K16 route at its `auto` default (on). Every arm censuses (`--replay-profile-out`, 8 profiled replays AFTER the timed window, timed number untouched — step_decomp's own contract).

| arm | batch | `--fuse-qkv` | what it is |
|---|---|---|---|
| `int4_b16` | 16 | no | the control — K16 P5's `int4_b16_smallm` arm verbatim (11.19 ms on that box) |
| `int4_b16_fqkv` | 16 | **yes** | the same bytes, q/k/v fused into one `Int4Linear` |
| `int4_b1` | 1 | no | control at B=1 |
| `int4_b1_fqkv` | 1 | **yes** | fused at B=1 |
| `int4_b16_series` | 16 | no | **untimed**: `--amort on --series-out` — the per-step touched-expert series per layer, for the distinct-expert count. Its `step_ms` is NOT quoted (per-layer counters and event syncs perturb it) |

**Each timed arm runs twice, interleaved (A B A B), and the position is read from the two arms' medians of `step_ms_clean`.** A ~3 % effect is inside the range of single-draw box noise this harness has shown (A/A spreads 1.0001–1.0166 across lanes); two draws per arm with the pair interleaved is the cheapest design that separates the effect from drift. The census is taken on the first draw of each arm only (it is a kernel list, not a timing).

Order on the box: `int4_b16` → `int4_b16_fqkv` → `int4_b16` → `int4_b16_fqkv` → `int4_b1` → `int4_b1_fqkv` → `int4_b1` → `int4_b1_fqkv` → `int4_b16_series`.

## Predictions, registered before the data

- **P1 — B=16: `int4_b16_fqkv` beats `int4_b16` by ≥ 0.25 ms/step** (median of two draws each), registered band **0.25–0.50 ms**. *Refuted by:* a saving under 0.10 ms (no effect: the fused launch costs what the three did, the bytes dominate) or over 0.60 ms (something other than launch count moved — read the census before believing it). Between 0.10 and 0.25 is "smaller than predicted, real", stated as such.
- **P2 — B=1: `int4_b1_fqkv` beats `int4_b1` by ≥ 0.20 ms/step**, band **0.20–0.45 ms**. Same refutation shape.
- **P3 — the census shows the mechanism and nothing else moved.** In `int4_b16_fqkv`, `_gemm_int4_b32_smallm` runs **96 calls/step** (2 × 48: qkv + o) against **192** in the control; in `int4_b1_fqkv`, the attention share of `_gemv_int4_b32` launches falls by 96/step. Every other kernel family within ±5 % of its control row. *Refuted by:* any other family moving more than 5 %, or the call counts not being exactly those.
- **P4 — the distinct-expert count** (`int4_b16_series`): the mean over layers and steps of distinct experts touched per layer at B=16 on the harness's natural prompts is **≤ 80** (128·(1−e⁻¹) = 80.9 is the uniform-random expectation; real routing is skewed — hot experts, several rows each — so fewer). Registered as a *measurement with a soft prior*, not a claim: #564's expert-tier roofline is 5.336 ms at 64 distinct and 6.670 at 80 against a measured 6.662, so this number decides whether the largest row in the step has ~1.3 ms of headroom or none. It is reported whatever it reads.
- **P5 — parity.** The fused arms' timed windows report the same first divergent-token step as their controls (the harness's own token-agreement reporting), or none. `Int4Linear.fuse` changes no byte, so any divergence is a harness or ordering effect and is investigated before any number is quoted. *Refuted by:* a fused arm diverging where its control does not.

## Decision rule, registered in advance

- P1 ∧ P3 ∧ P5 hold → `--fuse-qkv` becomes the int4 lanes' default in the harness (the same "default once the model-level read holds" rule K16 P5 used), the saving goes into `docs/SERVING-THROUGHPUT.md` as **measured** with the receipt, and the B=16 position is re-quoted from THIS box's control-vs-fused pair (a ratio within the box), never by subtracting from another box's 11.19.
- P1 refuted with P3 holding → the launches were not the cost; recorded, `--fuse-qkv` stays optional, and the finding (launch count is not the lever at this M) goes into #564.
- P3 refuted → nothing is read; the census names what moved and that is the next question.
- P4 is read regardless and its number goes into #564 as the measured replacement for "~80".

## What this lane cannot say

Nothing about training, nothing about any other family, no head-to-head against vLLM (P37's comparator ran on a different day and box class; the position is quoted as a same-box ratio only). It does not test the calibrated attention path (RTN attention is the same kernels — P42 amendment 1); a calibrated int4 attention K8 with fusion is a separate quality read if fusion ships as a default there.

## Budget and STOP rules

- **One box, RTX 5090 (verified/secure), ≤ 2.5 h wallclock, estimate ≤ $1.70 at $0.66/h; lane ceiling $3, hard stop $5.** A proving run is not required (guard ≤ 2.5 h; K16 P5 proved this exact path on 2026-09-19).
- **STOP-1** — `int4_b16` (first draw) outside ±8 % of K16 P5's 11.19 ms: record it, quote nothing across boxes, read only within-box ratios (the ratios are the position anyway).
- **STOP-2** — an arm cannot finish 10 min before the launcher's deadline: skipped, recorded host-limited, never shortened. The series arm is last so a slow host loses the untimed measurement first.
- **STOP-3** — a timed arm's census missing on its first draw, or the fused arm's census not showing 96 K16 calls: the run exits non-zero.
- **STOP-4** — no second box on a disappointing result.

## Receipts

Fetched to `receipts/experts4bit-qlora/<date>/p54/`: `logs/census_<arm>.txt` (first draws), `e4b_b{1,16}_<arm>[_r2].json` (both draws), `series_int4_b16.json.gz`, `summary.txt`, `forensics.txt`, `versions.txt`, per-arm VRAM traces. `RESULTS-p54.md` is generated by `p54_reduce.py` from those files and nothing else.
