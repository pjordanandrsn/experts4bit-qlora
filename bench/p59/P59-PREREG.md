# P59 — the gate on the B=16 fused-q/k/v default: KL at B=16 between the unfused and fused int4 stacks (registered 2026-09-22, before the run)

Owner directive (Jordan, 2026-09-22): *"go on"* — the throughput thread continues on its own ranked list; this is its cheapest registered gate. Record: [#652](https://github.com/pjordanandrsn/experts4bit-qlora/issues/652). Predecessors: **P54** ([`bench/p54/RESULTS-p54.md`](../p54/RESULTS-p54.md)): fusing q/k/v into one `Int4Linear` saves 0.516 ms/step at B=1 (token-identical, licensed) and **0.223 ms/step (2.0 %) at B=16, where the generated tokens diverge on 14 of 16 sequences** — so `--fuse-qkv` is the default at B=1 and opt-in at B=16 "until a KL-from-checkpoint or K8 read bounds the divergence". **P57** ([`bench/p57/RESULTS-p57.md`](../p57/RESULTS-p57.md)): with the round-2 glue forced off on both legs the divergence persists at the same indices — it is the K16 small-M GEMM (`Int4Linear.SMALLM_ROWS_MAX = 16`) accumulating over a 3× wider N, a route that exists only for 2..16 rows. **P44** ([`bench/p44/`](../p44/)): the register's KL instrument for served stacks scores decode-shaped at **batch = 1** — which never takes that route. Hence this lane.

## Claim under test

That fusing q/k/v on the int4 attention store at B=16 changes the served model's output distribution by no more than the register's own noise class for an arithmetic-order change — i.e. that the tokens diverge because two valid accumulation orders differ at bf16 rounding (which flips a top-k or an argmax here and there), not because the fused path computes something worse. If so, the 2 % B=16 saving is licensed as a default; if not, the fusion has a quality cost measured in nats and stays opt-in.

## Instrument (built and dry-run before registration)

[`kl_b16.py`](kl_b16.py) — teacher-forced KL between two **served** arms with **sixteen rows decoded together**: the first 384 tokens of each of the harness's 16 wikitext-2 rows (`step_decomp._k8_window`, the same `prompts_b16.json` P54/P57/P58 decoded, digest `f67e7e4d…`) are prefilled as one `[16, 384]` forward; the remaining 128 positions are decoded **one token per forward at B=16 with a carried KV cache**, the ground-truth token fed back (never the model's argmax), so every arm's logits are aligned token-for-token. The prefill path is shared by all arms (above 16 rows `Int4Linear` dequantises to bf16 and the fusion only concatenates), and its last-position logits are kept as the **prefill control**. Logits are saved fp32 `[16, 128, V]` per arm; [`p59_reduce.py`](p59_reduce.py) computes `KL(P_ref ‖ P_test)` with `kl_fidelity`'s primitives — fp64, full vocabulary, token-weighted over 16 × 128 = 2,048 positions — and top-1 agreement. Arms are built by P44's `serve_stack.build_served_model` (the model exactly as the harness builds it; the staged P42 hook applies the env lanes at load) and the fused arm calls `qkv_fuse.fuse_qkv`, **refusing unless all 48 attention modules fused**; each arm's census (Int4Linear count 192 → 96, int4 expert layers, glue/epilogue counts) is saved beside its logits.

**Dry runs, before any rental:** `kl_b16.py --self-test` (CPU, tiny random Qwen3-MoE, transformers 5.16.1): batched B=5 teacher forcing equals `kl_fidelity.decode_teacher_forced_logits` row by row to **2.2e-7**; the same model scored twice reads **KL exactly 0**; a perturbed copy reads KL 0.0246 with top-1 < 1 (the harness detects what it exists to detect). `p59_reduce.py` exercised on a synthetic four-arm set (all verdict branches). `p59_drive.sh` dry-run and drifted-pin refusal exercised; the runner reuses P58's install/tripwire/fetch/bake blocks verbatim plus `kl_fidelity --controls` (K0: no KL row without a passing control receipt on the box).

## Arms (one process each; the hook applies the env at load; all-VRAM; `GNF4_GEMV_FUSED_REDUCE` unset)

| arm | env | q/k/v | what it is |
|---|---|---|---|
| `nf4` | `E4B_SERVE_EXP_INT4=0`, no folds, no int4 attention | three projections (dense) | the register's NF4 control — the **anchor** |
| `int4` | P54's: `E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0`, glue r1/r2 + router epilogue | three `Int4Linear` | the shipped B=16 configuration — the **reference** for P1 |
| `int4_fqkv` | same | one fused `Int4Linear` (`fuse_qkv`, 48 modules) | **the lever under test** |
| `int4_aa` | same as `int4`, fresh process | three `Int4Linear` | the determinism control |

Order: `nf4`, `int4`, `int4_fqkv`, `int4_aa`. Cut: e4b = the launch manifest's `heads.e4b` (this PR's merge or later); grouped-nf4-gemm **65cb104** (K17 merge); transformers 5.16.1, torch 2.8 (image), triton 3.4; P42 hook, P39 harness pieces, P44 `serve_stack.py`, `bench/kl_fidelity.py` staged byte-identical (`staged.sha256`).

## Predictions (falsifiable; written before the data)

- **P1 — `KL(int4 ‖ int4_fqkv)` over the 2,048 decode positions: mean ≤ 0.01 nats/token AND top-1 ≥ 0.97.** Central expectation ~0.003: [`finding_moe_router_flips_are_the_parity_noise_floor`] measured bf16 reorderings at ~1.3e-3 nats on unflipped tokens and ~0.05 on the ~4.5 % of (layer, token) choices where a router top-k flips — the fusion is a reordering of exactly that kind. *Refuted by* mean > 0.03 (a quantisation-tier-sized difference: NF4-vs-bf16 experts measured 0.029) or top-1 < 0.95. Between the bands: not held, not refuted, stated as such.
- **P2 — the fusion does not move the stack's distance from the NF4 anchor:** `|KL(nf4 ‖ int4_fqkv) − KL(nf4 ‖ int4)| ≤ 0.005`. The anchors themselves are expected at 0.03–0.08 (RTN int4 experts + uncalibrated int4 attention vs NF4 experts + bf16 attention) and are reported, not gated.
- **P3 — determinism:** `KL(int4 ‖ int4_aa)` is **exactly 0** on every token (P57 read bit-identical tokens across draws; a fresh process must reproduce the logits bit for bit). *Refuted by* any non-zero token — which would mean the instrument, not the fusion, carries noise, and P1 could not be read.
- **P4 — prefill control:** `KL(int4 ‖ int4_fqkv)` over the 16 prefill-last logits ≤ 1e-4 (the shared > 16-row path is unchanged by the fusion).
- **Census:** `int4_fqkv` fused 48 modules and carries 96 `Int4Linear`; `int4` 192; `nf4` 0; all four arms scored the same `prompts_sha256`.

## Decision rule

**P1 ∧ P2 ∧ P3 ∧ census OK → `--fuse-qkv` is the default on the int4 serving lanes at B=16 as well**: P54's fused B=16 row (11.197 ms/step, 1429 tok/s on its box) becomes the quoted B=16 position, the register notes on `e4b.serve.p54.qwen3.b16.fqkv.5090.2026-09-21` and P57's/P58's B=16 rows say so, `docs/STATUS.md` and `docs/SERVING-THROUGHPUT.md` drop "opt-in at B=16". **¬P1 (refuted) → stays opt-in**, and the KL is recorded as the fusion's quality cost in nats. **Between bands → stays opt-in**, with the number, and the next instrument (a K8 two-text read at B=16 through the same batched scorer) is registered before any default moves. **¬P3 → nothing is read**: the lane records an instrument fault and stops. The shipped bar for KL-from-checkpoint (≤ 0.10 nats, top-1 ≥ 0.93, `docs/STATUS.md`) is the outer limit no verdict here may exceed; P1's bands are deliberately tighter because the comparison is the same model under two accumulation orders.

## What this lane does not say

No bf16 reference: Qwen3-30B-A3B in bf16 is 61 GB and does not fit the card; the NF4 control is the anchor, disclosed as such. No K8 perplexity; no held-out prompt set (the rows are the timed arms' own — the question is about two arithmetic paths of one model, and the prompts must be the ones whose tokens diverged). One box, one prompt set, B=16 only (the B=1 fusion is token-identical and needs no gate).

## Budget and STOP rules

One RTX 5090 (verified/secure), **≤ 1.0 h guard**, estimate **≤ $0.66** (install ~8 min + fetch ~6–15 + bake ~1 + K0 ~1 + four arms × ~3 min: load from the arena ~30 s, RTN int4 packs ~30 s, 384-token prefill + 128 batched decode steps ~1 min); lane ceiling $1, hard stop $2. **STOP-1** K0 controls fail → rc 14, no KL row. **STOP-2** `fuse_qkv` count ≠ 48 → the fused arm refuses (rc 41). **STOP-3** an arm's logits missing → rc 42, the pairs that need it read MISSING. **STOP-4** no second box on a disappointing result. Receipt + ledger row committed together (ledger `cp` backup, post-commit diff); the drive fetches summaries, censuses and logs, **not the logits** (~1.2 GB per arm stay on the box; the reducer ran there and its JSON travels).

Amendments, dated, go below this line before any data is read.
