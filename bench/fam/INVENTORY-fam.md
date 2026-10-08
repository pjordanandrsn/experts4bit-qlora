# FAM0 — which shipped defaults were read on which model family

Work item #1362. Read from `main` at e4b `b38866c6` (0.50.0) and grouped-nf4-gemm `3ce2ecd8` (0.43.0). Rows are the
model types in `docs/capabilities.json`. Columns are every row of the two STATUS defaults tables, plus P115's four knobs.

Each cell was proposed by a scan of both `claims.json` registers and the code's engagement predicates, then checked by
hand. "Read on a family" means a registered A/B of that default on that family. A run that merely had the default on
does not count.

| cell | meaning |
|---|---|
| ✓ | the read STATUS's defaults table cites (Qwen3-30B-A3B, one RTX 5090) |
| a lane tag (`p96`) | a read on this family; the claim ids are under [Reads beyond Qwen3](#reads-beyond-qwen3) |
| `E` + tag | engagement only: the default ran and was counted, never A/B'd |
| shape | cannot engage on this family (why: the notes under each table) |
| inert | acts only inside an opt-in that is off by default |
| n/a | the family has no such path |
| · | **unread**: it engages, and no read on this family exists |
| ? | whether it engages needs a probe on the real weights |
| ⚠ | a defect found by this audit |

**Arithmetic.** Some defaults are bitwise: their reads found identical outputs or tokens. These are KV step select, PDL,
K23, prebind, host reuse, the pinned ring, the tile rule and the lean delta, and slots for serial output. On another
family they need an engagement count, not a quality read. Every other default changes arithmetic and needs a quality
read per family.

## Serving (`serve_paged`'s default server, NF4 store, sm_120)

| family | graphs | slots | 1 graph/step | KV select | fp8 attn | BW GEMV | dot-pad | PDL | K19 | K23 | K21 | K25 | router cast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3_moe` | ✓ | ✓ | ✓ | ✓ | ✓ `m3` | ✓ | shadowed ⚠ | ✓ | ✓ | ✓ | shape | **·** | inert (✓ `p70` with the fold on) |
| `qwen3_5_moe` | `p101` | · | · | · | · | shape | shape | shape | opt-in store | · | shape | · | inert |
| `granitemoe` | `E p115c` | · | · | · | shape | shape | shape | shape | opt-in store | `p96` | shape | `p93` `p96` | inert |
| `olmoe` | · | · | · | · | · | shape | shape | shape | opt-in store | `p96` | shape | `p93` `p96` | inert |
| `mixtral` | · | · | · | · | · | shape | shape | shape | opt-in store | · | shape | · | inert ⚠ |
| `gemma4_text` | · | · | · | · | `fp8-share` `p108` | shape | shape | shape | opt-in store | · | shape | · | inert |
| `gpt_oss` | `E p115c` | · | · | · | shape | shape | shape | shape | shape | shape | `p90` | · | inert |
| `ernie4_5_moe` | · | · | · | · | · | shape | shape | shape | opt-in store | · | shape | · | inert |
| `qwen3_next` | · | · | · | · | · | shape | shape | shape | opt-in store | · | shape | · | inert |
| `lfm2_moe`, `granitemoehybrid`, `nemotron_h`, `deepseek_v4`, `deepseek_v2` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

- **Not served.** `paged_runner` refuses deepseek_v2 (latent attention), lfm2_moe (conv layers), granitemoehybrid and
  nemotron_h (Mamba layers), and deepseek_v4 (compressed attention).
- **fp8 attn.** fp8 compute needs key groups at least 32 wide. At head_dim 64 (Granite, gpt-oss) e4b's KV pool picks
  16-wide groups, so those two run f32. On Gemma-4 the read is a cost, not a pass: fp8 costs 0.046 nats one-shot, and
  the shipped per-layer groups recover 0.020 of it.
- **BW GEMV / dot-pad.** Both engage only at Qwen3-30B-A3B's two expert shapes on ≥ 160-SM parts. ⚠ The two shape sets
  are identical, and the BW route is checked first. So the dot-pad default never runs unless `GNF4_GEMV_BW=0`.
- **PDL** reaches the int4-b32 GEMV, the BW GEMV, the glue kernels and part of the MXFP4 decode, on sm_90+. With the
  folds off, the default NF4 server has no PDL-carrying kernel on any family but Qwen3.
- **K19** needs the opt-in int4-b32 store. **K21** needs gpt-oss's opt-in native MXFP4 store.
- **K23** (lean glue) also applies to K25 rows, though its docstring says K19 only. On Granite and OLMoE it ran inside
  P96's K25 arms, not alone.
- **K25** has no family or shape gate. It takes every NF4 decode step with T > 1 and at most 256 routed rows. That is
  buckets 2–32 at top-8, T ≤ 42 on ERNIE, T ≤ 64 on gpt-oss. On **Qwen3-30B-A3B it engages at W16 (128 routed rows) and
  was never A/B'd there**. Its licence (P96) read Granite and OLMoE only.
- **Router cast** acts only inside the fused router epilogue (`E4B_FUSE_ROUTER_EPI`, default `0`). ⚠ On Mixtral it is
  not upstream's function: Mixtral's router returns fp32 weights, but the cast sends them to bf16.
- **Slots** on Gemma-4: the estimate's per-layer KV geometry is unread.

## P115's fused B=1 stack (default `0` in code; #1354 decided a family allowlist)

The maintainer has since ruled that the allowlist takes a family only with a T == 1 read at reading size (bus,
2026-10-08T19:13Z).

| family | q/k/v | glue | glue r2 (layer / attention) | router epilogue |
|---|---|---|---|---|
| `qwen3_moe` | ✓ `p115` | ✓ `p115` | ✓ `p115` | ✓ `p115` |
| `qwen3_5_moe` | shape | shape | shape | `p115c` SANE at T == 12; T == 1 owed |
| `granitemoe` | shape | `p115c` quality at T == 12; T == 1 owed | same | same |
| `gpt_oss` | shape | `E p115c`, SANE failed (argmax 0.924) | `E p115c` / shape | `E p115c`, SANE failed |
| `olmoe` | shape | · | · / shape | · |
| `mixtral` | shape | · | · / · | · ⚠ (the cast above) |
| `gemma4_text` | shape | · | shape | · |
| `ernie4_5_moe` | shape | · | · / ⚠ #1367 | ? |
| `qwen3_next` | shape | shape | shape | · |
| not served (5) | n/a | n/a | n/a | n/a |

- **q/k/v** fuses only a module whose class is named `Qwen3MoeAttention`. ⚠ That gate is on a class name, against
  AGENTS.md §7.
- **glue** refuses centered `(1 + w)` norms (Qwen3.6, Qwen3-Next).
- **glue r2.**
  - The attention fold refuses gpt-oss's `sinks`.
  - It refuses OLMoE's hidden-width q/k norms.
  - It refuses Gemma-4's extra norms, `layer_scalar` and `v_norm`.
  - ⚠ #1367: it licenses ERNIE's attention and applies rotate-half rotary to ERNIE's interleaved rotary. On CPU the
    rotated q is wrong by 1.13 in relative norm.
- **router epilogue** on ERNIE: only the `topk_softmax` probe applies, and ERNIE selects with `e_score_correction_bias`.
- Census predicted from code, not run: OLMoE `0 / 65 / [16, 0] / 16`, Mixtral `0 / 65 / [32, 32] / 32`.

## Training (`enable_fast_train`)

| family | LM loss | absmax DQ | prebind | ckpt | eval loss | combine | pad buckets | compact | GEMM route | host reuse | pinned ring | tile rule | lean delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3_moe` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (sm_90) | ✓ | ✓ | ✓ | ✓ |
| `mixtral` | · | `tc1-a28` | · | · | · | · | · | · | `tc1-a22` (dense) | · | · | shape | · |
| `gemma4_text` | shape | · | · | · | shape | · | · | · | · | · | · | · | · |
| `olmoe`, `granitemoe`, `qwen3_5_moe`, `lfm2_moe`, `granitemoehybrid`, `ernie4_5_moe`, `nemotron_h` | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `gpt_oss` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `deepseek_v4`, `qwen3_next`, `deepseek_v2` | shape | ? | ? | ? | shape | ? | ? | ? | ? | ? | ? | ? | ? |

- **gpt_oss:** the loader builds its experts bare, so `enable_fast_train` patches nothing.
- **deepseek_v4, qwen3_next, deepseek_v2:** no fused-training parity read exists, so engagement is unproven.
- **LM loss / eval loss** need a class in the chunked loss's table. Gemma-4 is not in it and sets a final-logit softcap.
- **absmax DQ** keeps fp32 under offload. On Mixtral the read is a trade: 1.023 of the step for 2.04 GB less peak.
- **GEMM route.**
  - grouped_mm on sm_90 is unread off Qwen3.
  - Dense (≤ 16 present experts) is always taken on Mixtral and effectively never on the others.
  - Mixtral's dense route skips the fused GEMM, so the tile rule cannot reach it.
- **pad buckets** engage at ≥ 16,384 routed rows: tokens × top-k, at least 2,048 tokens per call at top-8.

## Reads beyond Qwen3

| tag | claim ids | kind |
|---|---|---|
| `p101` | `e4b.serve.p101.qwen36-hybrid-decode-graphs.5090.2026-10-03` | speed; graph tokens equal the padded eager step's |
| `p115c` | `e4b.serve.p115.fused-stack-engagement.gptoss-qwen36.5090.2026-10-08`, `e4b.serve.p115.fused-stack-quality.granite.5090.2026-10-08` | census, every bucket captured; SANE; Granite's quality |
| `fp8-share`, `p108` | `e4b.parity.gemma4.fp8-share`, `e4b.parity.gemma4.p108.paged-vs-floor.5090.2026-10-03` | quality: fp8's cost; the paged path at parity with its floor |
| `p90` | `e4b.serve.p90.gptoss.mxfp4.k21-b16.5090.2026-10-01` | speed, KL |
| `p93`, `p96` | `e4b.serve.p93.nf4-families.k25-tree-tf32-b16.5090.2026-10-01`, `e4b.serve.p96.nf4-families.k25-windowed-k8.5090.2026-10-02` | speed; windowed K8 |
| `tc1-a28` | `e4b.train.absmax-dq.mixtral.5090.2026-10-05` | speed, peak, held-out |
| `tc1-a22` | `e4b.train.dense-route.mixtral.5090.2026-10-04` | speed, held-out |
| `m3`, `p70`, `p115` (Qwen3) | `gnf4.serve.m3-defaults-on`, `e4b.serve.p70.qwen3.b1.router-weight-cast.5090.2026-09-25`, `e4b.serve.p115.fused-stack-speed.qwen3.5090.2026-10-07`, `e4b.serve.p115.fused-stack-quality.qwen3.5090.2026-10-07` | the cells that name them |

## What to buy first

Ranked by: default on (or waiting to flip), changes arithmetic, engages on a served or trained family, no read.

1. **The fused stack at T == 1 against each family's own floor:** gpt-oss-20b one knob per arm, then Granite and Qwen3.6.
   The allowlist waits on these (FAM2).
2. **K25 on Qwen3-30B-A3B.** It is on in every default W16 step, changes arithmetic (a TF32 select tree), and was never
   A/B'd on the flagship model. P96's windowed instrument fits.
3. **fp8 attention compute on head_dim-128 families** (OLMoE, Mixtral, ERNIE) and Qwen3.6's full-attention layers. It is
   on by default, read on Qwen3 (zero cost) and Gemma-4 (a cost). FAM1's instrument reads it as fp8 against f32.
4. **Decode graphs and the one-graph step on other families.** The padded-bucket arithmetic was read on Qwen3 only.
5. **Training: one defaults-on-against-off A/B per trained family** (held-out plus loss parity), Gemma-4 and OLMoE first.
   absmax DQ changes the stored scales and was read on Qwen3 and Mixtral only.
6. **The bitwise defaults** need no box of their own. Record their engagement and token identity inside the reads above.

**Defects for review**, not fixed here:
- #1367: ERNIE's rotary under glue r2.
- The dot-pad default is shadowed by the BW GEMV everywhere.
- The router cast is inert by default, and is not upstream's function on Mixtral.
- FUSE_QKV is gated on a class name.
- K23's docstring says K19 only.
