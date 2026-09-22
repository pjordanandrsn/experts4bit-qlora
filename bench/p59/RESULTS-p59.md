# Results — P59: KL at B=16 between the unfused and fused-q/k/v int4 stacks — exactly zero, because the K16 kernel is bitwise invariant to fusing; the divergence P54 saw lives in the >16-row prefill path (2026-09-22)

Pre-registration: [`P59-PREREG.md`](P59-PREREG.md) (#682, merged b2d6525 before the run; amendment 1 appended after this read). Receipts: [`receipts/`](receipts/) (per-arm censuses, trimmed logs, K0 controls, `p59_rep.json`, the prompt rows, the box's generated read, teardown proof; the fp32 logits stayed on the box). A2000 probe: [`probe/`](probe/). Every KL below is read from the receipts by [`p59_reduce.py`](p59_reduce.py).

Lane `p59-5090-1`: Vast instance 52092816, one RTX 5090; e4b **b2d6525**, grouped-nf4-gemm **65cb104** (K17 merge), transformers 5.16.1, torch 2.8, triton 3.4; rented 17:16Z, `TP_DONE` 17:53Z, destroyed 17:53:14Z (proven); **$0.34** against the $0.66 estimate. K0 controls passed on the box (`kl-fidelity/1.0.0`); the prompt rows reproduced P54/P57/P58's digest `f67e7e4d…`.

## The read

| pair (ref ‖ test) | positions | KL mean (nats/token) | top-1 | exactly zero |
|---|---|---|---|---|
| **`int4` ‖ `int4_fqkv`** (P1) | 2,048 decode | **0.000000** | **1.0000** | **True** |
| `nf4` ‖ `int4` (anchor) | 2,048 | 0.071728 (median 0.0235, p95 0.2549, max 4.69) | 0.9048 | False |
| `nf4` ‖ `int4_fqkv` (anchor) | 2,048 | 0.071728 | 0.9048 | False |
| `int4` ‖ `int4_fqkv`, prefill-last (P4) | 16 | 0.000000 | 1.0000 | True |
| `int4` ‖ `int4_aa` (P3) | — | **not run** — skipped by the deadline guard (STOP-2) | | |

- **P1 HOLDS, trivially: the fused and unfused stacks produced bit-identical logits on every one of 2,048 × 151,936 decode entries.** The census proves the lever was installed (48 attention modules fused, `Int4Linear` 192 → 96, the round-2 fold re-wrapped around the fused forward). A perfect 0.000 is the result an instrument's own docstring warns about ("the lever never ran"), so it was checked before it was believed — next section.
- **P2 HOLDS** (Δ +0.00000 against the anchors 0.07173 / 0.07173). **P4 HOLDS** (0.0). **Census OK.**
- **P3 UNREAD.** The four arms took ~5.5 min each on this host (build 130 s, 16-row teacher-forced scoring 200 s), and the fourth could not finish 10 minutes before the 1-hour guard; the runner skipped it as registered. With P1 exactly zero the determinism control could not have changed the P1 reading, but the decision rule names P3, so **the decision rule is not met as written** and no default moves on this run.
- The anchors: the int4 serving stack sits **0.0717 nats/token** from the NF4 control on these rows (top-1 0.905) — a heavy tail (p95 0.25, max 4.7), reported, not gated; well inside the shipped KL-from-checkpoint bar only in the sense that the NF4 control is not a checkpoint (no bf16 reference fits the card).

## Why exactly zero — the K16 kernel is bitwise invariant to fusing, and the divergence lives elsewhere

A free probe on the QNAP RTX A2000 ([`probe/p59_probe.py`](probe/p59_probe.py), output [`probe/probe.json`](probe/probe.json), e4b 0.36.4 @ b2d6525, grouped-nf4-gemm 0.32.1 @ 65cb104, torch 2.8.0+cu128, triton 3.4.0): three `Int4Linear` at Qwen3-30B-A3B's attention shapes (K 2048; N 4096 / 512 / 512) against `Int4Linear.fuse` of them, rows 1…64, K16 route on and off.

| route | R=1 | R=2 | R=8 | R=16 | R=17 | R=64 |
|---|---|---|---|---|---|---|
| K16 small-M GEMM on (`smallm=True`, the shipped default) | equal | **equal** | **equal** | **equal** | differs | differs |
| K16 off (cached-bf16 cuBLAS above one row) | equal | differs | differs | differs | differs | differs |

At 2–16 rows the K16 kernel's plan is the same for every N here (`block_n 64, kc 128, sk 4` for q, k, v and the fused 5120-wide projection) and every output column is computed independently, so fusing along N cannot change a bit. At one row the GEMV is likewise per-row. **Above 16 rows** `Int4Linear` runs cuBLAS on its cached bf16 weight, whose kernel selection depends on (M, N) — fused and unfused differ there by up to one bf16 ulp (max |Δ| 0.0039–0.0078). The A2000 is a correctness instrument here, not a timing one; bitwise identity is a correctness property, and the 5090 read of P1 (exactly zero at 16 rows) agrees with it.

**What that says about P54's and P57's token divergence.** Decode at B=16 is 16 rows — the invariant path — so the divergence cannot come from the fused projection at decode. The harness prefills in **128-token steps** (`step_decomp --chunk 128`, `max_prefill_tokens_per_step=128`): 128 rows per forward, the cuBLAS path, where fused and unfused differ at the ulp level. Those differences land in the KV cache, and greedy decoding amplifies them into different tokens (P54's earliest divergences are at generated tokens 2 and 8). This run prefilled all 384 tokens × 16 rows in one forward (6,144 rows) and read exactly zero on the prefill-last logits too, so on the 5090 cuBLAS picked matching kernels at that M — but the harness's M is 128. **The leading hypothesis for P54's divergence is now the prefill's cuBLAS path at the harness's step size; P57's conclusion ("the K16 GEMM's") is withdrawn** (corrected in `bench/p57/RESULTS-p57.md`, `docs/STATUS.md` and `CHANGELOG.md` in this change).

## What changes

Nothing in the defaults on this run. Register rows: `e4b.serve.p59.qwen3.b16.fqkv-kl.5090.2026-09-22` (0.0 nats, with the reason) and the two NF4-anchor rows. **Amendment 1** (in `P59-PREREG.md`, dated before any amendment data) re-runs the lane with the prefill in 8-token-per-row chunks (16 × 8 = **128 rows per forward**, the harness's step), a census of which `Int4Linear` carry the K16 route, and a guard long enough for the determinism arm: the KL it reads is the fusion's real cost through the path where it can change bits.

## Generated read (the box's `p59_reduce.py`, unedited)


Pre-registration: [`P59-PREREG.md`](P59-PREREG.md). Every number below is read from the fetched receipts by `p59_reduce.py`.

## Arms (census)

| arm | built | fuse_qkv_n | Int4Linear (before -> after fuse) | int4 expert layers | glue r1/r2/epi | decode logits | build s | score s |
|---|---|---|---|---|---|---|---|---|
| nf4 | yes | 0 | 0 -> 0 | 0 | 0/[0, 0]/0 | [16, 128, 151936] | 28.8 | 27.3 |
| int4 | yes | 0 | 192 -> 192 | 48 | 193/[48, 48]/48 | [16, 128, 151936] | 130.0 | 204.3 |
| int4_fqkv | yes | 48 | 192 -> 96 | 48 | 193/[48, 48]/48 | [16, 128, 151936] | 132.2 | 200.9 |
| int4_aa | MISSING | | | | | | | |

prompts_sha256 identical across arms: **True** (f67e7e4d592b002b)

## KL pairs (fp64, full vocab, token-weighted over 16 rows x 128 decode positions unless noted)

| pair (ref || test) | positions | kl_mean | kl_median | kl_p95 | kl_max | top-1 | exactly zero |
|---|---|---|---|---|---|---|---|
| int4 || int4_fqkv (P1) | 2048 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 1.0000 | True |
| nf4 || int4 | 2048 | 0.071728 | 0.023463 | 0.254881 | 4.6879 | 0.9048 | False |
| nf4 || int4_fqkv | 2048 | 0.071728 | 0.023463 | 0.254881 | 4.6879 | 0.9048 | False |
| int4 || int4_aa (P3, determinism) | -- | MISSING | | | | | |
| int4 || int4_fqkv, PREFILL-last (P4) | 16 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 1.0000 | True |
| nf4 || int4_aa | -- | MISSING | | | | | |

## Verdicts (pre-registered rules)

- **P1: HOLDS** -- kl_mean 0.00000 (hold <= 0.01, refute > 0.03), top-1 1.0000 (hold >= 0.97, refute < 0.95)
- **P2: HOLDS** -- KL(nf4||int4_fqkv) - KL(nf4||int4) = +0.00000 (band +-0.005); anchors 0.07173 / 0.07173
- **P3: UNREAD** (arm missing)
- **P4: HOLDS** -- prefill-last kl_mean 0.000e+00 (hold <= 1e-4)
- **census: OK** -- fused n=48 Int4Linear 96; int4 192; nf4 0; same prompts True

**Decision rule:** P1 and P2 hold and the census is OK -> --fuse-qkv becomes the default on the int4 lanes at B=16 too (P54's fused B=16 row is the quoted position)
