# Results -- P59: KL at B=16 between the unfused and fused-q/k/v int4 stacks (the gate on the B=16 fusion default)

Pre-registration: [`P59-PREREG.md`](P59-PREREG.md). Every number below is read from the fetched receipts by `p59_reduce.py`.

## Arms (census)

| arm | built | fuse_qkv_n | Int4Linear (before -> after fuse) | int4 expert layers | glue r1/r2/epi | decode logits | build s | score s |
|---|---|---|---|---|---|---|---|---|
| nf4 | yes | 0 | 0 -> 0 | 0 | 0/[0, 0]/0 | [16, 128, 151936] | 34.2 | 18.3 |
| int4 | yes | 0 | 192 -> 192 | 48 | 193/[48, 48]/48 | [16, 128, 151936] | 273.4 | 84.9 |
| int4_fqkv | yes | 48 | 192 -> 96 | 48 | 193/[48, 48]/48 | [16, 128, 151936] | 62.0 | 82.7 |
| int4_aa | yes | 0 | 192 -> 192 | 48 | 193/[48, 48]/48 | [16, 128, 151936] | 62.3 | 81.8 |

prompts_sha256 identical across arms: **True** (f67e7e4d592b002b)

## KL pairs (fp64, full vocab, token-weighted over 16 rows x 128 decode positions unless noted)

| pair (ref || test) | positions | kl_mean | kl_median | kl_p95 | kl_max | top-1 | exactly zero |
|---|---|---|---|---|---|---|---|
| int4 || int4_fqkv (P1) | 2048 | 0.004404 | 0.001508 | 0.014160 | 0.9790 | 0.9775 | False |
| nf4 || int4 | 2048 | 0.067151 | 0.023339 | 0.257067 | 2.0240 | 0.9028 | False |
| nf4 || int4_fqkv | 2048 | 0.070573 | 0.023393 | 0.259482 | 3.7486 | 0.9014 | False |
| int4 || int4_aa (P3, determinism) | 2048 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | 1.0000 | True |
| int4 || int4_fqkv, PREFILL-last (P4) | 16 | 0.029337 | 0.001059 | 0.117205 | 0.4427 | 0.9375 | False |
| nf4 || int4_aa | 2048 | 0.067151 | 0.023339 | 0.257067 | 2.0240 | 0.9028 | False |

## Verdicts (pre-registered rules)

- **P1: HOLDS** -- kl_mean 0.00440 (hold <= 0.01, refute > 0.03), top-1 0.9775 (hold >= 0.97, refute < 0.95)
- **P2: HOLDS** -- KL(nf4||int4_fqkv) - KL(nf4||int4) = +0.00342 (band +-0.005); anchors 0.06715 / 0.07057
- **P3: HOLDS** -- exactly_zero True, kl_max 0.000e+00
- **P4: REFUTED** -- prefill-last kl_mean 2.934e-02 (hold <= 1e-4)
- **census: OK** -- fused n=48 Int4Linear 96; int4 192; nf4 0; K16-routed int4/fused 192/96; prefill chunk 8; same prompts True

**Decision rule:** P1 and P2 hold and the census is OK -> --fuse-qkv becomes the default on the int4 lanes at B=16 too (P54's fused B=16 row is the quoted position)
