# Results -- P59: KL at B=16 between the unfused and fused-q/k/v int4 stacks (the gate on the B=16 fusion default)

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
