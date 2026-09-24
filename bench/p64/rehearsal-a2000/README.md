# P64 rehearsal on the NAS RTX A2000 — NOT a reading

Everything here is a rehearsal of `bench/p64/`. **None of it is P64's reading.** That is registered in
[`../P64-PREREG.md`](../P64-PREREG.md) for one rented RTX 5090 on Qwen3-30B-A3B with the licensed pack. Where it
departs from the registration:

- **Card:** RTX A2000 12 GB (sm_86, 70 W), not the 5090 class.
- **Model:** OLMoE-1B-7B-0924 **base** (P44's family, the local copy), not Qwen3-30B-A3B. P44 pins the Instruct
  checkpoint.
- **Pack:** calibrated on 8 × 512 C4 tokens with a 2 GB Hessian budget, not 128 × 512 and 24 GB. Its fingerprint is
  `sha256:0fc7bc44…`, not the licensed `0c9955a9…`.
- **Install:** skipped. The committed tree ran from `PYTHONPATH` with grouped-nf4-gemm v0.33.0 (`5ca1897`, the
  registered cut), in the image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (torch 2.8.0+cu128, triton 3.4.0,
  transformers 5.16.1, bitsandbytes 0.50.1).

`p64_run.sh` marked each run `REHEARSAL`, and the reducer titled its output so.

What it shows: the flag, the scorer, the census, the controls, the reducer and the runner's control flow work end to
end on real CUDA. What it cannot show: anything about Qwen3's int8 step, the 5090, or the licensed stack.

## Runs, in order

| dir | e4b tree | what ran | rc |
|---|---|---|---|
| `smoke-build/` | `94842a2` + the uncommitted lane (2026-09-23 ~21:00Z) | the runner up to the pack build, 2 rows. The registered build command (P55x's `step_decomp.py` licbuild) calibrated experts (2,048 gptq / 0 rtn over 16 layers) and attention (64 projections), dumped the artifact (manifest + 66 payloads), then died in its wikitext K8 cross-check: `type fp8e4nv not supported in this architecture` (the fp8 paged-KV kernel needs sm_89+). | 20 |
| `prove-curl-probe/` | `3062b9f` | the proving mode (`P64_PROVE=1`): every check passed, but the egress probe read **0.0 MB/s** because the image ships no `curl`. | 0 |
| `probe/` | `3062b9f` | `probe_p59_b16_route.py`: which expert route P59's B = 16 scorer runs (below). | — |
| `prove/` | `f2854d9` | the proving mode again, probe in python: **53.2 MB/s**; the flag check on real kernels passes. | 0 |
| `full/` | `f2854d9` | the whole runner, 16 rows × 2 texts, every pass. The smoke run's pack, which verifies, was reused (`BUILD reused (verify_artifact OK)`), and so was its NF4 arena. | **0** |

## What the rehearsals changed in the design (disclosed in the pre-registration)

1. **The pack's validity is `verify_artifact`, not the build process's exit code.**
   - `smoke-build` dumped a complete pack; then the K8 cross-check, which is not an input to anything, failed.
   - The runner, gating on the exit code, threw the pack away (rc 20).
   - It now verifies the artifact (every payload re-hashed, the fingerprint recomputed), continues when it verifies,
     and reuses a verified pack already on the box.
2. **The proving run's egress probe is python, not curl.**
   - It could only read 0 in this image: a probe that can only say no, P55x amendment 1's openrsync lesson again.
3. **The rehearsal harness itself** (not the lane) ran `datasets` online. `HF_DATASETS_OFFLINE=1` looks up a
   different `data_files` cache hash and refused the cached C4 shard. A rental box is online anyway.

## `full/`: the read, as a rehearsal

From `full/RESULTS-p64-generated.md`, the reducer's output, unedited:

- **Validity: VALID.**
  - `a8 ‖ a8_rep` is exactly 0 on all 2,048 positions of both texts.
  - The prefill-last logits of `a8`, `a16`, `a16_all` and `a8_rep` are bit-identical.
  - Every pass's decode counts are exactly the registered ones, per text: `a8` 65,536 expert quantises (2 × 16 × 2,048),
    0 dequants, 131,072 attention quantises (64 × 2,048); `a16` 0 / 524,288 (2 × 16 × 8 × 2,048) / 131,072;
    `a16_all` 0 / 524,288 / 0.

| text | G_exp = KL(a16 ‖ a8) | top-1 | floor F (pc64, pc384) | G/F | dNLL (95 % CI) | class |
|---|---|---|---|---|---|---|
| wikitext | 0.001735 | 0.9844 | 0.001990 (0.001937, 0.002042) | 0.87 | +0.00118 (−0.00209, +0.00433) | BELOW FLOOR |
| c4val1 | 0.001550 | 0.9819 | 0.001861 (0.001781, 0.001941) | 0.83 | −0.00077 (−0.00357, +0.00185) | BELOW FLOOR |

- **G_all** (experts + attention): 0.001734 and 0.001724, both below floor.
- **G_attn:** 0.001522 and 0.001365, both below floor.
- **Anchor** KL(nf4 ‖ a8): 0.0697 and 0.0619.
- **Verdict, as a rehearsal:** INDISTINGUISHABLE on all three statistics.

**Read the rows together, not as a sum.** G_all, G_exp and G_attn are all about F. At the floor, any bit-level
change to the decode arithmetic reads about F in this instrument: router flips carry it (METHODOLOGY §13.1). So a
G at the floor says the change is not resolvable here. It does not say the effects add.

**Timing, from the census** (seconds per decode step including the row's prefill; never a speed number):

| pass kind | s/step |
|---|---|
| a8 | 0.060–0.064 |
| a16 | 0.174 (about 2.8× a8) |
| pc64 | 0.082 |
| pc384 | 0.054 |
| nf4 | 0.041–0.046 |

The served process took 43.7 min for 12 pass-texts, its 115 s load included. The anchor process took 3.4 min.

## `probe/`: which route P59's B = 16 scorer runs

`probe_p59_b16_route.py` builds P59's `int4` arm (RTN int4 experts + RTN int4 attention + folds) through
`serve_stack.build_served_model`. It scores 16 rows through `kl_b16.batched_teacher_forced` exactly as P59 did:
prefill in 8-token-per-row chunks, then 8 decode steps at B = 16. Decode-phase counts:

| route | expert int8 quantises | expert dequants (bf16 branch) | attention int8 quantises |
|---|---|---|---|
| as P59 ran it (`DEVICE_GROUPING` off) | **0** | 13,484 | 0 |
| `DEVICE_GROUPING` on (the timed B = 16 route) | 256 (= 2 × 16 × 8) | 0 | 0 |

P59's KL read the bf16-activation expert branch at B = 16, not the int8 GEMV the timed arms run. Attention at 16
rows is K16 (bf16) in both.

## `prove/`: the flag on real kernels

`kl_a16.py --prove-flag`, at Qwen3-30B-A3B's expert shapes (hidden 2048, moe_intermediate 768), 16 random experts,
one token's 8 rows:

- Off calls `quant_x_rows` twice and never dequantises. On calls it 0 times with 16 dequants.
- On equals the prefill branch bit for bit.
- The off route still captures in a CUDA graph and replays bit-identical to eager.
- On refuses under capture with its sentence.
- Relative error against the fp64 product of the same int4 values and bf16 activations: **off 0.01224, on 0.00460**.
  Per call, the int8 step's error is 2.7× the bf16-dequant route's here.

Logs are stored as `*_log.txt` (the repository ignores `*.log`); the names are otherwise the runner's.
