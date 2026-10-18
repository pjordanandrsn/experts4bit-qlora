# P135 — what do Mixtral's ON_auto folds buy at one request on an RTX 5090? Track B1's read R-b

**Registered 2026-10-18, before any data.** Track B1's third paid read (bench/fastpath-portability/README.md section 6,
R-b: "Mixtral fused stack speed").

## Why

FAM's quality read licensed Mixtral ON_auto at T == 1 (`fam-mixtral-4`, all four ON configs PASS, resolution `mut090`;
bench/fam/PREREG-fam.md Amendments 5–8). FAM's registered consequence: a PASS licenses an allow-list pull request
**only together with a speed read of what the passing knobs buy**, by Amendment 4's interleaved method. Until that read
exists, Mixtral's folds stay off by default. P135 is that read.

## Design

FAM Amendment 4's box (`bench/fam/fam_speed.py`), which timed Qwen3.6's one engaged fold, widened to Mixtral's three:

- **One model, both settings.** The default `serve_paged` build with all four fusion knobs at `auto` (FAM's ON_auto). On
  Mixtral that engages census `0 / 65 / [32, 32] / 32`: the glue fold's 65 RMSNorms, round 2's 32 decoder layers and 32
  attentions, and the router epilogue's 32 routers, which makes **161** patched forwards (65 / 64 / 32 by fold module).
- **The swap.** Each fold patches an instance `forward` that keeps its original as the `_orig` default. `FoldRoute` puts
  all 161 originals back (OFF) or keeps the fused forwards (ON). It is exact, and it touches nothing else
  (tests/test_p135_speed.py: OFF reproduces the unfused model's logits bit for bit and runs no fold kernel).
- **Two blocks**, `a` (OFF first in every pair) and `b` (ON first). Each block holds one runner per setting, both alive,
  each with its own KV pool and its **bucket-1** graph captured under its own setting. Both runners prefill the same
  512-token wikitext prompt, then decode greedily in strict alternation: 5 warm steps, **256 timed** (each
  `run_decode`'s synchronised wall), then 32 more. Every token is recorded.
- **Bucket 1 only**, because the read is one row and Mixtral's NF4 weights take about 26.8 GiB of the card's 31.4. The
  server is built eager with 2 slots of 768 tokens (FAM Amendment 7's size).

**Pins.**
- Model: `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a2861cdc0098cc54bc9f17cb2c47eb61`, FAM Amendment 5's pin. The
  fetch excludes `consolidated*`; the arena is baked by P39's `k8_bake.py`.
- experts4bit-qlora: the launch commit, main after this registration merges.
- grouped-nf4-gemm: `724ccc454f006c1a46836e434e997f31f293747f` (v0.45.0), e4b CI's pin at registration.
- transformers 5.17.0, bitsandbytes 0.50.2, and torch held at the image's 2.8.0+cu128.

**The kernels are the ones FAM read.** FAM's Mixtral quality read ran grouped-nf4-gemm v0.43.0 (`6ee2e104`); P135 pins v0.45.0 (`724ccc45`). Between the two, the kernels Mixtral's folds call are bitwise unchanged on Mixtral: `rmsnorm_rows`, `rmsnorm_resid_rows` and `rope_heads` (the attention fold) are byte-identical. `router_epilogue` gained a `weights_dtype` argument, which is fp32 on Mixtral's fp32-upstream routers, so it is a no-op there. `combine_rows` gained `residual=`, licensed only where bitwise. `rope_norm_qk` needs a `q_norm`, which Mixtral lacks. ON is therefore bitwise FAM's ON_auto by construction (the independent review's check of the two releases).

**The folds are the ones FAM licensed.** Since `fam-mixtral-4` (e4b `68845bfc`), the fold modules have changed only by
#1583's two opt-in prefill knobs, `E4B_FUSE_PREFILL_GLUE` and `E4B_PREFILL_LEAN_DISPATCH`, both off by default. Decode
arithmetic is unchanged. The box refuses either knob set, and the tripwire asserts the 64-row decode gate.

**The premise**, on the card before anything is fetched: `tests/test_fusion_modes.py` and
`tests/test_fused_glue_decode_graphs_gpu.py`, all passed and none skipped, or rc 25.

## The rule (`p135_reduce.py`)

FAM Amendment 4's rule with Mixtral's constants. The first verdict that applies:

1. **VOID** on any of:
   - a record not ok;
   - another model, revision or e4b commit;
   - other knobs, or a prefill fold knob set;
   - a census other than `0 / 65 / [32, 32] / 32`, or other than 65 / 64 / 32 folds by module;
   - a setting that touched other than all 161;
   - the router epilogue off its fp32 path (32 patched, 32 `fp32_upstream`);
   - a block missing, out of order, on other slots, or with the wrong step counts;
   - bucket 1 not captured, replayed other than 293 times, or any eager step;
   - a setting whose two blocks emit different tokens.
2. **NOISY** if block a's ON/OFF ratio and block b's differ by more than **1.5 %**.
3. **FASTER** if both ratios ≤ **0.98**; **SLOWER** if both ≥ **1.02**; otherwise **NO_GAIN**.

`ratio_x` = median(ON) / median(OFF) of block x's timed steps: the within-box ABBA ratio. Reported and never gated: the
medians, the saving in ms, each block's median per-pair ratio, ON-vs-OFF token agreement and the first step where they
part (the folds change arithmetic, which FAM's quality read licensed), the GPU clock log, and peak memory.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Y1 | integrity holds (census, 161 folds, fp32 routers, bucket 1, no eager step, determinism): no VOID, about 85 % |
| Y2 | OFF's median step is within [17.0, 22.8] ms. P136's `p136-mixtral-5090-1` read 19.78 ms unprofiled at W1 with every fold off, on another host; 5090 hosts spread 13–15 % at one row. It sizes the guard only and is never a comparison anchor |
| Y3 | each block's ratio is within **[0.91, 0.975]**, a saving of 0.5–1.8 ms. The folds target P136's OFF classes `norm_elem` 1.106 ms, `moe_route` 0.440 ms, part of `attn` 0.341 ms, and the launch gaps (`idle_out` 0.445 ms). The low end is the router epilogue's measured 8.5 µs per router on Qwen3.6 (`fam-speed-1`) × 32, plus a small glue share. The high end is most of those classes |
| Y4 | FASTER about 70 %; NO_GAIN about 25 %; SLOWER or NOISY about 5 % |
| Y5 | ON's tokens agree with OFF's for at least the first 20 decode steps |
| Y6 | peak memory ≤ 31.0 GiB |

## Runs and budget

- **Run ids:** `p135-5090-1`, the reading. `-2` and `-3` are spares, used only after a refusal before create or a host
  fault. No proof run: P136's Mixtral box on the same class sizes the guard, and the CPU tests cover the swap and the
  rule.
- **The card:** one RTX 5090 on Vast verified-secure, direct SSH, reliability2 ≥ 0.98. Host RAM at the launcher's floor
  (98 GB). Disk ordered at 250 GB (the box's floor 220 + 30). Download ≤ 100 GB at ≤ $0.011/GB.
- **Guard 2 h.** The box's time-left checks (fetch 3,000 s, bake 1,500 s, speed 1,200 s with an 1,800 s alarm), each
  plus 600 s of fetch-back, fit inside it (tests/test_p135_staged_pin.py). Expected about 1 h.
- **Worst** = 2 h × the pinned rate (≤ $0.85/h) + 100 GB × the pinned $/GB, at most **$2.80**. The run falls under the
  lane's standing authority (worst ≤ $15).

## Consequence

- **FASTER**, with FAM's ON_auto PASS, licenses an allow-list pull request that turns Mixtral's folds on by default. The
  maintainer decides.
- **NO_GAIN** or **SLOWER**: Mixtral's folds stay off.
- **NOISY** or **VOID**: as registered, with a spare id.

## What this cannot say

Batches above one row (W16), other cards, prefill, the int4 route, and Mixtral's other checkpoints.

## Code

- `bench/p135/`: `p135_speed.py` (the box), `p135_reduce.py` (the rule), `p135_run.sh` (box side), `p135_drive.sh`
  (controller), `staged.sha256`. The files shared with FAM are at FAM's pinned bytes.
- `tests/test_p135_speed.py`: the self-tests; the census, folds and fp32 path at Mixtral's real depth; the swap's
  exactness; two alternating runners against each alone.
- `tests/test_p135_staged_pin.py`: the pin, the stage lists, the shared bytes, the constants and the guard.
