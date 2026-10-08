# P121 — results: **LICENSED**. K25 (`E4B_NF4_GROUPED_SMALLM=auto`, the default) decodes Qwen3-30B-A3B NF4's served W16 step 1.57× as fast as the NF4 M-tile (`0`), within P110's quality bar on wikitext and c4val1 (one RTX 5090, 2026-10-08)

Registration: `bench/p121/PREREG-p121.md` (#1377, reviewed by the maintainer, merged `e96e4be`, the launch commit).
Issue: #846. Asked for from FAM0's inventory (#1368). The instrument and the continuity row are the maintainer's choices.

Code under test:
- e4b 0.50.0 at `e96e4be`;
- grouped-nf4-gemm 0.43.0 at `56f90e3`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`, its NF4 arena baked on the box;
- the default `serve_paged` server with `E4B_PAGED_MAX_SEQS=16` named, buckets 1–16, decode graphs on, and the fusion
  knobs and the decode GEMV at their defaults.

**Verdict by `p121_reduce.py`: `LICENSED`.** No VOID condition fired:
- the commits, revision and server are the registered ones;
- every bucket was captured;
- the captures took the registered kernel: K1 took K25 1,152 times and the M-tile at ≤ 256 rows never; K0 the reverse;
- every quality pass made exactly the registered calls: the OFF phase took the M-tile 390,144 times and K25 never; the
  ON phase took K25 73,152 times (2 × 48 layers × 127 steps × 3 groups × 2 texts) and the M-tile at ≤ 256 rows never;
- **W1's tokens are identical in both arms** (T == 1 runs the decode GEMV either way);
- the mutant failed the bar on both texts.

## The reading (`p121-5090-1`)

**Host:** one RTX 5090 (sm_120, 170 SMs, driver 595.71.05, **450 W** power limit, 3,090 MHz) on an AMD EPYC 7C13 (256
threads, 1,007 GiB RAM), Vast instance 54908429. The 450 W limit is below SC2e's 575 W board; only ratios within this
box are read.

**Cost:** $1.056. Teardown proven at 21:07:55Z.

**Timeline:**
- premise passed 20:37:28Z (11 passed, none skipped);
- fetch until 20:41:40Z;
- the four speed arms 20:43–20:49Z;
- quality OFF until 20:59:30Z, ON until 21:01:35Z;
- the continuity row until 21:07Z.

### Speed (decode tok/s from the slope, 32 and 160 new tokens, 3 timed passes; ABBA)

| arm | `E4B_NF4_GROUPED_SMALLM` | W16 tok/s | W16 ms/step | W1 tok/s | W1 ms/step |
|---|---|---:|---:|---:|---:|
| K0a | `0` | 490.7 | 32.61 | 135.2 | 7.39 |
| K1a | `auto` | 770.3 | 20.77 | 134.9 | 7.41 |
| K1b | `auto` | 772.7 | 20.71 | 135.0 | 7.41 |
| K0b | `0` | 491.2 | 32.57 | 135.1 | 7.40 |

- **g16 = 1.5697** (pairs 1.5697 and 1.573; geomean 1.5713). K25 takes the 16-row step from 32.6 ms to 20.7 ms.
- **W1 is the null it should be:** 0.9977 and 0.9989.
- **Self-pairs** are within 0.3 % (K0b/K0a 1.001 and 0.999; K1b/K1a 1.003 and 1.000).
- **Tokens:** K1's W16 tokens differ from K0's on 9 of 16 rows at 32 tokens and 14 of 16 at 160, as two arithmetics
  should. Each arm's own two runs, and each run's timed reps, match exactly.

### Quality (P115 Phase B's instrument at 16 windows a pass: every decode step 128 routed rows)

48 windows × 128 positions per text, scored against R (K0's arithmetic):

| text | ON bias (nats) | SE | spread | bar (bias / spread) | floor B / S | argmax agree | KL | perplexity R → ON | mutant bias |
|---|---:|---:|---:|---|---|---:|---:|---|---:|
| wikitext | **+0.00020** | 0.0026 | 0.0134 | 0.0115 / 0.0317 | 0.0015 / 0.0158 | 0.966 | 0.0083 | 8.9692 → 8.9710 (+0.0018) | +1.059 |
| c4val1 | **+0.00223** | 0.0017 | 0.0094 | 0.0116 / 0.0235 | 0.0016 / 0.0117 | 0.955 | 0.0076 | 16.3665 → 16.4031 (+0.0366) | +0.783 |

- **Both texts pass with room.** The floor draws are `half` and `chunk`; R repeated bit for bit, so `rep` is not a draw.
- **The perplexity moves are reported, not gated** (the maintainer's ruling).

**Continuity** (P96's arms through the same instrument at T == 1, 12 wikitext windows, no bar): K25 against the M-tile
reads **−0.0027 nats**, argmax agreement 0.965, perplexity −0.024. That is K25's arithmetic at the shape P96 read, now
on Qwen3's weights; `auto` never runs it at T == 1.

**Memory** (peak allocated, reported): 22.17 GiB in every arm. K1 − K0 is 0.000 GiB allocated and −0.002 GiB reserved.
The two kernels' workspaces do not differ measurably.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: g16 in [1.10, 1.70] | **held** (1.5697) |
| Q2: W1 pairs in [0.98, 1.02] | **held** (0.998, 0.999) |
| Q3: self-pairs in [0.98, 1.02] | **held** (0.999–1.003) |
| Q4: ON bias wikitext in ±0.004, c4val1 in ±0.006; wikitext spread ≤ 0.015; wikitext ppl within ±0.03 | **held** (+0.0002, +0.0022, 0.0134, +0.0018) |
| Q5: the mutant's bias ≥ +0.3 on both texts | **held** (+1.059, +0.783) |

## What it means (the registered consequence)

- **LICENSED:** K25's `auto` default is now read on Qwen3-30B-A3B at the served W16 step. It is 1.57× as fast as the
  NF4 M-tile there, within P110's bar on both texts. No code change.
- **The register row** is `e4b.serve.p121.k25-w16.qwen3.5090.2026-10-08`. STATUS's K25 row cites it.
- **Still unread**, as registered:
  - other row counts: W2–W8, and bucket 32 (256 routed rows) on today's 64-slot default;
  - other families;
  - T == 1, where the continuity row is reported only.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p121-prove-1` | OK, PROVED | $0.189 | Granite (16 windows of 32 positions): every arm, both quality phases, the continuity row; engagement exact; reducer LICENSED (not a reading) |
| `p121-5090-1` | **OK, LICENSED** | $1.056 | the reading |

**The lane cost $1.245**, inside its $3.00 ceiling.

**Receipts** are in `receipts/p121-5090-1/`, with `SHA256SUMS`:
- the four speed arms' records, `quality_off.json`, `quality_on.json`, `continuity_off.json`, `continuity_on.json`;
- `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`, `prompts.json`;
- the logs and the teardown proof.

R's reference log-probs stayed on the box. These committed copies are the public record of the reading.

**Re-derive:**
```bash
python3 bench/p121/p121_reduce.py --dir bench/p121/receipts/p121-5090-1 --out /tmp/v.json --e4b-sha e96e4be2eadca3427ad867ce83c744ce1b33988f
```
The output equals the committed `verdict.json` byte for byte on Python 3.9 (macOS) and 3.13 (Windows).
