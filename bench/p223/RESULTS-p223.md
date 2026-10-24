# P223 results: both servers decode schedule-invariantly under the two knobs; the price is about 0.1 to 0.4 % at sixteen requests and about 3.5 to 3.9x at one (READ)

**Registration:** [PREREG-p223.md](PREREG-p223.md) (e4b#1955, Amendments 1 to 3), read once, as registered.
**Reading:** `p223-5090-1`, one RTX 5090 at 575 W, e4b `817acb94`, grouped-nf4-gemm v0.49.0, $1.003.
Records: [receipts/p223-5090-1/](receipts/p223-5090-1/README.md).

## The question

P149's served decode is not schedule-invariant: a request's tokens depend on which graph bucket each step replays, and
so on `max_new_tokens`. Two existing knobs removed that on P149's one-layer cut: one attention split at every bucket
(`E4B_ATTN_SPLIT_CAP=1`) and no bucket 1 (`E4B_PAGED_BUCKETS=2,4,8,16`). Do they make the full model's bf16-attention
server (S0) and int8-attention server (S1) schedule-invariant, and what do they cost in served decode throughput at one
and sixteen requests?

## The answer: READ

| step | result |
|---|---|
| VOID | no. The card was an RTX 5090 at cc 12.0 with power limit and enforced limit 575 W. Every engagement check held: the I arms ran split 1 at every bucket, the buckets (2, 4, 8, 16), zero bucket-1 replays, and every lone-row step in bucket 2 beside a pad row; the D arms ran the default plans (12, 12, 11, 5, 2 at buckets 1 to 16) with bucket-1 replays; D1 and I1 routed the int8 projections at their closed form (384 calls per captured bucket). Every record is present |
| NOISY | no. The W1 self-pairs read D0b/D0a 1.0000, I0b/I0a 1.0008, D1b/D1a 1.0011 and I1b/I1a 0.9995, all inside [0.985, 1.015] |
| **READ** | **S0: SCHEDULE_INVARIANT. S1: SCHEDULE_INVARIANT.** In every I process, every W16 and W1 row's 32-token stream equals the first 32 tokens of its 160-token stream (0 differing rows) |

**The price** (reported, never a bar), tok/s(I) / tok/s(D) over the adjacent pairs:

| server | W16 pairs | p16 | W1 pairs | p1 |
|---|---|---:|---|---:|
| S0 (bf16 attention) | I0a/D0a 0.9991, I0b/D0b 0.9991 | **0.9991** | I0a/D0a 0.2849, I0b/D0b 0.2852 | **0.2849** |
| S1 (int8 attention) | I1a/D1a 0.9984, I1b/D1b 0.9960 | **0.9960** | I1a/D1a 0.2572, I1b/D1b 0.2568 | **0.2568** |

At sixteen requests the knobs are nearly free: 320 to 328 tok/s in every arm. At one request they are not: S0 decodes
124.7 tok/s by default and 35.5 tok/s invariant, and S1 142.4 and 36.6. The W16 self-pairs, reported beside p16, read
D0b/D0a 0.9961, I0b/I0a 0.9961, D1b/D1a 1.0040 and I1b/I1a 1.0016.

**The default servers, reported beside the label:** neither is schedule-invariant.
- S0: W16 row 10 differs at step 29, in both D0 processes and all five reps.
- S1: W16 rows 8 and 12 differ, at steps 31 and 27, in both D1 processes.

P149's S0a differed in 1 of 16 rows and its S1a in 2, so the full model shows the same counts.

**The cross-W check** (reported only, not part of the label): P109's W1 prompt is W16's row 0. Under the knobs, W1's 160-token stream
**equals** W16 row 0's in every I process, although a lone row decodes in bucket 2 and the same row at W16 in bucket 16.
In every D process the two first differ at step 39.

## Predictions

| prediction | held? |
|---|---|
| W16, both servers: p16 in 0.97 to 1.00 | **yes** (0.9991, 0.9960) |
| W1, both servers: p1 in 0.60 to 0.95 | **no** (0.2849, 0.2568) |
| both servers SCHEDULE_INVARIANT | **yes** |
| cross-W under the knobs: equal (moderate confidence) | **yes**, in all four I processes |
| both D arms NOT invariant (a W16 row differs, 32 vs 160) | **yes** |

**The miss.** One request pays about 3.5x (S0) and 3.9x (S1), far beyond the predicted range. The read does not name
the cause, and two candidates remain:
- **The split cap.** By default a lone row's attention splits its KV over 12 programs (bucket 1's plan; bucket 2's is
  also 12), and the cap leaves it one program across 170 SMs. At sixteen requests bucket 16 already runs 2 splits, so
  the cap barely moves it. That fits both prices, but it is an inference.
- **The pad row.** A lone row decodes in bucket 2 beside a pad row, which may route experts the live row does not.

## What follows (as registered, never automatic)

- **Both servers are SCHEDULE_INVARIANT**, so I0 and I1 are the instrument for P149's owed routing-matched read. That is
  its own registration.
- **p1 is under 0.90 on both servers**, so the follow-up prices a cheaper invariant decode for one request: for
  example, a split count fixed by KV length rather than by row count, or a pad row that routes no experts. That is
  code, with its own PR and read, and the two candidates above are what it would separate.

## Run ids and spend

`p223-5090-1` was the lane's first and only reading box: no pre-create refusal and no host refusal. The lane is
answered. The shakedowns ran outside the lane's counts: `p223-shake-4` and `-5` on the one-layer cut, before the
reading.
- **`p223-shake-5`'s pin admitted a host with no recorded power cap.** Amendment 2 says a shake pins only hosts with a
  RECORDED cap of at least 575 W, so this was a deviation. The box measured 575 W enforced, so the shake ran at the
  reading's power.
- **This reading's pin** screened recorded caps of at least 575 W only, and unrecorded hosts were excluded.

## What this does not say

- Nothing about batch invariance in general. The cross-W check compares one prompt at W1 and W16, and it is
  reported only.
- Nothing about which of the knobs' two parts costs the W1 price; the read prices the pair.
- Nothing for other models, cards or stores, and nothing about quality. The knobs change the decode's arithmetic only
  inside the kernels' own tolerance, and this lane has no quality term.
- No default changes. The knobs stay opt-in.
