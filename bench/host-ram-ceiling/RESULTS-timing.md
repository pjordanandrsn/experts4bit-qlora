# What the arena costs in time — the load saving travels, the step cost does not

### 2026-08-13 · RTX 3090 (Ampere sm_86) **and** L40S (Ada sm_89), both SECURE on-demand · torch 2.8.0+cu128 · **published wheels** e4b 0.17.4 / gnf4 0.9.0 · drivers [`timing.py`](timing.py) + [`rounds.sh`](rounds.sh) · raw [`timing-3090.jsonl`](timing-3090.jsonl) / [`timing-l40s.jsonl`](timing-l40s.jsonl)

**Evidence tier: `measured`.** Pre-registered in [`PREREG-timing.md`](PREREG-timing.md)
before the first timed run, and amended once — [`PREREG-timing-AMENDMENT-1.md`](PREREG-timing-AMENDMENT-1.md) —
after the original gate proved mis-specified and **before** any re-run. Both are scored
below, including where they were wrong.

## Why two cards

[`feedback-falsified-component-rides-along`] says no timing claim ships on one machine's
evidence. That rule paid for itself here: **half of what the first card measured does not
travel**, and publishing off the 3090 alone would have understated the arena's step cost by
nearly 2× for anyone on a faster GPU.

## Results

Paired within each round, `host_self` (the same host config timed twice per round) as the
control. A ratio is reported only when its per-round range is **disjoint** from the
control's — all eight below are.

| card | model | **load** host/arena | **step** arena/host | control (step) |
|---|---|---|---|---|
| 3090 / Ampere | OLMoE-1B-7B | **3.39×** [3.05–3.75] | 1.331 [1.309–1.423] | 0.982 [0.948–1.144] |
| 3090 / Ampere | Qwen3-30B-A3B | **6.76×** [6.32–7.18] | 1.238 [1.117–1.299] | 1.013 [0.921–1.039] |
| L40S / Ada | OLMoE-1B-7B | **3.12×** [2.95–3.43] | **2.248** [2.224–2.322] | 1.005 [0.940–1.030] |
| L40S / Ada | Qwen3-30B-A3B | **5.87×** [5.63–5.96] | **1.708** [1.671–1.765] | 0.992 [0.960–1.027] |

### The load saving travels

**3.1–3.4× on OLMoE and 5.9–6.8× on Qwen3-30B, on both architectures.** The host path
fuses and quantizes every expert at load; the arena path reads a baked file. The gap grows
with expert count, which is what "quantize every expert" predicts.

This is the number worth quoting, and the one nobody had measured — it fell out of runs
that were not timing it.

### The step cost does not travel, and it is worse on faster hardware

| | host step | arena step |
|---|---|---|
| OLMoE, 3090 → L40S | 1.938 → **0.977 s** (2.0× faster) | 2.667 → **2.248 s** (1.19× faster) |
| Qwen3, 3090 → L40S | 9.464 → **4.858 s** (1.95× faster) | 11.661 → **8.283 s** (1.41× faster) |

The host arm nearly halves its step time on the faster card. The arena arm barely moves,
because part of every step is an NVMe read and the disk does not care which GPU you bought.
So the ratio **grows** with GPU speed: 1.331 → 2.248 on OLMoE, 1.238 → 1.708 on Qwen3.

**Quote the step cost with the card attached, or not at all.** "The arena costs ~1.3× per
step" is true of a 3090 and wrong by ~1.7× on an L40S.

### `hot_rows` above the routing floor buys nothing

Swept on the 3090, Qwen3-30B, paired:

| `hot_rows` | step vs host | load vs host |
|---|---|---|
| 128 (the routing floor) | 1.238 [1.117–1.299] | 0.148 |
| 384 | 1.188 [1.119–1.213] | 0.149 |
| 1024 | 1.281 [1.093–1.381] | **0.216** |

The three step ranges **overlap heavily**, so they are not distinguishable from each other
— raising residency does not resolvably change step time. Load time at 1024 *is*
resolvably worse (staging 8× the rows costs real seconds), and it costs ~8× the pinned RAM.

**So "size `hot_rows` as large as free RAM allows" is the wrong advice.** Size to the
routing floor: more than that costs load time and memory and returns nothing measurable.

### The protocol difference does not explain it

The two cards ran different protocols, and card 2 dropped 4 warmup steps against card 1's
2. Card 1's first scored step was visibly inflated, so the extra warmup alone would push the
ratio in exactly the direction being attributed to the card. That confound has to be bounded,
not waved at.

Recomputing card 1's paired ratios while discarding two further leading steps — imitating
card 2's warmup discipline on card 1's own data:

| | card 1 as measured | card 1, warmup-4 equivalent | card 2 |
|---|---|---|---|
| OLMoE | 1.377 | **1.384** | 2.293 |
| Qwen3-30B | 1.200 | **1.263** | 1.716 |

The protocol accounts for **0.007** of the OLMoE gap and **0.063** of the Qwen3 gap; the
remaining **0.91** and **0.45** are the card. (These recomputed figures differ in the third
digit from the table above because they take the median over the retained steps directly
rather than the per-run median already stored — same data, and the comparison is
like-for-like within this check.)

## Scoring the pre-registration

| registered | actual | verdict |
|---|---|---|
| T1 load, OLMoE **2–6×** | 3.39 / 3.12 | **hit**, both cards |
| T1 load, Qwen3 **4–15×** | 6.76 / 5.87 | **hit**, both cards |
| T1 "below 1.5× → do not publish" | 3.1–6.8× | clears easily |
| T2 step, OLMoE **1.1–1.8×** | 1.331 (Ampere) / **2.248 (Ada)** | **half miss** — Ada outside the band |
| T3 step falls monotonically with `hot_rows` | no resolvable difference | **miss** |

The T1 prediction deliberately sat *below* the QNAP's 6.3×/19.6× on the reasoning that
quantize-at-load parallelises and the pod has 256 cores against 12. That held.

The T2 miss is the interesting one: the band was written as if the ratio were a property of
the software. It is a property of the **pair** — software and card — which is precisely what
one card could not have shown.

### Scoring the amendment

The original gate — *"report a ratio only if the control spread is within ±8%"* — used
control spread as a proxy for precision and applied it regardless of effect size, and so
called a **239%** load effect "not resolvable" because a control moved 8.4%. The amendment
replaced it with **effect range disjoint from control range** and raised precision from 3
rounds × 6 steps to 5 × 12.

Both parts worked. Control spread fell from **11.6–19.9%** on the first card to
**6.7–9.0%** on the second — tight enough to pass even the original gate on Qwen3. And
every effect above is disjoint from its control.

The amendment also made a pre-committed prediction for the re-run: load 3.0–3.8× (**hit**,
3.12), step 1.25–1.45× (**miss**, 2.248), control under 10% on step (**hit**, 9.0%). The
step miss is the same mistake as T2 — the prediction was written for a re-run on the *same*
card and the re-run happened on a *different* one.

### A finding that did not survive

Round 1 of the `hot_rows` sweep showed 1.299 / 1.188 / 1.381 across 128 / 384 / 1024 — a
clean U-shape — and `ColdTier._victim` does a linear scan over every slot per eviction,
which would explain it. Three rounds dissolved it: the ranges overlap and the ordering is
not resolved. **The U-shape and its mechanism are withdrawn.** One round produced a
plausible story with code to back it; the control is what said no.

## What this does and does not establish

**Does:** the load saving is real, large, and portable across two architectures on
published wheels. The step cost is real and resolvable, and it is a function of the card.

**Does not:** absolute times are these pods'. Two cards are not a survey — both are
NVIDIA, both torch 2.8.0+cu128, and nothing here speaks to AMD, older CUDA, or a slower
disk. The `hot_rows` sweep ran on one card only. And nothing was measured **below** the
routing floor, so what spilling would cost in time is still unknown.

## The two ledgers were taken under different protocols

`timing-3090.jsonl` (32 rows, both models) is the **original** protocol: 3 scored rounds of
6 scored steps. `timing-l40s.jsonl` (36 rows) is the **amended** one: 5 scored rounds of 12.
That difference is the amendment doing its job, and it is why the second card's controls are
tighter — it is not a difference between the cards themselves. Every ratio in the table
above is computed the same way from whichever rows exist, and each is reported with its own
control alongside so a reader can judge it without taking the protocol on trust.

Row counts differ from rounds × arms because round 0 is a dropped warmup and the 3090's
Qwen3 set carried five arms (the `hot_rows` sweep) against the L40S's three.

## Cost

$1.30 of rented compute total: 3090 at $0.50/hr (~$0.58), L40S at $0.99/hr (~$0.60), plus
~$0.12 on a 4090 that reached `RUNNING` with no public IP and was torn down. Rates read off
`costPerHr` after create, never the listing. External teardown backstops armed on a
separate host before each run; both pods verified gone via the account listing, not the
`DELETE` status.
