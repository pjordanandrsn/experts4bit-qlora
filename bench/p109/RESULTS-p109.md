# P109 — results: **DIVERGENT**. On the default `serve_paged` server, bucketed decode graphs are 5.60× the eager default at 16 concurrent requests and 9.02× at one, and the replay is bit-identical to its own padded eager step; but the graph server's tokens leave the eager default's within 16 tokens on 7 of 16 rows, so the registered sanity bar fails and graphs stay opt-in

Registration: `bench/p109/PREREG-p109.md` (#988, `b27b73f`; Amendment 1 #989 `cab2ba5`, Amendment 2 #990 `3185b46`).
Issue: #770. Code under test:
- e4b 0.42.0 at `3185b46`;
- grouped-nf4-gemm 0.35.0 at `51a4916`;
- transformers 5.17.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`, NF4 experts baked on the box.

**Verdict by `p109_reduce.py`: `DIVERGENT`.** The rule's steps, in order:

| step | result |
|---|---|
| VOID | no. Commits, revision, prompts and lengths agree, and engagement is exact: G captured every bucket with device grouping on; E has no graphs and device grouping off; D has no graphs and device grouping on; P is `eager: capture=False` on every bucket with device grouping on |
| NOISY | no. Self-pairs E2/E1 0.954 (W16) and 0.941 (W1), G2/G1 1.023 and 1.001, all inside [0.93, 1.07] |
| FUNCTION_FAIL | no. **G1 ≡ G2 ≡ P1, bitwise**, on every row of both workloads at 32 and 160 tokens, and every timed rep of G and P digests the same |
| KEEP | no. S16 = **5.597** (bar 1.25), S1 = **9.020** (bar 0.97) |
| **DIVERGENT** | **yes. 9 of 16 W16 rows agree with E1 for 16 tokens (bar 12)** |

## The reading (`p109-5090-2`)

**Host:** one RTX 5090 (sm_120, driver 580.95.05) on an AMD EPYC 7C13 (256 threads, 818 GiB RAM), Vast instance
54064967 at $0.58/h. **Cost:** $0.2926. Teardown was proven at 21:46:43Z.

**Timeline (the box's own log):**
- install at 21:17:36Z;
- premise 7 passed, none skipped;
- fetch 21:18:28–21:23:09Z;
- bake and prompts until 21:25:27Z;
- six arms 21:25:27–21:45:44Z;
- reduce and TP_DONE at 21:45:44Z.

| arm | switch | W16 decode tok/s | W1 decode tok/s | W16 ms/step | W1 ms/step | load s | peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| E1 | none (the eager default) | 130.55 | 11.02 | 122.55 | 90.75 | 33.42 | 21.505 |
| G1 | `E4B_PAGED_GRAPHS=1` | 730.68 | 99.40 | 21.90 | 10.06 | 36.71 | 21.560 |
| G2 | `E4B_PAGED_GRAPHS=1` | 747.66 | 99.48 | 21.40 | 10.05 | 33.46 | 21.560 |
| E2 | none | 124.49 | 10.37 | 128.52 | 96.41 | 31.94 | 21.505 |
| D1 | eager, device grouping forced | 148.68 | 10.65 | 107.61 | 93.91 | 35.12 | 21.505 |
| P1 | graphs on, every bucket eager (`capture=False`) | 191.62 | 12.51 | 83.50 | 79.94 | 33.42 | 21.512 |

`decode tok/s` is p37's slope from the fastest of 3 timed passes at 32 and 160 new tokens. Every request runs to its
length.

- **The eager default is host-bound.** One request costs 91–96 ms per token (10–11 tok/s), and a 16-row step 123–129 ms.
  Under graphs they cost 10.1 ms and 21.4–21.9 ms.
- **Graphs cost +0.055 GiB of peak memory and +3.3 s of load** (the one-time capture).
- **The replay is its own padded eager function, exactly.** This is the second confirmation; `p109-prove-3` showed the
  same on Granite. P101 had read the same on Qwen3.6 against the same oracle.

## What the divergence is made of

The registered sanity pair, E against G, changes three things at once: the grouping (host to device), the padding
(unpadded to bucket-padded) and the replay. The other arms take them apart. Here is W16 at 160 tokens, from the arm
receipts:

| pair | what differs | rows identical | rows agreeing ≥ 16 tokens | median first divergence |
|---|---|---:|---:|---:|
| E1 vs D1 | grouping only (both eager, unpadded) | 2 | 10 | 42.5 |
| D1 vs P1 | padding only (both eager, device grouping) | 5 | 12 | 107 |
| E1 vs P1 | grouping and padding (both eager) | 1 | 9 | 23.5 |
| E1 vs G1 | the registered sanity pair | 1 | 9 | 23.5 |

W1, a single request, is identical across all six arms.

- **The graph replay adds no divergence.** E1-vs-G1 is E1-vs-P1, row for row.
- **Grouping alone fails the bar.** Today's eager default and eager decode with device grouping, neither one using
  graphs, agree for 16 tokens on only 10 of 16 rows.
- **The bar's premise is refuted on this model.** It held that "a broken function diverges at once; benign reordering
  of bf16 arithmetic does not, on most rows". Two eager arithmetic orders of e4b's own diverge within 16 tokens on 6 of
  16 rows. What the bar measured is Qwen3-30B-A3B's greedy-decoding sensitivity to bf16 order, not the graph path.
- **What no arm here can say:** whether device grouping's arithmetic, with or without padding, is better or worse than
  host grouping's. That is a quality question for a teacher-forced instrument (a K8-style NLL, or P108's floor method).
  A token-agreement bar cannot answer it.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: S16 between 2 and 15 | **yes** (5.597) |
| Q2: S1 between 1.5 and 8 | **no: 9.020**, above the range. The eager single-request step is more host-bound on this EPYC host than the band allowed |
| Q3 (Amendment 2): G1 ≡ G2 ≡ P1 on every row | **yes** |
| Q3 (registered, kept as reported): G ≡ D | **no**, as Amendment 2 expected: D differs from P on 11 of 16 W16 rows |
| Q4: E1 ≡ E2 | **yes** |
| Q5: E and G differ on some W16 rows, median first divergence ≥ 32, sanity bar holds | differ: **yes** (15 of 16). Median: **no** (23.5). Sanity: **no** (9 of 16) |
| Q6: G's peak ≤ E's + 0.5 GiB, load ≤ E's + 20 s | **yes** (+0.055 GiB, +3.3 s) |
| Q7: DEFAULT_GRAPHS | **no: DIVERGENT** |

## The registered consequence (DIVERGENT)

- **Graphs stay opt-in.** `E4B_PAGED_GRAPHS` keeps its default (off); the drafted `auto` default does not ship.
- **The divergence is the next investigation.** This read already localizes it to eager-side arithmetic (the grouping
  and the padding), not the replay. The question that would license the default is quality: is the device-grouped,
  bucket-padded arithmetic at least as good as host grouping's on a teacher-forced NLL? That question needs its own
  lane, with a floor in the manner of P108.
- **`docs/SERVING.md`** now states the measured ratio, and states that every registered serving number is the graph
  path.
- **A measured register row** records S16: `e4b.serve.p109.decode-graphs-vs-eager-default.qwen3.5090.2026-10-03`. The
  registration named a row only for DEFAULT_GRAPHS; this one is a measurement row, not that one.
- **#770 stays open** with this reading.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p109-prove-1` | HARNESS_ERROR | $0.0207 | the proof's time-left checks were sized for the reading (Amendment 1) |
| `p109-prove-2` | OK, PROVED | $0.0570 | Granite: G ≢ D, the unpadded step, on 7 of 16 rows (Amendment 2) |
| `p109-5090-1` | HARNESS_ERROR | $0.0150 | stopped 27 s into staging, before any data, once the wrong oracle was seen |
| `p109-prove-3` | OK, PROVED | $0.0746 | Granite: G ≡ P bitwise, D ≢ P on the same 7 rows |
| `p109-5090-2` | **OK, DIVERGENT** | $0.2926 | the reading |

**The lane cost $0.4599**, inside its $3.50 ceiling.

**Two lessons:**
- A proof that shares the reading's runner needs its own time checks.
- An identity read on bucket-sized active sets (P82, B771b: 16, 8, 4, 2, 1 rows) says nothing about padded steps, so a
  bucketed path's oracle is its padded eager step. P101 had used that oracle.

**Receipts** are in `receipts/p109-5090-2/`, with `SHA256SUMS`:
- the six arm receipts, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`;
- the install, premise, fetch, bake, prompts and arm logs;
- the teardown proof.

The launcher's receipts and ledger rows are in the receipt store: adertha-receipts `5809721` (`p109-prove-1`),
`45517df` (`p109-prove-2`), `af87536` (`p109-5090-1`), `a9650bc` (`p109-prove-3`) and `b987e65` (`p109-5090-2`).
