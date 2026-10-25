# P130 — results: **P1_LICENSED P2_LICENSED**. On SC2e's int4 stack, one RTX 5090 (2026-10-10):
- `E4B_PREFILL_LEAN_DISPATCH` (P2) is bit-identical on every window and no slower: ×0.968 [0.966, 0.970] on the served first-chunk prefill graph.
- `E4B_FUSE_PREFILL_GLUE` (P1) is ×1.118 [1.116, 1.119] faster, and at parity on P117's teacher-forced instrument: bias −0.0035 nats.

Registration: `bench/p130/PREREG-p130.md` (#1588, merged `afc56ba`) with Amendment 1 (#1590, `7a55c25`). The launch
commit `b14f593` also carries the driver's start probe (#1595). Issue: #846. This lane is not TC1's predictions P130–P133.

Code under test:
- e4b 0.53.0 at `b14f593`, with #1583's two knobs (default `0`);
- grouped-nf4-gemm 0.45.0 at `724ccc4`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, huggingface_hub 1.33.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`: SC2e's served stack (int4 experts on 48 layers, int4 RTN attention, the three T=1
  folds, fused q/k/v), NF4 arena baked on the box.

**Verdict by `p130_reduce.py`: `P1_LICENSED P2_LICENSED`.** No VOID condition fired, and no knob was held:
- the commits, the revision, every process's P1 setting (its record and glue round 1's fold report) and Phase B role are
  the registered ones; all eight processes read the same windows;
- **engagement is exact.** Every capture ran five forwards, with:
  - route `int4_k19|gt256` × 240 (5 × 48 layers);
  - K19 `chained|gather|gt256` (`p2_off`) or `chained|lean|gt256` (`p2_on`) × 240;
  - prefill folds `{norm 245, layer 240, attention 240}` with P1 set and none without it;
  - 192 timed replays per graph per process;
- **no capture was refused**, under either knob;
- **FUNCTION:** P2's replay was `torch.equal` to the default's in the logits and every layer's staged K and V, on every
  speed window of every process;
- **DETERMINISM:** the default's output digests agree across the processes of each P1 setting;
- **NOISY did not fire:** within each P1 setting the `p2_off` medians span ×1.0005 (P1 unset) and ×1.0013 (P1 set);
- Phase B passed every integrity check, and `mutant_scale` fails the bar.

## The reading (`p130-5090-1`)

**Host:** one RTX 5090 (sm_120, driver 595.91.07) on an AMD Ryzen 9 9950X (32 threads, 123 GiB RAM), Vast instance
redacted-aa4dfbd0. **Cost:** $1.094. Teardown proven at 18:15:54Z.

**Timeline:**
- pre-quote gates on the controller: the driver's dry run, and STOP-0's fetch gate passed;
- the lane started 17:25:14Z on its first start handshake;
- premise: `tests/test_prefill_graph_gpu.py`, 9 passed, none skipped (17:27:36Z);
- fetch under the watchdog until 17:55:43Z (unauthenticated, about 28 min);
- bake until 17:56:50Z;
- the eight processes (each load about 90 s with the int4 repack) until 18:15:49Z.

### Phase A — the served first-chunk prefill graph (T = 512), median ms per replay

| process | P1 | `p2_off` | `p2_on` | P2 ratio | eager `p2_off` / `p2_on` (reported) |
|---:|:---:|---:|---:|---:|---:|
| 1 | unset | 38.173 | 37.028 | 0.9700 | 47.53 / 46.68 |
| 2 | **set** | 34.184 | 33.025 | 0.9661 | 41.28 / 40.25 |
| 3 | **set** | 34.159 | 32.995 | 0.9659 | 41.48 / 40.74 |
| 4 | unset | 38.172 | 37.031 | 0.9701 | 46.58 / 45.65 |
| 5 | unset | 38.170 | 37.032 | 0.9702 | 47.17 / 46.17 |
| 6 | **set** | 34.146 | 32.984 | 0.9660 | 42.05 / 41.14 |
| 7 | **set** | 34.140 | 32.981 | 0.9661 | 41.11 / 40.33 |
| 8 | unset | 38.190 | 37.042 | 0.9699 | 47.32 / 46.34 |

- **P2 (equivalence, margin ×1.01):** the mean ratio is **0.9680**, with a 95 % t-interval of **[0.9662, 0.9698]** over the
  8 processes. The upper end clears the margin with room to spare: P2 is no slower, and in fact faster.
- **P1 (gain, threshold ×1.03):** pair gains 1.1167 (1/2), 1.1175 (4/3), 1.1179 (5/6) and 1.1186 (8/7). The mean is
  **1.1177**, with a 95 % t-interval of **[1.1164, 1.1190]**: FASTER.
- **Savings against the glue map's ceilings** (`bench/prefill-glue/DESIGN.md`):

  | knobs | saving per replay | ceiling | share of ceiling |
  |---|---:|---:|---:|
  | P1 | 4.02 ms | 5.47 ms | 73 % |
  | P2 | 1.15 ms | 1.34 ms | 86 % |
  | both (P1 set + P2 on vs the default) | 5.18 ms | 6.80 ms | 76 % |

  Together the two knobs take the 512-token first-chunk forward from 38.17 to 32.99 ms.
- **Reported, never gated.** These eager figures matter because a later chunk runs eagerly:
  - the same forward run eagerly (wall, synced) takes 46.6–47.5 ms at the default, so the graph saves about 9 ms of
    launch overhead;
  - P1 cuts the eager forward by about 6 ms (47.2 to 41.5 ms), and P2 cuts it by about 1 ms more;
  - the prefill graph's private pool is 448 MiB at the default, 446 with P1, and 428 with P2 (P2 drops the 4,096-row
    gather buffer).

### Phase B — P1's quality (P117's instrument; 64 wikitext windows, 512 prompt tokens, 128 teacher-forced positions)

| arm | what it is | bias vs R (nats) | spread | se | max \|d\| | mean KL vs R | argmax agree |
|---|---|---:|---:|---:|---:|---:|---:|
| rep | R again | +0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 1.0000 |
| half (floor) | buckets 1–8 | −0.00251 | 0.01449 | 0.00224 | 0.04477 | 1.1e-2 | 0.9583 |
| chunk (floor) | prompts prefilled in 256-token chunks | −0.00066 | 0.01478 | 0.00246 | 0.07201 | 1.3e-2 | 0.9537 |
| rev (floor) | windows bound and decoded in reverse | +0.00000 | 0.00000 | 0.00000 | 0.00000 | 0 | 1.0000 |
| **P1** | **R's buckets on the P1 model (process 2)** | **−0.00347** | **0.01441** | 0.00218 | 0.05255 | 1.2e-2 | 0.9557 |
| mutant_scale | R with the decode softmax scale halved | +1.14657 | 1.14657 | 0.03653 | 2.03586 | 1.27 | 0.5762 |

R's mean NLL is 2.1787 over the 64 windows (1.192 to 3.168).

- **The bar is P117's, unchanged:** B_floor 0.00251 and S_floor 0.01478, so bias ≤ 0.01251 and spread ≤ 0.02956. P1's
  bias (−0.00347) and spread (0.01441) are well inside it: **AT_PARITY**.
- **P1 sits inside the floor's own class.** Its bias is about 1.6 standard errors from zero, on the better side, and its
  spread and KL match the neutral perturbations'. P1 changes only the prefill's rounding (decode rows here are at most
  16), so it reaches the scored positions only through the K/V its prompt wrote and the first position's logits.
- **The floor reproduces P117's exactly** (`half`, `chunk`, `rep`, `rev`): the same instrument, the same windows and
  the same R arithmetic, read on another box two days later.
- **The instrument can fail:** a halved decode scale reads +1.147 nats.

### The windows' corpus (Amendment 1, reported)

The fetch gate resolved wikitext-2-raw-v1 at `main` = `b08601e`. Every process read that commit, both in `datasets`'
arrow cache and in the hub snapshot: `corpus.same` is true.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| Q1: engagement exact; no capture refused | **yes** |
| Q2: FUNCTION and DETERMINISM hold | **yes** (16 windows × 8 processes, 0 differ; digests agree) |
| Q3: P1 saves 2.5–4.5 ms (ceiling 5.47) | **yes**: 4.02 ms |
| Q4: P2 saves 0.5–1.3 ms (ceiling 1.34) | **yes**: 1.15 ms |
| Q5: together 3.0–5.5 ms (ceiling 6.80) | **yes**: 5.18 ms |
| Q6: NOISY does not fire; spans ≤ ×1.01 | **yes**: ×1.0005 and ×1.0013 |
| Q7: P1 bias in [−0.004, +0.004], spread ≤ 0.016; R repeats; floor B ≤ 0.003, S in [0.008, 0.020]; mutant > +0.3 | **yes** on all: −0.0035, 0.0144; R repeats; B 0.0025, S 0.0148; +1.147 |
| Q8: `P1_LICENSED P2_LICENSED` | **yes** |
| Q9: each process ≤ 4 min; the eight ≤ 35 min; peak ≤ 28 GiB | the eight **yes** (19 min in all) and peak **yes** (21.3 GB). Each process: about 2 min, but process 1 took about 5.6 min (its speed phase plus Phase B's six passes over 64 windows), so **no** for process 1 |

## The registered consequence

Each flip is its own small PR after this read. Each carries a $0 A2000 GPU smoke of the flipped default, which the
maintainer requires before merge, and ships in the next release wave:
- **P2:** `E4B_PREFILL_LEAN_DISPATCH` defaults to `1`, and `0` restores the gather.
  - Register row `e4b.serve.p130.prefill-lean-dispatch.qwen3.5090.2026-10-10`: ratio 0.968 [0.966, 0.970], and bitwise.
  - Scope: the int4 K19 prefill route, read on Qwen3-30B-A3B. Other families on that route ride its bitwise contract.
- **P1:** `E4B_FUSE_PREFILL_GLUE` defaults to `1` on `qwen3_moe` only, and `0` restores HF's prefill chain.
  - Register rows `e4b.serve.p130.prefill-glue-speed.qwen3.5090.2026-10-10` (×1.118 [1.116, 1.119]) and
    `e4b.serve.p130.prefill-glue-quality.qwen3.5090.2026-10-10` (bias −0.0035 nats).
  - Other families keep `0` until a read of their own, and the PR lists per family what was read.

The speed figures are this read's own: the first-chunk prefill graph at T = 512 on one RTX 5090 host. Nothing here is a
TTFT or TPOT figure under load; that is an SC lane's to read.

## What it took

| run | status | cost | note |
|---|---|---:|---|
| `p130-prove-1` | HARNESS_ERROR | $0.001 | the box's ssh proxy closed the lane-start connection after its banner. Recovery is now #1595's nonce probe |
| `p130-prove-2` | NOT_RUN | $0.005 | the instance stayed offline (stuck loading) for 600 s; the pre-flight refused |
| `p130-prove-3` | OK, PROVED | $0.861 | the whole box at the proof's sizes, from #1595's merge, on a track-record host; its verdict (not a reading) matched |
| `p130-5090-1` | **OK, P1_LICENSED P2_LICENSED** | $1.094 | the reading |

**The lane cost $1.961**, inside its $5.50 ceiling. $0.006 went to the two non-verdicts. Both were operational, not
scientific; the maintainer's two-strikes review accepted the fixes. Every rental was preceded by the fetch gate.

**Receipts** are in `receipts/p130-5090-1/`, with `SHA256SUMS`:
- `proc_1.json` … `proc_8.json`: every replay's ms, the FUNCTION records and digests, the per-capture counters, and
  Phase B's per-window scores and engagement;
- `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- `p130_fetch_gate.json`, the launch's STOP-0 report;
- the install, premise, fetch, bake and per-process logs;
- the teardown proof.

R's fp32 log-probs (about 5 GB) stayed on the box, as registered. The launcher's receipts and ledger rows for all four
runs are in the private receipt store.

## Reproduce

```
python bench/p130/p130_reduce.py --dir bench/p130/receipts/p130-5090-1 --out verdict.json \
  --e4b-sha b14f59325bc0c21d66a6bb2f17eb9f2609a6cd9c --gate-corpus b08601e04326c79dfdd32d625aee71d232d685c3
```

This reproduces the verdict, every gate, ratio, interval and prediction. A few aggregate floats (standard errors, mean
KLs) can differ in the last digit or two between the box's Python 3.11 and Python ≥ 3.12, whose built-in `sum()` of
floats is compensated. The maintainer's independent re-derivation found 7 of 135 fields differing at ≤ 7e-16, none
deciding.
