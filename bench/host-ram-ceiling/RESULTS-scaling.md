# The ratio scales with expert bytes — 2.56× on a 7B MoE, 6.40× on a 30B one

### 2026-08-13 · RTX A2000 12GB, sm_86 · torch 2.8.0+cu128 · **published wheels** e4b 0.17.3 / gnf4 0.9.0 · drivers [`runarm.sh`](runarm.sh) + [`cg.py`](cg.py) · raw [`ladder-qwen3-30b.jsonl`](ladder-qwen3-30b.jsonl)

**Evidence tier: `measured`.** Pre-registered in
[`PREREG-scaling.md`](PREREG-scaling.md), committed **before** the checkpoint was
downloaded. Three of its point predictions were wrong; the section that scores them is
below, unedited from what was registered.

## Why this exists

[`RESULTS-host-ram-ceiling.md`](RESULTS-host-ram-ceiling.md) measured **2.56×** on
OLMoE-1B-7B and stated plainly that scaling was *"the mechanism's prediction, not a
measurement, and nothing here measures it."* This measures it, on a model with **4.50×
the expert bytes**.

The mechanism's claim: the host path pins every expert of every layer, so it should track
total expert bytes; the arena path pins `hot_rows × row_stride`, so it should not.

## Method

Identical to the OLMoE run — same drivers, four steps, one seed, `docker --memory=N
--memory-swap=N` (swap-inclusive, positive-controlled in both directions), verdict from
`State.OOMKilled`, descend until two consecutive failures, report the bracket.

One parameter had to change, and that is itself a result — see **`hot_rows` is not
portable** below.

| | OLMoE-1B-7B-0924 | Qwen3-30B-A3B |
|---|---|---|
| layers × experts | 16 × 64 = 1024 | 48 × 128 = **6144** |
| top_k / vocab | 8 / 50304 | 8 / **151936** |
| arena (expert bytes) | 3.624 GB | **16.307 GB** |
| on-disk row | 3.539 MB | 2.654 MB |
| `hot_rows` used | 64 | 128 |

`hot_rows` = experts-per-layer in both, so the two runs are directly comparable.

## Results

| | host offload | arena `hot_rows=128` |
|---|---|---|
| **minimum host RAM to train** | **(23552, 24576] MiB = 24.70–25.77 GB** | **(3712, 3840] MiB = 3.89–4.03 GB** |
| steady RSS at `trained` (uncapped) | 24.87 GB | 4.08 GB |
| uncapped peak RSS | 43.69 GB | 4.36 GB |

**Ratio 6.40×**, bracketed 6.13–6.62× by the rungs either side.

### The headline cell

At **8192 MiB (8.59 GB)** — same model, seed, four steps, box and cap — the host path is
**OOM-killed (exit 137, `OOMKilled=true`)** and the arena path **trains to completion**.
On OLMoE that cell needed a 5 GiB cap to separate the arms; here an 8.59 GB machine
already cannot train the model any other way.

### Scaling

| | OLMoE | Qwen3-30B-A3B | factor |
|---|---|---|---|
| expert bytes | 3.624 GB | 16.307 GB | **×4.50** |
| host requirement | 6.17 GB | 25.77 GB | **×4.18** ← tracks expert bytes |
| arena requirement | 2.42 GB | 4.03 GB | **×1.66** ← does not |
| ratio | 2.56× | **6.40×** | |

The host arm's growth factor (4.18×) lands within 8% of the expert-byte growth (4.50×),
which is what "pins every expert" predicts. The arena arm grew far less, and most of that
growth is not the expert path at all — see the slope below.

## `hot_rows` is not portable — confirming a documented constraint

`hot_rows=64` — the value used for OLMoE — **refuses outright** on Qwen3:

```
ValueError: request of 97 unique rows exceeds hot_rows=64
```

This is the API behaving exactly as specified. `enable_nvme_train_residency`'s docstring
already states the hard floor — *"at least the number of unique experts one forward
routes… For a training batch of `T` tokens at top-`k` that approaches
`min(T*k, num_experts)`… Undersizing raises rather than thrashing."* For this batch that
is `min(384*8, 128) = 128`, and 97 distinct rows were observed. The prediction, the
refusal and the measurement agree; nothing needs changing.

What the run adds is the empirical point a caller cares about: **a `hot_rows` value does
not travel between models.** OLMoE has exactly 64 experts per layer, so its 64 silently
meant "all of them" and looked model-independent when it was really at the floor already.
Any wider MoE hits the refusal at the first training step. Sizing from the formula rather
than from a previous run's number is the portable habit.

## What a hot row actually costs in host RAM

Measured by re-running the whole ladder at `hot_rows=512`:

| `hot_rows` | requirement |
|---|---|
| 128 | (3712, 3840] MiB |
| 512 | (5376, 5632] MiB |

+384 rows costs **+(1536, 1920] MiB**, i.e. **4.19–5.24 MB per hot row** against an
on-disk row of **2.654 MB** — so a slot costs **1.58–1.98×** the bytes it holds. The cause
is not isolated here and no mechanism is offered for it.

Two consequences worth stating:

- **The fixed base dominates at the operating point.** At `hot_rows=128` the expert path
  accounts for roughly 0.5–0.7 GB of the ~3.9 GB requirement. The rest is the model's
  dense side and the runtime — which is why the arena requirement grew 1.66× between two
  models whose expert counts differ 6×: it was tracking the 30B dense side and a 3× larger
  vocabulary, not the experts.
- **Here the uncapped slope did not mislead.** The uncapped steady-RSS slope
  (4.13 MB/row) falls inside the capped bracket (4.19–5.24), unlike peak RSS, which
  overstates the host arm by 2.7× on OLMoE and by **1.70×** here (43.69 GB peak against a
  25.77 GB requirement). A differential of a reclaimable quantity can be sound even when
  its level is not.

## Scoring the pre-registration

Registered before the data, all three point predictions **missed**:

| registered | actual | verdict |
|---|---|---|
| host 18–21 GB | **24.70–25.77 GB** | too low |
| arena 2.2–3.0 GB, "should NOT scale" | **3.89–4.03 GB** | too low |
| ratio 7–9× | **6.40×** | too high |

The direction was right and the mechanism's qualitative claim held, but every quantity was
under-predicted because I sized both arms from OLMoE's non-expert baseline (~2.3 GB) and
Qwen3-30B's is far larger. Both arms being under-predicted is why the *ratio* came in below
its band while the *host* came in above its own.

The registered stop rule — *"if the arena arm's requirement exceeds 4 GB, stop and
investigate before reporting any ratio"* — **fired**, at 4.08 GB steady RSS. The
investigation is the `hot_rows=512` slope above: 6× the experts produced 1.74× the
requirement, and the expert path accounts for well under a fifth of it, so the defect the
rule guarded against (something scaling with expert *count*) is absent. The ratio is
reported only after that check.

## What this does and does not establish

**Does:** the ratio is a property of the model, not the feature, and it grows roughly with
expert bytes on the host side while the arena side stays near-flat. Two models, 4.5× apart
in expert bytes, on published wheels.

**Does not:** two points do not fix a curve; the host arm's 4.18× against 4.50× is
consistent with linearity but does not establish it. Nothing here measures a model whose
experts exceed host RAM on a *large* machine — Qwen3-30B-A3B still trains uncapped on this
box at 24.9 GB. Correctness testbed only: **no timing claim is made from this box**, and
the step times are not comparable across arms or models.

## Reproducing

As [`RESULTS-host-ram-ceiling.md`](RESULTS-host-ram-ceiling.md), with:

```bash
export E4B_MODEL=Qwen/Qwen3-30B-A3B
export E4B_ARENA=/work/arena/qwen3-30b-nf4.arena
./ladder.sh host  26624m 24576m 23552m 22528m
./ladder.sh arena  4608m  4096m  3840m  3712m 3584m -- --hot 128
```

The bake is 61.1 GB in and 16.3 GB out, ~4.5 min on an NVMe mirror.

`ladder-qwen3-30b.jsonl` predates the ledger's `extra_args` field, so rows killed by the
cap do not record their `hot_rows` — that gap is why the field now exists. The mapping used
for the brackets above is reconstructed from invocation order and asserted against every
*surviving* row, each of which does carry `hot`.
