# Training-side box anchor

Serving has `kernel/decode_anchor.py` (gnf4 M2: 7.37 ms ±4.2%). Training had
nothing — so a slow **box** and a slow **build** were indistinguishable.

**Why this exists.** 2026-08-26/27, identical config and identical stack:
**3.7 s/step on one RTX 5090, 6.1 s/step on another — 1.65×.** Both ran
`TRAIN_VRAM_FRAC=1.0` (all experts VRAM-resident, no PCIe expert traffic), so
transfer bandwidth does not explain it. TR1's census found this path
launch-bound (2.92M launches/step at 8–11% GPU busy).

## Usage

```
python train_anchor.py                      # writes /root/anchor.json
python train_anchor_gate.py anchor.json     # class the box
python train_anchor_gate.py anchor.json pcie-full/launch-fast   # enforce a class
python train_anchor_gate.py --calibrate a.json b.json c.json
```

Exit 0 accepted, exit 3 REFUSED. No model download — a gate nobody can afford
to run is a gate nobody runs.

## What it measures, and why three probes

| probe | catches | dispersion across 3 boxes |
|---|---|---|
| `flops` | dense bf16 GEMM, raw SM throughput | **1.025×** |
| `launch` | 20k trivial kernels, host+driver launch rate | **1.291×** |
| `h2d` | pinned host→device bandwidth | **1.945×** |

**A FLOPs-only gate is useless here.** It is the tightest probe (2.5%) and the
one that does *not* discriminate: it would have passed both boxes that later
differed 1.65× in training step time.

## The population is BIMODAL, so there is no single band

Calibrated across 3 distinct machines (9105, 137831, 137833):

| machine | power | launches/s | h2d GB/s |
|---|---|---|---|
| 137831 | 525 W | 128,453 | 25.95 |
| 137833 | 525 W | 129,370 | 25.97 |
| 9105 | 600 W | 100,226 | 13.35 |

The two 525 W boxes agree to **0.7% / 0.1%**. The third is 0.78× on launches
with **half** the PCIe bandwidth. `dispersion/2` on h2d would give a ±49% band,
which accepts everything.

So `launch` and `h2d` are **class labels**, not tolerance bands. A receipt
records the class it was measured on; comparing across classes is invalid.
Only `flops` is tight enough to band (±5% vs 2.5% observed).

M2 learned the mirror of this on the serving side — a ±3% gate over an
8.5%-dispersed class. A band narrower than its population is worse than none;
a band wider than the modes it separates is worse still.

## Self-consistency comes first

Each probe repeats 5×; a box whose own repeats disagree by >3% is refused on
its own terms. That check needs no reference constant, so it worked before
calibration existed. The 600 W outlier failed it (launch spread **1.116**) as
well as failing on class — **unstable and slow travel together**.
