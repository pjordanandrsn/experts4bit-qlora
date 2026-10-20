# Cross-request routing clustering at sixteen requests: small, real, not a lever (reported only)

**The question:** at sixteen concurrent requests, do the requests decoded in the same step share experts more than they
would without that alignment? Fewer distinct experts per layer-step would mean fewer expert bytes read and fuller
16-row tiles.

**The null was written down and pushed before any null draw:** [`NULL.md`](NULL.md), commit `20218ed3`.
- The statistic: `D`, the distinct experts among a layer-step's 16 rows x top-8 picks. At 16 rows this is exactly
  K25's 16-row tile count.
- Null A (primary): each row's 128-step sequence circularly shifted by its own random offset, per layer. Null B: each
  row's sequence permuted.
- 1,000 draws each, seed 20261020. The bands were stated in advance, and nothing here rules.

**Data:** P153's committed router replay, `bench/p153/router-replay/rr.json.pt` (`3c59e72f`): S0 (rows 0-15, primary)
and S1 (rows 16-31). **Positive control:** `clustering.py` reproduces P153's published mean `D` (57.107 S0, 57.052 S1)
from the same file, or stops.

## The read

| stream | null | D observed | D null (2.5-97.5 %) | ratio (interval) | rows per tile, observed / null |
|---|---|---|---|---|---|
| **S0** | **A, circular shift** | 57.107 | 57.257 (57.172-57.344) | **0.9974** (0.9959-0.9989) | 2.241 / 2.236 |
| S0 | B, permutation | 57.107 | 57.255 (57.192-57.320) | 0.9974 (0.9963-0.9985) | 2.241 / 2.236 |
| S1 | A, circular shift | 57.052 | 57.418 (57.338-57.503) | 0.9936 (0.9922-0.9950) | 2.244 / 2.229 |
| S1 | B, permutation | 57.052 | 57.417 (57.349-57.488) | 0.9937 (0.9924-0.9948) | 2.244 / 2.229 |

- **Small, and real.** Same-step requests share experts slightly more than the nulls: 0.26 % fewer distinct experts
  on S0 and 0.64 % on S1, both intervals clear of 1. In the bands stated in advance this is "small", far from the 2 %
  that would earn a scheduling look.
- **Not autocorrelation.** Null A keeps each request's step-to-step autocorrelation and null B breaks it. The two agree
  to 0.0001, so the effect is the same-step alignment itself.
- **Per layer (null A):**
  - **S0:** 0.9899 (layer 29) to 1.0045 (layer 37); no single layer's interval clears 1.
  - **S1:** 0.9855 (layer 27) to 0.9996; four layers clear below 1.
  - No layer is clear below 0.98. Layers 27 and 29 are the lowest in both streams.
- **What dominates instead is tile fill.** A step's 128 row assignments land on about 57 of 128 experts, so each
  non-empty 16-row tile holds about 2.24 rows, about 14 % full. Cross-request overlap moves that fill by under 1 %. The
  half-empty-tile cost is a property of top-8 routing over 128 experts at sixteen rows, not of which requests share a
  step.

## Limits

- **Scope:** one prompt set (P109's sixteen), teacher-forced bf16 picks (P153's replay, argmax fidelity 0.941 and
  0.936), and one model, Qwen3-30B-A3B.
- **Position:** all sixteen rows share a decode position in each step, so a position effect counts as alignment here.
  Null A moves rows onto different positions, so its ratio includes that effect; it is named, not separated.
- **Width:** at other widths (B > 16) the tile count and the distinct count differ, and this read does not cover them.

## Reproduce

```
python bench/routing-clustering/clustering.py --replay bench/p153/router-replay/rr.json.pt --out clustering.json
```

It needs torch (only to read the `.pt` file) and numpy, and takes about 50 s on a laptop CPU. The output equals
[`clustering.json`](clustering.json).
