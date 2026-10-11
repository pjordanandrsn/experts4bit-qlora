# Q235 Amendment 3: the depth test on Qwen3-30B-A3B (receipts)

**When and where.** 2026-10-10, 22:42–23:58 UTC, on one RTX A2000 12 GB (sm_86), correctness only, $0.

**Why.** Gate 1 set and checked the fidelity bands on a 4-layer slice. A full-depth test was registered before any number existed:
are the full-state bands, and a per-layer contribution statistic, able to see one wrong layer deep in a model?

**What ran.** e4b's NF4 training forward (experts in pinned host RAM, the fused kernels) at 450b07ff, `q235_box.py --repeat`, over
all 48 decoder layers on the calibration tokens (`../calibration-30b/`: 8 rows, 1,014 real tokens), in three fresh processes:
- the box, unplanted;
- the routing plant at layer 24 (`--plant routing`);
- the epilogue plant at layer 24 (`--plant epilogue`).

Each is read against Amendment 1's 48-layer layer-streamed bf16 reference.

**The reading lines,** registered before the runs:
1. **Full state, Amendment 1's bands** (median ≤ 0.25, max ≤ 0.50, minimum router overlap ≥ 0.80). The line fails iff either plant
   does not fail its registered must-fail line, and then the full-state bands are void for any deep box.
2. **Contribution,** u_i = ‖d_run,i − d_ref,i‖ / ‖d_ref,i‖ with d_i = h_i − h_(i−1), over real tokens, in fp32.
   - (2a) The primary line: it separates iff both plants' u_24 ≥ 2 × the box's u_24.
   - (2b) Reported: each plant's u_24 against 2 × the box's largest u_i over layers 4–44.

**The reader** (`depth_read.py`) was fixed before the numbers, at sha256 `0a8e2b3e…`, and applies those lines mechanically. The copy
here differs from it only in its docstring's first line (sha256 `61577986…`). Its output is `depth_read.json` (sha256 `6bf1f2fe…`). A
second implementation computed the same quantities blind, before reading these, and a third re-derived them on CPU. All three agree
on every quantity to three decimals. The one difference is the median definition, below.

| run | full state: median / max | full state at L24 | min overlap | u_24 | loss gap vs bf16 (1.8647) |
|---|---|---|---|---|---|
| box | 0.107 / 0.204 | 0.107 | 0.913 | 0.346 | −0.0055 |
| routing plant | 0.112 / 0.241 | 0.112 | 0.890 | 1.063 | +0.0440 |
| epilogue plant | 0.2555 / 0.371 | 0.287 | 0.834 | 8.505 | +0.1902 |

- **Line 1 fails.** The routing plant sits inside all three bands. The epilogue plant's maximum, 0.371, does not reach its must-fail
  line (> 0.50). Under Amendment 1's void rule the full-state bands are therefore void for any deep box.
- **The epilogue median.** `q235_compare` takes the upper middle of the 48 layers, 0.2555, which is outside 0.25. Averaging the two
  middle layers gives 0.196. With the plant at exactly layer 24 of 48, the upper middle is the first planted layer, so this is a knife
  edge of the plant's position and the median's definition. It does not enter line 1, which reads only the must-fail lines.
- **Line 2a separates.** The threshold is 2 × 0.346 = 0.692; the routing plant reads 3.07× the box and the epilogue plant 24.6×.
- **Line 2b.** The box's largest interior u is 0.388, at layer 34. Both plants clear 2 × that (0.775): the routing plant by 1.37×.
- **Determinism.** Layers 0–23 are bit for bit the same in all three processes, and the first difference is at layer 24 (`runs.json`).
- **Router overlap at layer 24** is 0.9275 in all three runs. The recorded top-k is the router's choice before the routing plant
  shifts it, so that band cannot see a routing error at its own layer.

**Why the state dilutes** (exploratory, not a registered line; `sink_read.py`, `sink_read.json`). Position 0 of each row is an
attention sink. Its hidden state's norm is about 1,217, against a median of 5–46 for the other positions. From layer 3 to 24 it holds
97–99.8 % of the reference's squared norm, so the full-state statistic over those layers is that one position's error: 0.106, set by
about layer 3 and flat after. Without position 0, the state at layer 24 reads 0.129 for the box, 0.222 for the routing plant (1.7×)
and 1.55 for the epilogue plant. That separates the plants better than the full state does, but the routing plant's ratio is still
below its u_24 ratio (3.07×).

**The in-process plants, rehearsed** (2026-10-11, 00:56–01:29 UTC, the same card). Amendment 3 moves the plants into the fresh-process
repeat: `q235_box.py --repeat --plants` at c9d28ac9 runs the clean step-0 forward (the null), then each plant in the same process,
removing it before the next. One such process on the same model and tokens reproduces the three separate processes above **bit for
bit** (`inproc_check.py`, `inproc_check.json`):
- the null's 48 layer outputs equal the box's, and its loss is the same (1.85921);
- each plant's 48 layer outputs and loss equal its separate-process run's;
- each plant leaves layers 0–23 equal to the null and changes layer 24.

The registered reader over these records (`inproc_read.json`) gives the same u_24: 0.3460, 1.0628 and 8.5047. Each in-process plant's
forward took about 5.5 s after a load of 1,962 s.

| file | what |
|---|---|
| `runs.json` | the three runs: plant, step-0 loss, load time, card, versions, token and per-layer sha256 (host paths removed) |
| `depth_read.py`, `depth_read.json` | the registered reader and its output |
| `sink_read.py`, `sink_read.json` | the exploratory per-position read |
| `inproc_check.py`, `inproc_check.json`, `inproc_read.json` | the in-process rehearsal against the separate runs, and its read |
