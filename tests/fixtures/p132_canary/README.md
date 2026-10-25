# p132_canary: the in-box canary records of `p132-5090-1`

These files are byte-identical to the records in the receipt of `p132-5090-1`. That run was lane P132's first
box: one RTX 5090 on 2026-10-11, e4b 88ae1cfd plus grouped-nf4-gemm 724ccc45, and SD2's read server. It stopped at the
canary with rc 26 (HARNESS_ERROR, $0.731, teardown verified). The tests that read these files check their sha256 first:

| file | sha256 |
|---|---|
| `canary.json` | `a5f5068ff57c6f5da5889031e46ba5e305384e1523326f52683a2f5a2f05e316` |
| `canary_census.json` | `1467f9d2a388e1d96d1caea8446a0644e89d6f372fccc3ec2c5f2bc42a128ee8` |
| `canary_drv_n2_graph.json` | `ba4db33afbd61ba8608c521e3ff0f0fdc6583a40980792a2760a2007b4293c37` |
| `canary_drv_n2_node.json` | `862123a5d959a70b62e43f2aec4da256904f7eecddfff7529beeeb0e487ebaf5` |
| `forensics.txt` | `8937620ac7926dec900fa58d1fe2c6978fcadcc953b2c45bc51c54733dc450eb` |

`forensics.txt` was added by Amendment 2. Its first line names the card, GPU-redacted-52f1f988, not the reference card
GPU-redacted-efc7954e. `tests/test_p132_card.py` uses it as the record the card-identity check must refuse.

**What they hold:**
- `canary.json` is `p132_box.py --mode canary`'s record, plus the census verdict that `p132_run.sh` merged into it.
  Buckets 1..4 were captured as graphs, and every plain and verify replay returned `ok`. The canary's status is
  `failed` only because its census is `void`.
- `canary_drv_n2_{graph,node}.json` are the box's records of the two short census captures at n = 2. Each shows
  `steps_bracketed` 8 after 2 skipped replays (bucket 2 replayed 10 times), with status `ok`.
- `canary_census.json` is SC1b's reading of the two Nsight exports. Both modes are void:
  `5 steady steps kept < 6 (dropped {})` and `5 steady launches kept < 6 (dropped {})`. The exports themselves stayed on
  the box (SC1b's precedent).

**Why the census is void.** SC1b makes one row for each period between consecutive steady replays, so 8 replays give
7 rows. It then trims both edges, which leaves 5. The registration's floor was `CANARY_STEPS - 2` = 6, so the canary
could not pass on any host. Amendment 1 (bench/p132/PREREG-p132.md) sets the floor to `CANARY_STEPS - 4` and
`CANARY_STEPS` to 12.

**Who reads them.** `tests/test_p132.py`, in its `test_a1_*` cases:
1. First, the registration's rule, run through the staged `sc1b_census.py` at the record's bracket count, must return
   these `why` strings byte for byte.
2. Then the amended floor is checked against what the reader keeps.
