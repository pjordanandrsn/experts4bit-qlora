# P65 rehearsal on the NAS RTX A2000 — NOT a reading

These files are the P65 census and reducer run end to end on real CUDA, before any box was rented. They are disclosed
in [`../P65-PREREG.md`](../P65-PREREG.md) ("The rehearsal"). They are **not** the lane's reading:

- **the models are not the registered ones:** `granite-3.0-1b-a400m-instruct` (Granite-MoE, prefused stacks, 32 experts
  × 24 layers) and `OLMoE-1B-7B-0924` (the BASE model, 64 × 16), read from the LAN model store. The lane registers
  Granite-3.1-3B-A800M-instruct, OLMoE-1B-7B-0924-Instruct and Mixtral-8x7B-Instruct-v0.1;
- **the card is an RTX A2000 12 GB** (sm_86, 70 W power limit), not the RTX 5090 class;
- **the tree is this branch's**, not a pinned cut;
- **the reducer's `premise` column applies the REGISTERED family's premise** (P44-a's Granite-3.1-3B read), not these
  models'.

What they show: the tap feeds the entropy what its definition says; the census is deterministic and complete on both
expert layouts; the reducer reads real rows; and the costs extrapolate to the rental estimate.

## Environment

- **Image:** `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, `--runtime nvidia-runtime`, the shared-GPU lock held for
  each job.
- **Packages** (`versions.txt`): torch 2.8.0+cu128, triton 3.4.0, transformers 5.16.1, bitsandbytes 0.50.1, gnf4 0.33.0
  (the e4b CI pin, `5ca18975`), e4b 0.37.2 from this branch.
- **Census code:**
  - Granite ran `p65_census.py` sha256 `041a3031…`, the commit `dd86a36` version;
  - OLMoE ran `52203b13…`, the registered version, which adds `--first-layers` (a layer-selection refactor; the math is
    the same);
  - both ran `expert_entropy.py` `e107db65…`. The registered `02a6c44b…` differs only in a docstring, which states
    the range as [0, 1] rather than (0, 1];
  - `p65_table.md` and `p65_verdicts.json` are the registered reducer (`bd0260e8…`) over the `.json.gz` files here:
    `python bench/p65/p65_reduce.py bench/p65/rehearsal-a2000` reproduces the table byte for byte.
- **Settings:** both at `--nseq 64` (two halves of 16,384 tokens per text, as registered). The Hessian budget was held
  small on purpose for Granite (2 GB: 5 chunks) so the chunk loop ran several passes. OLMoE's kept run used 16 GB (one
  4-layer chunk); its stopped first run used 8 GB (2-layer chunks).

## Files

- `census_<family>.json.gz`: every row (text × half × layer × expert × role), the text digests, the selfcheck and
  per-pass timing.
- `p65_table.md` and `p65_verdicts.json`: `p65_reduce.py` over the two censuses.
- `census_<family>_log.txt`, `bake_<family>_log.txt`, `forensics_<family>.txt`, `timing.txt` and `versions.txt`.
- `accbench_nas.json` / `accbench_mac.json` (`harness/accbench.py`): the host-side cost of one Hessian update at
  Mixtral's shapes on the NAS Xeon W-1250 and an M1 Max. This is the basis of the runner's host pre-flight.
- `prove_config_*`: the PROVING configuration (`P65_PROVE=1`: first layer, `nseq 8`) on Granite-1B, run 2026-09-24
  00:35Z under the lock with the registered census (`52203b13…`). rc 0: selfcheck 5.97e-8, 384 rows, both texts, both
  halves and the full rows, census 38.5 s, bake 74 s wall. It shows the cut-down census the proof runs is complete on
  CUDA. It is not the proof, which runs on the rented class.
- `harness/`: the rehearsal's own shell (`setup.sh`, the CPU install and text prefetch; `gpu_job.sh`, the shared-GPU
  lock plus one job; `gpu.sh`, bake and census; `ci.sh`, the CI job reproduced on Linux).

## Numbers

**Selfcheck:**
- Granite: moment-derived `rho` equal to the raw-row `rho` to a relative 4.5e-8 (experts 31 and 3 of layer 0, 1,277
  and 815 tap rows = captured rows), with bitwise repeats of `rho` and `rel_act`.
- OLMoE: agreement to 1.5e-8 (experts 28 and 41 of layer 0; 903 and 767 tap rows = captured rows), with bitwise
  repeats. The stopped first run printed the same 1.4525590898021376e-08, to every digit.

**Timings:**
- Granite: census 263.8 s, peak 1.4 GB reserved; bake ~23 s of work (`bake_granite_log.txt`: 13 s NF4 quantise, 5.8 s
  snapshot, 4 s arena). 20 half-passes: Hessians 130.5 s, rows 66.4 s, weight
  reads 15.2 s; the rest (build, selfcheck, texts) 51.7 s.
- OLMoE (the first 4 layers of plan order, {0, 1, 10, 11}): census 478.6 s, peak 5.0 GB reserved (6.5 GB used on the
  card by every tenant together); the bake was ~5.7 min of work, in the stopped first run (`bake_olmoe_log.txt`: 5 min 17 s NF4 quantise from the
  NAS disk, 17 s snapshot, 9 s arena). 4 half-passes: Hessians 258.3 s,
  rows 165.2 s, weight reads 14.6 s; the rest 40.5 s. No channel variance rounded below zero in either census.
- **Why OLMoE is 4 layers.** The first run (all 16 layers, 8 GB budget) passed its selfcheck and then took 162 s per
  half-pass of a 2-layer chunk, about 85 min for the whole census on a GPU three other agents share. It was stopped
  after its first half-pass (`census_olmoe_run1_stopped_log.txt`) and re-run with `--first-layers 4` and a 16 GB budget
  (one chunk).
- Every full row equals the row-weighted combination of its halves, in both censuses (3,072 Granite and 1,024 OLMoE
  full rows): max relative difference of `sq_err_act` 2.3e-16, and `rows` = the halves' sum. Every layer's routed
  total is exactly 16,384 × 8 (top-k) = 131,072 token-slots per half, in both.

See `p65_table.md` for the rankings. In one line:
- **Granite-1B:** every signal survives, and entropy is orthogonal to `rel_act`.
- **OLMoE-base:** routing frequency is largely redrawn by the domain (0.454); entropy degrades (0.783, penalty
  0.187 past the line); `rel_act` holds (0.938); Colla-Q's comparative claim replicates.

Both on models the lane does not register.
