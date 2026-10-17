# Q235 Amendment 9: the A2000 rehearsal of the method

Correctness only, $0, on an RTX A2000 12 GB (PCIe gen 3 x8). It sets no rate. `a2000-rehearsal-summary.json` holds the probe's
full record at the amendment's head (`72c0663a`) and the box steps' `h2d_staging`. The full box record stays with the run's
other receipts; its sha256 is in the summary. The A2000 host's kernel command line and PCI bus id are left out of this public copy.

**(ii) The probe** (`q235_hostrate.py` at `72c0663a`):
- the single 1 GiB buffer: H2D 6.10 GB/s, D2H 6.58 (the floors read the H2D figure, as before);
- by buffer, 8 × 1 GiB: 6.12 / 6.16 / 6.18 GB/s (min / median / max); buffer 0 first and last: 6.15 and 6.11, so no warm-up
  effect;
- placement could not be observed on this host, and the record says why: the kernel is built without NUMA (no
  `/sys/devices/system/node`, no `numa_node`, no `numa_maps`), and the container's seccomp filter refuses `move_pages` (EPERM).
  The buffers are therefore unbound, `by_observed_node_median_gb_s` is null, and `numa_why` and `gpu_numa_why` say so;
- the card's IOMMU group type: `identity` (passthrough).

**(i) The box** (`q235_box.py` at `3757430f`, byte-identical at `72c0663a`, sha256 `3a83354d…`), Shape B on the 4-layer slice:
- `h2d_staging.n` = 28 per step, which is 4 micro-batches × (2 × 4 − 1), and the bytes equal 28 × 1,358,954,496;
- 6.168 and 6.117 GB/s (steps 0 and 1), per call 6.0 to 6.25;
- `by_layer`: layer 0, 4 calls (recompute only); layers 1 to 3, 8 calls each, as the schedule says;
- Amendment 2's rules all hold.

On this host the staging rate sits +0.13 % (step 0) and −0.70 % (step 1) from the by-buffer median of 6.16 GB/s (R9.2's band is ±10 %). The host shows no single-buffer
gap, so R9.3 has nothing to classify here. That read is the next box's.
