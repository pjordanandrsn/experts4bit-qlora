### SC1 driver (#846): the receipt fetch leaves SC1g's staged reference rows on the box (bench and tests only)

- **The failure.** `sc1g-prove-a5-1`, box I's A5 proof, was refused on the box: 317 GB free against the 320 GB floor, after
  3.9 GB of box R's full-vocabulary rows were staged. Its fetch then pulled those same input rows back into the receipt:
  3.9 GB into the receipt store, and about 7 min of rented time.
- **The fix.** `sc1_drive.sh`'s `pull_box` (the mid-run pulls and the final fetch) now excludes `sc1g_ref_full/` and box I's
  `.f16` copies (`sc1g/ref_full_*.f16`). Those rows are inputs, registered by sha, never receipts.
- **The test.** It runs rsync with the driver's own filter list against a box-shaped tree.
- **The receipt.** The fetched-back copies were byte-identical to the registered rows. They were moved out of the store
  (`TRIMMED.txt`). The proof re-runs with more disk, so that 320 GB stays free after staging.
