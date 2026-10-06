### SC1g A5 box R read (#846): `R_OK`, conv1–conv4 gradable; box R's full-vocabulary rows registered for box I, and the reference is bit-reproducible across hosts (bench and tests only)

- **The run.** `sc1g-r5-2` (Vast H100 NVL, $0.826, from A5's merge `dfc5bdaf`) read `R_OK`:
  - all validity checks OK;
  - floor F 5.7e-3, 7.0e-3, 1.8e-3 and 1.9e-3 on conv1–conv4, all gradable; wikitext (2.2e-2) is never graded;
  - the fp16 storage error at most 4.8e-6, about 150× under its bar.
- **The registration.** `bench/sc1/sc1g_ref/` holds what box I stages:
  - the shas of the five `[2048, 201088]` fp16 row files (`ref_full_shas.json`);
  - R's `r_verdict.json` and `r_calib.json`.
  - A test pins all three to the committed receipt, and `--reverdict` re-derives `R_OK`.
- **The rows** live outside git: on the mini, and on QNAP Pool 3, each re-hashed.
- **Bit-reproducible.** Every number this run shares with A4's `sc1g-r-8` is bit-identical (25 of 25), across a different
  card, provider and driver.
- **Spend.** Box R totals $1.5164; the lane is at $5.308.
