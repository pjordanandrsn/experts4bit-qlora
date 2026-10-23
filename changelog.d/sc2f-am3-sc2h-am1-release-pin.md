### SC2f Amendment 3 and SC2h Amendment 1: the release under test moves to 0.57.0, kernels to grouped-nf4-gemm 0.50.0

- Both lanes were registered on e4b 0.54.0 and never read. They now run on the first `main` commit containing both
  `v0.57.0` and e4b#2011 (the paged server's single wakeup per step). That commit's diff against the tag and its commits
  are recorded at launch. `bench/sc2/sc2_defaults.py` dumps the serving defaults on CPU at `v0.57.0` and at the launch
  commit: the resolved levers, plus a source scan of every `E4B_*` / `GNF4_*` environment read with its default. Any
  difference refuses the launch by name. The exceptions are `--allow LEVER=reason` for an opt-in knob merged since the
  tag that defaults off and is unset in the lane, and the four levers of #2017 named in the amendment
  (`tests/test_sc2_defaults.py`, including two fake trees).
  Each amendment says which moved defaults reach its int4 server (both arms alike) and why the comparison still reads its
  knob alone. This changes the system under test; the old pin was never read.
- grouped-nf4-gemm moves to v0.50.0 (`41ef086e`) for boxes N and O (`bench/sc1/sc1_run.sh`, re-pinned in
  `bench/sc1/staged.sha256`). `tests/test_sc2f_box.py` and `tests/test_sc2h_box.py` read the new pin.
- The price ceiling rises to $1.05 an hour, the RTX 5090 policy rate. Lane totals, with the registered re-run counted as a
  reading: SC2f $11.11, SC2h $12.16; each run under $15.
- SC2f's four refused proofs (2026-10-18, $0) are explained: one ghost listing on the only listed bar machine. The
  launcher's offer-freshness check and same-pin ghost retry now cover that, so the boxes launch through it. The market
  half is still live: no bar machine was rentable at the 2026-10-23 screen. No candidate means no launch, and a $0 screen
  runs for at most 48 hours after the tag.
- SC2f, SC2h and SC2i run on separate boxes in that order, pinning the same machine while it stays listed. Each lane keeps
  its own verdict and strike count. Nothing in either lane's science (arms, knobs, ladder, knee rule, statistics,
  outcomes, bar) changes.
