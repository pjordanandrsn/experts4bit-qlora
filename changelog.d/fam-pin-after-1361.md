### FAM: re-pin `test_fusion_modes.py` after the family-scoped default

#1361 changed `tests/test_fusion_modes.py` and re-pinned P115 Phase C's manifest; #1380 had pinned the same file at
Phase C's old bytes, so main's `tests/test_fam_staged_pin.py` failed once both merged. FAM's pin moves to the new bytes,
again equal to Phase C's. FAM's arms set every fusion knob explicitly, so its measurement is unchanged; its premise now
runs the current knob tests.
