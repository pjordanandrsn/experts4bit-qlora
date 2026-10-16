### P132 Amendment 2 (#1313): the box refuses any card but the reference card

P132's rule reads absolute milliseconds against `sd2-5090-3`, which ran on one card (GPU-38c66e15). A matched pair in
the records (`p116-5090-1` vs `p116-5090-2`: same package bytes, kernels, model, prompts and box) shows another RTX 5090
taking 0.85–0.87× the time per B = 1 step, which would VOID P132 on the host, not the build.

Normalizing by a B = 1 anchor fails on the same pair: the one-row and multi-row phases scale differently, by up to
+10.8 % against the rule's 5 % band. So `bench/p132/p132_run.sh` now refuses with rc 18, before anything is installed or
fetched, unless the card's uuid is the reference card's. It records `card.json` either way.

`tests/test_p132_card.py` runs the check on real records: `sd2-5090-3`'s committed forensics passes, and
`p132-5090-1`'s refuses. `staged-p132.sha256` re-pins `p132_run.sh` only.
