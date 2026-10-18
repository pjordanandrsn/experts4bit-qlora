### P138, the verify split, registered (#1313): where SD2's verify increment goes, on one card's own numbers

P132 closes "not run": its reference card was listed in 0 of about 4,050 reads over 34.5 h, and its rule compared
absolute milliseconds with that card. The question moves to P138 in `bench/vsplit/`.

Every number P138's rule compares comes from the box that runs it:
- two unprofiled timing passes of SD2's V1 terms (the A/A anchors and a self-pair);
- the census's class shares at n = 1..4, converted with the unprofiled steps;
- need(n) from SD2's own formula with this box's terms.

An increment whose excess over the bytes is within the noise floor (the two passes' spread, or 5 % of the increment,
whichever is larger) reads NO_EXCESS: an answer that closes the lane, not a strike. Only a READ carries quotable
labels. The inputs and the tested outcome (the n >= 2 shares) are kept apart. A reducer test shows the outcome alone flips H
and BUILD with every input held fixed, and a profiler's uniform inflation moves no millisecond. P132's box and class
map are reused byte-identical, and the desk prediction is registered as fractions of the increment. It runs on any
RTX 5090; the verdict licenses BUILD or kills the lane, and ships no default.
