### The static guard fit counts the boot

`bench/common/README.md` ("Guard fit") and the docstring of `tests/test_lane_guard_fit.py` now state the per-lane
static check as (boot + fit) x 1.25 <= guard, where boot is the time from the launch, where the runner's deadline
starts, to the runner's check. The runner compares against the time left at the check, so a guard equal to fit x 1.25
would refuse every box. `tests/test_p155.py` already asserts the boot-inclusive form; the first P155 box
(p155-5090-1) reached its check 85 s after the launch. No runner or test assertion changes.
