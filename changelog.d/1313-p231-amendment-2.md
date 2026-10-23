### P231 Amendment 2 (after `p231-5090-1`, before any reading): the tripwire proves the switch's guard; shakedowns

- `bench/p231/p231_run.sh`: the tripwire's routing block first asserts that `E4B_NF4_SMALLM_BW=1`, unresolved, refuses with the guard's message, then makes the real `resolve_nf4_smallm_bw` call on stand-in modules at Mixtral's expert shapes on the box's device, reads the routing, and restores every switch in a `finally`. New ids `p231-shake-1` to `-3` stop with rc 0 before the premise.
- `tests/test_p231.py`: the routing block is extracted from the runner and run on CPU against the repository; the pre-amendment lines, a deleted guard and a non-engaging device each fail it.
- `bench/p231/p231_reduce.py`: shakedowns classify as `SHAKE_OK`/`SHAKE_FAIL`, never counted; `bench/p231/p231_drive.sh` accepts their ids.
