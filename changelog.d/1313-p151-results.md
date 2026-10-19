### P151 results (#1313): D1's served read closed at NOISY; `E4B_DENSE_GEMV_BW` stays opt-in

`p151-5090-4` read NOISY, a VOID and the lane's strike one. The box's first process ran 3.3 % slow (W1 252.86 tok/s
against about 261.5 for the other S0 arms, with no throttle), which put one W1 self-pair at 1.0345. By the maintainer's
ruling the lane closes without a re-run: the rule takes the minimum of four pairs, and the three clean pairs top out at
1.0247, under the 1.03 bar whatever the slow arm read. As context only, those pairs show about +2.45 % W1 decode on one
host, about 0.73 of P148's projection. Three earlier run ids were spent by refusals before any install, two of them
520 W hosts the power floor caught. Nothing changes by default. Record: `bench/p151/RESULTS-p151.md`.
