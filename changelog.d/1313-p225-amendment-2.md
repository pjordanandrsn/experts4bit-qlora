### Changed

- P225 Amendment 2 (pre-data): the guard's boot term rises from 300 s to 880 s (`bench/p225/p225_derive.py`,
  `bench/p225/PREREG-p225.md`): the launcher alone may wait 420 s for ssh before staging, and `p225-5090-r1` halted
  at the pre-install fit after a 443 s launch-to-check. The guard is 21,230 s (5.90 h, inside the 5.9 h fill limit) and
  the worst case USD 6.92 a box. No need, alarm, floor or runner byte changes, and rc 41 stays PRE_INSTALL_HALT. A new
  test keeps the boot allowance at least the launcher's ssh wait plus a 300 s staging margin.
