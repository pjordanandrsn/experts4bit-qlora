### Changed

- P225 Amendment 1 (pre-data): the retry run ids `p225-5090-r1` to `p225-5090-r8` for any box that follows one that did
  not start (HOST_REFUSED, or NOT_RUN when the launcher's pre-flight tore the instance down before the runner ran),
  accepted by the runner and driver at the same tier and terms (`bench/p225/PREREG-p225.md`). The ids are labels, not
  budget: the 3-box-started cap, the HOST_REFUSED caps (4 a trip, 8 in all), the strike rules and the stop are
  unchanged, and started-box accounting is by outcome, never by id. The Outcomes table gains the NOT_RUN row (no strike,
  no cap, one id), and `p225-5090-1` is recorded under it.
