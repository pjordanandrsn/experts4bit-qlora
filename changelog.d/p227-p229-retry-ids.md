### Changed

- P227 and P229 Amendment 4 (pre-data): the retry run ids `p227-5090-r1` to `-r8` and `p229-5090-r1` to `-r8`, for any
  box that follows one that did not start (a refusal at create, a pre-flight teardown, or the box's own refusal before
  the install), accepted by each runner and drive at the same tier and terms. The ids are labels, not budget: the
  registered caps, strike rules and stop are unchanged, and started-box accounting is by outcome, never by id. Each
  lane's stop never counts a refusal at create or a pre-flight teardown. Each PREREG gains the "refused at create"
  outcome row (nothing created, $0, no strike, no cap, one id) and a slack check (guard - FIT_NEED >= 420 s + 120 s);
  the runners' staged pins move.
