### Changed

- P227 and P229 Amendment 4 (pre-data): the retry run ids `p227-5090-r1` to `-r8` and `p229-5090-r1` to `-r8`, for any
  box that follows one that did not start (a refusal at create, a pre-flight teardown, or the box's own refusal before
  the install), accepted by each runner and drive at the same tier and terms. The ids are labels, not budget. A box
  STARTS when its runner passes the refusals before the install and begins the install; at most three boxes start
  (the registered budget), and only started boxes count toward the cap and toward any stop (P227's Budget and
  per-family stops, P229's stop). A single rc 41 is not a strike and the registered rc 41 pair rule is unchanged.
  Each PREREG gains the "refused at create" outcome row and a slack check (guard - FIT_NEED >= 420 s + 120 s); the
  runners' staged pins move.
