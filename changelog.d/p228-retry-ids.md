### Changed

- P228 Amendment 5 (pre-data): P227's and P229's Amendment 4 in its final form. The retry run ids `p228-5090-r1` to
  `-r8`, for any box that follows one that did not start (a refusal at create, a pre-flight teardown, or the box's own
  refusal before the install), accepted by the runner and drive at the same tier and terms. A box STARTS when its
  runner passes the refusals before the install and begins the install; at most three boxes start (the registered
  budget), and only started boxes count toward the cap and the stop. A single rc 41 is not a strike and the registered
  rc 41 pair rule is unchanged. The PREREG gains the "refused at create" outcome row and a slack check
  (guard - FIT_NEED = 600 s >= 420 s + 120 s); the runner's staged pin moves.
