### SD2 build 1 (#1313): the scheduler takes several tokens a step from a speculative runner

- **Several tokens a step.** A runner that sets `speculative = True` may return a list of tokens for a request.
  `ContinuousScheduler` emits them in order and stops at the token that finishes the sequence: a stop id once
  `min_tokens` are out, or its length. Later tokens are dropped and counted in `spec_dropped`, which `stats()` reports
  for a speculative runner only.
- **Budgets.** Before each decode the scheduler calls the runner's `decode_budgets({rid: tokens left})`, so a verify
  never runs past a request's length.
- **Refusals.** A speculative runner without `decode_budgets` is refused, and so is a speculative runner with the decode
  lookahead.
- **Unchanged.** A runner without `speculative` behaves exactly as before. Lane SD2 (`bench/sd2/PREREG-sd2.md`, B5).
