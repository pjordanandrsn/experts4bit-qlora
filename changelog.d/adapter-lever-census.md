### Changed

- The paged server's adapter lever census now classifies every lever name the server knows. That is the names
  `LEVER_ENV` declares, plus every environment name that `_apply_levers`, its calibration helper and the lane hook it
  mirrors (`bench/p42/hook/usercustomize.py`) read, whatever the prefix and in any spelling (`env(...)`,
  `os.environ.get`, `os.environ[...]`, `getenv`).
- Each name sits in exactly one table:
  - refused under an adapter;
  - allowed, with its reason;
  - `ADAPTER_LEVER_CONFIG` (new): the refused levers a configuration-only name configures;
  - `ADAPTER_NEUTRAL_KNOBS` (new): a fusion or the KV append, which changes no adapted weight.
- A lever declared in `LEVER_ENV`, or read by those functions, now fails the test until it is classified. A name read
  only somewhere else is outside the census.
