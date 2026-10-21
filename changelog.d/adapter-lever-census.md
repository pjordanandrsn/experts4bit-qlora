### Changed

- The paged server's adapter lever census now classifies every environment name that `_apply_levers` and the lane
  hook it mirrors (`bench/p42/hook/usercustomize.py`) read, whatever the prefix and in any spelling (`env(...)`,
  `os.environ.get`, `os.environ[...]`, `getenv`). Each name sits in exactly one table: refused under an adapter, allowed
  with its reason, or the new `ADAPTER_LEVER_CONFIG`, which names the refused levers that a configuration-only name
  configures. A new lever, under any name, now fails the test until it is classified.
