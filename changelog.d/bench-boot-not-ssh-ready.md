### Changed

- `bench/common/README.md` (Phase stamps): boot, from the create to a usable box, is read from the launcher's receipt (`<provider>_ssh_ready_after_s`, else `<provider>_preflight_seconds`), never from `ssh_ready_s`. That field is the configured readiness allowance, not a measurement. A receipt with neither measured field has an unknown boot.
