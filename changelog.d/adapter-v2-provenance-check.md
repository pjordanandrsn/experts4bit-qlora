### Changed

- The `backend-adapter/2` reader (`adapter_dir.verify_adapter_dir`) now refuses a manifest whose `run_id` is not a
  non-empty string, or whose `created_at` is not a real UTC time written as `YYYY-MM-DDTHH:MM:SSZ`. Before, it
  checked only that both keys were present, so a null passed. The Loggetta writer refuses a null or empty `run_id`,
  so each side refuses a manifest that names no run; this reader also refuses a whitespace-only one.
