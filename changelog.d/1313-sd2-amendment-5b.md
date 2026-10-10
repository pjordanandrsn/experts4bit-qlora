### SD2 Amendment 5b (#1313, operational): a canary and fail-soft reporting, after `sd2-5090-2`

- **The canary.** Right after V0 passes, V1, one E arm and one Q window each run once at minimal size in their own
  process. Their records are discarded. A canary that fails stops the lane (rc 26) before the paid stages. Stage V
  splits into V0 (`--read-v0`, `--gate-v0`) and V1 (`--read-v1`) around it.
- **Fail-soft reporting.** Every reporting-only value goes through `soft()` and records `report_unavailable` on an
  error, so a statistics bug cannot void a read. Verdict inputs stay strict.
- **Unchanged:** the rule, the verdict inputs and the budget (3.5 h, $3.66). The slow-host walk with the split stage still starts Q in time.
