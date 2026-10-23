### Changed

- P228 Amendment 6 and P229 Amendment 5 (pre-data): the pin screen admits RTX 5090 machines by their LATEST host
  registry power record at any age (575 W or more, or no record), never one whose latest record is below 575 W; it no
  longer requires a record read within 3 days, and each reading template sets pod-launch's `vast_power_limit_unknown`
  to "eligible" so the launcher admits the same way. The box's own power read before the install (rc 17) enforces the
  floor for hosts whose record is stale or missing, and three consecutive rc 17 boxes stop the lane. The floor, the
  measurement, the rows, the rule and the caps are unchanged; no staged pin moves and no runner byte changes. The
  screen and the templates are controller-side and change only after this merges, under their own review.
