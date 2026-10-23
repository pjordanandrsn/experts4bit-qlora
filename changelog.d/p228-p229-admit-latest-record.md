### Changed

- P228 Amendment 6 and P229 Amendment 5 (pre-data): the pin screen admits RTX 5090 machines by their LATEST host
  registry power record at any age (575 W or more, or no record), never one whose latest record is below 575 W; it no
  longer requires a record read within 3 days. The box's own power read before the install (rc 17) enforces the floor
  for hosts whose record is stale or missing, and three consecutive rc 17 boxes stop the lane. The floor, measurement,
  rows, rule and caps are unchanged; no staged pin moves and no runner byte changes. The screen is controller-side and
  changes only after this merges, under its own review.
