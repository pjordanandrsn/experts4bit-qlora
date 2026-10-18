# P138 canary records (`vsplit-5090-2`, 2026-10-18)

The box's own canary records, as fetched at teardown from `/root/vsplit/` on one RTX 5090 (card GPU-ecc38202):

| file | sha256 | what it is |
|---|---|---|
| `canary_census.json` | `4065b49988ec9b8ceaeed0060936a31ab555ff087e6ec4445c74f09ea54876d9` | SC1b's reading of the canary's two short captures at n = 2 (graph and node, 12 replays bracketed, 9 kept) |
| `canary.json` | `340eb4f59ee65ff87854b80924a5e6ac8fcdae9bf3cd3b31e39415c64c785dd4` | the canary's replays at n = 1..4, after the runner's canary check rewrote it |

The census reading is clean: both arms ok and 48 MoE layers segmented, with no blocking label. Its status is
`labelled` only because SC1b attaches `NO_UNPROFILED_TIMING` when no unprofiled time is passed, and none exists before
timing pass A. The runner's old canary check required status `ok`, so it failed this record (exit 26). That was P138's
first box-started attempt.

`canary.json` carries the old check's verdict (`status: failed` plus a `census` key). The box's pre-check record had
`status: ok`, as its own `P132_CANARY` line recorded (replays 1..4 ok, no missing bucket). The tests restore that record
before running the runner's canary block on it.
