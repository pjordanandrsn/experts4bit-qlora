# Fixtures: the controller-side fault record (`pgx2-4090-r1-2-1`)

`pgx2-4090-r1-2-1` was a healthy PGX2 reading box. A network outage at the controller stopped the controller's heartbeat, and
its guard destroyed the box on `heartbeat-loss` before anything was fetched. These files are that receipt's guard record,
redacted. They are the record `bench/common/controller_fault.py` is drawn from (PGX2 Amendment 1, P238 Amendment 7).

| file | from the receipt's | redaction |
|---|---|---|
| `guard-firing.json` | `guard-firing.json` | the provider instance id -> `INSTANCE` (1 place) |
| `teardown-proof.json` | `teardown-proof.json` (the final proof) | the provider instance id -> `INSTANCE` (2 places) |
| `guard.log` | `guard.log` | the controller's package name -> `controller` in its first line's module paths (3 places) |
| `receipt_status.json` | `receipt.json` | its `status` field alone, the only field the classifier reads |

Every other byte is the receipt's own. The provider's ids stay in the private receipt store (PREREG-pgx2, Receipts). A
reviewer with store access re-derives these files byte for byte by applying exactly the substitutions above.
