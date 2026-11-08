# PGX2: Paging stage 1, the successor registration of PGX (descriptive; registered before any of its runs)

Status: REGISTRATION DRAFT. It is not merged, so nothing is registered yet. Written 2026-11-06 (clock read 17:44Z).
`PGX2` is the lane's working name, not a reservation.

## Why this page exists

PGX (lane C, `bench/pgx/PREREG-pgx.md`) ended at two strikes under its registered stop:
- **`pgx-shake-1`** was a kit defect: the bake ran in the wrong interpreter. Amendment 2 fixed it.
- **`pgx-4090-1-2-1`**, the first reading box, lost the host's network mid-fetch. It STARTED, installed and fetched the
  bf16 and FP8 checkpoints. Then the GPTQ-Int4 fetch ended `httpx.ConnectError: [Errno 101] Network is unreachable`,
  and the runner exited 11 (`bench/pgx/receipts/pgx-4090-1-2-1`, committed byte-for-byte).

No reading exists. **PGX's verdict is not changed: it stays ended.** Changing a registered stop because of the outcome it
produced is what pre-registration exists to prevent. `pgx_box_end.py` still classes `pgx-4090-1-2-1` a strike, and a
test holds it there.

This page is registered after seeing those two records and no reading data. PGX2 is a new lane with its own ids and its
own strike count. It adds one rule, written in advance: how a host network failure after STARTED counts.

## What carries over, by reference and unchanged

- **PREREG-pgx.md and its Amendments 1 to 4:**
  - the question, the arms, the matched budget, the workload, the draws and passes;
  - every VOID and READ rule, the stability rule, the attribution, the readings and the predictions Q1 to Q7;
  - the tier counters and the instrument lifetimes;
  - Amendment 3's box plan: K = 2, its fit, its box pin and the `cross_host` label;
  - Amendment 4's box-end classes (a), (b), (c) and its `no_serve` stop.
  PGX2 reads them exactly as PGX would have. Its ids replace PGX's.
- **The code** is this page's merge: the same `bench/pgx` kit plus the changes listed under "The code".
- **The shakedown** is `pgx-shake-r1`, which finished 0 with `SHAKE_OK` (committed at `bench/pgx/receipts/pgx-shake-r1`).
  PGX2 runs no new shakedown box. The reason is under "Why no new shakedown".

## The ids

- The reading: **`pgx2-4090-1-2-1`** (draws 1 and 2) and **`pgx2-4090-1-2-2`** (draw 3). These are PGX's K = 2 boxes,
  unchanged.
- Retries: `pgx2-4090-r<n>-2-<i>`, with n from 1 to 3 across the lane. Each one names the box it replaces. A retry launches
  for a box that did not start (as registered) or for a host-fault box (below).
- Box 2 launches only after box 1 ends class (a), or class (b) without `no_serve`, as in Amendment 4.

## The one new rule: a host fault after STARTED

A box that STARTED and ended **rc 11** (a checkpoint fetch failed) is a **host fault, not a strike**, when its fetch log
proves a network-layer failure from this fixed list. The runner fetches bf16, FP8 and Int4 in that order and writes a
`FETCH <tag>` line to `summary.txt` after each. So:
- the failed fetch is the first tag without that line;
- no later tag's log may exist;
- that tag's `logs/fetch_<tag>.log` must hold a Python traceback whose last non-empty line **fully matches** one of these
  lines.

1. `httpx.ConnectError: [Errno 101] Network is unreachable`. Seen: it is the last line of `pgx-4090-1-2-1`'s
   `logs/fetch_int4.log`.
2. `httpx.ConnectError: [Errno -3] Temporary failure in name resolution`. The same failure's DNS form. **Not seen** as a
   final line in any receipt.
3. `httpx.ConnectTimeout: _ssl.c:999: The handshake operation timed out`, with any line number in place of 999. The TLS
   handshake to the hub timing out. The same record's retry lines show this error text. **Not seen** as a final line.

The list comes from the records first. The receipt store was searched for network errors in box-side logs. Only
`pgx-4090-1-2-1` holds a hub fetch that failed on one. The other hits are pip installs (`Connection reset by peer`, read
timeouts), not hub fetches, and none is listed. A connection reset or refusal at the hub is left off: no record shows what
its final line would be, and a guessed line would admit more than this page saw.

**Anything else is a strike, as registered.** This rule fails closed, and each case is tested:
- a network-shaped line not on the list;
- a listed line with any extra text;
- a truncated, empty, undecodable or unreadable log;
- a log with no traceback;
- a missing or out-of-order `summary.txt`;
- a later tag's log;
- every checkpoint fetched;
- any other exit code;
- a launcher that read another code.

An rc 11 without STARTED never reaches this rule: the registered pre-STARTED rules apply. The classifier is
`bench/pgx/pgx_box_end.py`, class `HOST_FAULT`, exit 14. It reads PGX2 ids only.

**Its limits:**
- **At most two host-fault boxes per lane.** The third counts as a strike.
- **Their spend is capped at USD 5.00 together**, read from their ledger rows. A host-fault box that takes that total past
  USD 5.00 counts as a strike. `pgx-4090-1-2-1` cost USD 1.13, about 14 minutes from STARTED.
- A host-fault box is retried under the next retry id. Its machine is excluded from the screen afterward.
- If box 1's machine is excluded this way, box 2 cannot be pinned to it. Box 2 then runs on another machine at once,
  labelled `cross_host`, as the registered pin rule labels a box that cannot run on box 1's machine.

## Strikes

**Strikes start at 0 for PGX2.** PGX's two strikes are PGX's. Two strikes end the lane, as before. Under Amendment 4, a
class (c) box is a strike, and so is a host fault beyond the limits above. A box refused before STARTED is not a strike.

## Budget

Each box's fit is Amendment 3's, from `pgx_schedule.py`, unchanged: box 1 52,713 s = 14.64 h, box 2 29,274 s = 8.13 h.
Its launch guard is that fit rounded **up** to 0.01 h, as PGX's reading kit did; PGX's Approved disclosed the rounding.
The worst is guard × USD 0.85 + 120 GB × USD 0.02.

| item | worst |
|---|---|
| box 1, `pgx2-4090-1-2-1` | guard 14.65 h, USD 14.85 |
| box 2, `pgx2-4090-1-2-2` | guard 8.14 h, USD 9.32 |
| the reading, K = 2 | USD 24.17 |
| host-fault boxes (at most two) | USD 5.00 together |
| boxes refused before STARTED (three retry ids, about USD 0.25 each) | USD 0.75 |
| **lane worst** | **USD 29.92**, under the USD 35 per-run cap |

The bound counts each allowance in full. In fact the three retry ids are shared, so a lane with two host faults has one
retry left for a refused box. Both boxes declare the policy's `wallclock_override` (≤ 18 h, worst ≤ USD 15).

**Launch rule:** no PGX2 box is created unless the lane's spend so far (its ledger rows) plus that box's worst is at most
USD 35. So the cap holds even if an allowance is misread.

PGX's own spend was USD 3.49: `pgx-shake-1` USD 0.96, `pgx-shake-r1` USD 1.40, `pgx-4090-1-2-1` USD 1.13. It is stated
for context and is not counted against PGX2.

## Why no new shakedown

`pgx-shake-r1` ran the staged box side on a 4090:
- the three venvs and e4b's install;
- the fetch, the bytes per expert and the K table;
- the NF4 bake;
- calibrate's `calib.json`;
- the card-clock burn;
- one lifetime per arm at 8 GB, all serving, with every counter shim on a real record.

`pgx-4090-1-2-1` then ran the reading kit's box side on another host. It reached STARTED, the installs, all three
tripwires, both box-side self-tests (`pgx_shake.py --self-test`, `rt/pgx/pgx_box.py --self-test`) and two of the three
fetches, and every step passed. PGX2 changes nothing the box runs except which run ids it accepts (below).

What no GPU has run is Amendment 3's read path: the 4 and 12 GB budgets, P3 and P4, the instrument lifetimes, the controls
and the part reduction. It has $0 evidence:
- the reducer's part mode on r1's real lifetimes;
- Amendment 4's classifier tests and the kit stubs.

A shakedown box cannot test that path more cheaply than box 1 does. Exercising it means running reading lifetimes, and a
defect there costs one strike either way. With PGX2's strikes at 0, such a defect leaves the lane able to amend and run
again. So no new shakedown box.

## The code

- `pgx_run.sh` and `pgx_drive.sh` accept `pgx2-4090-1-2-[12]` and `pgx2-4090-r[1-3]-2-[12]` as reading boxes, beside PGX's
  ids. That is one new case line each; the box derives its `BOX_SPEC` exactly as for PGX's ids.
- `pgx_schedule.py`:
  - `parse_run_id` accepts either lane's ids;
  - `run_lane` names the lane;
  - `box_ids` and `retry_id` take the lane.
  PGX's ids and outputs are unchanged.
- `pgx_box_end.py` adds `HOST_FAULT` (above) for PGX2 ids. A PGX id's rc 11 stays a strike.
- `staged-pgx.sha256` moves for `pgx_run.sh` and `pgx_schedule.py`.
- `bench/pgx/receipts/pgx-4090-1-2-1/` commits the record this rule is drawn from, byte-identical, with `SHA256SUMS`.

**The $0 tests** are in `tests/test_pgx2.py`. They start from the real record:
- its bytes match `SHA256SUMS`;
- its fetch log's sha256 is pinned;
- under its own PGX id it is still a strike;
- relabelled to a PGX2 id, the real log is a `HOST_FAULT`.

Then every fail-closed case above. Then:
- the runner and driver accept exactly both lanes' plan ids;
- this page lists the classifier's lines exactly;
- this page's budget is the schedule's.

Code mutants of the classifier were each killed. They are listed in the PR.

## Launch

As PGX's Amendment 4: one launch commit, this page's merge. A reading kit is cut from that merge and reviewed. Then comes
the maintainer's Approved, naming the commit and the kit's hash. The kit's stop counts PGX2's strikes and host faults
under this page. The kit's screen excludes a host-fault box's machine.

## Receipts

As PGX: every box's fetched files go to `bench/pgx/receipts/<run>/`, with `SHA256SUMS`. The provider's machine and
instance ids stay in the private receipt store.

## Amendment 1 (2026-11-07): a controller-side fault, from `pgx2-4090-r1-2-1`

Written 2026-11-07 after `pgx2-4090-r1-2-1`'s receipt, before any further PGX2 box. Nothing else changes: the ids, the
host-fault rule, the strikes, the budget's per-box worsts and the launch rule all stand.

**The record.** `pgx2-4090-r1-2-1` was box 1's first retry: box 1's plain id was refused rc 19 before STARTED (its hub download
measured below the floor). The retry ran for about four and a half hours and was healthy: draw 1's 21 lifetimes were done,
draw 2 was at lifetime 24 of 42, and every prompt record was clean. Then a network outage at the controller cut the
controller's own calls to the provider. The controller-side guard read the lost heartbeat as a dead controller and destroyed
the box at 12:03Z, before anything was fetched. The receipt is `ALARM`, USD 3.388. Its guard files are in
`tests/fixtures/ctlfault_pgx2_r1/`, redacted (the provider's ids stay in the private store). No lane directory came back, so
the record says nothing about the box, the host or the code under test.

Under the registered rules it is "a created box with no readable code", and the stop fails closed for a person. That is
right, and this amendment does not loosen it. It registers what such a record is, before any other box runs.

**The rule.** `bench/common/controller_fault.py` classes a box as a **controller fault** only when all five hold:
1. `guard-firing.json`: the guard fired, with `reason` exactly `heartbeat-loss`;
2. `teardown-proof.guard.json`, the guard's own proof written at the firing: its `reason` is `heartbeat-loss` and its
   evidence reads `list_after` `"UNKNOWN"` (the guard's listing right after its destroy got no answer, so the controller
   could not reach the provider then);
3. `teardown-proof.json`, the final proof: `complete` is true and its evidence reads `instance_absent` true;
4. `receipt.json`: `status` is `ALARM`;
5. no lane directory was fetched.

Anything else, an unreadable or malformed file included, is not a controller fault, and the lane's ordinary stop applies.
A heartbeat lost while the provider still answered, a stalled or crashed controller process, leaves `list_after` a list: it
is not a controller fault, and the lane stops for a person. The guard's log is not read: its network-failure lines carry no
time, and they also appear in runs with no outage.

A controller fault:
- is not a strike and not a read;
- is not a STARTED box (nothing came back, so whether it started is unknown): it is left out of the K = 2 count and the
  started-spec rule;
- is replaced under the next retry id, as a box that did not start is;
- counts its ledger spend toward the launch rule;
- happens **at most once** in the lane. A second controller fault ends the lane: the controller is then not fit to run it.

**This record.** `pgx2-4090-r1-2-1` is a controller fault under the rule (`tests/test_controller_fault.py` reads its redacted
record). It is the lane's one. Box 1's next id is `pgx2-4090-r2-2-1`, retry 2 of 3.

**Spend.** So far: `pgx2-4090-1-2-1` USD 0.019 and `pgx2-4090-r1-2-1` USD 3.388, USD 3.41 in all. The lane's worst becomes
3.41 + 14.85 (box 1) + 9.32 (box 2) + 5.00 (host faults) + 0.50 (two retry ids refused before STARTED) = **USD 33.08**, under
the USD 35 per-run cap, and the launch rule still holds the cap on the ledger's actual spend.

**The kit.** The stop reads the classifier from the launch commit (sha-pinned) and applies it before its "no readable exit
code" check. The kit is cut from this amendment's merge and approved afresh.

**Note (2026-11-08, no rule change): the cause chain of `pgx2-4090-r1-2-1`.** The record above says the guard read the lost
heartbeat as a dead controller. The full chain, read from the run's launch log and receipt: the driver held its start ssh to
the box open for the whole run (a driver defect, fixed in e4b#2181), and the outage at the controller cut it. The driver
logged "start failed" at 12:00:02Z and exited 21. The launcher refreshes its heartbeat only while the driver runs, so the
heartbeat's last write was 11:59:57Z; the launcher then began its own teardown, which could not reach the provider. The guard
fired on the stale heartbeat at 12:02:32Z and completed that teardown at 12:03:32Z. The box was healthy throughout. The five
conditions, the classifier and every rule of this amendment are unchanged, and the record is still this lane's one controller
fault. A kit cut at a launch commit after e4b#2181's merge starts the run detached, so an outage like this one no longer ends
the driver.
