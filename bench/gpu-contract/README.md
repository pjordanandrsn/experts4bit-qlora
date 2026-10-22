# bench/gpu-contract: a kernel PR's GPU contract on one rented card

A correctness contract, not a registered reading: nothing here is timed. A grouped-nf4-gemm PR that changes what runs by
default on a class of GPU (for example, auto engagement on a new SM count) needs its own tests run on that class, and the
lane machines may not have it (the RTX A2000 is sm_86). This tool runs, at two pinned commits on one card of a declared class:

- grouped-nf4-gemm's kernel tests `kernel/test_nf4_gemv_bw.py` and `kernel/test_dispatch_counts.py`, from a clean checkout
  of the grouped-nf4-gemm commit;
- experts4bit-qlora's served-route premise `tests/test_decode_graph_buckets.py`, `tests/test_kv_step_select.py` and
  `tests/test_gemv_bw_served_gpu.py` (P133's two premise suites), from a clean checkout of the experts4bit-qlora commit,
  with both packages installed at those commits.

Each suite runs with `-rs` and a junit report; the summary lists every skipped test and its reason.

## Inputs (refused unless well formed, before any connection)

| variable | meaning |
|---|---|
| `GC_GNF4_SHA` | the grouped-nf4-gemm commit under test: 40 lowercase hex (a branch, tag or short sha is refused) |
| `GC_E4B_SHA` | the experts4bit-qlora commit whose served tests run and which the box installs: 40 lowercase hex |
| `GC_CARD` | the declared class: `MAJOR.MINOR/SMS` (exact count) or `MAJOR.MINOR/>=SMS` (a floor), e.g. `8.9/128` for the RTX 4090 |
| `GC_REQUIRE` | optional: test ids (`<classname>::<name>`) that must PASS; see below |
| `GC_E4B_TESTS` | optional: the experts4bit-qlora test files to run INSTEAD of the served-route premise, space-separated `tests/test_<name>.py` paths, each refused (78) unless it has exactly that form; for a PR's own GPU smoke |
| `GC_NEED_E4B_S` | optional: the e4b suite's time need in whole seconds, 1-1800 without a leading zero (default 600), passed through to the box; a suite's alarm is capped at 1,800 s whatever the need |
| `E4B_RENT_RUN_ID` | `gc-<slug>-<n>`; a dry run's is `gc-dry-<slug>-<n>` with `GC_DRY=1` and `GC_BOX_EXEC` |

The launch tree must be clean, so the staged tool is the committed one; its commit is written to the box (`GC_TOOL_SHA`)
and travels back with the receipts.

## On the box (`gc_run.sh`)

Refusals: CUDA unusable 18, torch will not import 10, torch not the image's `2.8.0+cu128` 28, card outside the declared
class 15 (compute capability and SM count as torch reports them, never the card's name), disk 13. Then:

1. Install experts4bit-qlora at `GC_E4B_SHA` (transformers 5.17.0, bitsandbytes 0.50.2), then grouped-nf4-gemm at
   `GC_GNF4_SHA` over it.
2. Fetch each commit at depth 1 into a fresh checkout. Each checkout is refused (9) unless its HEAD is the pin and its tree
   is clean.
3. Run a tripwire. It checks that the installed packages' recorded commits are the pins, that torch is the pin, and that the
   installed `nf4_grouped.py` is byte-identical to the checkout's.
4. Run both suites. A failure or an error in either one exits 30.

`GC CONTRACT PASSED` with exit 0 needs both suites to pass with no failure or error.

## Launch

The launcher's `--command`, from a clean checkout of the tool's commit:

    GC_GNF4_SHA=<40 hex> GC_E4B_SHA=<40 hex> GC_CARD=8.9/128 bash bench/gpu-contract/gc_drive.sh

The receipts land in `<run dir>/gc/`: `summary.txt`, `versions.txt`, `forensics.txt`, `logs/` (both junit reports and both
pytest logs), `GC_TOOL_SHA`. The pin, the spend and the teardown proof are posted on the PR under review and on the bus, as
for any rental.

## Required tests and the time left

- `GC_REQUIRE` (optional, on the controller): the tests a review names as the PR's own evidence, in junit form
  `<classname>::<name>`, space-separated. For example, `kernel.test_dispatch_counts::test_on_a_real_rtx_4090_auto_engages_unmocked`.
  Each must PASS. A skip, an absence, a failure or an error exits 30, because a contract whose key test skipped on the
  declared card has not passed. The summary records each one's outcome (`GC_REQUIRE`).
- Each suite starts only when its need (`GC_NEED_GNF4_S` 900 s on the box, `GC_NEED_E4B_S` 600 s unless set on the controller) plus a 300 s receipt margin fits
  before `GC_DEADLINE_EPOCH`. Its alarm never runs past that margin, so a tight guard still leaves time to fetch the receipts.
  A suite skipped for time fails the contract and is listed (`SKIPPED <suite> host-limited deadline`).
