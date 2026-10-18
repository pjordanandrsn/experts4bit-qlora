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

## Before the next reuse (a follow-up PR)

- `GC_REQUIRE`: a list of test ids that must PASS (a skip or an absence exits 30). Today a pass allows any skip, so the
  first run's review reads the required test from `logs/gnf4.xml` by hand.
- A time-left check before each suite that reads `GC_DEADLINE_EPOCH` with a receipt margin, as P133's `can_run` does.
