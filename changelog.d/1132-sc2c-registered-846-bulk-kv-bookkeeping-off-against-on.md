### SC2c registered (#846): bulk KV bookkeeping OFF against ON on one 5090, with a per-step trace in both arms; the stall census behind it (bench and tests only)

- **The census** (`bench/stall-census-2026-10-05`, exploratory, $0).
  - **Batch growth.** SC2b's fitted per-prefill stall carried batch growth; bucket-controlled, it is 0.218 / 0.224 s
    on the ON servers, not 0.262 / 0.269.
  - **The prefill step does not grow under load.** Admission to first token holds at 157–170 ms.
  - **The bookkeeping.** One request's KV bookkeeping is ~13.5k host-issued launches at SC2b's geometry, against 66
    in bulk, bitwise identical (counted and checked on the NAS A2000, a correctness testbed: no A2000 timing is read,
    e4b#1133).
  - **Inferred from box F's traces and P107's 5090 receipt, not measured:** with the forward near ~42 ms of device
    time, ~120 ms of the prefill step is host work, most of it the prompt's flush (8,688 launches at box F's ~12 µs).
    The first graphed decode's block claims add ~55 ms.
  - **Projected:** `capsim.py`, a scheduler model calibrated on SC2b's rows, puts the ceiling at 2–4 req/s without the
    bookkeeping, depending on the 512-token forward's device time.
- **SC2c** (`bench/sc2/SC2c-PREREG.md`, reviewed and approved by the maintainer agent on #1132).
  - **The box.** Box H runs `E4B_PAGED_BULK_KV=0|1` paired, with the prefill graph at `auto` in both arms and
    `E4B_PAGED_STEP_TRACE` on.
  - **Gates.** ROUTES; ENGAGED (`/health`'s `kv_bookkeeping` counts); DETERMINISM; IDENTITY; PROMPTS.
  - **Predictions.** P1 stall ON/OFF ≤ 0.6; P2 TTFT ≥ 1.4×; P3 TPOT unchanged; P4 ceiling ≥ 2; P5 ceiling ≥ 4;
    P6 no regression.
  - **Licence:** gates + P6 + TTFT ≥ 1.10×.
  - **Tools.** `sc2c_reduce.py` (13 self-test cases) and `sc2c_census.py` (7) read the step trace's decomposition and
    the bucket-controlled stall. ROUTES also reads `prefill_routes.seen` (e4b#1129): every expert GEMM above 256 rows
    on K19 and every prefill attention call on flash, as the forward took them.
  - **Harness.** `sc1_run.sh` / `sc1_drive.sh` gain box H; `staged.sha256` regenerated.
