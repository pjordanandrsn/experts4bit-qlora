# `p147-a2000-1`: int8 and NF4 round trips of Mixtral-8x7B's attention projections, read against the W8A16 split-premise rule

The lab RTX A2000 12GB, 2026-10-19 03:37:50Z to 06:10:38Z, at $0. Code: e4b `15f8345f` (the registration's merge, #1790),
grouped-nf4-gemm 0.47.1 (`38b30888`); torch 2.8.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2 (`versions.txt`). The
staged closure matched `staged-p147.sha256` (`closure.log`).

**Verdict: READ** (`verdict_p147.json`): I8ATTN GO, NF4ATTN PARK; the instrument checks passed (`checks.json`), and A′ was
bit-identical to A. `RESULTS-p147.md` has the tables and the predictions.

The maintainer re-derived `verdict_p147.json` byte-identical from the receipt store with the registered reducer. Every
record the box wrote is here; the run used no rented host, so it has no launcher receipt or ledger row. `SHA256SUMS`
covers every file here.
