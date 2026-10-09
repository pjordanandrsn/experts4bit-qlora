# Registered Alpaca source and prepared-data gate

Run selected Python `-I -S -B ra_alpaca.py --manifest /absolute/manifest.json
--out /absolute/fresh-receipt`. The manifest has exactly `schema: 1`,
`input_spec`, `stage`, `input_lock`, and the reviewed `input_lock_sha256`.
It uses the existing common-input schema and complete inventories.

The gate reads literal constants from the verified staged `tp4_alpaca.py`;
it never executes that builder. The raw file must be `alpaca_data_cleaned.json`
in the reviewed `datasets` root, with the registered source SHA. An independent
projection shuffles all 51,760 rows with seed 3407, selects 1,200 train and
48 eval rows, preserves the builder's missing-input convention, and reconstructs
its exact sorted-key compact UTF-8 JSON. All prepared bytes must match,
including metadata, ordering and Unicode encoding. Source, prepared data,
helpers, pins, stage and complete common inventories are checked again.

A successful receipt proves registered Alpaca source-byte and prepared-data
projection equality. It does not prove WikiText or calibration authority,
tokenizer execution, actual runtime consumption, publisher signatures, GPU
engagement or launch clearance. There are no downloads, release imports or
caller source/model/command overrides. Failure receipts and logs are retained.

This standalone draft is not wired into the fixed worker inventory or ABBA.
The new helper remains an unknown extra to that inventory until an explicit
reviewed integration; never silently admit it. Synthetic source controls use
an owned expected source SHA and narrowed pinset only, not production data.
