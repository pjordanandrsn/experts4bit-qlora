### Read: SV3 (#1224): Qwen3.6-35B-A3B's linear-attention state pool measured beside its price, equal to the byte (bench and receipts only; licenses no change)

- `sv3-5090-1`: one RTX 5090, **$0.514**, teardown proven. Five arms OK (`bench/sv3/RESULTS-sv3.md`).
- **W2 HELD, exactly.** In all four Qwen3.6 arms the pool's `nbytes()` equalled the estimate's item to the byte (16,
  32, 17 and 2 slots, 30 layers).
- **W1, W3, W5 HELD.** Qwen3.6 eager +0.8% and with graphs +1.0%; gpt-oss-20b +1.8%. The peaks ran 0.2–0.3 GiB over
  their estimates, the allocator residual earlier serve runs measured.
- **W4 ALARM.** With one sequence and the default buckets, buckets 2–16 failed to capture on Qwen3.6, so that arm's
  peak is not used. The 968 MiB recorded against 931 MiB priced is a floor on the bucket cap's saving.
- **Process (maintainer note, 2026-10-06):** the box launched at 03:01:55Z from a registration that was not on main and under an
  open change request; #1225 and this read were then merged directly. The expectations were public before the data; no
  consequence was registered, so this read licenses no change. See `bench/sv3/RESULTS-sv3.md`.
