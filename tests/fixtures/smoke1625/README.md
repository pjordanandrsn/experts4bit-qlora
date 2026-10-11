# smoke1625: real K19 dispatch records from #1625's A2000 serve smoke

`pass2_summary.json` is byte-identical to the receipt of the GPU serve smoke that gated #1625
(`E4B_K19_PREFILL_BIG_M`), run on the project's RTX A2000 (sm_86) on 2026-10-11, correctness only (no timing in it is
read). sha256 `1f4960b1d6b05f774c7fb8c0c922f1b85aab84b333263b3e113c1404b32ab9bb`; the tests that read it check that
digest first.

Code under the smoke: e4b #1625 at 1d8c5b84, grouped-nf4-gemm #542's merge 57f53e19, torch 2.8.0+cu128.

**What it holds.** Twelve cells: the int4 stores of `qwen3_moe`, `granitemoe` and `mixtral` (two layers, four experts),
each at the knob `0`, `0r` (knob 0 again in a fresh process), `32` and `64`, with `DEVICE_GROUPING` forced and
`E4B_INT4_PREFILL=k19`. Per cell:
- `serve_k19_seen`: `K19_DISPATCH_SEEN` over the serve path (the scheduler's prefill and decode steps, two requests);
- `seen_knob0` and `seen_own`: the same counter over one forced 1,024-row forward at knob 0 and at the cell's knob, in the
  cell's own process;
- `routes`: `serve_paged.prefill_routes()` after the serve run.

At knob 0 the 1,024-row calls take the wide one-launch table (`one_launch|lean|gt256`; `E4B_INT4_WIDE_TILES=auto`,
lane P122's `next_pow2(E) * next_pow2(R) <= 128 * 512`), not the chained one. At 32 and 64 they take
`split|lean|gt256`: the split takes precedence over the wide table.

**The wrapper's `FAIL`.** The top-level `status` and every cell's `status` read `FAIL`. Each entry in `checks_failed` is
the smoke wrapper's own knob-0 expectation (`chained|lean|gt256`), which was wrong; the wide table above is the
release's default at this size. The assertions the smoke gated #1625 on hold in every cell: the split key at 32 and
64, and logits `torch.equal` to knob 0 in-process and across processes (`cross_process_logits`).

**Who reads it.** `tests/test_p131_real_records.py` runs these records through lane P131's decode check before its
synthetic departures. Records written to an author's belief about the route are how a gate encodes the wrong table;
these were written by the route.
