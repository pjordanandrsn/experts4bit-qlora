### `serve_paged`: an opt-in decode lookahead, `E4B_PAGED_DECODE_LOOKAHEAD=1` (lane P118, #1313; off by default)

- **What.** Each decode step used to read its tokens back before the step ended, so the GPU idled while the host
  emitted them, retired finished requests, planned the next step and copied its inputs in. With the switch on, the
  serving scheduler (`ContinuousScheduler(lookahead=True)`) issues the next decode step before it reads the previous
  one back. New runner entry points do the split: `PagedModelRunner.issue_decode` enqueues a step whose input ids are
  gathered on the device from a table of each slot's newest token, and `collect_decode` reads a queued step's tokens
  back through pinned memory and an event. `Fp8PagedKV.graph_bucket_load` takes an optional host `staging` buffer, so
  a step can be loaded while the one before it is still queued.
- **Contract.** A step's device inputs (ids, positions, slots, padding) are the synchronous step's, so the tokens and
  finish reasons are too. A sequence is never issued past `max_new_tokens`. A sequence that stops on a stop id has
  had one more step computed, whose token is discarded (`stats()["lookahead_discarded"]`), and frees its slot once
  that step is collected. When requests are waiting for a slot that the queued step frees, that step is collected
  before the next admission, so admission matches the synchronous path. `PagedModelRunner.run_decode` keeps its
  contract and refuses while a lookahead step is queued, as `free_slot` does for a sequence with one in flight.
- **Who is affected.** Nobody by default. The switch needs decode graphs (refused at startup otherwise) and does not
  run with a slot controller. `/health` reports `engine.decode_lookahead`.
- **Not measured.** No speed number yet: lane P118 reads it on an RTX 5090 (W1 and W16, tokens identical) before any
  default changes.
- **Tests.** `tests/test_decode_lookahead.py` (CPU, plus CUDA when present) checks:
  - the scheduler on a fake runner: the synchronous streams; the length bound; at most one discarded step per stop;
    the queued-request drain; an abort with a step queued;
  - the runner on a real `Fp8PagedKV` and the buckets `enable_decode_graphs` builds, with a stand-in graph: the
    synchronous streams and bucket statistics, chained replays included; a host edit resyncs; misuse is refused;
  - the switch's parsing, and an HTTP completion stopping on EOS both ways.

  `tests/test_decode_lookahead_gpu.py` (sm_89+) runs the tiny Qwen3 through captured graphs and the padded eager step:
  identical tokens, buckets and KV lengths, with the overlap engaged.
