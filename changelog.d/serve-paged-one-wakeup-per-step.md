### The paged server wakes the event loop once per step, not once per stream

- `PagedEngine._dispatch` now collects a step's token chunks and finishes and hands them to the event loop in ONE
  `call_soon_threadsafe`. A new `_fanout` puts them into each stream's queue on the loop thread, in the engine's order. Before,
  every stream with output cost its own cross-thread wakeup. At 64 slots that `dispatch` segment was the largest piece of
  the exposed host gap between decode replays (`bench/h2h-2026-10-02/sc2e/decode64/README.md`, section (c)). The step
  trace already records `dispatch`, so a served read shows the change before and after. No speed is claimed here.
- Errors, shutdown and refused submits still wake the loop at once, as before. Stream queues are unbounded, so one
  hand-off cannot fail part-way through a batch.
- `tests/test_serve_paged.py::test_one_cross_thread_wakeup_per_step_and_per_stream_order_is_kept` runs 64 streams
  through the real engine thread. It asserts at most one wakeup per non-empty step, that a full decode step carries all
  64 streams in one wakeup, and that every stream gets its tokens in order and then exactly one `done`. Against the
  previous dispatch the same test counts 384 wakeups over 6 steps.
