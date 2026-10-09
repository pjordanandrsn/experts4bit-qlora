# Capacity trace binding

After all three frozen SC2 processes finish and the server drains, the owned
wrapper stops its engine thread. The native engine closes and flushes its step
tracer before the supervisor reads or hashes either trace. A live thread or
buffered rows fail this barrier; the owned server is still cleaned up on failure.

`ra_trace.py` joins all 188 HTTP completion IDs to unique native request IDs and
scheduler IDs, checks token usage and derived timings, and checks that the three
workloads did not overlap. Client and server clock origins are never equated.
The SSE checks use only the frozen driver's six-decimal rounding precision.

Step rows must be contiguous and reconcile admissions, prefill tokens/replays,
decode rows/pieces, scheduler totals and graph counters. Missing or nonpositive
CUDA forward intervals, CUDA errors and eager fallbacks fail. The last bucket of
a chained decode step is checked against its final piece, as SC2e specifies.
Native step rows contain no request IDs: this is an aggregate step reconciliation,
not attribution of an individual step to a request.

`trace-close.json` and `trace-binding.json` retain the closure observations and
trace hashes. These checks do not establish wheel/import provenance, complete
feature coverage or release clearance. Missing native APIs or changed schemas
fail and need a reviewed adapter; the battery and instruments stay fixed.

CPU tests cover mutations, owned-process cleanup, actual native engine closure
and projection of the committed SC2e traces. The real HTTP/RTX 5090 executor has
not run; the CPU engine control has no CUDA events.
