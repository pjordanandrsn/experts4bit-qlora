### SD2 build 4 (#1313): the speculative loop in the paged runner

- **The loop.** `engines/spec_decode.py`'s `SpecDecoder` runs speculative greedy decode for the one request decoding
  alone (lane SD2, `bench/sd2/PREREG-sd2.md` B1, B2, B5):
  - `k` shrinks to the request's budget and the slot's capacity;
  - the verify is a decode bucket whose rows share the slot through alias slots;
  - on the device: the accept (leading matches only), the length `base + a + 1` written last, and the draft's
    extend-and-chain;
  - one host read per step.
- **The runner.** `PagedModelRunner.enable_speculation` comes before the graphs:
  - the verify buckets (2 .. k + 1 rows) are captured beside the decode buckets, and plain steps never pick one;
  - the post-verify step is captured per row count;
  - the auxiliary hooks write prefill and decode states;
  - a prompt that completes alone starts the draft.
- **Leaving speculation.** A request that is no longer alone, or has no room, drops its draft and decodes at T == 1
  for the rest of its life.
- **Off by default.** Nothing is engaged unless `enable_speculation` is called. The server's knobs come in build 5.
