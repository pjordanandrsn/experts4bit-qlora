### SD2 build 5 (#1313): `E4B_PAGED_SPEC` in the paged server (opt-in)

- **The knobs.** `E4B_PAGED_SPEC=eagle3` with `E4B_PAGED_SPEC_K` (1–3) and `E4B_PAGED_SPEC_HEAD` (the EAGLE-3 head's
  directory) serves a request decoding alone with speculative greedy decode (lane SD2, `bench/sd2/PREREG-sd2.md` B6).
  The default is `off`, which installs nothing.
- **Refusals.** The server refuses a missing K or head, eager decode (the verify is a decode bucket), the decode
  lookahead, and a K set with speculation off.
- **The build.**
  - It checks the head against the pin and against the target's width and vocabulary.
  - The KV pool gets k alias slots and at least k + 1 scratch slots.
  - The draft, the auxiliary hooks and the verify buckets are installed before any graph.
  - The MoE residual license covers the verify row counts.
- **Reporting.** `/health` and `/stats` report a `spec` block: what was asked, the build, the post-verify graphs and
  the census (steps, drafted, accepted, τ_live, requests dropped to one token a step).
