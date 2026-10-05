### Docs: A2000 timings out as speed evidence (testbed-policy audit; docs and comments only)

- **Why.** The RTX A2000 is a correctness-only testbed (policy standing since 2026-07-27, re-stated 2026-10-05): an A2000
  timing may not seed a prediction, filter a candidate, or appear as speed evidence, whatever its label. An audit of both
  repositories found such timings in current-facing text here. The full findings table, including what is left for the
  owner, is [`docs/audits/a2000-timing-2026-10-05.md`](docs/audits/a2000-timing-2026-10-05.md).
- **Code comments** (no behaviour change). The docstrings and comments of `engines/batched.py`, `engines/fast.py`,
  `engines/moe_keep.py`, `engines/paged_attention.py`, `engines/pipelined.py`, `engines/triton_prebind.py`, `serve.py` and
  `train.py` no longer quote A2000 step times, tok/s, per-call ratios or host microseconds. Where a rented reading exists it
  is cited instead: TC1 amendment 21's RTX 5090 step for `E4B_MOE_KEEP_LAYERS`, and the RunPod A5000 / L40S pair for the
  hot-set host dependence. `serve.py`'s offload-path advice for `E4B_HOT_PER_LAYER=0` now rests on memory alone (1.02 GB),
  which is all the A2000 read can support. Memory figures stay.
- **Docs.** `INFERENCE.md`'s decode table keeps peak GPU and drops tok/s. `RESIDENCY-ENGINES.md` cites the rented A5000's
  +56 % / +120 % in place of the A2000's +40 %. `SERVING.md` and `STORAGE-MODES.md` drop A2000 decode rates.
  `MOE_RUNTIME_PORTABILITY.md` retracts its "within-box ratios" label. `OFFLOAD-TRANSFER-NOTES.md` and `METHODOLOGY.md`
  gain testbed notes; their records stand as measured.
- **STATUS and register.** The TC3 A2000 row's 69.9 s/step and Kimi-K3's 90-92 s per decode token leave `STATUS.md` and
  their rows' claim and unit; the headline values (peak memory) are unchanged. Four rows' notes drop A2000 timings: the
  P66 gather and fixed-tax rows, `e4b.serve.informed-hot-sets` and the TC1c route row.
- **Left for the owner** (listed in the audit file): the A2000 energy rows (`e4b.train.energy-honest.*`, quoted in the
  README), the PREREG/RESULTS records, and the runtime warning in `enable_fast_train`, which still quotes the A2000's 36 %.
