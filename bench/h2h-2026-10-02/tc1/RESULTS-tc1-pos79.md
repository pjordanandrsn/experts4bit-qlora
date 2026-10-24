# TC1 amendment 79: e4b 0.58.0 against Unsloth 2026.10.8 at the field recipe, NOT_READ

Pre-registration: [`../../tc1/TC1-PREREG.md`](../../tc1/TC1-PREREG.md), amendment 79 (merged as experts4bit-qlora#2121; launch commit
`1245fd0e`). It is amendment 69's box at the release pins: e4b 0.58.0 (`f15cb7d6`), grouped-nf4-gemm 0.51.0 (`30adb1bf`), unsloth and
unsloth_zoo 2026.10.8. It runs two profiled draws a side in ABBA order, in venv-unsloth (torch 2.12.1+cu130), at TC1's field recipe over
60 steps. Lane spend: USD 2.123 actual, against USD 5.93 registered worst (one box with a retry id).

- **`tc1-5090-pos79` (NOT_RUN, USD 0.01):** the launcher's pre-flight found the instance still loading after 600 s and destroyed it, so no
  command ran.
- **`tc1-5090-pos79-r1` (read, USD 2.113):** all four arms ran 60 steps, and the release proof found Unsloth's installed wheels
  byte-identical to PyPI's. The box's RESULTS is the reducer's, verbatim: [`receipts/tc1-5090-pos79-r1/public/RESULTS-tc1.md`](receipts/tc1-5090-pos79-r1/public/RESULTS-tc1.md)
  (VOID rows marked VOID, as TC1's results always carry them; it rebuilds from the committed copy byte for byte).

## The verdict

The reducer's amendment 79 rule reads:

> **Amendment 79 rule: NOT_READ**
> - two VALID profiled draws a side are registered: e4b ok, Unsloth missing

Both e4b draws are VALID and both Unsloth draws are VOID, so no position, ratio or equivalence reading is quoted. That includes e4b's own
draws: the registration quotes a position only against a VALID Unsloth side, and no figure from a VOID row is read here.

## Why the Unsloth draws are VOID

Each is VOID on TC1's standing engagement predicate (F15, `tc1_reduce.py`), which reads `torch._grouped_mm 0 calls/step < 6*48*accum 4 = 1152`.

What the receipts show:
- F15's counter wraps `torch._grouped_mm` (`tc1_arm.py`'s `_install_torch_grouped_mm`), and on both Unsloth arms it read 0 every step.
- Unsloth's own backend counters on the same receipts are non-zero (`unsloth_backend_calls_per_step_*`: `unsloth_grouped_mm` and
  `moe_bnb4bit_backend`). The selected MoE backend is `grouped_mm`, as requested.
- Each Unsloth arm's profiled op list (`tc1/qwen3pos79_unsloth_ckpt_unsloth_m_fv{,_d2}_profile.json`) names
  `unsloth_zoo::grouped_mm_triton` and its device kernel `_grouped_gemm_kernel` among its top ops. Neither list has an
  `aten::_grouped_mm`.

**Hypothesis (UNTESTED; read from op names, not from Unsloth's source):** Unsloth 2026.10.8's `grouped_mm` backend sends the expert GEMMs
to its own Triton grouped-GEMM op rather than to `torch._grouped_mm`, so F15's torch-level check cannot see the backend engage. Nothing
here says the Unsloth arms ran incorrectly, and nothing says they ran as registered: the predicate is what decides validity, and it
reads VOID.

## What is next

This is for the next release, not 0.58.1. A $0 check of which op the 2026.10.8 `grouped_mm` backend actually calls, then an amendment
that counts the dispatch that runs, with a mutant test that VOIDs an arm whose experts do not engage. Until then, Unsloth 2026.10.8 has
no TC1 position.
