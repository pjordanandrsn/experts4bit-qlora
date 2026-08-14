## What this changes

<!-- One or two sentences. "Closes #123" if it fixes an issue. -->

## Why

---

### If it touches expert math (ExpertsLoRA, an epilogue, a kernel lane)

- [ ] **Parity test against the reference**, not a loss curve. This adapter
      re-implements expert math so the delta lands pre-activation, so it owns the
      nonlinearity — a clamped/gated base must supply `_apply_gate`. Getting this
      wrong fails silently: the model trains and optimises the wrong function
- [ ] Compared against the pure-torch oracle (`dequant_ref` / `dequantize_mxfp4`),
      not against another accelerated lane — that measures similarity of
      rounding, not truth

### If it makes a performance or memory claim

- [ ] Cites a committed receipt, or is marked "measuring now"
- [ ] Ships a **self-pair**; a ratio inside the instrument's spread is not a measurement
- [ ] Two devices, or names the single architecture it holds for
- [ ] Benchmarked on **real text** — random token ids route to fewer experts far
      more unevenly and understate the fused lane by ~1.6–1.7x
- [ ] Reports the cells that lose

### Always

- [ ] `python -m pytest tests -q -k "not gpu"` passes
- [ ] Enabler return values are asserted (`n = enable_*(...)`; `0` looks identical
      to "silently on the per-expert loop")
- [ ] No private-lane paths or markers
