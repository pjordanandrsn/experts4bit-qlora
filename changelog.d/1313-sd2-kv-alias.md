### SD2 build 2 (#1313): alias slots in the FP8 paged pool, for a speculative verify

- **Alias slots.** `Fp8PagedKV(..., alias_slots=n)` adds n slots after the scratch ones, with no block of their own.
  Lane SD2's verify of k + 1 rows runs as a decode bucket over the speculating slot and k aliases.
  - `alias_bind(slot, k)` copies the slot's block-table row into each alias and sets alias i's length to the slot's
    plus i + 1, in two device launches.
  - The batch append then writes row i into the slot's own blocks at its position plus i, and step-select reads the
    stagger.
- **Safeguards.** The aliases sit outside `scratch`, and that is asserted, so a padding row never lands on one.
- **After the step.** `set_len_device(slot, n)` sets the slot's length from a device scalar, the step's last length
  write. `note_len` updates the host mirror.
- **Unchanged without aliases.** With `alias_slots=0` (the default) nothing changes. `bench/sd2/PREREG-sd2.md` B1 and
  B2.
