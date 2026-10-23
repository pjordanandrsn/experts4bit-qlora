### serve_paged: a forced E4B_PAGED_GRAPHS=1 at the solver placement is refused at startup

- A tiered MoE layer syncs on the host every step, so decode graphs at `E4B_PAGED_PLACEMENT=solver` would capture
  nothing (every bucket raising and running eagerly). The forced knob is now refused by name, as every other
  unsupported graph mode at solver already was; `auto` and `0` keep eager decode there.
