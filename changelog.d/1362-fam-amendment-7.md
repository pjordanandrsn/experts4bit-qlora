### FAM Amendment 7 (#1362): Mixtral's server pool sized to the instrument

`fam-mixtral-prove-1` ran out of GPU memory building the server's own KV pool, which the instrument never uses, beside
Mixtral's 26.8 GiB of NF4 weights. Mixtral's processes now build that pool at 768 tokens a slot, which still covers the
instrument's prompt and positions, and its proof is guarded 2.0 h, sized from the failed run's measured fetch and bake.
