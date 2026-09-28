# bench/b511/mut_b511.py -- lane B511's ONE mutation (bench/b511/PREREG-b511.md). Run from the e4b clone's root.
# It removes the bound-bucket branch of Fp8PagedKV.kernel_args, so attention falls back to the per-tuple selector
# cache: a captured graph then bakes the capture-time (scratch) slot set, while the padded eager oracle still reads
# the live set. The replay-vs-padded-eager test MUST fail under it; if it passes, the test is inert.
p = 'experts4bit_qlora/engines/fp8_paged_kv.py'
t = open(p).read()
a = "        elif slots is not None and self._g_sel is not None \\\n"
if t.count(a) != 1:
    raise SystemExit(f"mutation target not found exactly once in {p}")
open(p, 'w').write(t.replace(a, "        elif False and slots is not None and self._g_sel is not None \\\n"))
print('mutated', p)
