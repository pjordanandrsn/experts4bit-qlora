"""Lane B771b's registered mutation: restore the shim's old routing, so a bound bucket of ONE row appends through
append_graph_t1 (graph_mode_init's scratch slot). The invariant test must then FAIL. Run from the e4b clone's root."""
from pathlib import Path

p = Path("experts4bit_qlora/engines/paged_attention.py")
t = p.read_text()
new = """            if (key.shape[0] > 1 or len(ctx.slots) > 1
                    or getattr(ctx.kv, "_g_sel", None) is not None):"""
assert t.count(new) == 1, "the fixed routing is not in this clone (wrong E4B_SHA?)"
p.write_text(t.replace(new, "            if key.shape[0] > 1 or len(ctx.slots) > 1:"))
print("mut_b771b: bound single-row buckets route to append_graph_t1 again")
