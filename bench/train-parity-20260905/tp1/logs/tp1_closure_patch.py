import pathlib, re, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text(); lines = t.split("\n")
# show the block
i0 = next(i for i, l in enumerate(lines) if "self.counts = {\"fused_grouped_lora\": 0" in l)
block = "\n".join(lines[i0:i0+26])
# late-binding fix: bind each wrapper's original as a default argument
new = block
new = re.sub(r"def w\(\*a, \*\*k\):", "def w(*a, _orig=orig, **k):", new)
new = re.sub(r"def w2\(\*a, \*\*k\):", "def w2(*a, _orig=orig, **k):", new)
new = new.replace("return orig(*a, **k)", "return _orig(*a, **k)")
assert new != block and new.count("_orig=orig") == 2 and new.count("_orig(*a, **k)") == 2, new
t = t.replace(block, new); p.write_text(t)
print("patched closure binding in", p); print("\n".join(l for l in new.split("\n") if "def w" in l or "_orig" in l or "orig = " in l))
