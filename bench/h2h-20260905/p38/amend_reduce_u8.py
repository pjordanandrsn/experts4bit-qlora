# P38 amendment 4 (reducer, applied ONLY after u8_proof.json shows the predicate is True on the innermost module and False on
# PEFT's wrapper): U8 reads the unwrapped count when the receipt carries it; a receipt written before amendment 4 (no unwrapped
# field) is judged by the proof (same load + PEFT call, fresh process) together with its own census (>= 96 Params4bit expert
# stacks) and its own per-step bnb4bit backend counter (>= 96) -- the function U8's predicate gates. Thresholds unchanged (48/96).
import pathlib, shutil, json, sys
p = pathlib.Path(sys.argv[1]); bak = p.with_suffix(".py.pre_amend4")
if not bak.exists(): shutil.copy(p, bak)
s = p.read_text()
old = '''        b = r.get("unsloth_bnb4bit_modules")
        if isinstance(b, dict) and b.get("n_bnb4bit", 0) < 48:
            why.append(f"bnb4bit expert modules {b.get('n_bnb4bit')} < 48 (silent fallback?)")
'''
new = '''        b = r.get("unsloth_bnb4bit_modules")
        if isinstance(b, dict):
            if "n_bnb4bit_unwrapped" in b:            # amendment 4: the predicate on the innermost experts module
                if b["n_bnb4bit_unwrapped"] < 48:
                    why.append(f"bnb4bit expert modules (innermost) {b['n_bnb4bit_unwrapped']} < 48 (silent fallback?)")
            elif b.get("n_bnb4bit", 0) < 48:
                proof = U8_PROOF or {}
                inner_ok = (proof.get("after_peft_innermost") or {}).get("n_pred_true", 0) >= 48
                wrapper_zero = (proof.get("after_peft_wrapper") or {}).get("n_pred_true", 1) == 0
                own = c.get("Params4bit_expert_stacks", 0) >= 96 and r.get("kernel_calls_per_step_min", 0) >= 96
                if not (inner_ok and wrapper_zero and own):
                    why.append(f"bnb4bit expert modules {b.get('n_bnb4bit')} < 48 (silent fallback?)")
'''
assert s.count(old) == 1, "U8 block not found once"
s = s.replace(old, new)
# module-level proof holder + CLI flag
s = s.replace("import glob", "import glob\nU8_PROOF = None  # amendment 4: u8_proof.json (wrapper vs innermost predicate), loaded from --u8-proof", 1)
assert 'ap.add_argument("--md"' in s or "add_argument(\"--md\"" in s
s = s.replace('ap.add_argument("--md"', 'ap.add_argument("--u8-proof", default=None, help="amendment 4: u8_proof.json for receipts written before the unwrapped U8 count")\n    ap.add_argument("--md"', 1)
# load it right after parse_args
import re
m = re.search(r"^(\s*)a = ap\.parse_args\(\);\s*", s, re.M)
assert m, "parse_args line not found"
s = s[:m.end()] + "_set_u8_proof(a.u8_proof); " + s[m.end():]
s = s.replace("U8_PROOF = None  # amendment 4", "def _set_u8_proof(path):\n    global U8_PROOF\n    U8_PROOF = json.load(open(path)) if path else None\nU8_PROOF = None  # amendment 4", 1)
p.write_text(s); print("reducer amendment 4 applied to", p)
