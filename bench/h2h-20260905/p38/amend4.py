# P38 amendment 4 (additive): U8 also evaluates unsloth_zoo's bnb4bit predicate on the INNERMOST experts module (PEFT wraps
# mlp.experts twice: experts.base_layer.base_layer holds the Params4bit stacks), and records both counts. Nothing removed.
import pathlib, shutil
p = pathlib.Path("/root/p38/p38_arm.py"); bak = pathlib.Path("/root/p38/p38_arm.py.pre_amend4")
if not bak.exists(): shutil.copy(p, bak)
s = p.read_text()
old = '''            bnb4 = {"n_experts_modules": len(bnb4), "n_bnb4bit": sum(bnb4)}
'''
new = '''            bnb4 = {"n_experts_modules": len(bnb4), "n_bnb4bit": sum(bnb4)}
            try:  # amendment 4: the same predicate on the innermost module (PEFT's ParamWrapper chain hides the Params4bit stacks)
                def _inner(m):
                    while hasattr(m, "base_layer"):
                        m = m.base_layer
                    return m
                _u = [bool(_moe_uses_bnb4bit_expert_weights(_inner(m))) for n, m in model.named_modules() if n.endswith("mlp.experts")]
                bnb4["n_bnb4bit_unwrapped"] = sum(_u)
                bnb4["inner_param_types"] = sorted({type(getattr(_inner(m), k, None)).__name__ for n, m in model.named_modules() if n.endswith("mlp.experts") for k in ("gate_up_proj", "down_proj")})
            except Exception as e:
                bnb4["unwrapped"] = f"unchecked: {e}"
'''
assert old in s and s.count(old) == 1; p.write_text(s.replace(old, new)); print("amend4 applied")
