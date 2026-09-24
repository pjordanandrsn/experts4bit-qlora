"""Scratch evidence (not committed): main's _fused_over_stack vs this branch's, flag OFF, over every route the
int4/NF4 CPU stubs can drive -- outputs must be torch.equal and the kernel-call sequence identical."""
import importlib.util, sys, types, json
import torch, torch.nn.functional as F
sys.path.insert(0, "/w/e4b/tests")
from int4_pack_ref import pack_int4_b32, dequant_int4_ref
import test_int4_decode_a16 as T                      # the committed test's reference stubs

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

old = load("/w/old_hot_residency.py", "old_hr")
from experts4bit_qlora.engines import hot_residency as new
assert new.DECODE_A16[0] is False
log = []
stub = types.ModuleType("int4_b32")
stub.quant_x_rows = lambda x: (log.append("quant") or T._quant_ref(x))
stub.gemv_int4_b32 = lambda *a, **k: (log.append("gemv") or T._gemv_ref(*a, **k))
sys.modules["int4_b32"] = stub
import nf4_grouped
def fake_nf4(x, p, a, sizes, eids):
    log.append("nf4")
    ids = (eids if torch.is_tensor(eids) else torch.as_tensor(eids)).float()
    return x * (ids.repeat_interleave(torch.as_tensor(sizes)).unsqueeze(1) + 2.0)
nf4_grouped.gemm_4bit_grouped = fake_nf4
stores = T._stores()
res = []
def run(tag, fn_kw, x, ids, int4):
    outs, seqs = [], []
    for mod in (old, new):
        log.clear()
        if int4:
            o = mod._fused_over_stack(x, ids, T.FREED_GU, T.FREED_A, T.FREED_DN, T.FREED_A, T.SHAPES, True, F.silu, int4_stores=stores, **fn_kw)
        else:
            o = mod._fused_over_stack(x.float(), ids, None, None, None, None, (8, 16, 8, 8), False, F.silu, **fn_kw)
        outs.append(o); seqs.append(list(log))
    res.append({"route": tag, "equal": bool(torch.equal(*outs)), "calls_equal": seqs[0] == seqs[1], "calls": seqs[1]})
g = torch.Generator().manual_seed(1)
for trial in range(5):
    x1, ids1 = T._decode_rows(seed=100 + trial)
    run(f"int4 T=1 singleton (trial {trial})", {"singleton_groups": True}, x1, ids1, True)
    xp = (torch.randn(24, T.H, generator=g) * 0.5).to(torch.bfloat16); idp = torch.randint(0, T.E, (24,), generator=g)
    run(f"int4 T>1 host-grouped prefill (trial {trial})", {}, xp, idp, True)
    run(f"int4 batched decode GEMV, device grouping <=256 rows (trial {trial})", {"device_grouping": True}, xp, idp, True)
    xn = torch.randn(8, 8, generator=g); idn = torch.randint(0, 6, (8,), generator=g)
    run(f"NF4 grouped (trial {trial})", {}, xn, idn, False)
    run(f"NF4 singleton (trial {trial})", {"singleton_groups": True}, xn, idn, False)
ok = all(r["equal"] and r["calls_equal"] for r in res)
print(json.dumps({"all_equal_and_same_calls": ok, "n_cases": len(res), "cases": res}, indent=1))
sys.exit(0 if ok else 1)
