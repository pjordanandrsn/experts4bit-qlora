import sys, torch
sys.path.insert(0, "/root/e4b/tests")
from experts4bit_qlora.engines import hot_residency
hot_residency.DEVICE_GROUPING[0] = True
import test_prefill_graph_gpu as t
from experts4bit_qlora.engines.paged_attention import set_context
model, prompts = t._model(), t._prompts()
r = t._runner(model)
r.enable_prefill_graph(t.T)
pg = r._prefill_graph
ctx = r.ctx
pos = torch.arange(t.T, device="cuda")[None]

def eager(p):
    ctx.mode, ctx.slots = "prefill", [99]; r._mode(True); prev = set_context(ctx)
    try:
        with torch.no_grad():
            ctx.drop(99)
            o = model(input_ids=torch.tensor(p, device="cuda")[None], position_ids=pos, use_cache=False)
            return o.logits.clone()
    finally:
        ctx.drop(99); set_context(prev); ctx.mode = "decode"; r._mode(False)

def replay(p):
    pg["ids"].copy_(torch.tensor(p, dtype=torch.long)[None]); pg["graph"].replay(); torch.cuda.synchronize()
    return pg["logits"].clone()

E = {n: eager(prompts[n][:16]) for n in "ABD"}
for seq in (["A", "D"], ["A", "B", "D"], ["D", "D", "A", "D"]):
    out = []
    for n in seq:
        g = replay(prompts[n][:16])
        out.append(f"{n}:{torch.equal(g, E[n])}:{(g.float()-E[n].float()).abs().max().item():.3g}")
    print("SEQ", seq, out)
# does an eager forward change after replays?
print("EAGER_STABLE", {n: torch.equal(eager(prompts[n][:16]), E[n]) for n in "ABD"})
print("PROMPTS_EQUAL_PREFIX", prompts["A"][:4], prompts["D"][:4], prompts["B"][:4])
