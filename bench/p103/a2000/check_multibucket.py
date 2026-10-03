"""The per-slot linear state under CUDA graphs as enable_decode_graphs drives it, on any CUDA card (no fp8): an all-linear
dense Qwen3.5, the pool warmed on a scratch slot and frozen, graphs captured for buckets 1, 2, 4 on scratch slots with
one selector per bucket (each warmed twice on a side stream first), then a scheduler-like trace -- prefills between
replays, a shrinking active set, a recycled slot -- replayed, against the same padded steps run EAGERLY on a twin
model through the same selector path. Bit for bit."""
import types

import torch

from experts4bit_qlora.engines import linear_state, paged_attention

LIN = "linear_attention"
NOMASK = {"linear_attention": None, "full_attention": None}
B, SCRATCH = 4, [4, 5, 6, 7]
BUCKETS = (1, 2, 4)


def model():
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
                            linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                            linear_conv_kernel_dim=4, layer_types=[LIN] * 3, max_position_embeddings=256)
    return Qwen3_5ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def ctx(slots, mode="decode", sel=None):
    c = paged_attention.PagedAttentionContext(kv=None, slots=list(slots), mode=mode)
    if sel is not None:
        c.kv = types.SimpleNamespace(_g_sel=sel)
    return c


def fwd(m, c, ids, pos):
    prev = paged_attention.set_context(c)
    try:
        return m(input_ids=ids, position_ids=pos, use_cache=False, attention_mask=NOMASK).logits[:, -1]
    finally:
        paged_attention.set_context(prev)


class Engine:
    def __init__(self, graphs):
        self.m = model()
        self.pool = linear_state.install(self.m, n_slots=B + len(SCRATCH))
        self.graphs = graphs
        with torch.no_grad():                                   # the warm-up: a 1-token prefill on a scratch slot
            fwd(self.m, ctx([SCRATCH[0]], "prefill"), torch.zeros(1, 1, dtype=torch.long, device="cuda"),
                torch.zeros(1, 1, dtype=torch.long, device="cuda"))
        self.pool.reset(SCRATCH[0])
        self.pool.frozen = True
        self.buf = {}
        for b in BUCKETS:
            sel = torch.tensor(SCRATCH[:b], device="cuda")
            ids = torch.zeros(b, 1, dtype=torch.long, device="cuda")
            pos = torch.zeros(b, 1, dtype=torch.long, device="cuda")
            c = ctx(SCRATCH[:b], sel=sel)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.no_grad(), torch.cuda.stream(side):
                for _ in range(2):
                    fwd(self.m, c, ids, pos)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = out = None
            if graphs:
                g = torch.cuda.CUDAGraph()
                with torch.no_grad(), torch.cuda.graph(g):
                    out = fwd(self.m, c, ids, pos)
                torch.cuda.synchronize()
            self.buf[b] = dict(sel=sel, ids=ids, pos=pos, ctx=c, g=g, out=out)

    def prefill(self, slot, prompt):
        self.pool.reset(slot)
        with torch.no_grad():
            fwd(self.m, ctx([slot], "prefill"), torch.tensor([prompt], device="cuda"),
                torch.arange(len(prompt), device="cuda")[None])
        self.pool.mark([slot])

    def decode(self, slots, toks, poss):
        n = len(slots)
        b = min(x for x in BUCKETS if x >= n)
        bf = self.buf[b]
        pad = b - n
        bf["sel"].copy_(torch.tensor(slots + SCRATCH[:pad], device="cuda"))
        bf["ids"].copy_(torch.tensor([[t] for t in toks] + [[0]] * pad, device="cuda"))
        bf["pos"].copy_(torch.tensor([[p] for p in poss] + [[0]] * pad, device="cuda"))
        with torch.no_grad():
            if self.graphs:
                bf["g"].replay()
                out = bf["out"]
            else:
                out = fwd(self.m, bf["ctx"], bf["ids"], bf["pos"])
        torch.cuda.synchronize()
        return out[:n].clone()


def trace(e):
    """Four sequences, staggered prefills between decode steps, one finishing early and its slot recycled."""
    g = torch.Generator().manual_seed(7)
    prompts = {s: torch.randint(0, 256, (int(torch.randint(3, 12, (1,), generator=g)),), generator=g).tolist()
               for s in range(5)}
    lens, outs, active = {}, [], []
    slot_of = {0: 0, 1: 1}
    for rid in (0, 1):
        e.prefill(slot_of[rid], prompts[rid]); lens[rid] = len(prompts[rid]); active.append(rid)
    for step in range(14):
        if step == 2:
            slot_of[2] = 2; e.prefill(2, prompts[2]); lens[2] = len(prompts[2]); active.append(2)
        if step == 4:
            slot_of[3] = 3; e.prefill(3, prompts[3]); lens[3] = len(prompts[3]); active.append(3)
        if step == 7:
            active.remove(1)                                    # rid 1 finishes; rid 4 takes its slot
            slot_of[4] = slot_of[1]; e.prefill(slot_of[4], prompts[4]); lens[4] = len(prompts[4]); active.append(4)
        if step == 10:
            active = active[:1]                                 # shrink to one row
        slots = [slot_of[r] for r in active]
        toks = [(17 * step + 5 * r) % 256 for r in active]
        poss = [lens[r] + step for r in active]
        outs.append((tuple(active), e.decode(slots, toks, poss)))
    return outs


g_outs = trace(Engine(graphs=True))
e_outs = trace(Engine(graphs=False))
bad = 0
for i, ((ra, a), (rb, b)) in enumerate(zip(g_outs, e_outs)):
    assert ra == rb
    if not torch.equal(a, b):
        bad += 1
        print(f"step {i} rows {ra}: replay != eager (maxabs {(a.float() - b.float()).abs().max():.3e})")
print("STEPS", len(g_outs), "BAD", bad)
import os
if os.environ.get("OUT"):
    torch.save({"g": [(r, o.float().cpu()) for r, o in g_outs], "e": [(r, o.float().cpu()) for r, o in e_outs]}, os.environ["OUT"])
    print("SAVED", os.environ["OUT"])
