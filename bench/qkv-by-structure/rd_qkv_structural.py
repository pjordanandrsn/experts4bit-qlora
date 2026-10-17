"""R-d: fused q/k/v selected by STRUCTURE instead of the class name, correctness only (no timing).

For each served family's committed fixture config, build ONE attention module (random weights, bf16) and compare
its output with two structural fusions against the unfused module:

  A  Qwen3-equivalent structure: children exactly {q_proj, k_proj, v_proj, o_proj, q_norm, k_norm}, unbiased
     projections, head_dim-wide q/k norms and a rotate-half rotary. Fused with e4b's own fused forward, as
     engines/qkv_fuse.py does for the class name it gates on today.
  B  Shared projection: q/k/v are nn.Linear over the same in_features. One fused GEMM serves all three when the
     module's OWN forward calls them on the same input tensor; any other call falls back to that projection's
     own slice. The family's forward is untouched (its norms, gated q, sinks, bias and rotary stay its own).

The bar is engines/qkv_fuse.py's: per output, max|delta| <= max|ref| * 2^-7 (the K6 relative frame). Bitwise
equality is reported beside it. The harness names classes to BUILD fixtures; neither selector reads a name.

    python rd_qkv_structural.py FIXTURES_DIR E4B_SRC OUT.json
"""
import json
import sys
import types

import torch

FIXTURES, E4B_SRC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, E4B_SRC)
from experts4bit_qlora.engines import qkv_fuse  # noqa: E402
from experts4bit_qlora.engines.glue_r2 import rotary_is_rotate_half  # noqa: E402
from transformers import AutoConfig  # noqa: E402

# fixture dir -> (modeling package, attention class, rotary class); the harness's build map, not a selector
SERVED = {
    "qwen3-30b-a3b": ("qwen3_moe", "Qwen3MoeAttention", "Qwen3MoeRotaryEmbedding"),
    "qwen3-235b-a22b-instruct-2507": ("qwen3_moe", "Qwen3MoeAttention", "Qwen3MoeRotaryEmbedding"),
    "qwen3.6-35b-a3b": ("qwen3_5_moe", "Qwen3_5MoeAttention", "Qwen3_5MoeTextRotaryEmbedding"),
    "olmoe-1b-7b-0924-instruct": ("olmoe", "OlmoeAttention", "OlmoeRotaryEmbedding"),
    "granite-3.1-3b-a800m-instruct": ("granitemoe", "GraniteMoeAttention", "GraniteMoeRotaryEmbedding"),
    "gpt-oss-20b": ("gpt_oss", "GptOssAttention", "GptOssRotaryEmbedding"),
    "gemma-4-26b-a4b-it": ("gemma4", "Gemma4TextAttention", "Gemma4TextRotaryEmbedding"),
    "mixtral-8x7b-instruct-v0.1": ("mixtral", "MixtralAttention", "MixtralRotaryEmbedding"),
    "ernie-4.5-21b-a3b-pt": ("ernie4_5_moe", "Ernie4_5_MoeAttention", "Ernie4_5_MoeRotaryEmbedding"),
}
BAR = 2.0 ** -7
DEV, DT = "cuda", torch.bfloat16


def selector_a(mod):
    kids = {n for n, _ in mod.named_children()}
    if kids != {"q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"}:
        return False, f"children {sorted(kids)}"
    if any(getattr(mod, p).bias is not None for p in ("q_proj", "k_proj", "v_proj")):
        return False, "biased projections"
    hd = int(getattr(mod, "head_dim", 0))
    if mod.q_norm.weight.shape[0] != hd or mod.k_norm.weight.shape[0] != hd:
        return False, "q/k norms are not head_dim wide"
    if not rotary_is_rotate_half(mod, hd):
        return False, "rotary is not rotate-half"
    return True, "matches"


def selector_a2(mod):
    """A plus a shape check: q/k/v out_features are heads x head_dim, so the fused forward's views are the module's."""
    ok, why = selector_a(mod)
    if not ok:
        return ok, why
    cfg, hd = mod.config, int(mod.head_dim)
    nh = getattr(cfg, "num_attention_heads", None)
    nkv = getattr(cfg, "num_key_value_heads", None) or nh
    want = (nh * hd, nkv * hd, nkv * hd)
    have = tuple(getattr(mod, p).out_features for p in ("q_proj", "k_proj", "v_proj"))
    if have != want:
        return False, f"q/k/v out {have} != heads x head_dim {want}"
    return True, "matches"


def fuse_a(mod):
    """engines/qkv_fuse.py's dense fusion body, minus its class-name check."""
    wq, wk, wv = mod.q_proj.weight, mod.k_proj.weight, mod.v_proj.weight
    qkv = torch.nn.Linear(wq.shape[1], wq.shape[0] + wk.shape[0] + wv.shape[0], bias=False, device=wq.device, dtype=wq.dtype)
    with torch.no_grad():
        qkv.weight.copy_(torch.cat([wq, wk, wv], 0))
    mod.qkv_proj, mod._fused_nq, mod._fused_nk, mod._fused_nv = qkv, wq.shape[0], wk.shape[0], wv.shape[0]
    del mod.q_proj, mod.k_proj, mod.v_proj
    mod.forward = types.MethodType(qkv_fuse._fused_forward, mod)


class _Shared:
    """One fused q/k/v GEMM, served to the three projections when the family's own forward calls them on the SAME
    input. Identity is object identity AND the tensor's ``_version``: the cache holds a reference to ``x``, so its
    id cannot be reused while cached, and ``_version`` catches an in-place write between the q and k calls. The
    cache is dropped once all three projections have taken their slice, so a later forward never reads it. Only
    the q call starts a fused GEMM; a k or v call that misses the cache computes its own slice (the unfused math)."""

    def __init__(self, parts):
        ws = [p.weight for p in parts]
        self.split = [w.shape[0] for w in ws]
        self.w = torch.cat(ws, 0).contiguous()
        bs = [p.bias if p.bias is not None else torch.zeros(w.shape[0], device=w.device, dtype=w.dtype) for p, w in zip(parts, ws)]
        self.b = torch.cat(bs, 0) if any(p.bias is not None for p in parts) else None
        self.x = self.ver = self.outs = None
        self.served = set()
        self.computes = self.hits = self.fallbacks = 0

    def drop(self):
        self.x = self.ver = self.outs = None
        self.served = set()


class _Slice(torch.nn.Module):
    def __init__(self, shared, i):
        super().__init__()
        self.s, self.i = shared, i
        lo = sum(shared.split[:i])
        self.lo, self.hi = lo, lo + shared.split[i]

    def forward(self, x):
        s = self.s
        cached = s.outs is not None and s.x is x and s.ver == x._version and self.i not in s.served
        if cached:
            s.hits += 1
            out = s.outs[self.i]
        elif self.i == 0:                                  # q starts one fused GEMM for all three
            s.drop()
            s.x, s.ver = x, x._version
            s.outs = torch.nn.functional.linear(x, s.w, s.b).split(s.split, dim=-1)
            s.computes += 1
            out = s.outs[0]
        else:                                              # a miss: this projection's own slice, the unfused math
            s.fallbacks += 1
            b = None if s.b is None else s.b[self.lo:self.hi]
            return torch.nn.functional.linear(x, s.w[self.lo:self.hi], b)
        s.served.add(self.i)
        if len(s.served) == 3:
            s.drop()                                       # consumed exactly once: nothing stale survives
        return out


def selector_b(mod):
    parts = [getattr(mod, p, None) for p in ("q_proj", "k_proj", "v_proj")]
    if not all(isinstance(p, torch.nn.Linear) for p in parts):
        return False, "q/k/v are not all nn.Linear"
    if len({p.in_features for p in parts}) != 1:
        return False, "different in_features"
    return True, "matches"


def fuse_b(mod):
    shared = _Shared([mod.q_proj, mod.k_proj, mod.v_proj])
    mod.q_proj, mod.k_proj, mod.v_proj = (_Slice(shared, i) for i in range(3))
    return shared


def build(name):
    pkg, attn_cls, rot_cls = SERVED[name]
    import importlib
    m = importlib.import_module(f"transformers.models.{pkg}.modeling_{pkg}")
    cfg = AutoConfig.from_pretrained(f"{FIXTURES}/{name}")
    tc = cfg.get_text_config()
    tc._attn_implementation = "eager"
    kinds = getattr(tc, "layer_types", None)
    layers = []
    if kinds:
        for want in ("full_attention", "sliding_attention"):
            if want in kinds:
                layers.append((kinds.index(want), want))
    else:
        layers = [(0, "full_attention")]
    return m, tc, getattr(m, attn_cls), getattr(m, rot_cls), layers


def rope(rot, tc, x, pos, kind):
    try:
        return rot(x, pos, layer_type=kind)
    except TypeError:
        return rot(x, pos)


def call(mod, x, pe):
    import inspect
    kw = {"hidden_states": x, "position_embeddings": pe, "attention_mask": None}
    if "shared_kv_states" in inspect.signature(mod.forward).parameters:
        kw["shared_kv_states"] = {}
    return mod(**kw)[0]


def proj_hooks(mod, names):
    got = {}
    hs = [getattr(mod, n).register_forward_hook(lambda m, a, o, n=n: got.setdefault(n, o.detach().clone())) for n in names]
    return got, hs


def rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30))


def run_one(name):
    m, tc, Attn, Rot, layers = build(name)
    rows = []
    for idx, kind in layers:
        torch.manual_seed(1234 + idx)
        ref = Attn(tc, idx).to(DEV, DT).eval()
        with torch.no_grad():
            for p in ref.parameters():
                if p.dim() == 2:
                    p.normal_(0, 0.02)
        rot = Rot(tc).to(DEV)
        state = {k: v.clone() for k, v in ref.state_dict().items()}
        for T in (1, 17):
            g = torch.Generator(device=DEV).manual_seed(7 + T)
            x = torch.randn(1, T, tc.hidden_size, device=DEV, dtype=DT, generator=g)
            pos = torch.arange(T, device=DEV)[None]
            pe = rope(rot, tc, x, pos, kind)
            names = [n for n in ("q_proj", "k_proj", "v_proj") if isinstance(getattr(ref, n, None), torch.nn.Module)]
            pref, hs = proj_hooks(ref, names)
            with torch.no_grad():
                out_ref = call(ref, x, pe)
                for h in hs:
                    h.remove()
                out_ref2 = call(ref, x, pe)
            row = {"family": name, "layer": idx, "kind": kind, "T": T, "ref_repeat_bitwise": bool(torch.equal(out_ref, out_ref2))}
            for tag, sel, fuse in (("A", selector_a, fuse_a), ("A2", selector_a2, fuse_a), ("B", selector_b, fuse_b)):
                mod = Attn(tc, idx).to(DEV, DT).eval()
                mod.load_state_dict(state)
                ok, why = sel(mod)
                cell = {"selected": ok, "why": why}
                if ok:
                    try:
                        shared = fuse(mod)
                        if tag == "B":
                            pgot, hs = proj_hooks(mod, ["q_proj", "k_proj", "v_proj"])
                        else:
                            qkv_out = {}
                            hs = [mod.qkv_proj.register_forward_hook(lambda m, a, o: qkv_out.setdefault("o", o.detach().clone()))]
                        with torch.no_grad():
                            out = call(mod, x, pe)
                        for h in hs:
                            h.remove()
                        if tag != "B":
                            q, k, v = qkv_out["o"].split([mod._fused_nq, mod._fused_nk, mod._fused_nv], dim=-1)
                            pgot = {"q_proj": q, "k_proj": k, "v_proj": v}
                        proj = max(rel(pgot[n], pref[n]) for n in names)
                        cell.update(proj_max_rel=proj, proj_within_bar=proj <= BAR,
                                    proj_bitwise=all(torch.equal(pgot[n], pref[n]) for n in names),
                                    out_rel=rel(out, out_ref), out_bitwise=bool(torch.equal(out, out_ref)))
                        if shared is not None:
                            cell.update(fused_computes=shared.computes, fused_hits=shared.hits, fallbacks=shared.fallbacks,
                                        cache_dropped=shared.outs is None)
                    except Exception as e:  # noqa: BLE001 -- a refusal or crash is a finding
                        cell.update(error=f"{type(e).__name__}: {e}"[:300])
                row[tag] = cell
            rows.append(row)
    return rows


def negative_tests():
    """B's safety cases on a stub: projections called on different tensors, an in-place write between calls, a
    repeated call after the cache was consumed. Each must fall back to the unfused math with no cache reuse."""
    torch.manual_seed(5)
    H, out = 256, {}
    lin = [torch.nn.Linear(H, n, bias=False).to(DEV, DT) for n in (256, 64, 64)]
    stub = torch.nn.Module()
    stub.q_proj, stub.k_proj, stub.v_proj = (torch.nn.Linear(H, n, bias=False).to(DEV, DT) for n in (256, 64, 64))
    with torch.no_grad():
        for a, b in zip((stub.q_proj, stub.k_proj, stub.v_proj), lin):
            a.weight.copy_(b.weight)
    shared = fuse_b(stub)
    x = torch.randn(1, 5, H, device=DEV, dtype=DT)
    y = torch.randn(1, 5, H, device=DEV, dtype=DT)
    with torch.no_grad():
        # cross-attention shape: q on x, k and v on y
        stub.q_proj(x)
        k, v = stub.k_proj(y), stub.v_proj(y)
        out["cross"] = {"hits": shared.hits, "fallbacks": shared.fallbacks,
                        "k_v_bitwise_unfused": bool(torch.equal(k, lin[1](y)) and torch.equal(v, lin[2](y)))}
        # an in-place write between the q and k calls
        h0, f0 = shared.hits, shared.fallbacks
        z = x.clone()
        stub.q_proj(z)
        z.add_(1.0)
        k = stub.k_proj(z)
        out["inplace"] = {"hits": shared.hits - h0, "fallbacks": shared.fallbacks - f0,
                          "k_bitwise_unfused": bool(torch.equal(k, lin[1](z)))}
        # a full q/k/v round, then k again: the cache was dropped, so it must not be reused
        h0, f0 = shared.hits, shared.fallbacks
        stub.q_proj(x), stub.k_proj(x), stub.v_proj(x)
        k2 = stub.k_proj(x)
        out["after_consume"] = {"hits": shared.hits - h0, "fallbacks": shared.fallbacks - f0, "cache_dropped": shared.outs is None,
                                "k_again_bitwise_unfused": bool(torch.equal(k2, lin[1](x)))}
    return out


def capture_test(name):
    """CUDA-graph capture of one B-fused attention module at T = 1, replayed on new data in the static input."""
    m, tc, Attn, Rot, layers = build(name)
    idx, kind = layers[0]
    torch.manual_seed(99)
    ref = Attn(tc, idx).to(DEV, DT).eval()
    with torch.no_grad():
        for prm in ref.parameters():
            if prm.dim() == 2:
                prm.normal_(0, 0.02)
    mod = Attn(tc, idx).to(DEV, DT).eval()
    mod.load_state_dict(ref.state_dict())
    shared = fuse_b(mod)
    rot = Rot(tc).to(DEV)
    x = torch.randn(1, 1, tc.hidden_size, device=DEV, dtype=DT)
    pos = torch.zeros(1, 1, dtype=torch.long, device=DEV)
    pe = tuple(t.clone() for t in rope(rot, tc, x, pos, kind))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(s):
        for _ in range(2):
            call(mod, x, pe)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        y = call(mod, x, pe)
    res = []
    for seed in (1, 2, 3):
        gen = torch.Generator(device=DEV).manual_seed(seed)
        with torch.no_grad():
            x.copy_(torch.randn(x.shape, device=DEV, dtype=DT, generator=gen))
            g.replay()
            torch.cuda.synchronize()
            r = call(ref, x, pe)
        res.append(rel(y, r))
    return {"family": name, "layer": idx, "replay_rel_vs_unfused_eager": res, "computes": shared.computes, "hits": shared.hits}


def main():
    rec = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "bar": BAR, "rows": []}
    import transformers
    rec["transformers"] = transformers.__version__
    for name in SERVED:
        try:
            rec["rows"] += run_one(name)
        except Exception as e:  # noqa: BLE001
            rec["rows"].append({"family": name, "build_error": f"{type(e).__name__}: {e}"[:300]})
        json.dump(rec, open(OUT, "w"), indent=1)
    rec["negative"] = negative_tests()
    rec["capture"] = []
    for name in ("qwen3-30b-a3b", "granite-3.1-3b-a800m-instruct"):
        try:
            rec["capture"].append(capture_test(name))
        except Exception as e:  # noqa: BLE001
            rec["capture"].append({"family": name, "error": f"{type(e).__name__}: {e}"[:300]})
    json.dump(rec, open(OUT, "w"), indent=1)
    print("negative:", json.dumps(rec["negative"]))
    for c in rec["capture"]:
        print("capture:", json.dumps(c))
    for r in rec["rows"]:
        if "build_error" in r:
            print(r["family"], "BUILD ERROR", r["build_error"])
            continue
        def c(t):
            x = r[t]
            if not x["selected"]:
                return f"{t}: no ({x['why']})"
            if "error" in x:
                return f"{t}: ERROR {x['error'][:80]}"
            extra = (f" c{x.get('fused_computes')}/h{x.get('fused_hits')}/fb{x.get('fallbacks')}{' dropped' if x.get('cache_dropped') else ' HELD'}"
                     if "fused_hits" in x else "")
            return (f"{t}: proj {x['proj_max_rel']:.1e} {'OK' if x['proj_within_bar'] else 'OVER'}{'=' if x['proj_bitwise'] else ''}"
                    f" out {x['out_rel']:.1e}{'=' if x['out_bitwise'] else ''}{extra}")
        print(f"{r['family'][:20]:20s} L{r['layer']:<2d} {r['kind'][:4]} T{r['T']:<2d} | {c('A')} | {c('A2')} | {c('B')}")


if __name__ == "__main__":
    main()
