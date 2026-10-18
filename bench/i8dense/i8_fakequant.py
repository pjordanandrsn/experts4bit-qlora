#!/usr/bin/env python3
"""i8_fakequant.py -- the int8 weight-only premise (e4b#1313): the numeric effect of W8A16 on the B = 1 dense path,
without a kernel.

Every attention projection (the fused ``qkv_proj`` when present, else ``q_proj`` / ``k_proj`` / ``v_proj``, and
``o_proj``) and the ``lm_head`` is replaced IN PLACE by its symmetric per-output-row int8 round trip:

    scale_r = max_c |W[r, c]| / 127      (fp32; a zero row keeps scale 1)
    W'[r, c] = round(W[r, c] / scale_r) clamped to [-127, 127], times scale_r, cast back to W's dtype

The modules stay ``nn.Linear`` in bf16 on the same cuBLAS path, so only the weights' values change. Storing the round
trip in bf16 adds bf16's own rounding (at most 2^-8 of each value) on top of the int8 error; a W8A16 kernel that
dequantizes in fp32 would not, so this premise slightly OVERSTATES int8's error (the conservative direction). Per-output-row
scaling commutes with the q/k/v fusion (rows never mix), so the fused and the unfused model round the same values. The
router gate, the embeddings, the norms and the experts are untouched. An ``Int4Linear`` (an int4 lever) is refused:
the int8 premise is measured on the bf16 default, never on top of another lever.

    python i8_fakequant.py --self-test
"""
import sys

QMAX = 127


def roundtrip_(weight) -> dict:
    """Replace ``weight`` (2-D, out x in) in place by its per-output-row int8 round trip; return error statistics."""
    import torch
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    with torch.no_grad():
        w = weight.detach().float()
        amax = w.abs().amax(dim=1, keepdim=True)
        scale = torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))
        q = torch.round(w / scale).clamp_(-QMAX, QMAX)
        deq = q * scale
        err = (deq - w).abs()
        weight.copy_(deq.to(weight.dtype))
    return {"shape": list(weight.shape), "max_abs_err": float(err.max()), "max_err_over_half_step": float(
        (err / (scale / 2)).max()), "rel_fro_err": float(torch.linalg.norm(deq - w) / max(float(torch.linalg.norm(w)),
                                                                                         1e-30))}


def targets(model) -> list:
    """(name, module) for every attention projection and the lm_head; refuses a model whose projections are not plain
    ``nn.Linear`` (an int4 lever is on) or that has no attention projections at all."""
    import torch
    out = []
    for name, mod in model.named_modules():
        if type(mod).__name__ not in ("Qwen3MoeAttention", "Qwen3Attention"):
            continue
        projs = ["qkv_proj"] if hasattr(mod, "qkv_proj") else ["q_proj", "k_proj", "v_proj"]
        for p in projs + ["o_proj"]:
            m = getattr(mod, p, None)
            if type(m) is not torch.nn.Linear:
                raise RuntimeError(f"{name}.{p} is {type(m).__name__}, not nn.Linear: the int8 premise runs on the bf16 "
                                   "default only (an int4 lever is set?)")
            out.append((f"{name}.{p}", m))
    if not out:
        raise RuntimeError("no Qwen3 attention module found")
    head = getattr(model, "lm_head", None)
    if type(head) is not torch.nn.Linear:
        raise RuntimeError(f"lm_head is {type(head).__name__}, not nn.Linear")
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if emb is not None and emb.weight.data_ptr() == head.weight.data_ptr():
        raise RuntimeError("the lm_head is tied to the input embeddings: rounding it would round the embeddings too")
    out.append(("lm_head", head))
    return out


def apply(model) -> dict:
    """Round every target in place; return a record of what was rounded."""
    mods = targets(model)
    stats = {name: roundtrip_(m.weight) for name, m in mods}
    worst = max(s["max_err_over_half_step"] for s in stats.values())
    return {"modules": len(mods), "attention_modules": len(mods) - 1, "lm_head": True,
            "max_err_over_half_step": worst, "rel_fro_err_max": max(s["rel_fro_err"] for s in stats.values()),
            "per_module": stats}


def self_test() -> int:
    import torch
    torch.manual_seed(0)
    cases = []
    w = torch.randn(64, 48, dtype=torch.bfloat16)
    w[3].zero_()
    ref = w.float().clone()
    s = roundtrip_(w)
    cases.append(("error is within half a step per row", s["max_err_over_half_step"] <= 1.0 + 1e-3, s))
    cases.append(("a zero row stays zero", bool((w[3] == 0).all()), None))
    scale = ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / QMAX
    grid = torch.round(ref / scale).clamp(-QMAX, QMAX) * scale
    off = ((w.float() - grid).abs() - grid.abs() * 2.0 ** -8).max()
    cases.append(("the stored values are the int8 grid up to bf16's own rounding", float(off) <= 1e-12, float(off)))
    before = w.clone()
    roundtrip_(w)
    cases.append(("idempotent: a second round trip changes nothing", bool(torch.equal(before, w)), None))
    a, b, c = (torch.randn(n, 32, dtype=torch.bfloat16) for n in (16, 4, 4))
    fused = torch.cat([a, b, c]).clone()
    for t in (a, b, c):
        roundtrip_(t)
    roundtrip_(fused)
    cases.append(("per-row rounding commutes with the q/k/v fusion", bool(torch.equal(fused, torch.cat([a, b, c]))), None))
    try:
        roundtrip_(torch.randn(4, dtype=torch.bfloat16))
        cases.append(("a 1-D tensor is refused", False, None))
    except ValueError:
        cases.append(("a 1-D tensor is refused", True, None))

    class Qwen3MoeAttention(torch.nn.Module):
        def __init__(self, fused):
            super().__init__()
            if fused:
                self.qkv_proj = torch.nn.Linear(8, 24, bias=False)
            else:
                self.q_proj, self.k_proj, self.v_proj = (torch.nn.Linear(8, n, bias=False) for n in (16, 4, 4))
            self.o_proj = torch.nn.Linear(16, 8, bias=False)

    class Toy(torch.nn.Module):
        def __init__(self, fused, tied=False):
            super().__init__()
            self.layers = torch.nn.ModuleList([Qwen3MoeAttention(fused) for _ in range(2)])
            self.gate = torch.nn.Linear(8, 4, bias=False)
            self.embed = torch.nn.Embedding(10, 8)
            self.lm_head = torch.nn.Linear(8, 10, bias=False)
            if tied:
                self.lm_head.weight = self.embed.weight

        def get_input_embeddings(self):
            return self.embed

    m = Toy(fused=True)
    gate0 = m.gate.weight.detach().clone()
    r = apply(m)
    cases.append(("fused: 2 qkv + 2 o + the head", r["modules"] == 5 and r["attention_modules"] == 4, r["modules"]))
    cases.append(("the router gate is untouched", bool(torch.equal(gate0, m.gate.weight)), None))
    cases.append(("unfused: 2 x (q, k, v, o) + the head", apply(Toy(fused=False))["modules"] == 9, None))
    for name, model in (("a tied head is refused", Toy(fused=True, tied=True)), ("no attention is refused", torch.nn.Linear(2, 2))):
        try:
            apply(model)
            cases.append((name, False, None))
        except RuntimeError:
            cases.append((name, True, None))

    class Int4Linear(torch.nn.Module):
        pass

    m = Toy(fused=True)
    m.layers[0].qkv_proj = Int4Linear()
    try:
        apply(m)
        cases.append(("an int4 lever is refused", False, None))
    except RuntimeError:
        cases.append(("an int4 lever is refused", True, None))
    bad = [(n, d) for n, ok, d in cases if not ok]
    print(f"i8_fakequant self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    print(__doc__)
