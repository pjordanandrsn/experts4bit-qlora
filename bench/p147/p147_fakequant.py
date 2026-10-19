#!/usr/bin/env python3
"""p147_fakequant.py -- P147's round trips (bench/p147/PREREG-p147.md): the numeric effect of storing Mixtral-8x7B's
attention projections (or its lm_head) in int8 or NF4, without a kernel.

The targets are Mixtral's attention projections (``q_proj``, ``k_proj``, ``v_proj`` and ``o_proj`` of every
``MixtralAttention``: 128 modules over 32 layers) or its ``lm_head``. Each target's weight is replaced IN PLACE by:
  - ``int8``: ``i8_fakequant.roundtrip_`` (the W8A16 premise's rule, at its pinned bytes): the per-output-row symmetric
    int8 round trip, scale = row max / 127;
  - ``nf4``: bitsandbytes' ``quantize_4bit`` then ``dequantize_4bit``, NF4, block size 64 (the rule e4b's experts use).
The modules stay bf16 ``nn.Linear``, so only the weights' values change. The router gate, the embeddings, the norms and
the experts are untouched; a module that is not a plain ``nn.Linear`` is refused.

``snapshot`` keeps a host copy and sha256 of every target, and ``restore`` puts them back and checks each sha256: an arm
never starts from another arm's rounded weights.

    python p147_fakequant.py --self-test
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "i8dense")]   # staged flat on the box; the repo keeps it in bench/i8dense
import i8_fakequant  # noqa: E402  (the W8A16 premise's round trip, staged at its pinned bytes)

ATTENTION_CLASSES = ("MixtralAttention",)
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
FORMATS = ("int8", "nf4")
SCOPES = ("attn", "head")
NF4_BLOCKSIZE = 64


def nf4_roundtrip_(weight) -> dict:
    """Replace ``weight`` (2-D) in place by its NF4 round trip (bitsandbytes, block size 64); return error statistics."""
    import torch
    import bitsandbytes.functional as F
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    with torch.no_grad():
        w = weight.detach()
        q, state = F.quantize_4bit(w.contiguous(), blocksize=NF4_BLOCKSIZE, quant_type="nf4")
        deq = F.dequantize_4bit(q, state, quant_type="nf4").to(w.dtype).reshape(w.shape)
        err = (deq.float() - w.float())
        rel = float(torch.linalg.norm(err) / max(float(torch.linalg.norm(w.float())), 1e-30))
        weight.copy_(deq)
    return {"shape": list(weight.shape), "max_abs_err": float(err.abs().max()), "rel_fro_err": rel}


ROUNDTRIP = {"int8": i8_fakequant.roundtrip_, "nf4": nf4_roundtrip_}


def targets(model, scope: str) -> list:
    """(name, module) for ``scope``: ``attn``, every Mixtral attention projection; ``head``, the lm_head only."""
    import torch
    if scope not in SCOPES:
        raise ValueError(f"scope {scope!r} is not one of {SCOPES}")
    if scope == "head":
        head = getattr(model, "lm_head", None)
        if type(head) is not torch.nn.Linear:
            raise RuntimeError(f"lm_head is {type(head).__name__}, not nn.Linear")
        emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if emb is not None and emb.weight.data_ptr() == head.weight.data_ptr():
            raise RuntimeError("the lm_head is tied to the input embeddings: rounding it would round the embeddings too")
        return [("lm_head", head)]
    out = []
    for name, mod in model.named_modules():
        if type(mod).__name__ not in ATTENTION_CLASSES:
            continue
        for p in PROJECTIONS:
            m = getattr(mod, p, None)
            if type(m) is not torch.nn.Linear:
                raise RuntimeError(f"{name}.{p} is {type(m).__name__}, not nn.Linear: the premise runs on the bf16 default")
            out.append((f"{name}.{p}", m))
    if not out:
        raise RuntimeError("no MixtralAttention module found")
    return out


def _sha(t) -> str:
    return hashlib.sha256(t.detach().contiguous().view(-1).view(dtype=__import__("torch").uint8).cpu().numpy().tobytes()).hexdigest()


def snapshot(model) -> dict:
    """A host copy and the sha256 of every weight any arm may touch (the 128 projections and the lm_head)."""
    mods = targets(model, "attn") + targets(model, "head")
    return {name: (m.weight.detach().to("cpu", copy=True), _sha(m.weight)) for name, m in mods}


def restore(model, snap: dict) -> dict:
    """Put every snapshotted weight back and check its sha256; returns {"restored": n, "mismatched": [names]}."""
    mods = dict(targets(model, "attn") + targets(model, "head"))
    bad = []
    for name, (copy, sha) in snap.items():
        w = mods[name].weight
        with __import__("torch").no_grad():
            w.copy_(copy.to(w.device))
        if _sha(w) != sha:
            bad.append(name)
    return {"restored": len(snap), "mismatched": bad}


def apply(model, fmt: str, scope: str) -> dict:
    """Round every target in ``scope`` to ``fmt`` in place; return the census and the error statistics."""
    if fmt not in FORMATS:
        raise ValueError(f"format {fmt!r} is not one of {FORMATS}")
    mods = targets(model, scope)
    stats = {name: ROUNDTRIP[fmt](m.weight) for name, m in mods}
    return {"format": fmt, "scope": scope, "modules": len(mods), "lm_head": scope == "head",
            "names": [n for n, _ in mods], "rel_fro_err_max": max(s["rel_fro_err"] for s in stats.values()),
            "per_module": stats}


def self_test() -> int:
    import torch
    torch.manual_seed(0)
    cases = []

    class MixtralAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(64, 64, bias=False)
            self.k_proj = torch.nn.Linear(64, 16, bias=False)
            self.v_proj = torch.nn.Linear(64, 16, bias=False)
            self.o_proj = torch.nn.Linear(64, 64, bias=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = MixtralAttention()
            self.gate = torch.nn.Linear(64, 8, bias=False)

    class Toy(torch.nn.Module):
        def __init__(self, n=3, tied=False):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer() for _ in range(n)])
            self.embed = torch.nn.Embedding(32, 64)
            self.lm_head = torch.nn.Linear(64, 32, bias=False)
            if tied:
                self.lm_head.weight = self.embed.weight
            self.to(torch.bfloat16)

        def get_input_embeddings(self):
            return self.embed

    m = Toy()
    gate0 = m.layers[0].gate.weight.detach().clone()
    head0 = m.lm_head.weight.detach().clone()
    snap = snapshot(m)
    cases.append(("the snapshot covers 4 x 3 projections + the head", len(snap) == 13, len(snap)))
    r = apply(m, "int8", "attn")
    cases.append(("int8 attn: the 12 projections, the head untouched", r["modules"] == 12 and not r["lm_head"]
                  and bool(torch.equal(head0, m.lm_head.weight)), r["modules"]))
    cases.append(("the router gate is untouched", bool(torch.equal(gate0, m.layers[0].gate.weight)), None))
    cases.append(("int8 changes the weights", not bool(torch.equal(snap["layers.0.self_attn.q_proj"][0],
                                                                      m.layers[0].self_attn.q_proj.weight)), None))
    back = restore(m, snap)
    cases.append(("restore puts every weight back, sha256-checked", back == {"restored": 13, "mismatched": []}, back))
    r = apply(m, "nf4", "attn")
    q0 = m.layers[0].self_attn.q_proj.weight.detach().clone()
    cases.append(("nf4 attn: 12 modules, error bounded", r["modules"] == 12 and r["rel_fro_err_max"] < 0.2, r["rel_fro_err_max"]))
    apply(m, "nf4", "attn")
    cases.append(("nf4 is idempotent on its own grid", bool(torch.equal(q0, m.layers[0].self_attn.q_proj.weight)), None))
    restore(m, snap)
    r = apply(m, "int8", "head")
    cases.append(("int8 head: the lm_head only", r["modules"] == 1 and r["lm_head"]
                  and bool(torch.equal(snap["layers.0.self_attn.q_proj"][0], m.layers[0].self_attn.q_proj.weight)), r))
    restore(m, snap)
    m.layers[1].self_attn.k_proj.weight.data[0, 0] += 1.0
    back = restore(m, snap)
    cases.append(("a weight edited after the snapshot is put back", back["mismatched"] == [], back))
    snap_bad = dict(snap)
    name = "layers.2.self_attn.o_proj"
    snap_bad[name] = (snap[name][0], "0" * 64)
    cases.append(("a restored weight whose sha differs is named", restore(m, snap_bad)["mismatched"] == [name], None))
    for label, call in (("an unknown format is refused", lambda: apply(Toy(), "fp8", "attn")),
                        ("an unknown scope is refused", lambda: apply(Toy(), "int8", "router"))):
        try:
            call()
            cases.append((label, False, None))
        except ValueError:
            cases.append((label, True, None))
    for label, model in (("a tied head is refused", Toy(tied=True)), ("no attention is refused", torch.nn.Linear(2, 2))):
        try:
            apply(model, "int8", "head" if "tied" in label else "attn")
            cases.append((label, False, None))
        except RuntimeError:
            cases.append((label, True, None))

    class Int4Linear(torch.nn.Module):
        pass

    m = Toy()
    m.layers[0].self_attn.q_proj = Int4Linear()
    try:
        apply(m, "int8", "attn")
        cases.append(("a non-Linear projection is refused", False, None))
    except RuntimeError:
        cases.append(("a non-Linear projection is refused", True, None))
    bad = [(n, d) for n, ok, d in cases if not ok]
    print(f"p147_fakequant self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    print(__doc__)
