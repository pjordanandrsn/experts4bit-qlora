"""HP1 (HP1-PREREG.md): the pieces the one evaluator shares with the TC1 harness.

- ``export_canonical``: every LoRA slot the harness's ``lora_slots`` maps (e4b ExpertsLoRA / LoRALinear, PEFT ParamWrapper /
  Linear) written to ONE layout, fp32: per (layer, kind) an expert stack ``A [E, r, in]`` / ``B [E, out, r]`` or an attention
  pair ``A [r, in]`` / ``B [out, r]``, plus each slot's scaling. delta(x) = scaling * (x A^T) B^T, the math every layout means.
- ``slot_fingerprint`` / ``fold_fingerprints``: an exact, deterministic integer fingerprint of a dequantized bf16 weight (GPU, no
  host copy), folded over canonical slot keys into one ``base_digest``. Two bases with the same digest present the same bf16
  bytes to the forward at every fingerprinted slot; a different quant scheme, block size or double-quant changes it.
- ``EagerExperts`` / ``EagerLinear``: the evaluator's own forward. Each expert weight is dequantized by the base's own
  dequantizer to bf16, the projection runs in bf16, and the canonical LoRA delta is computed in fp32 and added before the SwiGLU.
  No framework's fused kernel is on this path.
"""

import hashlib
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

CANONICAL_VERSION = 1
_P = 65521                                          # the largest prime below 2**16: the position weights cycle over it


def _slot_A(s):
    A, sel = s["A"].detach(), s["sel"]
    if sel[0] == "rows":
        return A[sel[1]:sel[2]]
    if sel[0] == "index":
        return A[sel[1]]
    return A


def _slot_B(s):
    """B for one slot, [out, r]. PEFT ParamWrapper: B [out, r*E] viewed [out, r, E] (peft 0.21.2 get_delta_factors), expert
    e = [:, :, e]. e4b ExpertsLoRA: B [E, out, r], expert e = B[e]. An attention pair's B is already [out, r]."""
    B, sel = s["B"].detach(), s["sel"]
    if sel[0] == "rows":
        r = sel[2] - sel[1]
        E = B.shape[1] // r
        return B.reshape(B.shape[0], r, E)[:, :, sel[1] // r]
    if sel[0] == "index":
        return B[sel[1]]
    return B


def slot_scaling(model, slot):
    """alpha / r as the framework applies it to this slot: e4b modules carry ``scaling`` (a float); PEFT modules a dict keyed
    by adapter name. The owning module is found by the slot's module path (an e4b ExpertsLoRA path carries the parameter name
    after the module)."""
    mods = dict(model.named_modules())
    path = slot["module"]
    m = mods.get(path)
    if m is None and "." in path:
        m = mods.get(path.rsplit(".", 1)[0])
    sc = getattr(m, "scaling", None)
    if isinstance(sc, dict):
        if len(sc) != 1:
            raise ValueError(f"{path}: {len(sc)} adapters in scaling; one is mappable")
        sc = next(iter(sc.values()))
    if not isinstance(sc, (int, float)):
        raise ValueError(f"{path}: no scalar scaling found (got {type(sc).__name__})")
    return float(sc)


def export_canonical(model, slots, path, meta=None):
    """Write the canonical adapter of ``model`` (``slots`` = ``lora_slots(model, hidden)[0]``) to ``path`` (safetensors) and
    return its record: {path, sha256, n_slots, keys, scalings, meta}. Refuses (ValueError) on a duplicate slot key, a slot
    whose A/B shapes do not chain, or mixed scalings within one (layer, kind) stack."""
    from safetensors.torch import save_file
    groups, seen = {}, set()
    for s in slots:
        L, kind, e = s["key"]
        if s["key"] in seen:
            raise ValueError(f"duplicate canonical slot {s['key']} ({s['module']})")
        seen.add(s["key"])
        A, B = _slot_A(s).float().cpu(), _slot_B(s).float().cpu()
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
            raise ValueError(f"slot {s['key']}: A {list(A.shape)} and B {list(B.shape)} do not chain")
        groups.setdefault((L, kind), {})[e] = (A, B, slot_scaling(model, s))
    tensors, scalings = {}, {}
    for (L, kind), by_e in sorted(groups.items()):
        sc = {v[2] for v in by_e.values()}
        if len(sc) != 1:
            raise ValueError(f"layer {L} {kind}: mixed scalings {sorted(sc)}")
        key = f"{L}.{kind}"
        scalings[key] = sc.pop()
        if set(by_e) == {-1}:
            tensors[key + ".A"], tensors[key + ".B"] = by_e[-1][0].contiguous(), by_e[-1][1].contiguous()
        else:
            E = len(by_e)
            if sorted(by_e) != list(range(E)):
                raise ValueError(f"{key}: experts {sorted(by_e)[:5]}... are not 0..{E - 1}")
            tensors[key + ".A"] = torch.stack([by_e[e][0] for e in range(E)]).contiguous()
            tensors[key + ".B"] = torch.stack([by_e[e][1] for e in range(E)]).contiguous()
    md = {"hp1_canonical": str(CANONICAL_VERSION), "scalings": json.dumps(scalings, sort_keys=True), "meta": json.dumps(meta or {}, sort_keys=True)}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(tensors, path, metadata=md)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return {"path": path, "sha256": h.hexdigest(), "n_slots": len(seen), "n_tensors": len(tensors), "scalings": sorted(set(scalings.values())),
            "meta": meta or {}}


def load_canonical(path, device):
    """(tensors {"L.kind.A"/"L.kind.B": fp32 tensor on device}, scalings {"L.kind": float}, meta)."""
    from safetensors import safe_open
    out = {}
    with safe_open(path, "pt", device=str(device)) as fh:
        md = fh.metadata() or {}
        if md.get("hp1_canonical") != str(CANONICAL_VERSION):
            raise ValueError(f"{path}: not an HP1 canonical adapter (metadata {md.get('hp1_canonical')!r})")
        for k in fh.keys():
            out[k] = fh.get_tensor(k).float()
    return out, json.loads(md["scalings"]), json.loads(md.get("meta", "{}"))


def slot_fingerprint(w):
    """An exact integer fingerprint of a bf16 (or any 2-byte) tensor: (sum of the int16 bit patterns, the same sum weighted by
    (position mod 65521) + 1), both in int64 on the tensor's device. Deterministic (integer adds), sensitive to any changed
    value and to a permutation; not a cryptographic hash (the digest over all slots is sha256)."""
    w = w.detach().contiguous()
    if w.element_size() != 2:
        w = w.to(torch.bfloat16)
    v = w.view(torch.int16).reshape(-1).to(torch.int64)
    pos = (torch.arange(v.numel(), device=v.device, dtype=torch.int64) % _P) + 1
    return int(v.sum().item()), int((v * pos).sum().item())


def fold_fingerprints(items):
    """sha256 over `L|kind|e|n|s1|s2` lines in sorted key order -> (digest hex, n_slots)."""
    h = hashlib.sha256()
    n = 0
    for (L, kind, e), (numel, s1, s2) in sorted(items.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        h.update(f"{L}|{kind}|{e}|{numel}|{s1}|{s2}\n".encode())
        n += 1
    return h.hexdigest(), n


def lora_delta(x, A, B, scaling):
    """scaling * (x A^T) B^T in fp32, returned in fp32."""
    return (x.float() @ A.t()) @ B.t() * scaling


class EagerExperts(nn.Module):
    """The evaluator's expert forward (HF's experts signature: hidden_states [T, H], top_k_index [T, k], top_k_weights [T, k]).
    ``deq(kind, e)`` returns expert e's dequantized bf16 weight [out, in] for kind in {gate_up, down}. The canonical delta is
    added in fp32 to each projection's output, the gate_up output is split chunk(2) -> (gate, up) as HF Qwen3-MoE and e4b do,
    then act(gate) * up, then down. Routing weights multiply in fp32 and accumulate in fp32 (as e4b's reference and HF do)."""

    def __init__(self, deq, num_experts, act_fn, adapter=None):
        super().__init__()
        self.deq, self.num_experts, self.act_fn, self.adapter = deq, int(num_experts), act_fn, adapter

    def forward(self, hidden_states, top_k_index, top_k_weights):
        in_dtype = hidden_states.dtype
        x_all = hidden_states.to(torch.bfloat16)
        out = torch.zeros_like(x_all, dtype=torch.float32)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1).tolist()
        ad = self.adapter
        for e in hit:
            pos, tok = torch.where(mask[e])
            x = x_all[tok]
            h = F.linear(x, self.deq("gate_up", e)).float()
            if ad is not None:
                h = h + lora_delta(x, ad["gate_up"][0][e], ad["gate_up"][1][e], ad["gate_up"][2])
            gate, up = h.to(torch.bfloat16).chunk(2, dim=-1)
            mid = self.act_fn(gate) * up
            y = F.linear(mid, self.deq("down", e)).float()
            if ad is not None:
                y = y + lora_delta(mid, ad["down"][0][e], ad["down"][1][e], ad["down"][2])
            y = y.to(torch.bfloat16).float() * top_k_weights[tok, pos, None].float()
            out.index_add_(0, tok, y)
        return out.to(in_dtype)


class EagerLinear(nn.Module):
    """The evaluator's attention projection: the dequantized bf16 weight (``deq()``), an optional bias, plus the canonical
    fp32 delta, returned in bf16."""

    def __init__(self, deq, bias=None, adapter=None):
        super().__init__()
        self.deq, self.bias, self.adapter = deq, bias, adapter

    def forward(self, x):
        xb = x.to(torch.bfloat16)
        y = F.linear(xb, self.deq(), None if self.bias is None else self.bias.to(torch.bfloat16)).float()
        if self.adapter is not None:
            A, B, sc = self.adapter
            y = y + lora_delta(xb, A, B, sc)
        return y.to(torch.bfloat16)


def dequant_raw(r, device):
    """The evaluator's ONE dequantizer: a framework-neutral stored slot (see tc1_arm.hp1_expert_source) -> bf16 [out, in] on
    ``device``. 4-bit: bitsandbytes.functional.dequantize_4bit with a QuantState rebuilt from the stored absmax (fp32, already
    un-nested), code, blocksize, quant type and the dtype the framework dequantized in, then cast to bf16 (and transposed back
    for a transposed stack). bf16: the bytes."""
    if r["fmt"] == "bf16":
        return r["w"].to(device=device, dtype=torch.bfloat16)
    from bitsandbytes.functional import QuantState, dequantize_4bit, get_4bit_type
    code = r["code"].to(device) if r.get("code") is not None else get_4bit_type(r["fmt"], device=device)
    dt = getattr(torch, r.get("dtype") or "bfloat16")
    st = QuantState(absmax=r["absmax"].to(device=device, dtype=torch.float32), shape=torch.Size(r["shape"]), code=code,
                    blocksize=int(r["blocksize"]), quant_type=r["fmt"], dtype=dt)
    w = dequantize_4bit(r["packed"].to(device).reshape(-1, 1), quant_state=st).to(torch.bfloat16)
    return w.t().contiguous() if r.get("transposed") else w


def _key_str(key):
    L, kind, e = key
    return f"{L}|{kind}|{e}"


def export_base(slots, out_dir, meta):
    """Write a base AS STORED: one safetensors file per layer (plus `global` for layer -1), each slot's raw tensors under
    `<L|kind|e>.<field>` and its scalar fields in the file's JSON metadata. -> {dir, files, n_slots, bytes, sha256 of the index}."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    by_layer = {}
    for key in slots:
        by_layer.setdefault(key[0], []).append(key)
    files, total, index = [], 0, {}
    with torch.no_grad():
        for L, keys in sorted(by_layer.items()):
            tensors, fields = {}, {}
            for key in sorted(keys, key=lambda k: (str(k[1]), k[2])):
                r = slots[key][0]()
                ks = _key_str(key)
                f = {"fmt": r["fmt"]}
                if r["fmt"] == "bf16":
                    tensors[ks + ".w"] = r["w"].detach().to(torch.bfloat16).cpu().contiguous()
                else:
                    tensors[ks + ".packed"] = r["packed"].detach().reshape(-1).cpu().contiguous()
                    tensors[ks + ".absmax"] = r["absmax"].detach().float().reshape(-1).cpu().contiguous()
                    if r.get("code") is not None:
                        tensors[ks + ".code"] = r["code"].detach().float().cpu().contiguous()
                    f.update(shape=list(r["shape"]), blocksize=int(r["blocksize"]), transposed=bool(r.get("transposed")), dtype=r.get("dtype") or "bfloat16")
                fields[ks] = f
            name = "global.safetensors" if L < 0 else f"layer{L:03d}.safetensors"
            path = os.path.join(out_dir, name)
            save_file(tensors, path, metadata={"hp1_base": str(CANONICAL_VERSION), "fields": json.dumps(fields, sort_keys=True)})
            files.append(name)
            total += os.path.getsize(path)
            index[name] = sorted(fields)
    idx = {"hp1_base": CANONICAL_VERSION, "meta": meta, "files": index}
    body = json.dumps(idx, sort_keys=True, indent=1)
    open(os.path.join(out_dir, "index.json"), "w").write(body)
    return {"dir": out_dir, "files": len(files), "n_slots": sum(len(v) for v in index.values()), "bytes": total,
            "index_sha256": hashlib.sha256(body.encode()).hexdigest()}


class BaseStore:
    """A base written by export_base, loaded onto ``device`` (packed NF4 stays packed): ``raw(key)`` rebuilds the slot's raw
    dict, ``deq(key)`` = dequant_raw(raw(key)), ``digest()`` re-derives the HP1 base digest through the evaluator's own
    dequantizer (STRICT: it must equal the training run's `hp1.train_base.digest`)."""

    def __init__(self, base_dir, device):
        from safetensors import safe_open
        self.dir, self.device = base_dir, device
        self.index = json.load(open(os.path.join(base_dir, "index.json")))
        self.meta, self.t, self.fields = self.index["meta"], {}, {}
        for name in self.index["files"]:
            with safe_open(os.path.join(base_dir, name), "pt", device=str(device)) as fh:
                md = fh.metadata() or {}
                if md.get("hp1_base") != str(CANONICAL_VERSION):
                    raise ValueError(f"{name}: not an HP1 base file")
                self.fields.update(json.loads(md["fields"]))
                for k in fh.keys():
                    self.t[k] = fh.get_tensor(k)

    def keys(self):
        out = []
        for ks in self.fields:
            L, kind, e = ks.split("|")
            out.append((int(L), kind, int(e)))
        return out

    def raw(self, key):
        ks = _key_str(key)
        f = self.fields[ks]
        if f["fmt"] == "bf16":
            return {"fmt": "bf16", "w": self.t[ks + ".w"]}
        return {"fmt": f["fmt"], "packed": self.t[ks + ".packed"], "absmax": self.t[ks + ".absmax"], "code": self.t.get(ks + ".code"),
                "shape": f["shape"], "blocksize": f["blocksize"], "transposed": f["transposed"], "dtype": f["dtype"]}

    def deq(self, key):
        return dequant_raw(self.raw(key), self.device)

    def digest(self):
        fps = {}
        with torch.no_grad():
            for key in self.keys():
                w = self.deq(key)
                fps[key] = (int(w.numel()), *slot_fingerprint(w))
        return fold_fingerprints(fps)
