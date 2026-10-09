"""P129 Amendment 1: the first step's gradients, reported (not gated): per seed, each variant's worst relative difference from B."""
import json, os, torch
out = []
for s in (211, 223, 227, 229, 233):
    b = torch.load(os.path.join(os.environ.get("P129_OUT", "p129a1"), f"g0_B_{s}.pt"))
    for c in ("F1", "F2", "F3", "FUSED"):
        v = torch.load(os.path.join(os.environ.get("P129_OUT", "p129a1"), f"g0_{c}_{s}.pt"))
        worst, name, nb = 0.0, None, 0
        for n, r in b.items():
            g = v.get(n)
            if g is None:
                continue
            m = r.abs().max().item()
            d = (g - r).abs().max().item()
            nb += int(d > 0)
            rel = d / m if m > 0 else 0.0
            if rel > worst:
                worst, name = rel, n
        out.append({"cfg": c, "seed": s, "worst_rel": worst, "worst_name": name, "tensors_not_bitwise": nb, "tensors": len(b)})
print("G0 " + json.dumps(out))
