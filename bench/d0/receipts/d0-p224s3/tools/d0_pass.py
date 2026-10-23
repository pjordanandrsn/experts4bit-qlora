"""D0 Step 1 + Step 2 in one layer-major walk (PREREG-d0.md). Per layer, every expert's repacks are built once: the
Step 1 sums come from them in fp32, and (with --heldout) every arm's hidden states advance through the layer on them.

Arms (with --heldout): BASE0 (no adapter), Q (attention LoRA), REF (Q + the expert LoRA unmerged, fp32), and for each
cell T x S: "<T>-<S>-base" = T(S) and "<T>-<S>-merged" = T(S + dW), all with the attention LoRA. Expert weights enter
the forward as bf16 values of each arm's dequantized weights; compute is fp32.

    python d0_pass.py --ckpt DIR --fixture-base DIR [--dq] --adapter DIR --out DIR [--heldout heldout.json]

Outputs per layer: cells_LNN.csv, checks_LNN.csv, routing_LNN.csv (REF's per-expert counts); at the end arms.csv
(arm, row, nll, n, dmax, dmean vs REF). A resume point (hidden states) is saved after every layer.
"""
import argparse
import csv
import json
import os
import time

import torch

import d0_forward as fw
import d0lib
from d0_retention import code_diff_share

CELLS = [(t, s) for t in ("NF4", "INT4", "MXFP4") for s in ("Q", "W")]


def write_csv(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def attn_lora(ad: d0lib.Adapter, layer: int):
    names = [n for n in ad.names if f"layers.{layer}.self_attn." in n]
    if any("qkv" in n for n in names):
        raise NotImplementedError(f"fused qkv adapter names in layer {layer}: {names[:3]}")
    s = ad.man.get("lora_scale_applied", {}).get("attention") if isinstance(ad.man.get("lora_scale_applied"), dict) else None
    s = float(s) if s is not None else ad.scale()
    out = {}
    for p in "qkvo":
        a = [n for n in names if n.endswith(f"{p}_proj.lora_A")]
        b = [n for n in names if n.endswith(f"{p}_proj.lora_B")]
        if a and b:
            out[p] = (ad.st.get_tensor(a[0]).float(), ad.st.get_tensor(b[0]).float(), s)
    return out


class Stack:
    """experts(e) -> (gate_up, down) bf16, from per-projection lists."""

    def __init__(self, gu, dn):
        self.gu, self.dn, self.n = gu, dn, len(gu)

    def __call__(self, e):
        return self.gu[e], self.dn[e]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fixture-base", required=True)
    ap.add_argument("--dq", action="store_true")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout")
    ap.add_argument("--layers", type=int, default=0, help="first N layers only (plumbing tests)")
    ap.add_argument("--experts", type=int, default=0, help="Step 1 sums for the first N experts only (plumbing)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck, fb, ad = d0lib.Checkpoint(a.ckpt), d0lib.FixtureBase(a.fixture_base, a.dq), d0lib.Adapter(a.adapter)
    T = d0lib.targets()
    cfg = d0lib.load_cfg(a.ckpt)
    H, inter, E = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_experts"]
    shapes = {"gate_up": (2 * inter, H), "down": (H, inter)}
    n_layers = a.layers or cfg["num_hidden_layers"]
    s_exp = ad.scale()

    arms, hidden, offsets, ids = [], {}, None, None
    resume = os.path.join(a.out, "resume.pt")
    if a.heldout:
        rows = [torch.tensor(r["ids"]) for r in json.load(open(a.heldout))["rows"]]
        ids = torch.cat(rows)
        offsets = [0]
        for r in rows:
            offsets.append(offsets[-1] + len(r))
        arms = ["BASE0", "Q", "REF"] + [f"{t}-{s}-{b}" for t, s in CELLS for b in ("base", "merged")]
        emb = ck.get("model.embed_tokens.weight").float()
        hidden = {arm: emb[ids].clone() for arm in arms}
        del emb
        hd = H // cfg["num_attention_heads"]
        cos, sin = fw.rope_tables(max(len(r) for r in rows), hd, cfg["rope_theta"])
    start = 0
    if a.heldout and os.path.exists(resume):
        st = torch.load(resume)
        start, hidden = st["next_layer"], st["hidden"]
        print(f"resuming at layer {start}", flush=True)

    for layer in range(start, n_layers):
        t0 = time.time()
        if a.heldout:
            lw = fw.LayerWeights(ck, layer, attn_lora(ad, layer))
            routed = {}
            for arm in arms:
                hidden[arm] = fw.attention(hidden[arm], offsets, lw, cfg, cos, sin, use_lora=(arm != "BASE0"))
                routed[arm] = fw.route(hidden[arm], lw, cfg)
        Q = {w: fb.q_train(layer, w, *shapes[w]) for w in d0lib.WHICH}
        A = {w: ad.st.get_tensor(ad.expert_name(layer, w, "A")).float() for w in d0lib.WHICH}
        B = {w: ad.st.get_tensor(ad.expert_name(layer, w, "B")).float() for w in d0lib.WHICH}
        dW = lambda w, e: s_exp * (B[w][e] @ A[w][e])  # noqa: E731
        chk_rows, cell_rows = [], []
        n_sum = a.experts or E
        for w in d0lib.WHICH:
            packed = fb.codes(layer, w)
            for e in range(n_sum):
                q, d = Q[w][e], dW(w, e)
                diff, ncodes = code_diff_share(ck.expert(layer, e, w), packed[e])
                chk_rows.append({"layer": layer, "which": w, "expert": e,
                                 "nf4_q_exact": bool(torch.equal(T["NF4"](q), q)), "cpu_codes_diff": diff,
                                 "codes": ncodes, "dW_norm2": float(d.double().pow(2).sum()),
                                 "Q_norm2": float(q.double().pow(2).sum())})
        if a.heldout:
            qs = Stack([Q["gate_up"][e].bfloat16() for e in range(E)], [Q["down"][e].bfloat16() for e in range(E)])
            lora = lambda e: ((A["gate_up"][e], B["gate_up"][e], s_exp), (A["down"][e], B["down"][e], s_exp))  # noqa: E731
            counts = None
            for arm in ("BASE0", "Q", "REF"):
                h, tv, ti = routed[arm]
                hidden[arm], c = fw.moe(hidden[arm], h, tv, ti, qs, lora if arm == "REF" else None)
                if arm == "REF":
                    counts = c
            del qs
            write_csv(os.path.join(a.out, f"routing_L{layer:02d}.csv"),
                      [{"layer": layer, "expert": e, "routed": int(counts[e])} for e in range(E)])
        for tname, sname in CELLS:
            fn = T[tname]
            built = {"base": {"gate_up": [], "down": []}, "merged": {"gate_up": [], "down": []}}
            for w in d0lib.WHICH:
                for e in range(E):
                    q, d = Q[w][e], dW(w, e)
                    src = q if sname == "Q" else ck.expert(layer, e, w)
                    base = fn(src)
                    merged = fn(src + d)
                    if e < n_sum:
                        D = merged - base
                        cell_rows.append({"layer": layer, "which": w, "expert": e, "T": tname, "S": sname,
                                          "dot": float((D.double() * d.double()).sum()),
                                          "dd": float(d.double().pow(2).sum()), "DD": float(D.double().pow(2).sum()),
                                          "ee": float((D - d).double().pow(2).sum()), "changed": int((D != 0).sum()),
                                          "numel": D.numel(), "bb": float((base - q).double().pow(2).sum()),
                                          "qq": float(q.double().pow(2).sum())})
                    if a.heldout:
                        built["base"][w].append(base.bfloat16())
                        built["merged"][w].append(merged.bfloat16())
                    del base, merged
            if a.heldout:
                for b in ("base", "merged"):
                    arm = f"{tname}-{sname}-{b}"
                    h, tv, ti = routed[arm]
                    hidden[arm], _ = fw.moe(hidden[arm], h, tv, ti, Stack(built[b]["gate_up"], built[b]["down"]))
            del built
        Q = A = B = None  # drop the layer's stacks before the next one loads
        write_csv(os.path.join(a.out, f"cells_L{layer:02d}.csv"), cell_rows)
        write_csv(os.path.join(a.out, f"checks_L{layer:02d}.csv"), chk_rows)
        if a.heldout:
            torch.save({"next_layer": layer + 1, "hidden": hidden}, resume + ".tmp")
            os.replace(resume + ".tmp", resume)
        print(f"L{layer:02d} done in {time.time() - t0:.0f} s", flush=True)

    if a.heldout:
        norm_w, head = ck.get("model.norm.weight").float(), ck.get("lm_head.weight").float()
        eps, out_rows = cfg["rms_norm_eps"], []
        for r, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
            tgt = ids[lo + 1:hi]
            ref = torch.nn.functional.linear(fw.rms(hidden["REF"][lo:hi], norm_w, eps), head)[:-1]
            for arm in arms:
                lg = torch.nn.functional.linear(fw.rms(hidden[arm][lo:hi], norm_w, eps), head)[:-1]
                diff = (lg - ref).abs()
                out_rows.append({"arm": arm, "row": r, "nll": torch.nn.functional.cross_entropy(lg, tgt, reduction="sum").item(),
                                 "n": hi - lo - 1, "dmax": diff.max().item(), "dmean": diff.mean().item()})
        write_csv(os.path.join(a.out, "arms.csv"), out_rows)
    print("D0_PASS done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
