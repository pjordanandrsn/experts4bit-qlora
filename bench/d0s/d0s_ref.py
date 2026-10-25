# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""REF's and Q's per-position argmax on D0's held-out rows, by D0's own pinned forward (CPU, USD 0; PREREG-d0s.md).

The served read's canary compares each served arm's teacher-forced argmax with REF's (and the attention-only arm's
with Q's). D0 recorded only per-row NLL (``pass/arms.csv``), so the argmax is computed here BEFORE any rental, by D0's
own tools as they ran (``bench/d0/receipts/d0-p224s3/tools/``, checked against their SHA256SUMS before import): the
same layer-major walk ``d0_pass.py`` makes for its ``Q`` and ``REF`` arms (Q = the trained base plus the attention LoRA;
REF = Q plus the expert LoRA unmerged, fp32), nothing else. Every row's NLL must reproduce ``arms.csv`` to a relative
1e-6, or the output says VOID and the read does not rent.

    python bench/d0s/d0s_ref.py --ckpt DIR --fixture-base DIR [--dq] --adapter DIR --heldout heldout.json \\
        --arms-csv arms.csv --out ref-argmax.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "bench", "d0", "receipts", "d0-p224s3", "tools")
REL_TOL = 1e-6
ARMS = ("Q", "REF")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_tools(tools: str) -> str:
    """Every file D0's SHA256SUMS lists is byte-identical; returns that list's own sha256."""
    sums = os.path.join(tools, "SHA256SUMS")
    bad = []
    with open(sums) as f:
        for line in f:
            digest, name = line.rstrip("\n").split("  ", 1)
            if sha256_file(os.path.join(tools, name)) != digest:
                bad.append(name)
    if bad:
        raise SystemExit(f"D0S_REF REFUSED: D0's tools differ from their SHA256SUMS: {bad}")
    return sha256_file(sums)


def recorded_nll(arms_csv: str) -> dict:
    """``{(arm, row): (nll, n)}`` for the Q and REF arms D0 recorded."""
    out = {}
    with open(arms_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["arm"] in ARMS:
                out[(r["arm"], int(r["row"]))] = (float(r["nll"]), int(r["n"]))
    return out


def walk(ckpt: str, fixture_base: str, dq: bool, adapter: str, heldout: str, layers: int = 0) -> dict:
    """``{arm: [per-row (nll, n, argmax list)]}`` for Q and REF, by d0_pass.py's own walk restricted to those arms."""
    import torch

    import d0_forward as fw
    import d0_pass
    import d0lib

    ck, fb, ad = d0lib.Checkpoint(ckpt), d0lib.FixtureBase(fixture_base, dq), d0lib.Adapter(adapter)
    cfg = d0lib.load_cfg(ckpt)
    H, inter, E = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_experts"]
    shapes = {"gate_up": (2 * inter, H), "down": (H, inter)}
    n_layers = layers or cfg["num_hidden_layers"]
    s_exp = ad.scale()
    rows = [torch.tensor(r["ids"]) for r in json.load(open(heldout))["rows"]]
    ids = torch.cat(rows)
    offsets = [0]
    for r in rows:
        offsets.append(offsets[-1] + len(r))
    emb = ck.get("model.embed_tokens.weight").float()
    hidden = {arm: emb[ids].clone() for arm in ARMS}
    del emb
    cos, sin = fw.rope_tables(max(len(r) for r in rows), H // cfg["num_attention_heads"], cfg["rope_theta"])
    for layer in range(n_layers):
        lw = fw.LayerWeights(ck, layer, d0_pass.attn_lora(ad, layer))
        routed = {}
        for arm in ARMS:
            hidden[arm] = fw.attention(hidden[arm], offsets, lw, cfg, cos, sin, use_lora=True)
            routed[arm] = fw.route(hidden[arm], lw, cfg)
        Q = {w: fb.q_train(layer, w, *shapes[w]) for w in d0lib.WHICH}
        A = {w: ad.st.get_tensor(ad.expert_name(layer, w, "A")).float() for w in d0lib.WHICH}
        B = {w: ad.st.get_tensor(ad.expert_name(layer, w, "B")).float() for w in d0lib.WHICH}
        qs = d0_pass.Stack([Q["gate_up"][e].bfloat16() for e in range(E)], [Q["down"][e].bfloat16() for e in range(E)])
        lora = lambda e: ((A["gate_up"][e], B["gate_up"][e], s_exp), (A["down"][e], B["down"][e], s_exp))  # noqa: E731
        for arm in ARMS:
            h, tv, ti = routed[arm]
            hidden[arm], _ = fw.moe(hidden[arm], h, tv, ti, qs, lora if arm == "REF" else None)
        qs = Q = A = B = None   # drop the layer's stacks before the next one loads (d0_pass.py's own way)
        print(f"L{layer:02d}", flush=True)
    norm_w, head = ck.get("model.norm.weight").float(), ck.get("lm_head.weight").float()
    eps, out = cfg["rms_norm_eps"], {arm: [] for arm in ARMS}
    for r, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        tgt = ids[lo + 1:hi]
        for arm in ARMS:
            lg = torch.nn.functional.linear(fw.rms(hidden[arm][lo:hi], norm_w, eps), head)[:-1]
            out[arm].append((torch.nn.functional.cross_entropy(lg, tgt, reduction="sum").item(), hi - lo - 1,
                             lg.argmax(-1).tolist()))
    return out


def compare(got: dict, recorded: dict) -> dict:
    """The largest relative NLL difference per arm against D0's record; refuses a row count or token count that
    differs."""
    worst = {}
    for arm in ARMS:
        rec_rows = sorted(r for a, r in recorded if a == arm)
        if rec_rows != list(range(len(got[arm]))):
            raise SystemExit(f"D0S_REF VOID: arms.csv holds rows {rec_rows[:3]}... for {arm}, the walk {len(got[arm])}")
        w = 0.0
        for r, (nll, n, _am) in enumerate(got[arm]):
            ref_nll, ref_n = recorded[(arm, r)]
            if n != ref_n:
                raise SystemExit(f"D0S_REF VOID: {arm} row {r} scores {n} tokens, D0 {ref_n}")
            w = max(w, abs(nll - ref_nll) / max(abs(ref_nll), 1e-12))
        worst[arm] = w
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("ckpt", "fixture-base", "adapter", "heldout", "arms-csv", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--dq", action="store_true")
    ap.add_argument("--tools", default=TOOLS, help="D0's pinned tools (default: the receipt's copy in this tree)")
    ap.add_argument("--layers", type=int, default=0, help="first N layers only (plumbing tests)")
    a = ap.parse_args(argv)
    tools_sums = check_tools(a.tools)
    sys.path.insert(0, a.tools)
    got = walk(a.ckpt, a.fixture_base, a.dq, a.adapter, a.heldout, a.layers)
    worst = compare(got, recorded_nll(a.arms_csv))
    status = "OK" if all(v <= REL_TOL for v in worst.values()) else "VOID"
    record = {"scheme": "d0s-ref/1", "status": status, "rel_tol": REL_TOL, "max_rel_nll_err": worst,
              "tools_sha256sums_sha256": tools_sums, "layers": a.layers or None,
              "inputs": {"heldout": sha256_file(a.heldout), "arms_csv": sha256_file(a.arms_csv),
                         "adapter_safetensors": sha256_file(os.path.join(a.adapter, "adapter.safetensors"))},
              "rows": [{"row": r, "n": got["REF"][r][1], "ref_nll": got["REF"][r][0], "q_nll": got["Q"][r][0],
                        "ref_argmax": got["REF"][r][2], "q_argmax": got["Q"][r][2]} for r in range(len(got["REF"]))]}
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(tmp, a.out)
    print(f"D0S_REF {status}: max relative NLL error {worst} (tolerance {REL_TOL}); {a.out}")
    return 0 if status == "OK" else 3


if __name__ == "__main__":
    raise SystemExit(main())
