#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""P152's routing census (bench/p152/PREREG-p152.md): the experts each MoE layer's router touched, per layer and per step,
over the per-name census's own bracketed W16 window (``census_window``: the same prompts, ramp, skip and steps), run EAGER
(``E4B_PAGED_GRAPHS=0``, nothing else set) so a forward hook on every layer's router fires. Reported only: it gives the
expert byte floor on real traffic (the maintainer's term), never a speed reading.

The hook reads the router's third output (``_, weights, idx = mlp.gate(x)``, the transformers 5.x sparse-MoE block's
body, as e4b's fused router epilogue keeps it). Every bracketed step must call all ``moe_layers`` routers in layer order,
each with ``batch`` rows of ``top_k`` distinct experts; anything else is a fault and the record says so.

Record (``--out``): per step, the unique experts per layer; per layer, the expert hit counts over the window; and the
summary the reducer reads (unique per layer-step: mean, min, max; the per-step total; the hottest expert's share).

    python p152_route.py --prompts prompts.json --skip 32 --steps 64 --out route_b16.json   (engine from E4B_PAGED_*)
    python p152_route.py --self-test
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOKENS = 160          # each row's budget, as the census driver (p136_census.py) sets it
GATE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")


def group_steps(calls, n_layers):
    """``calls`` is the hook's (layer, rows, k, idx-list) in call order. Returns (steps, faults): each step is the
    ``n_layers`` calls of one decode step, layers 0..n-1 in order."""
    faults, steps = [], []
    if len(calls) % n_layers:
        faults.append(f"{len(calls)} router calls is not a whole number of {n_layers}-layer steps")
    for s in range(len(calls) // n_layers):
        chunk = calls[s * n_layers:(s + 1) * n_layers]
        layers = [c[0] for c in chunk]
        if layers != list(range(n_layers)):
            faults.append(f"step {s}: routers called out of layer order ({layers[:4]}...)")
        steps.append(chunk)
    return steps, faults


def summarize(steps, n_experts, batch, top_k):
    faults, per_step, hits = [], [], {}
    for s, chunk in enumerate(steps):
        uniq = []
        for layer, rows, k, idx in chunk:
            if rows != batch or k != top_k:
                faults.append(f"step {s} layer {layer}: {rows} rows x top-{k}, registered {batch} x {top_k}")
            flat = [e for row in idx for e in row]
            if any(len(set(row)) != len(row) for row in idx):
                faults.append(f"step {s} layer {layer}: a row picked one expert twice")
            if any(not 0 <= e < n_experts for e in flat):
                faults.append(f"step {s} layer {layer}: an expert id out of range")
            uniq.append(len(set(flat)))
            h = hits.setdefault(layer, [0] * n_experts)
            for e in flat:
                if 0 <= e < n_experts:
                    h[e] += 1
        per_step.append(uniq)
    flat_u = [u for st in per_step for u in st]
    totals = [sum(st) for st in per_step]
    hot = max((max(h) / sum(h) for h in hits.values() if sum(h)), default=None)
    summary = {"steps": len(per_step), "layers": len(per_step[0]) if per_step else 0,
               "unique_per_layer_step": {"mean": round(sum(flat_u) / len(flat_u), 4) if flat_u else None,
                                         "min": min(flat_u, default=None), "max": max(flat_u, default=None)},
               "unique_per_step_total": {"mean": round(sum(totals) / len(totals), 2) if totals else None,
                                         "min": min(totals, default=None), "max": max(totals, default=None)},
               "hottest_expert_share_of_a_layer": round(hot, 5) if hot is not None else None,
               "uniform_expectation_per_layer": round(n_experts * (1 - (1 - top_k / n_experts) ** batch), 3)}
    return {"summary": summary, "unique_per_step": per_step, "hits_per_layer": {str(k): v for k, v in sorted(hits.items())},
            "faults": faults}


class Recorder:
    """Forward hooks on the routers; records only while ``on``."""

    def __init__(self):
        self.on, self.calls = False, []

    def hook(self, layer):
        def _h(_mod, _inp, out):
            if not self.on:
                return
            idx = out[-1] if isinstance(out, (tuple, list)) else None
            if idx is None or getattr(idx, "dim", lambda: 0)() != 2:
                self.calls.append((layer, -1, -1, []))
                return
            self.calls.append((layer, int(idx.shape[0]), int(idx.shape[1]), idx.detach().to("cpu").tolist()))
        return _h

    def start(self):
        self.on = True

    def stop(self):
        self.on = False


def attach(model, rec):
    gates = sorted(((int(m.group(1)), mod) for name, mod in model.named_modules() if (m := GATE_RE.search(name))),
                   key=lambda x: x[0])
    for layer, mod in gates:
        mod.register_forward_hook(rec.hook(layer))
    return [g[0] for g in gates]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts")
    ap.add_argument("--skip", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    import p109_box
    from p136_box import default_env_ok
    from sc1b_e4b_census import census_window

    env = {k: v for k, v in os.environ.items() if k != "E4B_PAGED_GRAPHS"}
    ok, why = default_env_ok(env)
    if not ok or os.environ.get("E4B_PAGED_GRAPHS") != "0":
        raise SystemExit(f"REFUSED: the routing census runs the default server with graphs off and nothing else set ({why})")
    batch = int(os.environ["P136_BATCH"])
    with open(a.prompts) as f:
        pf = json.load(f)
    rows = pf["rows"][:batch]
    if p109_box.digest(pf["rows"]) != pf["prompts_sha256"] or len(rows) != batch:
        raise SystemExit("REFUSED: prompts.json does not match its own digest, or has too few rows")
    import torch

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or (cfg.max_seqs, cfg.placement) != (16, "all-vram"):
        raise SystemExit(f"REFUSED: graphs={cfg.graphs} max_seqs={cfg.max_seqs} placement={cfg.placement}")
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    info, model = parts.info, parts.runner.model
    rec = Recorder()
    layers = attach(model, rec)
    n_layers, n_experts = int(info.get("moe_layers") or 0), int(getattr(model.config, "num_experts", 0) or 0)
    top_k = int(getattr(model.config, "num_experts_per_tok", 0) or 0)
    out = {"engine": "e4b", "batch": batch, "mode": "routing", "graphs": False, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model, "revision": cfg.revision,
           "load_s": round(time.perf_counter() - t0, 1), "prompts_sha256": pf["prompts_sha256"],
           "moe_layers": n_layers, "num_experts": n_experts, "top_k": top_k, "routers_hooked": len(layers)}
    faults = [] if layers == list(range(n_layers)) else [f"{len(layers)} routers hooked, {n_layers} MoE layers"]
    p109_box.run_pass(parts, torch, rows, 32)                      # warm (untimed, hooks off)
    sched = parts.scheduler
    for r in rows:
        sched.add_request(list(r), max_new_tokens=TOKENS)
    out.update(census_window(sched, batch, a.skip, a.steps, rec.start, rec.stop, torch.cuda.synchronize, tokens=TOKENS))
    sched.run_until_idle()
    steps, f2 = group_steps(rec.calls, n_layers)
    s = summarize(steps, n_experts, batch, top_k)
    out.update(s)
    out["faults"] = faults + f2 + s["faults"] + ([] if len(steps) == a.steps else [f"{len(steps)} steps recorded, {a.steps} bracketed"])
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("P152_ROUTE " + json.dumps({"steps": len(steps), "summary": s["summary"], "faults": len(out["faults"])}), flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------- self-test --
def self_test() -> int:
    import random
    cases = []
    rng = random.Random(7)
    n_layers, n_exp, batch, k = 3, 16, 4, 2
    rec = Recorder()

    class _T:                                          # a stand-in for the router's index tensor
        def __init__(self, rows):
            self.rows, self.shape = rows, (len(rows), len(rows[0]) if rows else 0)

        def dim(self):
            return 2

        def detach(self):
            return self

        def to(self, _dev):
            return self

        def tolist(self):
            return self.rows

    def step(fixed=None):
        for layer in range(n_layers):
            rows = fixed or [rng.sample(range(n_exp), k) for _ in range(batch)]
            rec.hook(layer)(None, None, (None, None, _T(rows)))

    step()                                             # off: not recorded
    rec.start()
    for _ in range(5):
        step()
    rec.stop()
    step()
    steps, f = group_steps(rec.calls, n_layers)
    cases.append(("records only between start and stop, grouped by layer", len(steps) == 5 and not f))
    s = summarize(steps, n_exp, batch, k)
    flat = [u for st in s["unique_per_step"] for u in st]
    cases.append(("unique experts per layer-step lie in [k, batch*k]", all(k <= u <= batch * k for u in flat) and not s["faults"]))
    cases.append(("hits sum to batch*k per layer-step", all(sum(h) == 5 * batch * k for h in s["hits_per_layer"].values())))
    same = [[0, 1]] * batch                             # every row the same two experts: 2 unique, the hottest is half
    s2 = summarize([[(layer, batch, k, same) for layer in range(n_layers)]], n_exp, batch, k)
    cases.append(("identical rows read k unique and a 0.5 hottest share", s2["summary"]["unique_per_layer_step"]["max"] == 2
                  and s2["summary"]["hottest_expert_share_of_a_layer"] == 0.5))
    cases.append(("the uniform expectation is the closed form", s2["summary"]["uniform_expectation_per_layer"]
                  == round(16 * (1 - (1 - 2 / 16) ** 4), 3)))
    _, f3 = group_steps([(0, batch, k, same), (2, batch, k, same), (1, batch, k, same)], n_layers)
    cases.append(("routers out of layer order is a fault", any("out of layer order" in x for x in f3)))
    _, f4 = group_steps([(0, batch, k, same)] * 4, n_layers)
    cases.append(("a partial step is a fault", any("whole number" in x for x in f4)))
    s5 = summarize([[(0, batch - 1, k, same[:-1])]], n_exp, batch, k)
    cases.append(("a step with other than the batch's rows is a fault", any("rows x top" in x for x in s5["faults"])))
    s6 = summarize([[(0, 1, k, [[3, 3]])]], n_exp, 1, k)
    cases.append(("a row picking one expert twice is a fault", any("twice" in x for x in s6["faults"])))
    cases.append(("the gate name pattern matches transformers' layer path", GATE_RE.search("model.layers.47.mlp.gate")
                  and not GATE_RE.search("model.layers.3.mlp.experts.gate")))
    bad = [n for n, ok in cases if not ok]
    print(f"p152_route self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
