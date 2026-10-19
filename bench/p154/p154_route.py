#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""P154's routing pass (bench/p154/PREREG-p154.md): the experts each MoE layer's router touched, per layer and per step,
over P152's bracketed W16 window (``census_window``: the same prompts, ramp, skip and steps), run eager so a forward hook
on every router fires, and ON THE REPLAY'S ROUTE: the pass sets ``hot_residency.DEVICE_GROUPING`` itself, as the server
sets it under graphs, so every 16-row expert call is device-grouped and served by K25 (the grouped NF4 small-M GEMM),
the route P152's W16 census timed.

It also records, on this box:
- the route: ``_collapsed_grouping(16, store)``, K25's mode, the grouping flags, a count of K25's calls inside the
  window (two a MoE layer a step: gate_up and down), and the library's own route record over the window
  (``hot_residency.ROUTE_SEEN``: one ``nf4_k25|le256`` a MoE layer a step, and no other route);
- the store: on every MoE layer, the NF4 stacks the K25 route reads (``_hot_residency.h_gu_p/h_gu_a/h_dn_p/h_dn_a``),
  their dtypes, shapes, devices and bytes per expert, and the arena they were loaded from (its size, and that it was
  written after this run started: the box's own bake, not a shipped file);
- a device copy rate (reported only).

Any other route, store or count is a fault, and P154's rule reads the floor VOID.

    python p154_route.py --prompts prompts.json --skip 32 --steps 64 --out route_b16.json   (engine from E4B_PAGED_*)
    python p154_route.py --self-test
"""
import argparse
import json
import os
import re
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOKENS = 160          # each row's budget, as the census driver (p136_census.py) sets it
GATE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")
K25_PER_LAYER_STEP = 2   # gate_up and down
COPY_BYTES = 1 << 29     # 512 MiB a side for the copy-rate read
COPY_REPS = 20


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
    """Forward hooks on the routers, and the K25 call count; records only while ``on``."""

    def __init__(self):
        self.on, self.calls, self.k25_calls = False, [], 0

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

    def count_k25(self, mod):
        """Wrap ``mod.gemm_nf4_grouped_smallm`` (looked up on the module at each MoE call) to count calls while on."""
        orig = mod.gemm_nf4_grouped_smallm

        def counted(*args, **kwargs):
            if self.on:
                self.k25_calls += 1
            return orig(*args, **kwargs)
        counted.__wrapped__ = orig
        mod.gemm_nf4_grouped_smallm = counted
        return orig

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


def _nbytes(t):
    return int(t.numel()) * int(t.element_size())


def store_record(model, n_layers, arena, started_at):
    """The NF4 stacks the K25 route reads, on every MoE layer's ``_hot_residency`` state (a plain object, read as
    ``serve_paged`` reads it), and the arena they were loaded from. Returns (record, faults)."""
    faults, layers = [], []
    for m in model.modules():
        hr = getattr(m, "_hot_residency", None)
        if hr is None:
            continue
        row = {"int4_stores": getattr(hr, "_int4_stores", None) is not None}
        try:
            gp, ga, dp, da = hr.h_gu_p, hr.h_gu_a, hr.h_dn_p, hr.h_dn_a
            e = int(gp.shape[0])
            row.update({"experts": e, "device": str(gp.device),
                        "dtypes": [str(t.dtype) for t in (gp, ga, dp, da)],
                        "shapes": [list(t.shape) for t in (gp, ga, dp, da)],
                        "gate_up_bytes_per_expert": (_nbytes(gp) + _nbytes(ga)) / e if e else None,
                        "down_bytes_per_expert": (_nbytes(dp) + _nbytes(da)) / e if e else None})
        except Exception as ex:  # noqa: BLE001 -- an unreadable stack is a fault, never a crash
            row["error"] = f"{type(ex).__name__}: {ex}"
        layers.append(row)
    rec = {"layers_with_state": len(layers), "host": socket.gethostname(), "arena": None}
    if len(layers) != n_layers:
        faults.append(f"store: {len(layers)} MoE layers carry a residency state, {n_layers} MoE layers")
    bad = [i for i, r in enumerate(layers) if r.get("error") or r["int4_stores"] or not r.get("experts")
           or r.get("dtypes", [None])[0] != "torch.uint8" or not str(r.get("device", "")).startswith("cuda")]
    if bad:
        faults.append(f"store: layers {bad[:6]} are not an NF4 store on the GPU (packed uint8 with no int4 store)")
    kinds = {(r.get("experts"), r.get("gate_up_bytes_per_expert"), r.get("down_bytes_per_expert"),
              tuple(r.get("dtypes", ()))) for r in layers}
    if len(kinds) != 1:
        faults.append(f"store: {len(kinds)} different layouts across the MoE layers")
    if layers and not bad and len(kinds) == 1:
        r0 = layers[0]
        rec.update({"kind": "nf4", "experts_per_layer": r0["experts"], "dtypes": r0["dtypes"], "shapes": r0["shapes"],
                    "device": r0["device"], "gate_up_bytes_per_expert": r0["gate_up_bytes_per_expert"],
                    "down_bytes_per_expert": r0["down_bytes_per_expert"]})
    try:
        st = os.stat(arena)
        rec["arena"] = {"bytes": st.st_size, "mtime": int(st.st_mtime), "started_at": int(started_at)}
        if st.st_mtime < started_at:
            faults.append("store: the arena predates this run (not this box's own bake)")
    except (OSError, TypeError, ValueError) as ex:
        faults.append(f"store: the arena is unreadable ({type(ex).__name__})")
    return rec, faults


def route_record(hr, batch, store_kind, k25_mode_ok):
    """The route the eager pass's 16-row expert calls take, read from ``hot_residency`` (``hr``) after the pass set
    ``DEVICE_GROUPING``: (singleton, device) must be (False, True) on an NF4 store, with K25 not switched off."""
    rec = {"k25_mode": hr._k25_mode_env(), "device_grouping_flag": bool(hr.DEVICE_GROUPING[0]),
           "force_singleton_groups": bool(hr.FORCE_SINGLETON_GROUPS[0]), "store_kind": store_kind,
           "collapsed_grouping": list(hr._collapsed_grouping(batch, None)), "collapsed_grouping_rows": batch,
           "k25_importable": bool(k25_mode_ok)}
    faults = []
    if store_kind != "nf4":
        faults.append(f"route: the store is {store_kind!r}, the registered route is K25 on NF4")
    if rec["collapsed_grouping"] != [False, True]:
        faults.append(f"route: {batch} rows group as (singleton, device) = {tuple(rec['collapsed_grouping'])}, "
                      "not device-grouped")
    if rec["k25_mode"] == "0" or not k25_mode_ok:
        faults.append(f"route: K25 is off or absent (mode {rec['k25_mode']!r}, importable {bool(k25_mode_ok)})")
    return rec, faults


def head_record(model, env_value):
    """P152's server had the bf16 lm_head; the int8 head (``Int8HeadLinear``, the qwen3_moe default since e4b#1838) rounds
    the logits, which changes tokens and so the routing. The pass runs with ``E4B_INT8_HEAD_BW=0``; an int8 head is a fault."""
    try:
        from experts4bit_qlora.engines.int8_head_bw import Int8HeadLinear
        n = sum(1 for m in model.modules() if isinstance(m, Int8HeadLinear))
    except ImportError:
        n = 0                                       # a commit before the int8 head: the head is bf16
    rec = {"E4B_INT8_HEAD_BW": env_value, "int8_head_modules": n}
    faults = [] if (n == 0 and env_value == "0") else [f"head: {n} int8 head modules with E4B_INT8_HEAD_BW={env_value!r}; "
                                                         "P152's server had the bf16 head"]
    return rec, faults


def k25_faults(count, steps, n_layers):
    want = K25_PER_LAYER_STEP * n_layers * steps
    return [] if count == want else [f"route: {count} K25 calls in the window, {want} registered "
                                     f"({K25_PER_LAYER_STEP} a layer a step)"]


def route_seen_delta(before, after):
    return {k: after.get(k, 0) - before.get(k, 0) for k in sorted(set(before) | set(after))
            if after.get(k, 0) != before.get(k, 0)}


def route_seen_faults(delta, steps, n_layers):
    want = {"nf4_k25|le256": n_layers * steps}
    return [] if delta == want else [f"route: the library's route record over the window is {delta}, {want} registered"]


def copy_gbps(total_bytes_moved, ms):
    """A device copy's rate as P141 counts it: bytes read plus bytes written, over the elapsed time."""
    return round(total_bytes_moved / (ms / 1e3) / 1e9, 1) if ms and ms > 0 else None


def copy_rate(torch):
    src = torch.empty(COPY_BYTES, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(COPY_REPS):
        dst.copy_(src)
    b.record()
    torch.cuda.synchronize()
    ms = a.elapsed_time(b)
    del src, dst
    return {"bytes_a_side": COPY_BYTES, "reps": COPY_REPS, "ms": round(ms, 4),
            "gbps": copy_gbps(2 * COPY_BYTES * COPY_REPS, ms)}


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
        raise SystemExit(f"REFUSED: the routing pass runs the default server with graphs off and nothing else set ({why})")
    batch = int(os.environ["P136_BATCH"])
    started_at = float(os.environ["P154_STARTED_AT"])
    with open(a.prompts) as f:
        pf = json.load(f)
    rows = pf["rows"][:batch]
    if p109_box.digest(pf["rows"]) != pf["prompts_sha256"] or len(rows) != batch:
        raise SystemExit("REFUSED: prompts.json does not match its own digest, or has too few rows")
    import torch

    from experts4bit_qlora.engines import hot_residency
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or (cfg.max_seqs, cfg.placement) != (16, "all-vram"):
        raise SystemExit(f"REFUSED: graphs={cfg.graphs} max_seqs={cfg.max_seqs} placement={cfg.placement}")
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    info, model = parts.info, parts.runner.model
    # the replay's route: the switches serve_paged sets for a server that captures batched decode graphs
    hot_residency.DEVICE_GROUPING[0] = True
    hot_residency.FORCE_SINGLETON_GROUPS[0] = False
    rec = Recorder()
    layers = attach(model, rec)
    try:
        import nf4_smallm
        rec.count_k25(nf4_smallm)
        k25_ok = True
    except (ImportError, AttributeError):
        k25_ok = False
    n_layers, n_experts = int(info.get("moe_layers") or 0), int(getattr(model.config, "num_experts", 0) or 0)
    top_k = int(getattr(model.config, "num_experts_per_tok", 0) or 0)
    out = {"engine": "e4b", "batch": batch, "mode": "routing", "graphs": False, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model, "revision": cfg.revision,
           "load_s": round(time.perf_counter() - t0, 1), "prompts_sha256": pf["prompts_sha256"],
           "moe_layers": n_layers, "num_experts": n_experts, "top_k": top_k, "routers_hooked": len(layers)}
    faults = [] if layers == list(range(n_layers)) else [f"{len(layers)} routers hooked, {n_layers} MoE layers"]
    out["store"], store_faults = store_record(model, n_layers, os.environ.get("E4B_PAGED_ARENA"), started_at)
    out["route"], route_faults = route_record(hot_residency, batch, out["store"].get("kind"), k25_ok)
    faults += store_faults + route_faults
    out["head"], head_faults = head_record(model, os.environ.get("E4B_INT8_HEAD_BW"))
    faults += head_faults
    p109_box.run_pass(parts, torch, rows, 32)                      # warm (untimed, hooks off)
    sched = parts.scheduler
    for r in rows:
        sched.add_request(list(r), max_new_tokens=TOKENS)
    seen = {}

    def _start():
        seen["before"] = dict(hot_residency.ROUTE_SEEN)
        rec.start()

    def _stop():
        rec.stop()
        seen["after"] = dict(hot_residency.ROUTE_SEEN)
    out.update(census_window(sched, batch, a.skip, a.steps, _start, _stop, torch.cuda.synchronize, tokens=TOKENS))
    sched.run_until_idle()
    steps, f2 = group_steps(rec.calls, n_layers)
    s = summarize(steps, n_experts, batch, top_k)
    out.update(s)
    out["route"]["k25_calls_in_window"] = rec.k25_calls
    out["route"]["route_seen_in_window"] = route_seen_delta(seen.get("before", {}), seen.get("after", {}))
    try:
        out["copy_rate"] = copy_rate(torch)
    except Exception as ex:  # noqa: BLE001 -- reported only: an unread rate never ends the pass
        out["copy_rate"] = {"error": f"{type(ex).__name__}: {ex}"}
    out["faults"] = (faults + f2 + s["faults"] + k25_faults(rec.k25_calls, len(steps), n_layers)
                     + route_seen_faults(out["route"]["route_seen_in_window"], len(steps), n_layers)
                     + ([] if len(steps) == a.steps else [f"{len(steps)} steps recorded, {a.steps} bracketed"]))
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("P154_ROUTE " + json.dumps({"steps": len(steps), "summary": s["summary"], "k25_calls": rec.k25_calls,
                                      "store": {k: out["store"].get(k) for k in ("kind", "gate_up_bytes_per_expert",
                                                                                "down_bytes_per_expert")},
                                      "copy_gbps": out["copy_rate"].get("gbps"), "faults": len(out["faults"])}), flush=True)
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
    same = [[0, 1]] * batch
    _, f3 = group_steps([(0, batch, k, same), (2, batch, k, same), (1, batch, k, same)], n_layers)
    cases.append(("routers out of layer order is a fault", any("out of layer order" in x for x in f3)))
    s5 = summarize([[(0, batch - 1, k, same[:-1])]], n_exp, batch, k)
    cases.append(("a step with other than the batch's rows is a fault", any("rows x top" in x for x in s5["faults"])))

    # the K25 count: only calls inside the window count, and the registered count is 2 a layer a step
    class _Mod:
        def gemm_nf4_grouped_smallm(self, *a, **kw):
            return ("ran", a, kw)
    mod, rc = _Mod(), Recorder()
    rc.count_k25(mod)
    mod.gemm_nf4_grouped_smallm(1, x=2)
    rc.start()
    for _ in range(2 * n_layers * 5):
        r = mod.gemm_nf4_grouped_smallm(1, x=2)
    rc.stop()
    cases.append(("the K25 wrapper counts only inside the window and passes calls through",
                  rc.k25_calls == 2 * n_layers * 5 and r == ("ran", (1,), {"x": 2})))
    cases.append(("the registered K25 count is exact", not k25_faults(2 * n_layers * 5, 5, n_layers)
                  and k25_faults(2 * n_layers * 5 - 1, 5, n_layers) and k25_faults(0, 5, n_layers)))
    d = route_seen_delta({"nf4_k25|le256": 7, "nf4_mtile|gt256": 2}, {"nf4_k25|le256": 7 + n_layers * 5,
                                                                      "nf4_mtile|gt256": 2})
    cases.append(("the library's route record over the window is one nf4_k25 call a layer a step",
                  d == {"nf4_k25|le256": n_layers * 5} and not route_seen_faults(d, 5, n_layers)))
    cases.append(("another route inside the window is a fault", bool(route_seen_faults(
        route_seen_delta({}, {"nf4_k25|le256": n_layers * 5, "nf4_host|le256": 1}), 5, n_layers))))
    cases.append(("host grouping instead of K25 is a fault (P152's eager route)",
                  bool(route_seen_faults({"nf4_host|le256": n_layers * 5}, 5, n_layers))))

    # the route, from hot_residency's switches after the pass set DEVICE_GROUPING
    class _HR:
        def __init__(self, device=True, k25="auto"):
            self.DEVICE_GROUPING, self.FORCE_SINGLETON_GROUPS, self.k25 = [device], [False], k25

        def _k25_mode_env(self):
            return self.k25

        def _collapsed_grouping(self, T, st):           # the library's rule for an NF4 store (no int4 store) at T > 1
            return (T == 1, self.DEVICE_GROUPING[0] and T > 1)
    r0, f0 = route_record(_HR(), 16, "nf4", True)
    cases.append(("the replay route: device-grouped K25 on NF4", not f0 and r0["collapsed_grouping"] == [False, True]))
    cases.append(("device grouping left off (P152's eager route) is a fault",
                  bool(route_record(_HR(device=False), 16, "nf4", True)[1])))
    cases.append(("K25 switched off is a fault", bool(route_record(_HR(k25="0"), 16, "nf4", True)[1])))
    cases.append(("K25 absent is a fault", bool(route_record(_HR(), 16, "nf4", False)[1])))
    cases.append(("a store that is not NF4 is a fault", bool(route_record(_HR(), 16, None, True)[1])))

    # the store: NF4 stacks on every MoE layer's _hot_residency, the box's own arena
    class _Ten:
        def __init__(self, shape, esize, dtype="torch.uint8", device="cuda:0"):
            self.shape, self.esize, self.dtype, self.device = shape, esize, dtype, device

        def numel(self):
            n = 1
            for d in self.shape:
                n *= d
            return n

        def element_size(self):
            return self.esize

    def _state(int4=None, dtype="torch.uint8", device="cuda:0", e=8):
        hr = type("HR", (), {})()
        hr._int4_stores = int4
        hr.h_gu_p = _Ten([e, 1536, 1024], 1, dtype, device)
        hr.h_gu_a = _Ten([e, 1536 * 2048 // 64], 4, "torch.float32", device)
        hr.h_dn_p = _Ten([e, 2048, 384], 1, dtype, device)
        hr.h_dn_a = _Ten([e, 2048 * 768 // 64], 4, "torch.float32", device)
        return hr

    class _Model:
        def __init__(self, states):
            self.states = states

        def modules(self):
            return [type("Attn", (), {})()] + [type("Experts", (), {"_hot_residency": st})() for st in self.states]
    import tempfile
    with tempfile.NamedTemporaryFile() as arena:
        now = time.time()
        sr, sf = store_record(_Model([_state() for _ in range(n_layers)]), n_layers, arena.name, now - 60)
        cases.append(("the NF4 store reads 0.5625 B a weight: 1,769,472 and 884,736 bytes an expert",
                      not sf and sr["kind"] == "nf4" and sr["gate_up_bytes_per_expert"] == 1769472
                      and sr["down_bytes_per_expert"] == 884736))
        cases.append(("an arena older than the run is a fault (a shipped file, not this box's bake)",
                      any("predates" in x for x in store_record(_Model([_state()] * n_layers), n_layers, arena.name,
                                                                now + 3600)[1])))
        cases.append(("an int4 store is a fault", bool(store_record(_Model([_state(int4={"gu": {}})] * n_layers),
                                                                     n_layers, arena.name, now - 60)[1])))
        cases.append(("a store off the GPU is a fault", bool(store_record(_Model([_state(device="cpu")] * n_layers),
                                                                          n_layers, arena.name, now - 60)[1])))
        cases.append(("a missing layer is a fault", bool(store_record(_Model([_state()] * (n_layers - 1)), n_layers,
                                                                      arena.name, now - 60)[1])))
        cases.append(("layers with different layouts are a fault",
                      bool(store_record(_Model([_state(), _state(e=4), _state()]), n_layers, arena.name, now - 60)[1])))
    cases.append(("a missing arena is a fault", bool(store_record(_Model([_state()] * n_layers), n_layers,
                                                                  "/nonexistent/nf4.arena", 0)[1])))
    class _Head:
        def modules(self):
            return [type("Linear", (), {})()]
    cases.append(("P152's bf16 head with the knob at 0 is clean", head_record(_Head(), "0")[1] == []))
    cases.append(("the knob unset (today's default: the int8 head) is a fault", bool(head_record(_Head(), None)[1])))
    cases.append(("the copy rate counts bytes read plus written", copy_gbps(2 * 10**9, 2.0) == 1000.0
                  and copy_gbps(1, 0) is None))
    bad = [n for n, ok in cases if not ok]
    print(f"p154_route self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
