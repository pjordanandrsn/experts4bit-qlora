#!/usr/bin/env python3
"""Lane P119's reducer (bench/p119/PREREG-p119.md): checks the census ran as registered, then tabulates it. Descriptive:
the verdict is READ or VOID, never a speed or quality claim.

**VOID**, any of:
- the e4b or grouped-nf4-gemm commit or the model revision is not the registered one;
- the reading's stack is not SC2e's (every MoE layer on int4 experts, int4 attention, the three folds and fused q/k/v);
- a decode bracket did not run as registered: its profiled bucket statistics are not ``steps`` x its registered split
  (``SPLITS``), or a bucket is not ``eager: capture=False``, or the KV bookkeeping is not bulk;
- the prefill OFF bracket is missing or refused, or runs other than ``last_logits`` off;
- any bracket's profiler saw no device kernel (``device_ms`` 0).

A refused prefill ON bracket (the model's forward has no explicit keyword) is reported, not VOID.

**READ** tabulates, per bracket, device ms and calls per step by class, the marginal device ms per decode row by class
from 16 to 32 to 64 rows, the chained 64-row step against the one-piece one, device-to-device copies per layer, the
LM head's device ms at each row count, and the prefill ON - OFF difference. Floats are summed with ``math.fsum``, so
the output is byte-identical on any Python.

    p119_reduce.py --dir RUN --out verdict.json [--e4b-sha SHA]
    p119_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

GNF4_SHA = "b4f93f1c62d1e3436ed45bec8ccd608c90433737"
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "ibm-granite/granite-3.1-3b-a800m-instruct": "a02780686e08a03fe0d2679a293b5c74a90efa89"}
READING_MODEL = "Qwen/Qwen3-30B-A3B"
# bracket -> {bucket: pieces per step} with the registered rows (64; the proof may run fewer, see split_of)
DECODE = {"d16": ((1, 2, 4, 8, 16), 16), "d32": ((1, 2, 4, 8, 16, 32), 32), "d64": ((1, 2, 4, 8, 16, 32, 64), 64),
          "d64x4": ((1, 2, 4, 8, 16), 64)}


def split_of(buckets, rows) -> dict:
    """The runner's split of ``rows`` active rows: pieces of the largest bucket, each padded to its bucket."""
    top, out, left = max(buckets), {}, rows
    while left > 0:
        n = min(left, top)
        b = min(x for x in buckets if x >= n)
        s = out.setdefault(str(b), {"pieces": 0, "rows": 0, "pad_rows": 0})
        s["pieces"] += 1
        s["rows"] += n
        s["pad_rows"] += b - n
        left -= n
    return out


def _fsum(xs) -> float:
    return math.fsum(float(x) for x in xs)


def faults(box: dict, e4b_sha: str) -> list:
    out = []
    if box.get("e4b_sha") != e4b_sha:
        out.append(f"e4b {box.get('e4b_sha')} != {e4b_sha}")
    if box.get("gnf4_sha") != GNF4_SHA:
        out.append(f"grouped-nf4-gemm {box.get('gnf4_sha')} != {GNF4_SHA}")
    if REVS.get(box.get("model")) != box.get("revision"):
        out.append(f"model {box.get('model')}@{box.get('revision')} is not a registered revision")
    cb = box.get("census_build") or {}
    if box.get("model") == READING_MODEL:
        if not (cb.get("int4_expert_layers") and cb.get("int4_expert_layers") == cb.get("moe_layers")):
            out.append(f"int4 experts on {cb.get('int4_expert_layers')} of {cb.get('moe_layers')} MoE layers")
        if not cb.get("int4_attn_projections"):
            out.append("int4 attention not engaged")
        for k in ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_router_epilogue_n"):
            if not cb.get(k):
                out.append(f"{k} is {cb.get(k)}: SC2e's stack folds it")
    dec = box.get("decode") or {}
    for label, (buckets, rows) in DECODE.items():
        d = dec.get(label)
        if d is None:
            out.append(f"decode {label} missing")
            continue
        r = min(rows, int(d.get("rows") or 0)) or rows
        want = split_of(buckets, r)
        steps = int(d.get("steps") or 0)
        got = d.get("profiled") or {}
        for b, w in want.items():
            g = got.get(b) or {}
            if (g.get("eager_steps"), g.get("replays"), g.get("rows"), g.get("pad_rows")) != \
                    (w["pieces"] * steps, 0, w["rows"] * steps, w["pad_rows"] * steps):
                out.append(f"decode {label}: bucket {b} profiled {g}, registered {w} x {steps} steps")
        extra = [b for b, g in got.items() if b not in want and (g.get("eager_steps") or g.get("replays"))]
        if extra:
            out.append(f"decode {label}: buckets {extra} ran outside the registered split")
        if any(v != "eager: capture=False" for v in (d.get("graph_status") or {"?": None}).values()):
            out.append(f"decode {label}: graph status {d.get('graph_status')}")
        if d.get("bulk_kv") is not True:
            out.append(f"decode {label}: KV bookkeeping not bulk (the server's default)")
        if not d.get("device_ms"):
            out.append(f"decode {label}: the profiler saw no device kernel")
    pf = box.get("prefill") or {}
    off = pf.get("p512_off")
    if not off or off.get("refused"):
        out.append(f"prefill OFF {'missing' if not off else 'refused: ' + str(off.get('refused'))}")
    else:
        if (off.get("last_logits_stats") or {}).get("status") != "off":
            out.append(f"prefill OFF ran {off.get('last_logits_stats')}")
        if not off.get("device_ms"):
            out.append("prefill OFF: the profiler saw no device kernel")
        if off.get("bulk_kv") is not True:
            out.append("prefill OFF: KV bookkeeping not bulk")
    on = pf.get("p512_on") or {}
    if on and not on.get("refused") and (on.get("last_logits_stats") or {}).get("status") != "on":
        out.append(f"prefill ON ran {on.get('last_logits_stats')}")
    return out


def _classes(b) -> dict:
    return {k: v[1] for k, v in (b.get("classes") or {}).items()}


def tabulate(box: dict) -> dict:
    dec, pf = box["decode"], box["prefill"]
    t = {"decode_ms_per_step": {k: v["device_ms"] for k, v in dec.items()},
         "decode_classes": {k: _classes(v) for k, v in dec.items()},
         "dtod_per_layer": {**{k: v.get("dtod_per_layer") for k, v in dec.items()},
                            **{k: v.get("dtod_per_layer") for k, v in pf.items() if not v.get("refused")}}}
    rows = {k: dec[k]["rows"] for k in ("d16", "d32", "d64")}
    marg = {}
    for lo, hi in (("d16", "d32"), ("d32", "d64"), ("d16", "d64")):
        dr = rows[hi] - rows[lo]
        if dr <= 0:
            continue
        cl = set(t["decode_classes"][lo]) | set(t["decode_classes"][hi])
        marg[f"{lo}->{hi}"] = {"per_row_ms": round((dec[hi]["device_ms"] - dec[lo]["device_ms"]) / dr, 5),
                               "by_class": {c: round((t["decode_classes"][hi].get(c, 0.0)
                                                      - t["decode_classes"][lo].get(c, 0.0)) / dr, 5) for c in sorted(cl)}}
    t["marginal_per_row"] = marg
    t["chained_over_one_piece"] = round(dec["d64x4"]["device_ms"] / dec["d64"]["device_ms"], 4)
    off = pf["p512_off"]
    t["prefill_ms"] = off["device_ms"]
    t["prefill_classes"] = _classes(off)
    on = pf.get("p512_on") or {}
    if on and not on.get("refused"):
        t["prefill_on_minus_off_ms"] = round(on["device_ms"] - off["device_ms"], 4)
        t["prefill_on_classes"] = _classes(on)
    else:
        t["prefill_on"] = on.get("refused", "not run")
    head = box.get("head") or {}
    t["head_ms"] = {k: v.get("device_ms") for k, v in (head.get("rows") or {}).items()}
    t["head_weight"] = {"shape": head.get("weight_shape"), "dtype": head.get("weight_dtype"), "module": head.get("module")}
    t["decode_sum_check"] = {k: round(_fsum(v.values()) - dec[k]["device_ms"], 4) for k, v in t["decode_classes"].items()}
    return t


def reduce(run: Path, e4b_sha: str) -> dict:
    p = run / "box.json"
    if not p.is_file():
        return {"lane": "P119", "verdict": "NO_READING", "reasons": ["no box record"]}
    return reduce_obj(json.loads(p.read_text()), e4b_sha)


# ------------------------------------------------------------------------------------------------ self-test --

def _bracket(buckets, rows, steps=8, ms=10.0, extra=None):
    st = {b: {"replays": 0, "eager_steps": w["pieces"] * steps, "rows": w["rows"] * steps, "pad_rows": w["pad_rows"] * steps}
          for b, w in split_of(buckets, rows).items()}
    d = {"buckets": list(buckets), "rows": rows, "steps": steps, "warm": 3, "layers": 48, "bulk_kv": True,
         "graph_status": {str(b): "eager: capture=False" for b in buckets}, "profiled": st, "device_ms": ms,
         "classes": {"expert_int4": [96.0, ms * 0.6], "dense_gemm": [40.0, ms * 0.3], "attn": [48.0, ms * 0.1]},
         "dtod": 4.0, "dtod_per_layer": round(4 / 48, 3)}
    d.update(extra or {})
    return d


def _box(model=READING_MODEL, e4b="a" * 40):
    ms = {"d16": 8.4, "d32": 11.2, "d64": 16.0, "d64x4": 34.0}
    return {"model": model, "revision": REVS[model], "e4b_sha": e4b, "gnf4_sha": GNF4_SHA,
            "census_build": {"moe_layers": 48, "int4_expert_layers": 48, "int4_attn_projections": 96, "fuse_qkv_n": 48,
                             "fuse_t1_glue_n": 193, "fuse_router_epilogue_n": 48},
            "decode": {k: _bracket(b, r, ms=ms[k]) for k, (b, r) in DECODE.items()},
            "prefill": {"p512_off": {"last_logits": False, "last_logits_stats": {"status": "off"}, "device_ms": 39.0,
                                     "bulk_kv": True, "classes": {"expert_int4": [1.0, 17.0], "dense_gemm": [1.0, 15.0]},
                                     "dtod": 30.0, "dtod_per_layer": 0.625},
                        "p512_on": {"last_logits": True, "last_logits_stats": {"status": "on"}, "device_ms": 36.5,
                                    "bulk_kv": True, "classes": {"expert_int4": [1.0, 17.0], "dense_gemm": [1.0, 12.5]},
                                    "dtod": 30.0, "dtod_per_layer": 0.625}},
            "head": {"module": "Linear", "weight_shape": [151936, 2048], "weight_dtype": "torch.bfloat16",
                     "rows": {"1": {"device_ms": 0.2}, "512": {"device_ms": 2.6}}}}


def self_test() -> int:
    cases = []

    def run(box, e4b="a" * 40):
        return reduce_obj(box, e4b)

    ok = run(_box())
    cases.append(("a registered census reads", ok["verdict"] == "READ"))
    t = ok.get("tables") or {}
    cases.append(("marginal per row 16->64", t.get("marginal_per_row", {}).get("d16->d64", {}).get("per_row_ms") == 0.15833))
    cases.append(("chained over one piece", t.get("chained_over_one_piece") == 2.125))
    cases.append(("prefill ON - OFF", t.get("prefill_on_minus_off_ms") == -2.5))
    cases.append(("class sums tie out", all(abs(v) < 1e-9 for v in t.get("decode_sum_check", {}).values())))
    cases.append(("split of 64 on buckets to 16", split_of((1, 2, 4, 8, 16), 64) == {"16": {"pieces": 4, "rows": 64, "pad_rows": 0}}))
    cases.append(("split of 40 on buckets to 32", split_of((1, 2, 4, 8, 16, 32), 40) ==
                  {"32": {"pieces": 1, "rows": 32, "pad_rows": 0}, "8": {"pieces": 1, "rows": 8, "pad_rows": 0}}))
    cases.append(("wrong e4b", run(_box(), e4b="b" * 40)["verdict"] == "VOID"))
    b = _box()
    b["revision"] = "0" * 40
    cases.append(("wrong revision", run(b)["verdict"] == "VOID"))
    b = _box()
    b["census_build"]["int4_expert_layers"] = 40
    cases.append(("int4 not on every layer", run(b)["verdict"] == "VOID"))
    b = _box()
    b["census_build"]["fuse_qkv_n"] = 0
    cases.append(("a fold missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["decode"]["d64"]["profiled"]["64"]["eager_steps"] -= 1
    cases.append(("a profiled step missing", run(b)["verdict"] == "VOID"))
    b = _box()
    b["decode"]["d64x4"]["profiled"] = {"64": {"replays": 0, "eager_steps": 8, "rows": 512, "pad_rows": 0}}
    cases.append(("d64x4 ran as one piece", run(b)["verdict"] == "VOID"))
    b = _box()
    b["decode"]["d32"]["profiled"]["32"]["replays"] = 8
    cases.append(("a replay where an eager step was registered", run(b)["verdict"] == "VOID"))
    b = _box()
    b["decode"]["d16"]["graph_status"]["16"] = "graph"
    cases.append(("graph status", run(b)["verdict"] == "VOID"))
    b = _box()
    b["decode"]["d32"]["bulk_kv"] = False
    cases.append(("per-layer KV bookkeeping", run(b)["verdict"] == "VOID"))
    b = _box()
    b["decode"]["d16"]["device_ms"] = 0
    cases.append(("no device kernels", run(b)["verdict"] == "VOID"))
    b = _box()
    b["prefill"]["p512_off"] = {"last_logits": False, "refused": "x"}
    cases.append(("prefill OFF refused", run(b)["verdict"] == "VOID"))
    b = _box()
    b["prefill"]["p512_on"] = {"last_logits": True, "refused": "no keyword"}
    r = run(b)
    cases.append(("prefill ON refused is reported", r["verdict"] == "READ" and r["tables"]["prefill_on"] == "no keyword"))
    b = _box()
    b["prefill"]["p512_on"]["last_logits_stats"] = {"status": "off"}
    cases.append(("prefill ON ran off", run(b)["verdict"] == "VOID"))
    b = _box(model="ibm-granite/granite-3.1-3b-a800m-instruct")
    b["census_build"] = {"moe_layers": 32}
    cases.append(("the proof's NF4 Granite needs no int4 stack", run(b)["verdict"] == "READ"))
    b = copy.deepcopy(_box())
    del b["decode"]["d32"]
    cases.append(("a bracket missing", run(b)["verdict"] == "VOID"))
    bad = [n for n, okk in cases if not okk]
    if bad:
        print("p119_reduce self-test FAILED:", bad)
        return 1
    print(f"p119_reduce self-test OK ({len(cases)} cases)")
    return 0


def reduce_obj(box: dict, e4b_sha: str) -> dict:
    void = faults(box, e4b_sha)
    if void:
        return {"lane": "P119", "verdict": "VOID", "reasons": void}
    return {"lane": "P119", "verdict": "READ", "model": box["model"], "tables": tabulate(box)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    v = reduce(Path(a.dir), a.e4b_sha)
    s = json.dumps(v, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(s + "\n")
    print(f"P119_VERDICT {v['verdict']} {json.dumps(v.get('reasons') or [])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
