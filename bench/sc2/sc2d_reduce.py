#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2d_reduce.py -- lane SC2d's registered rule (bench/sc2/SC2d-PREREG.md; #846): the E4B_PAGED_BULK_KV default's
engagement reads, from box K's run files. Correctness only: no time is read.

Files in DIR, per model M in (gptoss, qw36) and arm A in (off, on), from sc2_driver.py ``run`` and the box:
  M_A_warm.json, M_A_serial.json           4 warm requests, then the serial plan (seed 1); both arms of M, same pool
  M_off_serial_repeat.json                 the determinism control (OFF only)
  health_M_A_start.json / _end.json        the server's own /health at ready and after its runs
  arch_M.json                              the checkpoint's own facts (box K reads config.json and the weight index)

Per model, the gates in order (the first that fails names the model's outcome):
- **SERVED**: OFF served every request VALID (warm, serial, repeat), else UNREAD (the knob is not implicated: a
  harness or model failure, relaunched after a fix); ON served every request VALID, else NOT_SERVED_ON (a finding).
- **ARCH**: the checkpoint is what the read claims: Qwen3.6 has attention layers on a strict subset of its layers (a
  hybrid; the pool holds those only), gpt-oss has a sinks tensor on every layer. Otherwise VOID.
- **PROMPTS**: both arms ran the same prompt pool and plan, else VOID.
- **ENGAGED**, from each server's end /health ``kv_bookkeeping``, with n = the requests that server admitted:
  OFF: ``requested`` and ``bulk`` false, ``flush_layers`` n, ``flush_bulk`` = ``flush_bulk_fallback`` = ``ready_bulk``
  = ``ready_at_flush`` = 0. ON: ``requested`` and ``bulk`` true, ``flush_bulk`` n, ``flush_bulk_fallback`` 0 (the bulk
  path WROTE every prompt, e4b#1174), ``flush_layers`` = ``ready_layers`` = 0. Otherwise NOT_ENGAGED.
- **DETERMINISM**: OFF serial against its repeat is IDENTICAL (sc2_identity), else VOID (identity unreadable).
- **IDENTITY**: OFF serial against ON serial is IDENTICAL, else DIFFERENT.
A model that passes all of them is ENGAGED_IDENTICAL.

**The flip**, as registered: FLIP_LICENSED iff both models are ENGAGED_IDENTICAL. Otherwise FLIP_HELD, naming each
model's outcome. UNREAD on either model is a harness result: the lane relaunches after a fix, and no reading is made.

  sc2d_reduce.py --dir DIR [--out verdict_sc2d.json]
  sc2d_reduce.py --dir DIR --proof MODEL     the proof: MODEL's gates without ARCH; rc 0 iff ENGAGED_IDENTICAL
  sc2d_reduce.py --self-test
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ("gptoss", "qw36")
ARMS = ("off", "on")
RUNS = {"off": ("warm", "serial", "serial_repeat"), "on": ("warm", "serial")}


def _sib(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def served(d, m, arm) -> dict:
    """Every run of this server present and every request VALID."""
    n, bad = 0, []
    for run in RUNS[arm]:
        x = _load(d, f"{m}_{arm}_{run}.json")
        if not x or not x.get("requests"):
            bad.append(f"{run}: missing")
            continue
        inv = [i for i, r in enumerate(x["requests"]) if not r.get("valid")]
        if inv:
            bad.append(f"{run}: requests not VALID {inv[:5]}")
        n += len(x["requests"])
    return {"ok": not bad, "n": n, "bad": bad}


def arch_ok(m, a) -> dict:
    """The checkpoint is what the read claims (box K writes ``arch_<m>.json`` from config.json and the weight index)."""
    if not a:
        return {"ok": False, "why": "no arch record"}
    if m == "qw36":
        attn, layers = a.get("attention_layers") or [], a.get("n_layers") or 0
        ok = bool(attn) and len(attn) < layers and a.get("linear_layers", 0) == layers - len(attn)
        return {"ok": ok, "why": "" if ok else f"not a hybrid: {len(attn)} attention of {layers} layers"}
    ok = a.get("n_layers", 0) > 0 and a.get("sinks_layers") == a.get("n_layers")
    return {"ok": ok, "why": "" if ok else f"sinks on {a.get('sinks_layers')} of {a.get('n_layers')} layers"}


def engagement(arm, h, n) -> dict:
    """The server's own record of which path wrote each prompt (and its first-decode claims)."""
    k = (h or {}).get("kv_bookkeeping") or {}
    want = ({"requested": False, "bulk": False, "flush_layers": n, "flush_bulk": 0, "flush_bulk_fallback": 0,
             "ready_bulk": 0, "ready_at_flush": 0} if arm == "off" else
            {"requested": True, "bulk": True, "flush_layers": 0, "flush_bulk": n, "flush_bulk_fallback": 0,
             "ready_layers": 0})
    bad = {x: k.get(x, "<missing>") for x, v in want.items() if k.get(x, "<missing>") != v}
    return {"ok": not bad, "bad": bad, "kv_bookkeeping": k}


def reduce_model(d, m, ident, need_arch=True) -> dict:
    out = {"gates": {}}
    s = {a: served(d, m, a) for a in ARMS}
    out["gates"]["served"] = s
    if not s["off"]["ok"]:
        out["outcome"] = "UNREAD"
        return out
    if not s["on"]["ok"]:
        out["outcome"] = "NOT_SERVED_ON"
        return out
    ar = arch_ok(m, _load(d, f"arch_{m}.json")) if need_arch else {"ok": True, "why": "the proof model: not read"}
    out["gates"]["arch"] = ar
    if not ar["ok"]:
        out["outcome"] = "VOID"
        return out
    pools = {a: [(_load(d, f"{m}_{a}_{r}.json") or {}).get("prompts_sha256") for r in RUNS[a]] for a in ARMS}
    same = len({x for v in pools.values() for x in v}) == 1
    out["gates"]["prompts"] = {"ok": same, "pools": pools}
    if not same:
        out["outcome"] = "VOID"
        return out
    eng = {a: engagement(a, _load(d, f"health_{m}_{a}_end.json"), s[a]["n"]) for a in ARMS}
    out["gates"]["engaged"] = eng
    if not all(e["ok"] for e in eng.values()):
        out["outcome"] = "NOT_ENGAGED"
        return out
    det = ident.compare(_load(d, f"{m}_off_serial.json"), _load(d, f"{m}_off_serial_repeat.json"))
    out["gates"]["determinism"] = det
    if det["verdict"] != "IDENTICAL":
        out["outcome"] = "VOID"
        return out
    idt = ident.compare(_load(d, f"{m}_off_serial.json"), _load(d, f"{m}_on_serial.json"))
    out["gates"]["identity"] = idt
    out["outcome"] = "ENGAGED_IDENTICAL" if idt["verdict"] == "IDENTICAL" else "DIFFERENT"
    return out


def reduce_dir(d) -> dict:
    ident = _sib("sc2_identity")
    models = {m: reduce_model(d, m, ident) for m in MODELS}
    if any(v["outcome"] == "UNREAD" for v in models.values()):
        flip = "UNREAD"
    else:
        flip = "FLIP_LICENSED" if all(v["outcome"] == "ENGAGED_IDENTICAL" for v in models.values()) else "FLIP_HELD"
    return {"models": models, "flip": flip, "outcomes": {m: v["outcome"] for m, v in models.items()}}


# ------------------------------------------------------------------------------------------------- self-test --

def self_test() -> int:
    def write(d, m, *, on_fallback=0, off_bulk=0, on_text="t", repeat_text="t", on_missing=False, off_missing=False,
              arch=None, pool_on="p", on_bulk=True):
        def run(arm, name, text, n, sha):
            return {"prompts_sha256": sha, "mode": "serial", "plan": [[0.0, i, 8] for i in range(n)],
                    "requests": [{"valid": True, "text": f"{text}{i}", "prompt_tokens": 512} for i in range(n)]}
        files = {}
        if not off_missing:
            files[f"{m}_off_warm.json"] = run("off", "warm", "w", 4, "p")
            files[f"{m}_off_serial.json"] = run("off", "serial", "t", 16, "p")
            files[f"{m}_off_serial_repeat.json"] = run("off", "serial_repeat", repeat_text, 16, "p")
        if not on_missing:
            files[f"{m}_on_warm.json"] = run("on", "warm", "w", 4, pool_on)
            files[f"{m}_on_serial.json"] = run("on", "serial", on_text, 16, pool_on)
        n_off, n_on = 36, 20
        files[f"health_{m}_off_end.json"] = {"kv_bookkeeping": {
            "requested": False, "bulk": False, "flush_layers": n_off - off_bulk, "flush_bulk": off_bulk,
            "flush_bulk_fallback": 0, "ready_layers": n_off, "ready_bulk": 0, "ready_at_flush": 0}}
        files[f"health_{m}_on_end.json"] = {"kv_bookkeeping": {
            "requested": True, "bulk": on_bulk, "flush_layers": 0, "flush_bulk": n_on,
            "flush_bulk_fallback": on_fallback, "ready_layers": 0, "ready_bulk": 0, "ready_at_flush": n_on}}
        files[f"arch_{m}.json"] = arch if arch is not None else (
            {"n_layers": 40, "attention_layers": list(range(3, 40, 4)), "linear_layers": 30} if m == "qw36" else
            {"n_layers": 24, "sinks_layers": 24})
        for name, body in files.items():
            json.dump(body, open(os.path.join(d, name), "w"))

    def case(gpt=None, qw=None):
        with tempfile.TemporaryDirectory() as d:
            write(d, "gptoss", **(gpt or {}))
            write(d, "qw36", **(qw or {}))
            return reduce_dir(d)

    checks = []
    o = case()
    checks.append(("all pass", o["flip"] == "FLIP_LICENSED" and set(o["outcomes"].values()) == {"ENGAGED_IDENTICAL"}))
    o = case(gpt={"on_fallback": 3})
    checks.append(("a bulk call written per layer is not engagement",
                   o["outcomes"]["gptoss"] == "NOT_ENGAGED" and o["flip"] == "FLIP_HELD"
                   and o["models"]["gptoss"]["gates"]["engaged"]["on"]["bad"] == {"flush_bulk_fallback": 3}))
    o = case(qw={"on_bulk": False})
    checks.append(("bulk requested but not on", o["outcomes"]["qw36"] == "NOT_ENGAGED"))
    o = case(gpt={"off_bulk": 2})
    checks.append(("OFF took the bulk path", o["outcomes"]["gptoss"] == "NOT_ENGAGED"))
    o = case(qw={"repeat_text": "x"})
    checks.append(("OFF not deterministic: VOID", o["outcomes"]["qw36"] == "VOID" and o["flip"] == "FLIP_HELD"))
    o = case(qw={"on_text": "x"})
    checks.append(("OFF and ON differ", o["outcomes"]["qw36"] == "DIFFERENT"
                   and o["models"]["qw36"]["gates"]["identity"]["verdict"] == "DIFFERS"))
    o = case(gpt={"on_missing": True})
    checks.append(("ON did not serve", o["outcomes"]["gptoss"] == "NOT_SERVED_ON" and o["flip"] == "FLIP_HELD"))
    o = case(qw={"off_missing": True})
    checks.append(("OFF did not serve: UNREAD", o["outcomes"]["qw36"] == "UNREAD" and o["flip"] == "UNREAD"))
    o = case(qw={"arch": {"n_layers": 40, "attention_layers": list(range(40)), "linear_layers": 0}})
    checks.append(("not a hybrid: VOID", o["outcomes"]["qw36"] == "VOID"))
    o = case(gpt={"arch": {"n_layers": 24, "sinks_layers": 12}})
    checks.append(("sinks missing: VOID", o["outcomes"]["gptoss"] == "VOID"))
    o = case(gpt={"pool_on": "q"})
    checks.append(("different prompt pools: VOID", o["outcomes"]["gptoss"] == "VOID"))
    with tempfile.TemporaryDirectory() as d:
        write(d, "granite")
        os.remove(os.path.join(d, "arch_granite.json"))
        x = reduce_model(d, "granite", _sib("sc2_identity"), need_arch=False)
    checks.append(("the proof reads without an arch record", x["outcome"] == "ENGAGED_IDENTICAL"))
    bad = [n for n, ok in checks if not ok]
    print(f"sc2d_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(checks)} cases)")
    return 1 if bad else 0


def _brief(g):
    """A gate's one-word reading for the summary line: its verdict, its ok, or per arm."""
    if "verdict" in g:
        return g["verdict"]
    if "ok" in g:
        return g["ok"]
    return {a: y.get("ok") for a, y in g.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--proof", metavar="MODEL")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.dir:
        ap.error("--dir is required")
    if a.proof:
        x = reduce_model(a.dir, a.proof, _sib("sc2_identity"), need_arch=False)
        print(f"SC2D_PROOF {a.proof} {x['outcome']} " + json.dumps({k: _brief(g) for k, g in x["gates"].items()}), flush=True)
        return 0 if x["outcome"] == "ENGAGED_IDENTICAL" else 1
    v = reduce_dir(a.dir)
    for m, x in v["models"].items():
        print(f"SC2D_MODEL {m} {x['outcome']} " + json.dumps({k: _brief(g) for k, g in x["gates"].items()}), flush=True)
    print(f"SC2D_FLIP {v['flip']} " + json.dumps(v["outcomes"]), flush=True)
    if a.out:
        with open(a.out, "w", newline="\n") as f:
            json.dump(v, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
