#!/usr/bin/env python3
"""Lane SV5's reducer (bench/sv5/SV5-PREREG.md): the planner's all-VRAM 8 x 8192 plan for Qwen3-30B-A3B on one
RTX 4090, read from the two arm receipts that ``bench/sv4/sv4_measure.py`` writes (``sv4-arm/1``).

Inputs in the run directory (tc1_drive's copy of the box's ``/root/tc1``):
- ``receipts/a5_short.json`` and ``receipts/a5_long.json``;
- ``forensics.txt``: its first line is nvidia-smi's ``name, memory.total, ...`` (the card's total for Z1).

The verdict:
- NO_READING: no ``a5_short`` receipt, or a receipt whose schema is not ``sv4-arm/1``;
- VOID, any arm:
  - the model, revision, server setup, prompt length or new-token count is not as registered;
  - the estimate's device total is not the registered 22.150 GiB (the estimate under test changed);
  - the card is not an RTX 4090;
- READ otherwise, with each reading HELD, MISSED or NO_READING:
  - integrity, per arm: ALARM unless status is OK with all 8 requests done, or (``a5_long`` only) status is OOM; and
    every captured decode bucket reports ``graph``. An ALARM arm's numbers are not read;
  - Z1 (``a5_long``): HELD if OK and its driver peak is at or under the card's total; MISSED on OOM or a driver peak
    over the card's total;
  - Z2 (``a5_long``): its driver peak against the plan (22.74 GiB): HELD at or under, MISSED over. On OOM the
    error's "this process has X in use" plus "Tried to allocate Y" is a lower bound on the need: MISSED if it is over
    the plan, NO_READING if not (or if the error does not parse);
  - Z3 (``a5_long``) and Z4 (``a5_short``): the allocator peak against the estimate, HELD within +/-5% (inclusive),
    MISSED outside it (``above`` or ``below``). On OOM the larger of the receipt's allocator peak and the error's
    "X is allocated by PyTorch" plus "Tried to allocate Y" is a lower bound: MISSED (above) if it is over +5%,
    NO_READING if not, because the true peak is unknown.

    python sv5_reduce.py --dir RUN_DIR --out verdict.json
    python sv5_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

SCHEMA = "sv4-arm/1"
MODEL, REV = "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
SETUP = {"placement": "all-vram", "max_seqs": 8, "max_tokens_per_seq": 8192, "chunk_tokens": 512, "graphs": True,
         "buckets": [1, 2, 4, 8], "prefill_graph": "0", "hot_rows": 64, "exp_int4": False, "attn_int4": False}
ARMS = {"a5_short": 1024, "a5_long": 8000}
NEW_TOKENS, N_REQ = 32, 8
GIB = 2 ** 30
EST_BYTES = 23783815168                 # estimate_serve_footprint's device total at fd70f75b: 22.150 GiB
PLAN_BYTES = round(22.74 * GIB)         # the planner's plan: the estimate + 0.09 GiB learned reserve + 0.5 GiB context
BAND = 0.05
CARD = "RTX 4090"
_UNIT = {"KiB": 2 ** 10, "MiB": 2 ** 20, "GiB": 2 ** 30}


_TRIED = r"Tried to allocate ([\d.]+) (KiB|MiB|GiB)"
_IN_USE = r"this process has ([\d.]+) (KiB|MiB|GiB) memory in use"
_ALLOCATED = r"([\d.]+) (KiB|MiB|GiB) is allocated by PyTorch"


def _amount(regex, text):
    m = re.search(regex, text or "")
    return round(float(m.group(1)) * _UNIT[m.group(2)]) if m else None


def oom_numbers(error):
    """(tried, process in use, allocated by PyTorch) in bytes from a CUDA OOM message; None for what doesn't parse."""
    return _amount(_TRIED, error), _amount(_IN_USE, error), _amount(_ALLOCATED, error)


def card_total_bytes(forensics):
    first = (forensics or "").splitlines()[:1]
    m = re.search(r",\s*(\d+) MiB", first[0]) if first else None
    return int(m.group(1)) * 2 ** 20 if m else None


def void_reasons(tag, r):
    errs = []
    a = r.get("args", {})
    if a.get("model") != MODEL or a.get("revision") != REV:
        errs.append(f"{tag}: model {a.get('model')}@{a.get('revision')} is not {MODEL}@{REV}")
    s = r.get("setup", {})
    for k, want in SETUP.items():
        if s.get(k) != want:
            errs.append(f"{tag}: setup.{k} = {s.get(k)!r}, registered {want!r}")
    if a.get("prompt_tokens") != ARMS[tag] or a.get("new_tokens") != NEW_TOKENS:
        errs.append(f"{tag}: prompts {a.get('prompt_tokens')} / new {a.get('new_tokens')}, registered "
                    f"{ARMS[tag]} / {NEW_TOKENS}")
    if r.get("estimate", {}).get("device_total") != EST_BYTES:
        errs.append(f"{tag}: estimate device_total {r.get('estimate', {}).get('device_total')} != {EST_BYTES}")
    if CARD not in str(r.get("gpu", "")):
        errs.append(f"{tag}: card {r.get('gpu')!r} is not an {CARD}")
    return errs


def integrity(tag, r):
    status = r.get("status")
    bad = []
    if status == "OK":
        if r.get("measured", {}).get("requests_done") != N_REQ:
            bad.append(f"requests_done {r.get('measured', {}).get('requests_done')} != {N_REQ}")
    elif not (tag == "a5_long" and status == "OOM"):
        bad.append(f"status {status}")
    graphs = r.get("graph_status") or {}
    if not graphs:
        bad.append("no decode bucket captured")
    bad += [f"bucket {b} = {v}" for b, v in sorted(graphs.items()) if v != "graph"]
    return ("ALARM", bad) if bad else ("CLEAN", [])


def band(peak, est=EST_BYTES):
    rel = peak / est - 1
    return rel, ("HELD" if abs(rel) <= BAND + 1e-12 else "MISSED"), ("above" if rel > 0 else "below")


def read_allocator(r):
    """Z3/Z4 for one arm: (reading, detail)."""
    m = r.get("measured", {})
    if r.get("status") == "OK":
        rel, reading, side = band(m["device_peak_bytes"])
        return reading, {"peak_bytes": m["device_peak_bytes"], "rel": rel, "bound": "peak",
                         **({"side": side} if reading == "MISSED" else {})}
    tried, _, allocated = oom_numbers(r.get("error"))
    floor = max(m.get("device_peak_bytes") or 0, (allocated + tried) if allocated and tried else 0)
    rel = floor / EST_BYTES - 1
    reading = "MISSED" if rel > BAND + 1e-12 else "NO_READING"
    return reading, {"floor_bytes": floor, "rel": rel, "bound": "lower",
                     **({"side": "above"} if reading == "MISSED" else {})}


def reduce(receipts, forensics):
    """receipts: {tag: dict or None}; forensics: forensics.txt's text. Returns the verdict dict."""
    short, long_ = receipts.get("a5_short"), receipts.get("a5_long")
    if short is None:
        return {"verdict": "NO_READING", "why": ["no a5_short receipt"]}
    present = {t: r for t, r in receipts.items() if r is not None}
    wrong = [f"{t}: schema {r.get('schema')!r}" for t, r in present.items() if r.get("schema") != SCHEMA]
    if wrong:
        return {"verdict": "NO_READING", "why": wrong}
    voids = [e for t, r in present.items() for e in void_reasons(t, r)]
    if voids:
        return {"verdict": "VOID", "why": voids}

    out = {"verdict": "READ", "integrity": {}, "readings": {}}
    clean = {}
    for t, r in present.items():
        state, bad = integrity(t, r)
        out["integrity"][t] = {"state": state, **({"why": bad} if bad else {})}
        clean[t] = state == "CLEAN"
    rd = out["readings"]

    rd["Z4"] = dict(zip(("reading", "detail"), read_allocator(short))) if clean["a5_short"] else \
        {"reading": "NO_READING", "detail": "a5_short failed integrity"}

    if long_ is None or not clean.get("a5_long"):
        why = "no a5_long receipt" if long_ is None else "a5_long failed integrity"
        for z in ("Z1", "Z2", "Z3"):
            rd[z] = {"reading": "NO_READING", "detail": why}
        return out

    m = long_.get("measured", {})
    card = card_total_bytes(forensics)
    if long_["status"] == "OK":
        drv = m.get("driver_process_peak_bytes")
        if drv is None or card is None:
            rd["Z1"] = {"reading": "NO_READING", "detail": "no driver peak or no card total"}
        else:
            rd["Z1"] = {"reading": "HELD" if drv <= card else "MISSED", "detail": {"driver_peak_bytes": drv,
                                                                                 "card_total_bytes": card}}
        rd["Z2"] = ({"reading": "NO_READING", "detail": "no driver peak"} if drv is None else
                    {"reading": "HELD" if drv <= PLAN_BYTES else "MISSED",
                     "detail": {"driver_peak_bytes": drv, "plan_bytes": PLAN_BYTES, "bound": "peak"}})
    else:
        rd["Z1"] = {"reading": "MISSED", "detail": {"status": "OOM", "error": (long_.get("error") or "")[:200]}}
        tried, in_use, _ = oom_numbers(long_.get("error"))
        if tried is None or in_use is None:
            rd["Z2"] = {"reading": "NO_READING", "detail": "the OOM message does not parse"}
        else:
            floor = in_use + tried
            rd["Z2"] = {"reading": "MISSED" if floor > PLAN_BYTES else "NO_READING",
                        "detail": {"floor_bytes": floor, "plan_bytes": PLAN_BYTES, "bound": "lower"}}
    rd["Z3"] = dict(zip(("reading", "detail"), read_allocator(long_)))
    return out


def load_dir(d):
    d = Path(d)
    rec = {}
    for t in ARMS:
        p = d / "receipts" / f"{t}.json"
        rec[t] = json.loads(p.read_text()) if p.is_file() else None
    fx = d / "forensics.txt"
    return rec, fx.read_text() if fx.is_file() else ""


# ----------------------------------------------------------------------------------------------------- self-test
_FORENSICS = "NVIDIA GeForce RTX 4090, 24564 MiB, 595.84, 3, 16\n"
_OOM = ("CUDA out of memory. Tried to allocate 32.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 9.75 MiB "
        "is free. Including non-PyTorch memory, this process has 23.50 GiB memory in use. Of the allocated memory "
        "22.59 GiB is allocated by PyTorch, with 79.87 MiB allocated in private pools (e.g., CUDA Graphs), and 352.99 "
        "MiB is reserved by PyTorch but unallocated.")


def _arm(tag, peak_gib=21.7, driver_gib=22.4, status="OK"):
    r = {"schema": SCHEMA, "args": {"model": MODEL, "revision": REV, "prompt_tokens": ARMS[tag],
                                    "new_tokens": NEW_TOKENS},
         "setup": copy.deepcopy(SETUP), "estimate": {"device_total": EST_BYTES}, "gpu": "NVIDIA GeForce RTX 4090",
         "graph_status": {"1": "graph", "2": "graph", "4": "graph", "8": "graph"}, "status": status,
         "measured": {"device_peak_bytes": round(peak_gib * GIB)}}
    if status == "OK":
        r["measured"].update(requests_done=N_REQ, driver_process_peak_bytes=round(driver_gib * GIB))
    else:
        r["error"] = _OOM
    return r


def _case(name, receipts, check, forensics=_FORENSICS):
    v = reduce(receipts, forensics)
    ok = check(v)
    return name, ok, v


def self_test():
    S, L = (lambda **k: _arm("a5_short", **k)), (lambda **k: _arm("a5_long", **k))
    rd = lambda v, z: v.get("readings", {}).get(z, {}).get("reading")  # noqa: E731

    def mut(r, f):
        r = copy.deepcopy(r)
        f(r)
        return r

    cases = [
        _case("all held", {"a5_short": S(), "a5_long": L(peak_gib=22.3, driver_gib=22.6)},
              lambda v: v["verdict"] == "READ" and all(rd(v, z) == "HELD" for z in ("Z1", "Z2", "Z3", "Z4"))),
        _case("OOM, floor inside the band", {"a5_short": S(), "a5_long": L(peak_gib=22.535, status="OOM")},
              lambda v: (rd(v, "Z1"), rd(v, "Z2"), rd(v, "Z3")) == ("MISSED", "MISSED", "NO_READING")
              and v["integrity"]["a5_long"]["state"] == "CLEAN"),
        _case("OOM, floor above the band", {"a5_short": S(), "a5_long": L(peak_gib=23.4, status="OOM")},
              lambda v: rd(v, "Z3") == "MISSED" and v["readings"]["Z3"]["detail"]["side"] == "above"),
        _case("OOM message unparsable", {"a5_short": S(), "a5_long": mut(L(peak_gib=22.5, status="OOM"),
                                                                       lambda r: r.update(error="CUDA OOM"))},
              lambda v: rd(v, "Z1") == "MISSED" and rd(v, "Z2") == "NO_READING" and rd(v, "Z3") == "NO_READING"),
        _case("OOM floor under the plan", {"a5_short": S(), "a5_long": mut(L(peak_gib=22.0, status="OOM"), lambda r: r.update(
            error=_OOM.replace("has 23.50 GiB", "has 22.10 GiB")))},
              lambda v: rd(v, "Z1") == "MISSED" and rd(v, "Z2") == "NO_READING"),
        _case("driver over the plan, under the card", {"a5_short": S(), "a5_long": L(peak_gib=22.3, driver_gib=23.2)},
              lambda v: rd(v, "Z1") == "HELD" and rd(v, "Z2") == "MISSED"),
        _case("driver over the card", {"a5_short": S(), "a5_long": L(peak_gib=22.3, driver_gib=24.1)},
              lambda v: rd(v, "Z1") == "MISSED" and rd(v, "Z2") == "MISSED"),
        _case("no card total", {"a5_short": S(), "a5_long": L(peak_gib=22.3)},
              lambda v: rd(v, "Z1") == "NO_READING" and rd(v, "Z2") == "HELD", forensics=""),
        _case("Z3 above the band (OK arm)", {"a5_short": S(), "a5_long": L(peak_gib=23.3)},
              lambda v: rd(v, "Z3") == "MISSED" and v["readings"]["Z3"]["detail"]["side"] == "above"),
        _case("Z4 below the band", {"a5_short": S(peak_gib=20.9), "a5_long": L()},
              lambda v: rd(v, "Z4") == "MISSED" and v["readings"]["Z4"]["detail"]["side"] == "below"),
        _case("band edge inclusive", {"a5_short": S(peak_gib=EST_BYTES * 1.05 / GIB), "a5_long": L()},
              lambda v: rd(v, "Z4") == "HELD"),
        _case("no a5_short", {"a5_short": None, "a5_long": L()}, lambda v: v["verdict"] == "NO_READING"),
        _case("no a5_long", {"a5_short": S(), "a5_long": None},
              lambda v: v["verdict"] == "READ" and rd(v, "Z4") == "HELD"
              and all(rd(v, z) == "NO_READING" for z in ("Z1", "Z2", "Z3"))),
        _case("wrong schema", {"a5_short": mut(S(), lambda r: r.update(schema="sv4-arm/0")), "a5_long": L()},
              lambda v: v["verdict"] == "NO_READING"),
        _case("wrong revision", {"a5_short": S(), "a5_long": mut(L(), lambda r: r["args"].update(revision="x"))},
              lambda v: v["verdict"] == "VOID"),
        _case("graphs off", {"a5_short": mut(S(), lambda r: r["setup"].update(graphs=False)), "a5_long": L()},
              lambda v: v["verdict"] == "VOID"),
        _case("wrong prompt length", {"a5_short": S(), "a5_long": mut(L(), lambda r: r["args"].update(
            prompt_tokens=4096))}, lambda v: v["verdict"] == "VOID"),
        _case("estimate changed", {"a5_short": mut(S(), lambda r: r["estimate"].update(device_total=EST_BYTES + 1)),
                                   "a5_long": L()}, lambda v: v["verdict"] == "VOID"),
        _case("not a 4090", {"a5_short": mut(S(), lambda r: r.update(gpu="NVIDIA GeForce RTX 5090")), "a5_long": L()},
              lambda v: v["verdict"] == "VOID"),
        _case("short arm missing a request", {"a5_short": mut(S(), lambda r: r["measured"].update(requests_done=7)),
                                              "a5_long": L()},
              lambda v: v["integrity"]["a5_short"]["state"] == "ALARM" and rd(v, "Z4") == "NO_READING"),
        _case("eager bucket", {"a5_short": mut(S(), lambda r: r["graph_status"].update({"8": "eager"})),
                               "a5_long": L()}, lambda v: v["integrity"]["a5_short"]["state"] == "ALARM"),
        _case("no bucket captured", {"a5_short": S(), "a5_long": mut(L(), lambda r: r.update(graph_status={}))},
              lambda v: v["integrity"]["a5_long"]["state"] == "ALARM" and rd(v, "Z1") == "NO_READING"),
        _case("short arm OOM is an ALARM", {"a5_short": S(status="OOM"), "a5_long": L()},
              lambda v: v["integrity"]["a5_short"]["state"] == "ALARM" and rd(v, "Z4") == "NO_READING"),
        _case("long arm ERROR", {"a5_short": S(), "a5_long": L(status="ERROR")},
              lambda v: v["integrity"]["a5_long"]["state"] == "ALARM"
              and all(rd(v, z) == "NO_READING" for z in ("Z1", "Z2", "Z3"))),
        _case("OOM numbers parse", {"a5_short": S(), "a5_long": L(status="OOM")},
              lambda v: oom_numbers(_OOM) == (32 * 2 ** 20, round(23.50 * GIB), round(22.59 * GIB))),
    ]
    failed = [(n, v) for n, ok, v in cases if not ok]
    for n, v in failed:
        print("FAIL", n, json.dumps(v, default=str)[:400])
    print(f"sv5_reduce self-test: {len(cases) - len(failed)}/{len(cases)} passed")
    return 0 if not failed else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.dir:
        ap.error("--dir is required")
    v = reduce(*load_dir(a.dir))
    text = json.dumps(v, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
