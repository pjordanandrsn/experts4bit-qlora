#!/usr/bin/env python3
"""Lane SV6's reducer (bench/sv6/SV6-PREREG.md): the planner's solver-tier plan for Qwen3-30B-A3B at 8 x 8192 on one
RTX 4090, read from the two arm receipts that ``bench/sv4/sv4_measure.py`` writes (``sv4-arm/1``).

Inputs in the run directory (tc1_drive's copy of the box's ``/root/tc1``):
- ``receipts/b6_short.json`` and ``receipts/b6_long.json``;
- ``forensics.txt``: its first line is nvidia-smi's ``name, memory.total, ...`` (the card's total for Y1).

The verdict:
- NO_READING: no ``b6_short`` receipt, or a receipt whose schema is not ``sv4-arm/1``;
- VOID, any arm:
  - the model, revision, server setup, prompt length or new-token count is not as registered;
  - the estimate's device total is not the registered 21,985,437,184 bytes (the estimate under test changed);
  - the card is not an RTX 4090;
- READ otherwise, with each reading HELD, MISSED or NO_READING:
  - integrity, per arm: ALARM unless status is OK with all 8 requests done, or (``b6_long`` only) status is OOM. An
    ALARM arm's numbers are not read;
  - Y1 (``b6_long``): HELD if OK and its driver peak is at or under the card's total; MISSED on OOM or a driver peak
    over the card's total;
  - Y2 (``b6_long``): its driver peak against the plan (23,990,209,701 bytes): HELD at or under, MISSED over. On OOM
    the error's "this process has X in use" plus "Tried to allocate Y" is a lower bound on the need: MISSED if it is
    over the plan, NO_READING if not (or if the error does not parse);
  - Y3 (``b6_long``) and Y4 (``b6_short``): the allocator peak against the estimate, HELD within +/-5% (inclusive),
    MISSED outside it (``above`` or ``below``). On OOM the larger of the receipt's allocator peak and the error's
    "X is allocated by PyTorch" plus Y is a lower bound: MISSED (above) if it is over +5%, NO_READING if not;
  - Y5 (the split): every clean arm that records the server's tier rows has exactly 5,109 VRAM / 1,035 DRAM / 0 NVMe:
    HELD if all do, MISSED if any differs, NO_READING if no clean arm records them;
  - Y6 (the prompt-length items): ``b6_long``'s allocator peak minus ``b6_short``'s against the 1,052,688,384 bytes
    the estimate's formulas give for 8,000- against 1,024-token prompts (bulk flush + prefill staging), HELD within
    +/-15% (inclusive), MISSED outside it. On ``b6_long``'s OOM its floor (as in Y3) minus ``b6_short``'s peak is a
    lower bound: MISSED (above) if over +15%, NO_READING if not. NO_READING if either arm is not clean.

    python sv6_reduce.py --dir RUN_DIR --out verdict.json
    python sv6_reduce.py --self-test
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
SETUP = {"placement": "solver", "max_seqs": 8, "max_tokens_per_seq": 8192, "chunk_tokens": 512, "graphs": False,
         "buckets": [1, 2, 4, 8], "prefill_graph": "0", "vram_gb": 12.631, "dram_gb": 15.187, "hot_rows": 1,
         "exp_int4": False, "attn_int4": False}
ARMS = {"b6_short": 1024, "b6_long": 8000}
NEW_TOKENS, N_REQ = 32, 8
GIB = 2 ** 30
EST_BYTES = 21985437184                 # estimate_serve_footprint's device total at 50e24f3c: 20.476 GiB
PLAN_BYTES = 23990209701                # loggetta dd4783f's plan: the estimate + 1.367 GiB measured reserve + 0.5 GiB context
SPLIT = {"vram": 5109, "dram": 1035, "nvme": 0}
DELTA_BYTES = 1052688384                # flush(8000) - flush(1024) + staging(8512 - 1536 tokens), from the code's formulas
BAND, DELTA_BAND = 0.05, 0.15
CARD = "RTX 4090"
_UNIT = {"KiB": 2 ** 10, "MiB": 2 ** 20, "GiB": 2 ** 30}
_TRIED = r"Tried to allocate ([\d.]+) (KiB|MiB|GiB)"
_IN_USE = r"this process has ([\d.]+) (KiB|MiB|GiB) memory in use"
_ALLOCATED = r"([\d.]+) (KiB|MiB|GiB) is allocated by PyTorch"
_EPS = 1e-9                             # inclusive band edges, robust to a byte of rounding


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
    elif not (tag == "b6_long" and status == "OOM"):
        bad.append(f"status {status}")
    return ("ALARM", bad) if bad else ("CLEAN", [])


def allocator_floor(r):
    """An OOM arm's allocator lower bound: the larger of its recorded peak and allocated + requested."""
    tried, _, allocated = oom_numbers(r.get("error"))
    return max(r.get("measured", {}).get("device_peak_bytes") or 0, (allocated + tried) if allocated and tried else 0)


def read_allocator(r):
    """Y3/Y4 for one clean arm: (reading, detail)."""
    if r.get("status") == "OK":
        peak = r["measured"]["device_peak_bytes"]
        rel = peak / EST_BYTES - 1
        reading = "HELD" if abs(rel) <= BAND + _EPS else "MISSED"
        return reading, {"peak_bytes": peak, "rel": rel, "bound": "peak",
                         **({"side": "above" if rel > 0 else "below"} if reading == "MISSED" else {})}
    floor = allocator_floor(r)
    rel = floor / EST_BYTES - 1
    reading = "MISSED" if rel > BAND + _EPS else "NO_READING"
    return reading, {"floor_bytes": floor, "rel": rel, "bound": "lower",
                     **({"side": "above"} if reading == "MISSED" else {})}


def reduce(receipts, forensics):
    """receipts: {tag: dict or None}; forensics: forensics.txt's text. Returns the verdict dict."""
    short, long_ = receipts.get("b6_short"), receipts.get("b6_long")
    if short is None:
        return {"verdict": "NO_READING", "why": ["no b6_short receipt"]}
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

    rd["Y4"] = dict(zip(("reading", "detail"), read_allocator(short))) if clean["b6_short"] else \
        {"reading": "NO_READING", "detail": "b6_short failed integrity"}

    tiers = {t: r.get("server_tiers") for t, r in present.items() if clean.get(t) and r.get("server_tiers")}
    if not tiers:
        rd["Y5"] = {"reading": "NO_READING", "detail": "no clean arm records the server's tier rows"}
    else:
        rd["Y5"] = {"reading": "HELD" if all(v == SPLIT for v in tiers.values()) else "MISSED",
                    "detail": {"registered": SPLIT, **tiers}}

    if long_ is None or not clean.get("b6_long"):
        why = "no b6_long receipt" if long_ is None else "b6_long failed integrity"
        for z in ("Y1", "Y2", "Y3", "Y6"):
            rd[z] = {"reading": "NO_READING", "detail": why}
        return out

    m = long_.get("measured", {})
    card = card_total_bytes(forensics)
    if long_["status"] == "OK":
        drv = m.get("driver_process_peak_bytes")
        if drv is None or card is None:
            rd["Y1"] = {"reading": "NO_READING", "detail": "no driver peak or no card total"}
        else:
            rd["Y1"] = {"reading": "HELD" if drv <= card else "MISSED", "detail": {"driver_peak_bytes": drv,
                                                                                 "card_total_bytes": card}}
        rd["Y2"] = ({"reading": "NO_READING", "detail": "no driver peak"} if drv is None else
                    {"reading": "HELD" if drv <= PLAN_BYTES else "MISSED",
                     "detail": {"driver_peak_bytes": drv, "plan_bytes": PLAN_BYTES, "bound": "peak"}})
    else:
        rd["Y1"] = {"reading": "MISSED", "detail": {"status": "OOM", "error": (long_.get("error") or "")[:200]}}
        tried, in_use, _ = oom_numbers(long_.get("error"))
        if tried is None or in_use is None:
            rd["Y2"] = {"reading": "NO_READING", "detail": "the OOM message does not parse"}
        else:
            floor = in_use + tried
            rd["Y2"] = {"reading": "MISSED" if floor > PLAN_BYTES else "NO_READING",
                        "detail": {"floor_bytes": floor, "plan_bytes": PLAN_BYTES, "bound": "lower"}}
    rd["Y3"] = dict(zip(("reading", "detail"), read_allocator(long_)))

    if not clean["b6_short"]:
        rd["Y6"] = {"reading": "NO_READING", "detail": "b6_short failed integrity"}
    elif long_["status"] == "OK":
        delta = m["device_peak_bytes"] - short["measured"]["device_peak_bytes"]
        rel = delta / DELTA_BYTES - 1
        reading = "HELD" if abs(rel) <= DELTA_BAND + _EPS else "MISSED"
        rd["Y6"] = {"reading": reading, "detail": {"delta_bytes": delta, "predicted_bytes": DELTA_BYTES, "rel": rel,
                                                   "bound": "peak", **({"side": "above" if rel > 0 else "below"}
                                                                       if reading == "MISSED" else {})}}
    else:
        delta = allocator_floor(long_) - short["measured"]["device_peak_bytes"]
        rel = delta / DELTA_BYTES - 1
        reading = "MISSED" if rel > DELTA_BAND + _EPS else "NO_READING"
        rd["Y6"] = {"reading": reading, "detail": {"delta_floor_bytes": delta, "predicted_bytes": DELTA_BYTES,
                                                   "rel": rel, "bound": "lower",
                                                   **({"side": "above"} if reading == "MISSED" else {})}}
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
_SHORT_GIB = 19.6
_LONG_GIB = _SHORT_GIB + DELTA_BYTES / GIB


def _arm(tag, peak_gib=None, driver_gib=21.6, status="OK"):
    peak_gib = (_SHORT_GIB if tag == "b6_short" else _LONG_GIB) if peak_gib is None else peak_gib
    r = {"schema": SCHEMA, "args": {"model": MODEL, "revision": REV, "prompt_tokens": ARMS[tag],
                                    "new_tokens": NEW_TOKENS},
         "setup": copy.deepcopy(SETUP), "estimate": {"device_total": EST_BYTES}, "gpu": "NVIDIA GeForce RTX 4090",
         "graph_status": {}, "server_tiers": dict(SPLIT), "status": status,
         "measured": {"device_peak_bytes": round(peak_gib * GIB)}}
    if status == "OK":
        r["measured"].update(requests_done=N_REQ, driver_process_peak_bytes=round(driver_gib * GIB))
    else:
        r["error"] = _OOM
    return r


def _mut(r, f):
    r = copy.deepcopy(r)
    f(r)
    return r


def self_test():
    S, L = (lambda **k: _arm("b6_short", **k)), (lambda **k: _arm("b6_long", **k))
    rd = lambda v, z: v.get("readings", {}).get(z, {}).get("reading")  # noqa: E731
    side = lambda v, z: v["readings"][z]["detail"].get("side")  # noqa: E731
    est = EST_BYTES / GIB
    cases = [
        ("all held", {"b6_short": S(), "b6_long": L()}, _FORENSICS,
         lambda v: v["verdict"] == "READ" and all(rd(v, z) == "HELD" for z in ("Y1", "Y2", "Y3", "Y4", "Y5", "Y6"))),
        ("OOM, floors inside the bands", {"b6_short": S(peak_gib=20.2), "b6_long": _mut(L(peak_gib=20.9, status="OOM"),
                                                                                        lambda r: r.update(error=_OOM.replace("22.59 GiB is", "21.00 GiB is")))},
         _FORENSICS, lambda v: (rd(v, "Y1"), rd(v, "Y2"), rd(v, "Y3"), rd(v, "Y6")) ==
         ("MISSED", "MISSED", "NO_READING", "NO_READING") and v["integrity"]["b6_long"]["state"] == "CLEAN"),
        ("OOM, floors above the bands", {"b6_short": S(), "b6_long": L(peak_gib=22.9, status="OOM")}, _FORENSICS,
         lambda v: rd(v, "Y3") == "MISSED" and side(v, "Y3") == "above" and rd(v, "Y6") == "MISSED"
         and side(v, "Y6") == "above"),
        ("OOM message unparsable", {"b6_short": S(), "b6_long": _mut(L(peak_gib=20.5, status="OOM"),
                                                                    lambda r: r.update(error="CUDA OOM"))},
         _FORENSICS, lambda v: rd(v, "Y1") == "MISSED" and rd(v, "Y2") == "NO_READING" and rd(v, "Y3") == "NO_READING"),
        ("OOM floor under the plan", {"b6_short": S(), "b6_long": _mut(L(peak_gib=20.5, status="OOM"), lambda r: r.update(
            error=_OOM.replace("has 23.50 GiB", "has 22.00 GiB")))}, _FORENSICS,
         lambda v: rd(v, "Y1") == "MISSED" and rd(v, "Y2") == "NO_READING"),
        ("driver over the plan, under the card", {"b6_short": S(), "b6_long": L(driver_gib=22.6)}, _FORENSICS,
         lambda v: rd(v, "Y1") == "HELD" and rd(v, "Y2") == "MISSED"),
        ("driver over the card", {"b6_short": S(), "b6_long": L(driver_gib=24.1)}, _FORENSICS,
         lambda v: rd(v, "Y1") == "MISSED" and rd(v, "Y2") == "MISSED"),
        ("no card total", {"b6_short": S(), "b6_long": L()}, "",
         lambda v: rd(v, "Y1") == "NO_READING" and rd(v, "Y2") == "HELD"),
        ("Y3 above the band", {"b6_short": S(peak_gib=est * 1.06 - DELTA_BYTES / GIB), "b6_long": L(peak_gib=est * 1.06)},
         _FORENSICS, lambda v: rd(v, "Y3") == "MISSED" and side(v, "Y3") == "above" and rd(v, "Y6") == "HELD"),
        ("Y4 below the band", {"b6_short": S(peak_gib=est * 0.94), "b6_long": L(peak_gib=est * 0.94 + DELTA_BYTES / GIB)},
         _FORENSICS, lambda v: rd(v, "Y4") == "MISSED" and side(v, "Y4") == "below"),
        ("band edge inclusive", {"b6_short": S(peak_gib=est * 1.05 - DELTA_BYTES / GIB), "b6_long": L(peak_gib=est * 1.05)},
         _FORENSICS, lambda v: rd(v, "Y3") == "HELD"),
        ("Y6 above", {"b6_short": S(), "b6_long": L(peak_gib=_SHORT_GIB + 1.2 * DELTA_BYTES / GIB)}, _FORENSICS,
         lambda v: rd(v, "Y6") == "MISSED" and side(v, "Y6") == "above"),
        ("Y6 below", {"b6_short": S(), "b6_long": L(peak_gib=_SHORT_GIB + 0.8 * DELTA_BYTES / GIB)}, _FORENSICS,
         lambda v: rd(v, "Y6") == "MISSED" and side(v, "Y6") == "below"),
        ("Y6 edge inclusive", {"b6_short": S(), "b6_long": L(peak_gib=_SHORT_GIB + 1.15 * DELTA_BYTES / GIB)},
         _FORENSICS, lambda v: rd(v, "Y6") == "HELD"),
        ("split differs", {"b6_short": _mut(S(), lambda r: r.update(server_tiers={"vram": 5108, "dram": 1036, "nvme": 0})),
                           "b6_long": L()}, _FORENSICS, lambda v: rd(v, "Y5") == "MISSED"),
        ("split unrecorded", {"b6_short": _mut(S(), lambda r: r.pop("server_tiers")),
                              "b6_long": _mut(L(), lambda r: r.pop("server_tiers"))}, _FORENSICS,
         lambda v: rd(v, "Y5") == "NO_READING"),
        ("no b6_short", {"b6_short": None, "b6_long": L()}, _FORENSICS, lambda v: v["verdict"] == "NO_READING"),
        ("no b6_long", {"b6_short": S(), "b6_long": None}, _FORENSICS,
         lambda v: v["verdict"] == "READ" and rd(v, "Y4") == "HELD" and rd(v, "Y5") == "HELD"
         and all(rd(v, z) == "NO_READING" for z in ("Y1", "Y2", "Y3", "Y6"))),
        ("wrong schema", {"b6_short": _mut(S(), lambda r: r.update(schema="sv4-arm/0")), "b6_long": L()}, _FORENSICS,
         lambda v: v["verdict"] == "NO_READING"),
        ("wrong revision", {"b6_short": S(), "b6_long": _mut(L(), lambda r: r["args"].update(revision="x"))},
         _FORENSICS, lambda v: v["verdict"] == "VOID"),
        ("all-VRAM placement", {"b6_short": _mut(S(), lambda r: r["setup"].update(placement="all-vram")),
                                "b6_long": L()}, _FORENSICS, lambda v: v["verdict"] == "VOID"),
        ("other tier budget", {"b6_short": S(), "b6_long": _mut(L(), lambda r: r["setup"].update(vram_gb=10.933))},
         _FORENSICS, lambda v: v["verdict"] == "VOID"),
        ("wrong prompt length", {"b6_short": S(), "b6_long": _mut(L(), lambda r: r["args"].update(prompt_tokens=4096))},
         _FORENSICS, lambda v: v["verdict"] == "VOID"),
        ("estimate changed", {"b6_short": _mut(S(), lambda r: r["estimate"].update(device_total=EST_BYTES + 1)),
                              "b6_long": L()}, _FORENSICS, lambda v: v["verdict"] == "VOID"),
        ("not a 4090", {"b6_short": _mut(S(), lambda r: r.update(gpu="NVIDIA GeForce RTX 5090")), "b6_long": L()},
         _FORENSICS, lambda v: v["verdict"] == "VOID"),
        ("short arm missing a request", {"b6_short": _mut(S(), lambda r: r["measured"].update(requests_done=7)),
                                         "b6_long": L()}, _FORENSICS,
         lambda v: v["integrity"]["b6_short"]["state"] == "ALARM" and rd(v, "Y4") == "NO_READING"
         and rd(v, "Y6") == "NO_READING" and rd(v, "Y1") == "HELD"),
        ("short arm OOM is an ALARM", {"b6_short": S(status="OOM"), "b6_long": L()}, _FORENSICS,
         lambda v: v["integrity"]["b6_short"]["state"] == "ALARM" and rd(v, "Y4") == "NO_READING"),
        ("long arm ERROR", {"b6_short": S(), "b6_long": L(status="ERROR")}, _FORENSICS,
         lambda v: v["integrity"]["b6_long"]["state"] == "ALARM"
         and all(rd(v, z) == "NO_READING" for z in ("Y1", "Y2", "Y3", "Y6"))),
        ("an ALARM arm's split is not read", {"b6_short": S(), "b6_long": _mut(L(status="ERROR"), lambda r: r.update(
            server_tiers={"vram": 1, "dram": 1, "nvme": 1}))}, _FORENSICS, lambda v: rd(v, "Y5") == "HELD"),
        ("OOM numbers parse", {"b6_short": S(), "b6_long": L()}, _FORENSICS,
         lambda v: oom_numbers(_OOM) == (32 * 2 ** 20, round(23.50 * GIB), round(22.59 * GIB))),
    ]
    failed = []
    for name, receipts, fx, check in cases:
        v = reduce(receipts, fx)
        if not check(v):
            failed.append(name)
            print("FAIL", name, json.dumps(v, default=str)[:500])
    print(f"sv6_reduce self-test: {len(cases) - len(failed)}/{len(cases)} passed")
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
