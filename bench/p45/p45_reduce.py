#!/usr/bin/env python3
"""bench/p45/p45_reduce.py -- read P45 (where a training step's time goes) from the profile receipts and dmon samples.

    python bench/p45/p45_reduce.py <run-dir>/tp4 [--md RESULTS-p45.md] [--json out.json]

Rules (P45-PREREG.md, in code): P1 device_busy_fraction(e4b) <= 0.30 (refuted >= 0.60); P2 routing+autograd >= 0.50 of
CPU self time (alternatives fused_kernel >= 0.50, memcpy >= 0.30, optimizer >= 0.30); P3 device_events_per_step >= 20000
(refuted < 5000); P4 busy(unsloth) >= 2x busy(e4b) (alternative <= 1.3x); P5 PCIe rx+tx mean < 2 GB/s (refuted >= 8).
A missing receipt is NOT_READ, never a pass.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ARMS = {"e4b": "qwen3_e4b_fused_attn4", "unsloth": "qwen3_unsloth_ckpt_unsloth"}


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def dmon_mean_pcie_gbps(path: str):
    """nvidia-smi dmon -s ut -o T: columns time, gpu, sm, mem, enc, dec, [jpg, ofa,] rxpci, txpci (MB/s). Header lines start with '#'."""
    if not os.path.exists(path):
        return None
    hdr, rows = None, []
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0].startswith("#"):
            if "rxpci" in parts:
                hdr = [x.lstrip("#") for x in parts]          # "#Time gpu sm mem ... rxpci txpci"
            continue
        if hdr is None:
            continue
        rec = dict(zip(hdr, parts))
        try:
            rx, tx = float(rec.get("rxpci", "nan")), float(rec.get("txpci", "nan"))
            sm = float(rec.get("sm", "nan"))
        except ValueError:
            continue
        if rx == rx and tx == tx:
            rows.append((rx, tx, sm))
    if not rows:
        return {"samples": 0}
    n = len(rows)
    return {"samples": n, "pcie_mean_gbps": sum(r[0] + r[1] for r in rows) / n / 1e3,
            "pcie_max_gbps": max(r[0] + r[1] for r in rows) / 1e3,
            "sm_mean_pct": sum(r[2] for r in rows if r[2] == r[2]) / max(1, sum(1 for r in rows if r[2] == r[2]))}


def reduce(d: str) -> dict:
    out = {"arms": {}}
    for fw, base in ARMS.items():
        cell = _load(os.path.join(d, base + ".json"))
        prof = _load(os.path.join(d, base + "_profile.json"))
        dm = dmon_mean_pcie_gbps(os.path.join(d, "logs", f"dmon_{base}.txt"))
        out["arms"][fw] = {"status": (cell or {}).get("status", "MISSING"), "s_per_step": (cell or {}).get("s_per_step"),
                           "profile": (prof or {}).get("summary"), "dmon": dm,
                           "top_cpu": [(r["name"], r["family"], r["self_cpu_ms"]) for r in (prof or {}).get("top_cpu", [])[:15]],
                           "top_device": [(r["name"], r["family"], r["self_device_ms"]) for r in (prof or {}).get("top_device", [])[:15]]}
    e = out["arms"]["e4b"]["profile"]
    u = out["arms"]["unsloth"]["profile"]
    dm = out["arms"]["e4b"]["dmon"]
    v = {}
    if e:
        b = e["device_busy_fraction"]
        fam = e.get("cpu_self_by_family_fraction", {})
        v["P1"] = {"device_busy_fraction": b, "verdict": "HOLDS" if b is not None and b <= 0.30 else "REFUTED" if b is not None and b >= 0.60 else "INCONCLUSIVE (0.30 < busy < 0.60)"}
        ra = fam.get("routing", 0) + fam.get("autograd", 0)
        alt = [k for k, thr in (("fused_kernel", 0.5), ("memcpy", 0.3), ("optimizer", 0.3)) if fam.get(k, 0) >= thr]
        v["P2"] = {"routing_plus_autograd": round(ra, 4), "by_family": fam, "verdict": "HOLDS (routing+autograd)" if ra >= 0.5 else (f"ALTERNATIVE: {alt}" if alt else "NONE OF THE REGISTERED FAMILIES DOMINATES")}
        ev = e["device_events_per_step"]
        v["P3"] = {"device_events_per_step": ev, "verdict": "HOLDS" if ev >= 20000 else "REFUTED" if ev < 5000 else "INCONCLUSIVE (5k..20k)"}
        if u:
            r = (u["device_busy_fraction"] or 0) / b if b else None
            v["P4"] = {"busy_e4b": b, "busy_unsloth": u["device_busy_fraction"], "ratio": r, "verdict": "HOLDS" if r is not None and r >= 2.0 else "ALTERNATIVE (shared structure)" if r is not None and r <= 1.3 else "INCONCLUSIVE (1.3..2.0)" if r is not None else "NOT_READ"}
        else:
            v["P4"] = {"verdict": "NOT_READ (no Unsloth profile)"}
    else:
        v["P1"] = v["P2"] = v["P3"] = v["P4"] = {"verdict": "NOT_READ (no e4b profile)"}
    if dm and dm.get("samples"):
        g = dm["pcie_mean_gbps"]
        v["P5"] = {"pcie_mean_gbps": round(g, 3), "pcie_max_gbps": round(dm["pcie_max_gbps"], 3), "sm_mean_pct": round(dm["sm_mean_pct"], 1), "verdict": "HOLDS" if g < 2.0 else "REFUTED" if g >= 8.0 else "INCONCLUSIVE (2..8 GB/s)"}
    else:
        v["P5"] = {"verdict": "NOT_READ (no dmon samples)"}
    out["verdicts"] = v
    return out


def render_md(o: dict) -> str:
    L = ["# P45 — where the training step's time goes (Qwen3 field recipe, RTX 5090)", ""]
    for fw, a in o["arms"].items():
        p = a["profile"] or {}
        L += [f"## {fw} — status {a['status']}, s/step {a['s_per_step']}", "",
              f"- profiled steps {p.get('profiled_steps')}, wall/step {p.get('wall_ms_per_step')} ms, device busy {p.get('device_busy_fraction')}, memcpy share of device {p.get('memcpy_fraction_of_device')}, device events/step {p.get('device_events_per_step')}, CPU ops/step {p.get('cpu_ops_per_step')}, CPU self/wall {p.get('cpu_self_fraction_of_wall')}",
              f"- CPU self by family: {p.get('cpu_self_by_family_fraction')}", f"- device by family: {p.get('device_by_family_fraction')}",
              f"- dmon: {a['dmon']}", "", "| top CPU op | family | self ms |", "|---|---|---|"]
        L += [f"| `{n[:70]}` | {f} | {ms} |" for n, f, ms in a["top_cpu"]]
        L += ["", "| top device kernel | family | self ms |", "|---|---|---|"] + [f"| `{n[:70]}` | {f} | {ms} |" for n, f, ms in a["top_device"]] + [""]
    L += ["## Verdicts", ""] + [f"- **{k}**: {json.dumps(x)}" for k, x in o["verdicts"].items()] + [""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    o = reduce(a.run_dir)
    md = render_md(o)
    print(md)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(o, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
