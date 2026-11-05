"""PGX box helper: one server lifetime's ``lifetime.json`` (the reducer's provenance record), assembled from what the box
kept for it. Stdlib only, so the same code runs at $0 on fixtures (``--self-test``) and on the box.

    python pgx_box.py lifetime --arm ARM --lt DIR --wall-s S --samples FILE [arm inputs]
    python pgx_box.py --self-test

Arm inputs, one counter source per engine family (PREREG-pgx.md, "Measures per pass"):
- S4, S8: ``--route-dump DIR`` (the untimed graphs-off instrument lifetime's ``route-*.jsonl``, pass ``I1``) and
  ``--k K``. The dump is replayed through SGLang's own rule (``pgx_residency.sglang_residency``) and self-checked against
  the engine's host-path loads (``host_path_check``). Counters fire when the dump has rows and no call mismatches.
- V8: ``--pager-log FILE --probes O0 O1 O2`` (the server log's byte offsets after each eager probe: before P1, between
  P1 and P2, after P2). Each pass's counts are the difference of the cumulative per-layer counters at its two probes
  (``pager_pass_counts``); counters fire when both differences read. Give-ups come from the same log.
- E-hyb: ``--health FILE --placement FILE --step-trace FILE`` (``/health`` after the passes, the placement-proof hook's
  record, and ``E4B_PAGED_STEP_TRACE``). The VRAM-tier bytes are the placed VRAM experts times the arena's bytes per
  expert (``--bpe``). Counters fire when the proof and a non-empty step trace exist.
Every arm: ``--samples FILE`` (one line per second, ``<epoch> <vram_mib> <rss_kib>``, the server's process group) gives
the peaks, and ``--startup-failed MSG`` records a server that never served.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

NF4_BPE = 2_654_208


def peaks(samples: str) -> tuple:
    vram, rss = [], []
    try:
        with open(samples, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3 and all(p.replace(".", "", 1).isdigit() for p in parts):
                    vram.append(float(parts[1]))
                    rss.append(float(parts[2]))
    except FileNotFoundError:
        pass
    return (max(vram) if vram else None, round(max(rss) / 2**20, 2) if rss else None)


def s_counters(dump_dir: str, k: int) -> dict:
    import pgx_residency as res
    rows = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "route-*.jsonl"))):
        rows += res.read_dump(path)
    if not rows:
        return {"counters_ok": False, "why": f"no routing rows under {dump_dir}"}
    check = res.host_path_check(rows, k)
    resid = res.sglang_residency(rows, k, "I1")
    ok = check["mismatched"] == 0 and check["checked"] > 0
    return {"counters_ok": ok, "host_path_check": check, "residency_I1": {x: resid[x] for x in resid if x != "layers"},
            "why": None if ok else f"host-path check: {check['mismatched']} mismatched of {check['checked']}"}


def v_counters(log: str, probes: list) -> dict:
    import pgx_residency as res
    with open(log, encoding="utf-8", errors="replace") as f:
        text = f.read()
    out = {"pager_giveups": res.pager_giveups(text), "passes": {}}
    try:
        o0, o1, o2 = (int(x) for x in probes)
        out["passes"]["P1"] = res.pager_pass_counts(text[:o0], text[:o1])
        out["passes"]["P2"] = res.pager_pass_counts(text[:o1], text[:o2])
        for p in out["passes"].values():
            p.pop("layers", None)
        out["counters_ok"], out["why"] = True, None
    except (ValueError, TypeError) as exc:
        out["counters_ok"], out["why"] = False, f"pager counters: {exc}"
    return out


def e_counters(health: str, placement: str, step_trace: str, bpe: int) -> dict:
    out = {"counters_ok": False, "why": None, "profile": None, "vram_tier_bytes": None, "nvme_experts": None,
           "placed_experts": None}
    try:
        with open(health, encoding="utf-8") as f:
            prof = ((json.load(f).get("levers") or {}).get("placement_profile")) or {}
        out["profile"] = {"kind": prof.get("kind"), "sha256": prof.get("sha256")}
    except (FileNotFoundError, ValueError) as exc:
        out["why"] = f"/health unread: {exc}"
    try:
        with open(placement, encoding="utf-8") as f:
            pl = json.load(f)
        out["placed_experts"] = pl["experts"]
        out["nvme_experts"] = int(pl["experts"]["nvme"])
        out["vram_tier_bytes"] = int(pl["experts"]["vram"]) * bpe
    except (FileNotFoundError, ValueError, KeyError) as exc:
        out["why"] = out["why"] or f"placement proof unread: {exc}"
    trace_ok = os.path.exists(step_trace) and os.path.getsize(step_trace) > 0
    if not trace_ok:
        out["why"] = out["why"] or f"no step trace at {step_trace}"
    out["counters_ok"] = out["why"] is None and trace_ok
    return out


def lifetime(a) -> dict:
    vram, rss = peaks(a.samples)
    lt = {"arm": a.arm, "budget": 8, "draw": 1, "k_registered": a.k if a.arm in ("S4", "S8", "V8") else None,
          "k_measured": a.k_measured if a.arm in ("S4", "S8", "V8") else None, "wall_s": a.wall_s,
          "peak_vram_mib": vram, "peak_rss_gib": rss, "counters_ok": False, "pager_giveups": None,
          "startup_failed": a.startup_failed, "counters": None}
    if a.startup_failed:
        return lt
    if a.arm in ("S4", "S8"):
        c = s_counters(a.route_dump, a.k)
    elif a.arm == "V8":
        c = v_counters(a.pager_log, a.probes)
        lt["pager_giveups"] = c["pager_giveups"]
    else:
        c = e_counters(a.health, a.placement, a.step_trace, a.bpe)
        for key in ("profile", "vram_tier_bytes", "nvme_experts"):
            lt[key] = c[key]
    lt["counters_ok"] = bool(c["counters_ok"])
    lt["counters"] = c
    return lt


def _args(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("lifetime")
    p.add_argument("--arm", required=True, choices=("S4", "S8", "V8", "E-hyb"))
    p.add_argument("--lt", required=True)
    p.add_argument("--wall-s", type=float)
    p.add_argument("--samples", default="")
    p.add_argument("--startup-failed", default=None)
    p.add_argument("--k", type=int)
    p.add_argument("--k-measured", type=int)
    p.add_argument("--route-dump")
    p.add_argument("--pager-log")
    p.add_argument("--probes", nargs=3)
    p.add_argument("--health")
    p.add_argument("--placement")
    p.add_argument("--step-trace")
    p.add_argument("--bpe", type=int, default=NF4_BPE)
    ap.add_argument("--self-test", action="store_true")
    return ap, ap.parse_args(argv)


def _pager_line(layer, vh, vt, rh, rt):
    return (f"vllm-expert-pager {layer}: vram hit {100 * vh / max(vt, 1):.1f}% ({vh}/{vt}), "
            f"ram hit {100 * rh / max(rt, 1):.1f}% ({rh}/{rt})\n")


def self_test() -> int:
    bad = 0
    d = tempfile.mkdtemp()
    try:
        samples = os.path.join(d, "s.txt")
        with open(samples, "w") as f:
            f.write("1 9000 41943040\n2 9100 42991616\nnot a sample\n")
        assert peaks(samples) == (9100.0, 41.0), peaks(samples)
        # V8: three probes over cumulative counters; a counter going down fails closed
        log = os.path.join(d, "v.log")
        parts = [_pager_line("model.layers.0.mlp.experts", 10, 20, 20, 20), _pager_line("model.layers.0.mlp.experts", 50, 80, 80, 80),
                 _pager_line("model.layers.0.mlp.experts", 90, 130, 130, 130)]
        with open(log, "w") as f:
            f.write("".join(parts))
        offs = [len(parts[0]), len(parts[0]) + len(parts[1]), len("".join(parts))]
        v = v_counters(log, offs)
        bad += not (v["counters_ok"] and v["passes"]["P1"]["vram_hits"] == 40 and v["passes"]["P2"]["vram_misses"] == 10)
        with open(log, "w") as f:
            f.write(parts[1] + parts[0] + parts[2])
        bad += v_counters(log, [len(parts[1]), len(parts[1]) + len(parts[0]), len("".join(parts))])["counters_ok"] is not False
        # E-hyb: the proof, the profile and a non-empty trace; then a missing trace
        health, pl, tr = (os.path.join(d, n) for n in ("h.json", "p.json", "t.jsonl"))
        with open(health, "w") as f:
            json.dump({"levers": {"placement_profile": {"kind": "measured", "sha256": "ab" * 32, "path": "x"}}}, f)
        with open(pl, "w") as f:
            json.dump({"experts": {"vram": 2900, "dram": 3244, "nvme": 0}}, f)
        with open(tr, "w") as f:
            f.write("{}\n")
        e = e_counters(health, pl, tr, NF4_BPE)
        bad += not (e["counters_ok"] and e["vram_tier_bytes"] == 2900 * NF4_BPE and e["nvme_experts"] == 0)
        bad += e_counters(health, pl, os.path.join(d, "none"), NF4_BPE)["counters_ok"] is not False
        # S: an empty dump directory fails closed
        bad += s_counters(d, 35)["counters_ok"] is not False
        # a server that never started records no counters
        _, a = _args(["lifetime", "--arm", "V8", "--lt", d, "--startup-failed", "rc 45", "--samples", samples, "--k", "35"])
        lt = lifetime(a)
        bad += not (lt["startup_failed"] == "rc 45" and lt["counters_ok"] is False and lt["peak_vram_mib"] == 9100.0)
    except AssertionError as exc:
        print(f"  BAD {exc}")
        bad += 1
    finally:
        shutil.rmtree(d)
    print(f"PGX_BOX_SELFTEST {'OK' if not bad else f'FAIL {bad}'}")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap, a = _args(argv)
    if a.self_test:
        return self_test()
    if a.cmd != "lifetime":
        ap.error("lifetime, or --self-test")
    lt = lifetime(a)
    os.makedirs(a.lt, exist_ok=True)
    with open(os.path.join(a.lt, "lifetime.json"), "w", encoding="utf-8") as f:
        json.dump(lt, f, indent=1, default=str)
    print(f"PGX_LIFETIME {a.arm} counters_ok={lt['counters_ok']} wall={lt['wall_s']} peak_vram={lt['peak_vram_mib']} "
          f"peak_rss={lt['peak_rss_gib']} {('why=' + str((lt['counters'] or {}).get('why'))) if not lt['counters_ok'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
