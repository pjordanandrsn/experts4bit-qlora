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


def s_counters(dump_dir: str, k: int, passes=("I1",)) -> dict:
    """Each pass's residency from the instrument lifetime's dump (warm state carried across its passes, as the server
    carries it), plus the dump's host-path self-check. The shakedown scored its one pass I1; a reading lifetime scores
    each of its passes (Amendment 3: the instrument lifetime runs the timed lifetime's whole pass sequence)."""
    import pgx_residency as res
    rows = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "route-*.jsonl"))):
        rows += res.read_dump(path)
    if not rows:
        return {"counters_ok": False, "why": f"no routing rows under {dump_dir}"}
    check = res.host_path_check(rows, k)
    seen = {r["pass"] for r in rows}
    missing = [p for p in passes if p not in seen]
    out = {"host_path_check": check}
    for p in passes:
        if p in seen:
            resid = res.sglang_residency(rows, k, p)
            out[f"residency_{p}"] = {x: resid[x] for x in resid if x != "layers"}
    ok = check["mismatched"] == 0 and check["checked"] > 0 and not missing
    why = None if ok else (f"no dump rows for pass(es) {missing}" if missing else
                           f"host-path check: {check['mismatched']} mismatched of {check['checked']}")
    out.update({"counters_ok": ok, "why": why})
    return out


def read_windows(path: str) -> dict:
    """The runner's pass windows: one line per pass, ``<pass> <start epoch> <end epoch>``."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                out[parts[0]] = (int(parts[1]), int(parts[2]))
    return out


def running_reqs(log: str, window: tuple) -> list:
    """SGLang's logged ``Decode batch, #running-req: N`` values (one line per 40 decode tokens at the default log level)
    whose timestamp falls strictly inside the window: the boundary seconds are excluded (Amendment 3, the maintainer's
    condition (ii)). Log times are the box's local time, read with the box's own clock rules."""
    import re
    import time
    pat = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Decode batch, #running-req: (\d+)")
    out = []
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.match(line)
            if m:
                t = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                if window[0] < t < window[1]:
                    out.append(int(m.group(2)))
    return out


def batch_check(timed_log: str, instr_log: str, timed_win: dict, instr_win: dict, passes) -> dict:
    """Per pass: the timed (graphs on) and instrument (graphs off) servers' logged batch sizes. Equal sequences =
    batch-matched; otherwise the cell's residency is APPROXIMATE (reported, never a VOID)."""
    out = {}
    for p in passes:
        if p not in timed_win or p not in instr_win:
            out[p] = {"state": "APPROXIMATE", "why": "no pass window"}
            continue
        a, b = running_reqs(timed_log, timed_win[p]), running_reqs(instr_log, instr_win[p])
        out[p] = {"state": "BATCH_MATCHED" if a == b and a else "APPROXIMATE", "timed": a, "instr": b}
    return out


def e_tiers(path: str, windows: dict, decode_tokens: dict, layers: int, top_k: int) -> dict:
    """E arms, per pass, from the hook's once-a-second cumulative tier counters (``dram_time.jsonl``): the last line at
    or before the window's start against the first at or after its end. Per pass:
    - ``dram_s_per_token``: the DRAM bus's host seconds (``_dram_contrib``) per decode token;
    - ``routed``, ``want``: the decode routed slots the hook counted against the pass's decode tokens x MoE layers x
      top_k (the maintainer's self-check: unequal means a route the hook does not see);
    - ``hit_rate``: the VRAM-tier hit rate, 1 - nonvram / routed, ONLY when routed == want, else None ("not measured");
    - ``nonvram_is_dram``: the non-VRAM rows equal the DRAM rows plus the host NVMe rows (reported).
    A pass with no bracketing lines, or no decode tokens, reads None throughout (not measured, never a VOID)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["t"])
    out = {}
    for p, (t0, t1) in windows.items():
        before = [r for r in rows if r["t"] <= t0]
        after = [r for r in rows if r["t"] >= t1]
        n = decode_tokens.get(p)
        if not before or not after or not n:
            out[p] = {"dram_s_per_token": None, "hit_rate": None, "routed": None, "want": None, "nonvram_is_dram": None}
            continue
        d = {k: after[0][k] - before[-1][k] for k in after[0] if k != "t"}
        want = n * layers * top_k
        ok = d["decode_routed"] == want
        out[p] = {"dram_s_per_token": d["decode_ns"] / 1e9 / n, "routed": d["decode_routed"], "want": want,
                  "hit_rate": (1 - d["decode_nonvram"] / d["decode_routed"]) if ok and d["decode_routed"] else None,
                  "nonvram_is_dram": d["decode_nonvram"] == d["decode_dram_rows"] + d["decode_nvme_cpu_rows"]}
    return out


def v_counters(log: str, probes: list, passes=("P1", "P2")) -> dict:
    """One eager probe before the first pass and after each pass (len(passes) + 1 offsets); each pass's counts are the
    difference of the cumulative counters at its two probes. Anything else fails closed."""
    import pgx_residency as res
    with open(log, encoding="utf-8", errors="replace") as f:
        text = f.read()
    out = {"pager_giveups": res.pager_giveups(text), "passes": {}}
    try:
        offs = [int(x) for x in (probes or [])]
        if len(offs) != len(passes) + 1:
            raise ValueError(f"{len(offs)} probe offset(s) for {len(passes)} pass(es)")
        for j, p in enumerate(passes):
            out["passes"][p] = res.pager_pass_counts(text[:offs[j]], text[:offs[j + 1]])
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


E_ARMS = ("E-hyb", "E-hyb-u", "E-allvram-g", "E-allvram-e")


def _decode_tokens(lt_dir: str, passes) -> dict:
    """Each pass's decode tokens from its own record: requests x (max_tokens - 1), the first token being prefill's."""
    out = {}
    for p in passes:
        try:
            with open(os.path.join(lt_dir, f"{p}.json"), encoding="utf-8") as f:
                r = json.load(f)
            out[p] = int(r["n"]) * (int(r["max_tokens"]) - 1)
        except (FileNotFoundError, ValueError, KeyError):
            out[p] = None
    return out


def lifetime(a) -> dict:
    vram, rss = peaks(a.samples)
    budget = a.budget if a.budget == "all" else int(a.budget)
    passes = tuple(a.passes)
    lt = {"arm": a.arm, "budget": budget, "draw": a.draw, "passes": list(passes),
          "k_registered": a.k if a.arm in ("S4", "S8", "V8") else None,
          "k_measured": a.k_measured if a.arm in ("S4", "S8", "V8") else None, "wall_s": a.wall_s,
          "peak_vram_mib": vram, "peak_rss_gib": rss, "counters_ok": False, "pager_giveups": None,
          "startup_failed": a.startup_failed, "counters": None}
    if a.startup_failed:
        return lt
    if a.arm in ("S4", "S8"):
        # the shakedown scored its instrument's one pass I1; a reading lifetime scores its own passes
        c = s_counters(a.route_dump, a.k, passes if a.windows else ("I1",))
        if a.windows and a.instr_windows and a.instr_log and a.timed_log:
            c["batch"] = batch_check(a.timed_log, a.instr_log, read_windows(a.windows), read_windows(a.instr_windows),
                                     passes)
    elif a.arm == "V8":
        c = v_counters(a.pager_log, a.probes, passes)
        lt["pager_giveups"] = c["pager_giveups"]
    else:
        c = e_counters(a.health, a.placement, a.step_trace, a.bpe)
        for key in ("profile", "vram_tier_bytes", "nvme_experts"):
            lt[key] = c[key]
        if a.arm in ("E-allvram-g", "E-allvram-e"):
            lt["profile"] = None          # all-vram reads no profile: the proof's nvme count and the trace still count
        if a.arm in ("E-hyb", "E-hyb-u") and a.windows:
            # Amendment 3: the tier counters (the hook), per pass, each behind its invariant. Missing or unreadable ->
            # None for that pass: the dram_host branch and the hit rate read "not measured" there, never a VOID.
            try:
                with open(a.health, encoding="utf-8") as f:
                    lev = json.load(f).get("levers") or {}
                tiers = e_tiers(a.dram_time, read_windows(a.windows), _decode_tokens(a.lt, passes),
                                int(lev["moe_layers"]), int(lev["top_k"]))
            except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
                tiers = {p: {"dram_s_per_token": None, "hit_rate": None, "why": f"{type(exc).__name__}: {exc}"}
                         for p in passes}
            lt["tiers"] = tiers
            lt["dram_tier_s_per_token"] = {p: tiers[p]["dram_s_per_token"] for p in passes if p in tiers}
            lt["vram_hit_rate"] = {p: tiers[p]["hit_rate"] for p in passes if p in tiers}
    lt["counters_ok"] = bool(c["counters_ok"])
    lt["counters"] = c
    return lt


def _args(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("lifetime")
    p.add_argument("--arm", required=True, choices=("S4", "S8", "V8") + E_ARMS)
    p.add_argument("--lt", required=True)
    p.add_argument("--budget", default="8")
    p.add_argument("--draw", type=int, default=1)
    p.add_argument("--passes", nargs="+", default=["P1", "P2"])
    p.add_argument("--windows", help="this lifetime's pass windows (Amendment 3)")
    p.add_argument("--instr-windows", help="S arms: the instrument lifetime's pass windows")
    p.add_argument("--instr-log", help="S arms: the instrument server's log")
    p.add_argument("--timed-log", help="S arms: this timed server's log")
    p.add_argument("--dram-time", help="E arms: the hook's dram_time.jsonl")
    p.add_argument("--wall-s", type=float)
    p.add_argument("--samples", default="")
    p.add_argument("--startup-failed", default=None)
    p.add_argument("--k", type=int)
    p.add_argument("--k-measured", type=int)
    p.add_argument("--route-dump")
    p.add_argument("--pager-log")
    p.add_argument("--probes", nargs="+")
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
