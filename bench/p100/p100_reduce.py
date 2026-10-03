#!/usr/bin/env python3
"""Lane P100's reducer (bench/p100/PREREG-p100.md; e4b#916): is the paged prefill's cost on the int4 expert store a
fixed cost per CHUNK, paid in a per-expert host loop, as the code trace says?

Inputs in --dir (the box's $W):
- ``ttft_<arm>.json`` -- ``bench/sc1/sc1_e4b_sched.py --ttft`` records, for arms ``c512_t512`` (512 tokens, chunk 512),
  ``c512_t4096``, ``c1024_t4096`` and ``c2048_t4096`` (4096 tokens at chunk 512 / 1024 / 2048);
- ``census_<arm>.json`` -- ``p100_box.py``'s dispatch census of one more request in the same process;
- ``census_prof.json``, ``cprofile_b1.txt``, ``logs/arm_prof.log`` -- ``step_decomp.py --cprofile-out`` at ``--batch 1``.

**VOID** if any of these holds (each reason listed):
- the run is a REHEARSAL;
- a TTFT record is missing, not ``ok``, or off the registered shape (model, revision, B=1, chunk, prompt length, SC1's
  prompt digest, all-VRAM, graph bucket 1 captured, fused q/k/v, SC1's route knobs, 48 int4 expert layers, the int4_b32
  store, ``device_grouping`` False -- the configuration the trace read);
- a census is missing;
- the prof arm did not run its levers (no ``INT4EXP enabled: 48 layers``), has no cProfile window, or its cProfile text
  names ``dequant_int4_ref`` with an ``ncalls`` that differs from the census's count for the same window.

**SCALING** (a1 = TTFT-512 at chunk 512; T8, T4, T2 = TTFT-4096 at chunk 512, 1024, 2048; rho = T8 / T2):
- CONFIRMED iff rho >= 3.0 AND T2 <= 3 * a1 AND T2 < T4 < T8;
- REFUTED iff rho < 2.0;
- INDETERMINATE otherwise.

**MECHANISM** CONFIRMED iff all of:
- c512_t512's census: 48 grouped (T > 1) calls, all of them the host-grouped int4 loop (none device-grouped), with
  >= 9,600 ``dequant_int4_ref`` calls in them, exactly two per distinct expert per call;
- every TTFT census: 48 x n_chunks loop calls (paid per chunk, per layer), none device-grouped;
- the prof arm's cProfile window: 48 loop calls (one 512-token chunk), >= 9,600 ``dequant_int4_ref`` calls, and none
  inside a singleton (T == 1 decode) call.
NOT_CONFIRMED otherwise.

**Verdict:** CONFIRMED iff SCALING and MECHANISM are both CONFIRMED; REFUTED iff SCALING is REFUTED or MECHANISM is
NOT_CONFIRMED; INDETERMINATE otherwise. The kernel count, the fit T(n) = alpha * n + beta over the three 4096-token
arms, the distinct experts per layer and the cProfile's own ``ncalls`` are reported beside it.

    python p100_reduce.py --dir RUN_DIR --out verdict.json
    python p100_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3-30B-A3B"
REV = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
PROMPT_SHA = {512: "a8e6ea1d7d140dbe726c94f0e6325eb11457a93f196419483de79507272f95e3",
              4096: "cd70a142d533eb88a3d2bec79bc4b11539f3a08b30c20ba5fbba248fb9dbafdd"}
ARMS = {"c512_t512": (512, 512), "c512_t4096": (512, 4096), "c1024_t4096": (1024, 4096), "c2048_t4096": (2048, 4096)}
ROUTE = {"E4B_INT4_GROUPED_SMALLM": "auto", "E4B_INT4_LEAN_GLUE": "auto", "E4B_NF4_GROUPED_SMALLM": "0",
         "E4B_MXFP4_GROUPED_SMALLM": "auto"}
LAYERS = 48
MIN_DEQUANT = 9600            # >= 100 distinct experts per layer, two GEMMs each, over 48 layers
SC1 = {"c512_t512": 2.0782, "c512_t4096": 16.5965}   # sc1b-5090-1 medians (adertha-receipts cabef5a)


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def check_ttft(name, r) -> list:
    chunk, plen = ARMS[name]
    if r is None:
        return [f"{name}: no record"]
    bad = []
    cfg, cen = r.get("config") or {}, r.get("census") or {}
    want = {"status": "ok", "model": MODEL, "revision": REV, "batch": 1, "ttft_prompt_tokens": plen,
            "prompts_sha256": PROMPT_SHA[plen], "route_env": ROUTE, "graph_status": {"1": "graph"}}
    for k, v in want.items():
        if r.get(k) != v:
            bad.append(f"{name}: {k}={str(r.get(k))[:70]} (want {str(v)[:70]})")
    for k, v in {"chunk_tokens": chunk, "max_seqs": 1, "placement": "all-vram", "graphs": True, "buckets": [1],
                 "fuse_qkv": True}.items():
        if cfg.get(k) != v:
            bad.append(f"{name}: config.{k}={cfg.get(k)} (want {v})")
    if cen.get("int4_expert_layers") != LAYERS or cen.get("int4_store_kinds") != ["int4_b32"]:
        bad.append(f"{name}: int4 store not engaged ({cen.get('int4_expert_layers')}, {cen.get('int4_store_kinds')})")
    if (cen.get("grouping") or {}).get("device_grouping") is not False:
        bad.append(f"{name}: device_grouping={(cen.get('grouping') or {}).get('device_grouping')} (the trace reads False)")
    if len(r.get("ttft_walls_s") or []) != 3:
        bad.append(f"{name}: {len(r.get('ttft_walls_s') or [])} timed walls (want 3)")
    return bad


def cprofile_ncalls(text: str):
    """``ncalls`` of ``int4_pack_ref``'s ``dequant_int4_ref`` row in a pstats listing (``a/b`` -> a), or None when
    absent (the listing holds the top 60 by cumulative time only)."""
    for line in (text or "").splitlines():
        if "int4_pack_ref.py:" in line and "(dequant_int4_ref)" in line:
            m = re.match(r"\s*(\d+)(?:/\d+)?\s", line)
            if m:
                return int(m.group(1))
    return None


def reduce(d: Path) -> dict:
    void, notes = [], []
    if (d / "REHEARSAL").exists():
        void.append("REHEARSAL: a knob is off its registered default")
    recs = {n: _load(d / f"ttft_{n}.json") for n in ARMS}
    cens = {n: _load(d / f"census_{n}.json") for n in ARMS}
    for n in ARMS:
        void += check_ttft(n, recs[n])
        if cens[n] is None or "summary" not in cens[n]:
            void.append(f"{n}: no census")
    prof = _load(d / "census_prof.json")
    log = (d / "logs" / "arm_prof.log").read_text(errors="replace") if (d / "logs" / "arm_prof.log").exists() else ""
    cpt = (d / "cprofile_b1.txt").read_text(errors="replace") if (d / "cprofile_b1.txt").exists() else None
    if prof is None or prof.get("rc") != 0:
        void.append(f"prof: no census or rc={None if prof is None else prof.get('rc')}")
    elif "summary_cprofile_window" not in prof:
        void.append("prof: no cProfile window in the census")
    if "INT4EXP enabled: 48 layers" not in log:
        void.append("prof: the int4 expert lever did not report 48 layers (P42's hook not engaged)")
    if cpt is None:
        void.append("prof: no cprofile_b1.txt")
    nc = cprofile_ncalls(cpt)
    win = (prof or {}).get("cprofile_window_dequant")
    if nc is not None and win is not None and nc != win:
        void.append(f"prof: cProfile ncalls {nc} != census window count {win} (the instruments disagree)")
    out = {"lane": "P100", "void_reasons": void, "notes": notes}
    med = {n: (r or {}).get("ttft_s_median") for n, r in recs.items()}
    out["ttft_s_median"] = med
    out["sc1_box_b"] = SC1
    out["cprofile_dequant_ncalls"] = nc
    out["cprofile_in_top60"] = nc is not None
    out["cprofile_window_dequant"] = win
    if void:
        out["verdict"] = "VOID"
        return out
    # ---- SCALING
    a1, t8, t4, t2 = med["c512_t512"], med["c512_t4096"], med["c1024_t4096"], med["c2048_t4096"]
    rho = t8 / t2
    pts = [(8, t8), (4, t4), (2, t2)]
    mx, my = statistics.mean(p[0] for p in pts), statistics.mean(p[1] for p in pts)
    alpha = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    beta = my - alpha * mx
    if rho >= 3.0 and t2 <= 3 * a1 and t2 < t4 < t8:
        scaling = "CONFIRMED"
    elif rho < 2.0:
        scaling = "REFUTED"
    else:
        scaling = "INDETERMINATE"
    out.update(scaling=scaling, rho=round(rho, 3), alpha_s_per_chunk=round(alpha, 4), beta_s=round(beta, 4),
               t2_over_two_a1=round(t2 / (2 * a1), 3), a1_over_sc1=round(a1 / SC1["c512_t512"], 3),
               t8_over_sc1=round(t8 / SC1["c512_t4096"], 3))
    # ---- MECHANISM
    why = []
    s1 = cens["c512_t512"]["summary"]
    if not (s1["grouped_calls"] == s1["loop_calls"] == LAYERS and s1["grouped_device_calls"] == 0):
        why.append(f"c512_t512: grouped {s1['grouped_calls']}, loop {s1['loop_calls']}, device {s1['grouped_device_calls']} "
                   f"(want {LAYERS}/{LAYERS}/0)")
    if s1["loop_dequant"] < MIN_DEQUANT or not s1["loop_dequant_is_two_per_distinct"]:
        why.append(f"c512_t512: {s1['loop_dequant']} dequant calls, two per distinct = {s1['loop_dequant_is_two_per_distinct']}")
    per_chunk = {}
    for n, (chunk, plen) in ARMS.items():
        s = cens[n]["summary"]
        k = -(-plen // chunk)
        per_chunk[n] = {"chunks": k, "loop_calls": s["loop_calls"], "loop_rows": s["loop_rows"],
                        "loop_dequant": s["loop_dequant"], "loop_distinct_mean": s["loop_distinct_mean"],
                        "grouped_device_calls": s["grouped_device_calls"],
                        "census_request_wall_s": cens[n].get("census_request_wall_s")}
        if s["loop_calls"] != LAYERS * k or s["grouped_device_calls"] != 0:
            why.append(f"{n}: {s['loop_calls']} loop calls (want {LAYERS} x {k}), {s['grouped_device_calls']} device-grouped")
    sw = prof["summary_cprofile_window"]
    if sw["loop_calls"] != LAYERS or (win or 0) < MIN_DEQUANT or sw["singleton_dequant"] != 0:
        why.append(f"prof window: {sw['loop_calls']} loop calls, {win} dequant, {sw['singleton_dequant']} in T == 1 calls")
    mechanism = "CONFIRMED" if not why else "NOT_CONFIRMED"
    out.update(mechanism=mechanism, mechanism_failures=why, per_arm=per_chunk,
               prof_window={k: sw[k] for k in ("loop_calls", "loop_dequant", "singleton_calls", "singleton_dequant",
                                               "loop_distinct_mean")},
               launches=cens["c512_t512"].get("launches"))
    if nc is None:
        notes.append("dequant_int4_ref is not in step_decomp's top-60 cProfile listing; the census window count stands alone")
    if scaling == "CONFIRMED" and mechanism == "CONFIRMED":
        out["verdict"] = "CONFIRMED"
    elif scaling == "REFUTED" or mechanism == "NOT_CONFIRMED":
        out["verdict"] = "REFUTED"
    else:
        out["verdict"] = "INDETERMINATE"
    return out


# ---------------------------------------------------------------------------------------------------- self-test
def _rec(name, med):
    chunk, plen = ARMS[name]
    return {"status": "ok", "model": MODEL, "revision": REV, "batch": 1, "ttft_prompt_tokens": plen,
            "prompts_sha256": PROMPT_SHA[plen], "route_env": dict(ROUTE), "graph_status": {"1": "graph"},
            "config": {"chunk_tokens": chunk, "max_seqs": 1, "placement": "all-vram", "graphs": True, "buckets": [1],
                       "fuse_qkv": True},
            "census": {"int4_expert_layers": 48, "int4_store_kinds": ["int4_b32"],
                       "grouping": {"device_grouping": False, "force_singleton_groups": False}},
            "ttft_walls_s": [med, med, med], "ttft_s_median": med}


def _summ(loop_calls, rows, dq_per_call=254, device=0):
    return {"grouped_calls": loop_calls + device, "loop_calls": loop_calls, "grouped_device_calls": device,
            "loop_rows": [rows], "loop_dequant": loop_calls * dq_per_call, "loop_dequant_is_two_per_distinct": True,
            "loop_distinct_mean": dq_per_call / 2, "singleton_calls": 0, "singleton_dequant": 0}


def _write(d: Path, meds, mutate=None):
    for n, m in meds.items():
        (d / f"ttft_{n}.json").write_text(json.dumps(_rec(n, m)))
        chunk, plen = ARMS[n]
        k = -(-plen // chunk)
        (d / f"census_{n}.json").write_text(json.dumps({"summary": _summ(48 * k, chunk * 8), "census_request_wall_s": m}))
    sw = _summ(48, 4096)
    sw.update(singleton_calls=1488, singleton_dequant=0)
    (d / "census_prof.json").write_text(json.dumps({"rc": 0, "cprofile_window_dequant": 48 * 254,
                                                    "summary_cprofile_window": sw}))
    (d / "logs").mkdir(exist_ok=True)
    (d / "logs" / "arm_prof.log").write_text("GEN HOOK: lanes attached\nINT4EXP enabled: 48 layers (model_type=qwen3_moe)\n")
    (d / "cprofile_b1.txt").write_text("   12192    0.400    0.000    1.900    0.000 int4_pack_ref.py:35(dequant_int4_ref)\n")
    if mutate:
        mutate(d)


def selftest() -> int:
    import tempfile
    good = {"c512_t512": 2.08, "c512_t4096": 16.6, "c1024_t4096": 8.4, "c2048_t4096": 4.3}
    cases = []

    def case(label, meds, want, mutate=None, check=None):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _write(d, meds, mutate)
            r = reduce(d)
            assert r["verdict"] == want, (label, r["verdict"], r.get("void_reasons"), r.get("mechanism_failures"))
            if check:
                assert check(r), (label, r)
            cases.append(label)

    def edit(name, fn):
        def m(d):
            p = d / name
            o = json.loads(p.read_text())
            fn(o)
            p.write_text(json.dumps(o))
        return m

    case("per-chunk cost: CONFIRMED", good, "CONFIRMED",
         check=lambda r: r["rho"] > 3.8 and 2.0 < r["alpha_s_per_chunk"] < 2.2 and abs(r["beta_s"]) < 0.3)
    case("flat in chunk count: REFUTED", dict(good, c1024_t4096=16.4, c2048_t4096=16.3), "REFUTED",
         check=lambda r: r["scaling"] == "REFUTED")
    case("rho 2.5: INDETERMINATE", dict(good, c1024_t4096=10.0, c2048_t4096=6.6), "INDETERMINATE")
    case("T2 above 3 x a1: INDETERMINATE", dict(good, c512_t512=1.0, c1024_t4096=6.0, c2048_t4096=4.3), "INDETERMINATE")
    case("missing arm: VOID", good, "VOID", mutate=lambda d: (d / "ttft_c1024_t4096.json").unlink())
    case("foreign prompts: VOID", good, "VOID",
         mutate=edit("ttft_c512_t4096.json", lambda o: o.update(prompts_sha256="0" * 64)))
    case("device grouping on at build: VOID", good, "VOID",
         mutate=edit("ttft_c2048_t4096.json", lambda o: o["census"]["grouping"].update(device_grouping=True)))
    case("rehearsal: VOID", good, "VOID", mutate=lambda d: (d / "REHEARSAL").write_text(""))
    case("hook not engaged: VOID", good, "VOID", mutate=lambda d: (d / "logs" / "arm_prof.log").write_text("GEN HOOK\n"))
    case("instruments disagree: VOID", good, "VOID", mutate=lambda d: (d / "cprofile_b1.txt").write_text(
        "   999    0.4    0.0    1.9    0.0 int4_pack_ref.py:35(dequant_int4_ref)\n"))
    case("paid once per request: REFUTED", good, "REFUTED",
         mutate=edit("census_c512_t4096.json", lambda o: o["summary"].update(loop_calls=48)),
         check=lambda r: r["mechanism"] == "NOT_CONFIRMED")
    case("device-grouped prefill: REFUTED", good, "REFUTED",
         mutate=edit("census_c512_t512.json", lambda o: o.update(summary=_summ(0, 4096, device=48))))
    case("not in the top 60: still CONFIRMED", good, "CONFIRMED",
         mutate=lambda d: (d / "cprofile_b1.txt").write_text("nothing here\n"),
         check=lambda r: r["cprofile_in_top60"] is False and r["notes"])
    assert cprofile_ncalls("  12288/12288  0.1  0.0  1.0  0.0 /x/int4_pack_ref.py:35(dequant_int4_ref)") == 12288
    assert cprofile_ncalls("  12288  0.1  0.0  1.0  0.0 /root/p100/p100_box.py:60(_counted_dequant_int4_ref)") is None
    print(f"self-test OK ({len(cases)} cases)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return selftest()
    r = reduce(Path(a.dir))
    Path(a.out).write_text(json.dumps(r, indent=1, default=str))
    print("P100_VERDICT " + json.dumps({k: r.get(k) for k in ("verdict", "scaling", "mechanism", "rho", "alpha_s_per_chunk",
                                                              "beta_s", "ttft_s_median", "cprofile_dequant_ncalls",
                                                              "cprofile_window_dequant", "void_reasons")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
