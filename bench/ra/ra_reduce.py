#!/usr/bin/env python3
"""RA's fixed ABBA rule. Stdlib; no imports from either measured installation.

Input contract: run.json + four arm_TAG.json RA envelopes. Envelopes embed
native training/profile/decode/capacity/quality records. The executor must
prove its envelope matches raw files before a rental; it is not in this PR.
This module validates the envelope and reduces raw samples, not typed metrics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

TAGS = ("old_a", "new_a", "new_b", "old_b")
PAIRS = (("new_a", "old_a"), ("new_b", "old_b"))
SELF_PAIRS = (("old_b", "old_a"), ("new_b", "new_a"))
BOUNDS = {"train_wall_s": (0.05, False), "train_device_ms": (0.05, False),
          "train_peak_gb": (0.03, False), "decode_W1_tok_s": (0.07, True),
          "decode_W16_tok_s": (0.07, True), "capacity_goodput_rps": (0.10, True)}
IDENTITY = ("host_boot_id", "gpu_uuid", "gpu_name", "gpu_sm_count", "driver", "cuda_runtime", "python",
            "wheel_lock_sha256", "torch", "triton", "transformers", "bitsandbytes", "peft", "datasets",
            "allocator", "physical_cores", "threads", "checkpoint_sha256", "tokenizer_sha256",
            "model_revision", "dataset_sha256", "training_tokens_sha256", "prompts_sha256",
            "capacity_plan_sha256", "quality_windows_sha256", "calibration_inputs_sha256",
            "harness_sha256", "prereg_sha256", "source_pins_sha256")
HEX40 = set("0123456789abcdef")


class Invalid(ValueError):
    """Evidence cannot be reduced."""


def require(ok, why):
    if not ok:
        raise Invalid(why)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def metric(values, bound, higher, *, zero=False):
    """Symmetric fractional spread; cost ratios >1 are worse. Boundary equality passes."""
    require(all(finite(x) and (x >= 0 if zero else x > 0) for x in values.values()), "invalid metric sample")
    spreads, noisy = [], False
    for a, b in SELF_PAIRS:
        hi, lo = max(values[a], values[b]), min(values[a], values[b])
        spreads.append((hi / lo - 1) if lo else (0.0 if not hi else None))
        noisy |= hi > lo * (1 + bound)
    ratios, states = [], []
    for new, old in PAIRS:
        n, o = values[new], values[old]
        if not n or not o:
            require(zero and higher, "zero only supported for goodput")
            state = "equal_zero" if n == o else ("new_zero" if n == 0 else "old_zero")
            ratios.append(None)
            states.append(state)
        else:
            ratios.append(o / n if higher else n / o)
            states.append("finite")
    worse = [s == "new_zero" or (r is not None and r > 1 + bound) for r, s in zip(ratios, states)]
    better = [s == "old_zero" or (r is not None and r < 1 / (1 + bound)) for r, s in zip(ratios, states)]
    verdict = ("NOISY" if noisy else
               "REGRESSION" if all(worse) else "IMPROVED" if all(better) else "WITHIN_NOISE")
    return {"verdict": verdict, "values": values, "bound": bound, "higher_is_better": higher,
            "self_spread": spreads, "cost_ratios": ratios, "ratio_states": states}


def quality_pair(a, b):
    """a minus b, in nats; argmax equality is calculated from IDs, not claimed."""
    delta = statistics.fmean(x["nll"] - y["nll"] for x, y in zip(a, b))
    agree = statistics.fmean(statistics.fmean(int(u == v) for u, v in zip(x["argmax_ids"], y["argmax_ids"]))
                             for x, y in zip(a, b))
    return {"bias_nats": delta, "argmax_agree": agree, "sane": abs(delta) <= 0.02 and agree >= 0.95}


def quality_metric(rows):
    selfs = [quality_pair(rows[a], rows[b]) for a, b in SELF_PAIRS]
    pairs = [quality_pair(rows[a], rows[b]) for a, b in PAIRS]
    noisy = any(abs(s["bias_nats"]) > 0.005 or s["argmax_agree"] < 0.999 for s in selfs)
    failures = sum(not s["sane"] for s in pairs)
    improved = all(s["sane"] and s["bias_nats"] < -0.005 for s in pairs)
    verdict = "NOISY" if noisy else "REGRESSION" if failures == 2 else "IMPROVED" if improved else "WITHIN_NOISE"
    return {"verdict": verdict, "pairs": pairs, "same_version": selfs,
            "unsettled": not noisy and failures == 1, "bias_bar_nats": 0.02, "agreement_bar": 0.95}


def released(r):
    require(set(r) == {"e4b", "gnf4"}, "release pair must name both packages")
    for pkg, pin in r.items():
        require(bool(pin["version"]), f"{pkg}: version missing")
        require(len(pin["commit"]) == 40 and set(pin["commit"]) <= HEX40, f"{pkg}: invalid commit")
        require(len(pin["wheel_sha256"]) == 64 and set(pin["wheel_sha256"]) <= HEX40, f"{pkg}: invalid wheel digest")


def features(rec):
    """Executor records resolved state and observed calls per enabled path."""
    fs = rec["features"]
    require(isinstance(fs, dict) and bool(fs), "resolved feature evidence missing")
    for name, f in fs.items():
        require(f["mode"] in ("on", "off", "inapplicable"), f"{name}: unresolved mode")
        require(finite(f["calls"]) and f["calls"] >= 0, f"{name}: invalid calls")
        require(f["fallback_calls"] == 0, f"{name}: fallback")
        if f["mode"] == "on":
            require(f["patched"] > 0 and f["calls"] > 0, f"{name}: vacuous enable")
        else:
            require(f["calls"] == 0, f"{name}: disabled path called")


def train_metrics(a, steps, expected):
    t, p = a["training"], a["training_profile"]
    for rec, prof in ((t, False), (p, True)):
        require(rec["status"] == "ok", "training not ok")
        require(rec["steps"] == steps == 20 and len(rec["step_ms"]) == steps, "wrong training steps")
        require(all(finite(v) and v > 0 for v in rec["step_ms"]), "invalid training times")
        require(len(rec["losses"]) == steps and all(finite(v) for v in rec["losses"]), "invalid training losses")
        require(rec["trainable_params"] == expected and rec["trainable_mismatch"] is None, "trainable mismatch")
        require(rec["matched_init"]["complete"] is True and rec["matched_init_sha"] == a["matched_init_sha"], "init mismatch")
        require(len(rec["kernel_calls_per_step"]) == steps and min(rec["kernel_calls_per_step"]) > 0, "no fused training")
        require(rec["profile_steps"] == (10 if prof else 0), "profile mixed into wall arm")
        require(rec["profile_warm"] == 10 if prof else rec["profile"] is None, "wrong profile window")
        features(rec)
    pr = p["profile"]
    require(pr["profiled_steps"] == 10 and finite(pr["device_ms"]) and pr["device_ms"] > 0, "invalid device profile")
    require(finite(t["peak_vram_gb"]) and t["peak_vram_gb"] > 0, "invalid peak memory")
    function_fault = any(not rec["C1_bit_exact"] or not rec["C1_control_detects_flipped_byte"] for rec in (t, p))
    return {"train_wall_s": statistics.median(t["step_ms"][10:]) / 1000,
            "train_device_ms": pr["device_ms"] / pr["profiled_steps"],
            "train_peak_gb": t["peak_vram_gb"]}, function_fault


def decode_metrics(a, short, long_, reps):
    r = a["decode"]
    require(r["status"] == "ok" and (r["short"], r["long"], r["reps"]) == (short, long_, reps), "decode fixture mismatch")
    require(r["prompts_sha256"] == a["identity"]["prompts_sha256"], "decode prompt mismatch")
    require(r["resolved_buckets"] and len(set(r["resolved_buckets"])) == len(r["resolved_buckets"]), "bad buckets")
    if r["graphs"]:
        require(all(r["graph_status"].get(str(b)) == "graph" for b in r["resolved_buckets"]), "graph not captured")
        require(not any(g["eager_steps"] for g in r["graph_stats"].values()), "eager fallback")
        require(sum(g["replays"] for g in r["graph_stats"].values()) > 0, "graphs unused")
    features(r)
    values, fn = {}, False
    for w, batch in (("W1", 1), ("W16", 16)):
        x = r["workloads"][w]
        require(x["batch"] == batch, f"{w}: batch mismatch")
        for n in (short, long_):
            key = str(n)
            times = x["walls"][key]
            require(len(times) == reps and all(finite(v) and v > 0 for v in times), f"{w}: invalid walls")
            toks, ds = x["tokens"][key], x["rep_digests"][key]
            require(len(toks) == batch and all(len(row) == n and all(type(v) is int and v >= 0 for v in row) for row in toks),
                    f"{w}: wrong request lengths")
            require(len(ds) == reps and ds[-1] == digest(toks), f"{w}: digest mismatch")
            fn |= len(set(ds)) != 1
        d = min(x["walls"][str(long_)]) - min(x["walls"][str(short)])
        require(d > 0, f"{w}: nonpositive slope")
        values[f"decode_{w}_tok_s"] = batch * (long_ - short) / d
    return values, fn


def capacity_metrics(a):
    c = a["capacity"]
    reqs, plan = c["requests"], c["plan"]
    require(c["status"] == "ok" and c["offered_rate"] == 12 and len(reqs) == len(plan) == 120, "capacity fixture mismatch")
    require(digest(plan) == a["identity"]["capacity_plan_sha256"], "capacity plan mismatch")
    require(c["slo"] == {"ttft_s": 1.0, "tpot_s": 0.1}, "wrong SLO")
    require(c["prompt_tokens"] == 512 and c["invalid"] == 0, "invalid capacity request")
    good = 0
    for r, item in zip(reqs, plan):
        require(r["valid"] is True and r["prompt_tokens"] == 512 and r["completion_tokens"] == item[2], "invalid request")
        require(64 <= item[2] <= 256 and finite(item[0]) and item[0] >= 0 and type(item[1]) is int, "wrong plan")
        require(r["finish_reason"] == "length" and finite(r["ttft_s"]) and finite(r["tpot_s"]) and
                r["ttft_s"] >= 0 and r["tpot_s"] >= 0, "invalid stream")
        good += r["ttft_s"] <= 1 and r["tpot_s"] <= 0.1
    require(c["burst"]["valid"] == 64 and c["burst"]["widest_bucket_replays"] > 0, "widest bucket not exercised")
    h = c["health_end"]
    require(h["prefill_replays"] == h["admitted"] == h["kv_bulk_flush"] == 188 and
            h["prefill_eager_chunks"] == h["decode_eager_steps"] == 0, "capacity counters disagree")
    features(c)
    return {"capacity_goodput_rps": 12 * good / len(reqs)}


def quality_rows(a, group, cont, layers):
    q = a[f"quality_{group}"]
    require(q["status"] == "ok" and (q["group"], q["prompt"], q["cont"], q["layers"]) ==
            (group, 512, cont, layers), "quality fixture mismatch")
    require(q["windows_sha256"] == a["identity"]["quality_windows_sha256"], "quality input mismatch")
    require(q["rehearsal"]["stand_in_attention"] is False, "stand-in attention")
    rows = q["per_window"]["wikitext"]["R"]
    require(len(rows) == 12 and [r["window"] for r in rows] == list(range(12)), "wrong/duplicate quality windows")
    for r in rows:
        require(finite(r["nll"]) and r["nll"] >= 0 and len(r["argmax_ids"]) == cont and
                all(type(x) is int and x >= 0 for x in r["argmax_ids"]), "invalid quality score")
    passes = q["engagement"]["wikitext"]["R"]
    require(len(passes) == 12 // group, "missing quality passes")
    b = 1 if group == 1 else 16
    for e in passes:
        require(e["decode_calls"] == (cont - 1) * layers and e["grouping_flags_in_pass"]["device_grouping"] is True,
                "quality attention not engaged")
        gs = e["graph_stats"][str(b)]
        require(e["graph_status"][str(b)] == "eager: capture=False" and gs["replays"] == 0 and
                gs["eager_steps"] == cont - 1 and gs["pad_rows"] == (b - group) * (cont - 1), "quality bucket mismatch")
        features(e)
    return rows


def reduce(run, arms):
    out = {"schema": 1, "lane": "RA", "kind": run.get("kind"), "verdict": "VOID", "reasons": [], "metrics": {}}
    try:
        require(run["schema"] == 1 and run["order"] == list(TAGS), "wrong ABBA order/schema")
        require(run["kind"] in ("PROOF", "READING"), "unknown run kind")
        require(set(arms) == set(TAGS), "missing/extra arm")
        proof = run["kind"] == "PROOF"
        require((run["short"], run["long"], run["reps"], run["quality_cont"], run["training_steps"]) ==
                ((8, 24, 1, 32, 20) if proof else (32, 160, 3, 128, 20)), "wrong battery sizes")
        require(run["layers"] == (32 if proof else 48) and run["trainable_params"] > 0, "wrong model topology")
        require(set(run["identity"]) == set(IDENTITY) and all(v is not None and v != "" for v in run["identity"].values()),
                "incomplete common identity")
        require("RTX 5090" in run["identity"]["gpu_name"], "speed box must be RTX 5090")
        require(proof or run["old"] != run["new"], "reading has no successor release")
        require(not proof or run["old"] == run["new"], "proof must compare baseline with itself")
        released(run["old"])
        released(run["new"])
        samples, rows, function_faults = {}, {1: {}, 12: {}}, []
        for tag in TAGS:
            a = arms[tag]
            require(a["tag"] == tag and a["status"] == "ok", f"{tag}: arm not ok")
            require(a["identity"] == run["identity"], f"{tag}: non-release identity mismatch")
            require(a["release"] == run["old" if tag.startswith("old") else "new"], f"{tag}: release mismatch")
            require(a["feature_env"] == {}, f"{tag}: inherited feature environment")
            require(a["matched_init_sha"] == run["matched_init_sha"], f"{tag}: init changed")
            require(all(a["imports"][pkg]["venv"] == a["venv"] and a["imports"][pkg]["wheel_verified"] is True and
                        bool(a["imports"][pkg]["tree_sha256"]) for pkg in ("e4b", "gnf4")), f"{tag}: import contamination")
            require(bool(a["arena_sha256"]) and bool(a["resolved_defaults"]), f"{tag}: output evidence missing")
            tm, tf = train_metrics(a, run["training_steps"], run["trainable_params"])
            dm, df = decode_metrics(a, run["short"], run["long"], run["reps"])
            samples[tag] = {**tm, **dm, **capacity_metrics(a)}
            for group in (1, 12):
                rows[group][tag] = quality_rows(a, group, run["quality_cont"], run["layers"])
            if tf or df:
                function_faults.append(f"{tag}: frozen bytes/control or timed decode determinism")
        for a, b in SELF_PAIRS:
            require(arms[a]["resolved_defaults"] == arms[b]["resolved_defaults"], "same-version defaults changed")
            require(arms[a]["arena_sha256"] == arms[b]["arena_sha256"], "same-version arena changed")
            for w in ("W1", "W16"):
                if arms[a]["decode"]["workloads"][w]["tokens"] != arms[b]["decode"]["workloads"][w]["tokens"]:
                    function_faults.append(f"{a}/{b}: {w} tokens differ")
        for name, (bound, higher) in BOUNDS.items():
            out["metrics"][name] = metric({tag: samples[tag][name] for tag in TAGS}, bound, higher,
                                          zero=name == "capacity_goodput_rps")
        for group in (1, 12):
            out["metrics"][f"quality_group_{group}"] = quality_metric(rows[group])
        statuses = [m["verdict"] for m in out["metrics"].values()]
        out["verdict"] = ("FUNCTION_FAIL" if function_faults else "REGRESSION" if "REGRESSION" in statuses else
                          "QUALITY_UNSETTLED" if any(m.get("unsettled") for m in out["metrics"].values()) else
                          "NOISY" if "NOISY" in statuses else "CLEAR")
        out["reasons"] = function_faults
        out["release_clearance"] = not proof and out["verdict"] == "CLEAR"
        out["regression_blocks_next_release"] = not proof and out["verdict"] == "REGRESSION"
    except (Invalid, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, OverflowError) as exc:
        out.update(verdict="VOID", reasons=[str(exc)], metrics={}, release_clearance=False,
                   regression_blocks_next_release=False)
    return out


def verified_files(root):
    """Every receipt byte is checked before reduction; no absolute/escaping links."""
    records = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(None, 1)
        name = name.lstrip(" *")
        p = root / name
        require(not Path(name).is_absolute() and p.resolve().is_relative_to(root.resolve()), "checksum path escapes receipt")
        require(name not in records and p.is_file(), "missing/duplicate checksum file")
        require(hashlib.sha256(p.read_bytes()).hexdigest() == expected, f"checksum mismatch: {name}")
        records[name] = expected
    need = {"run.json", *(f"arm_{t}.json" for t in TAGS)}
    require(need <= records.keys(), "mandatory record not checksummed")
    all_files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(all_files - {"SHA256SUMS", "verdict.json"} <= records.keys(), "unchecksummed raw file")
    return records


def fixture():
    """Synthetic test data only, never a receipt or release measurement."""
    run = {"schema": 1, "kind": "READING", "order": list(TAGS), "short": 32, "long": 160, "reps": 3,
           "quality_cont": 128, "training_steps": 20, "trainable_params": 100, "matched_init_sha": "synthetic-init",
           "layers": 48, "identity": {k: "synthetic" for k in IDENTITY}}
    run["identity"]["gpu_name"] = "NVIDIA GeForce RTX 5090"
    run["old"] = {p: {"version": "synthetic-old", "commit": "a" * 40, "wheel_sha256": "a" * 64} for p in ("e4b", "gnf4")}
    run["new"] = {p: {"version": "synthetic-new", "commit": "b" * 40, "wheel_sha256": "b" * 64} for p in ("e4b", "gnf4")}
    plan = [[i / 12, i, 64] for i in range(120)]
    run["identity"]["capacity_plan_sha256"] = digest(plan)
    fs = {"synthetic_kernel": {"mode": "on", "patched": 1, "calls": 1, "fallback_calls": 0}}
    arms = {}
    for tag in TAGS:
        t = {"status": "ok", "steps": 20, "step_ms": [1000.] * 20, "losses": [2.] * 20,
             "trainable_params": 100, "trainable_mismatch": None, "matched_init": {"complete": True},
             "matched_init_sha": "synthetic-init", "kernel_calls_per_step": [1] * 20,
             "C1_bit_exact": True, "C1_control_detects_flipped_byte": True, "profile_steps": 0,
             "profile_warm": 10, "profile": None, "peak_vram_gb": 20., "features": copy.deepcopy(fs)}
        prof = copy.deepcopy(t)
        prof.update(profile_steps=10, profile={"profiled_steps": 10, "device_ms": 1000.})
        dec = {"status": "ok", "short": 32, "long": 160, "reps": 3,
               "prompts_sha256": "synthetic", "graphs": True, "resolved_buckets": [1, 2, 4, 8, 16],
               "graph_status": {str(b): "graph" for b in (1, 2, 4, 8, 16)},
               "graph_stats": {"16": {"replays": 5, "eager_steps": 0}}, "features": copy.deepcopy(fs), "workloads": {}}
        for w, batch in (("W1", 1), ("W16", 16)):
            toks = {str(n): [[1] * n for _ in range(batch)] for n in (32, 160)}
            dec["workloads"][w] = {"batch": batch, "walls": {"32": [1.] * 3, "160": [2.] * 3},
                                   "tokens": toks, "rep_digests": {n: [digest(ts)] * 3 for n, ts in toks.items()}}
        cap = {"status": "ok", "offered_rate": 12, "plan": plan, "prompt_tokens": 512, "invalid": 0,
               "slo": {"ttft_s": 1., "tpot_s": 0.1}, "burst": {"valid": 64, "widest_bucket_replays": 1},
               "health_end": {"prefill_replays": 188, "admitted": 188, "kv_bulk_flush": 188,
                              "prefill_eager_chunks": 0, "decode_eager_steps": 0}, "features": copy.deepcopy(fs),
               "requests": [{"valid": True, "prompt_tokens": 512, "completion_tokens": 64, "finish_reason": "length",
                             "ttft_s": 0.5, "tpot_s": 0.05} for _ in plan]}
        a = {"tag": tag, "status": "ok", "identity": copy.deepcopy(run["identity"]),
             "release": copy.deepcopy(run["old" if tag.startswith("old") else "new"]), "feature_env": {},
             "matched_init_sha": "synthetic-init", "venv": tag[:3],
             "imports": {p: {"venv": tag[:3], "wheel_verified": True, "tree_sha256": "synthetic"} for p in ("e4b", "gnf4")},
             "arena_sha256": "old-arena" if tag.startswith("old") else "new-arena", "resolved_defaults": {"mode": tag[:3]},
             "training": t, "training_profile": prof, "decode": dec, "capacity": cap}
        for group in (1, 12):
            b = 1 if group == 1 else 16
            e = {"decode_calls": 127 * 48, "grouping_flags_in_pass": {"device_grouping": True},
                 "graph_status": {str(b): "eager: capture=False"},
                 "graph_stats": {str(b): {"replays": 0, "eager_steps": 127, "pad_rows": (b - group) * 127}},
                 "features": copy.deepcopy(fs)}
            a[f"quality_{group}"] = {"status": "ok", "group": group, "prompt": 512, "cont": 128, "layers": 48,
                                     "windows_sha256": "synthetic", "rehearsal": {"stand_in_attention": False},
                                     "per_window": {"wikitext": {"R": [{"window": i, "nll": 2., "argmax_ids": [1] * 128}
                                                                            for i in range(12)]}},
                                     "engagement": {"wikitext": {"R": [copy.deepcopy(e) for _ in range(12 // group)]}}}
        arms[tag] = a
    return run, arms


def self_test():
    tests = []

    def check(name, ok):
        tests.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} {name}")

    def mutate(name, change, want="VOID"):
        r, a = fixture()
        change(r, a)
        check(name, reduce(r, a)["verdict"] == want)

    r, a = fixture()
    check("complete envelope CLEAR; different release-produced arenas allowed", reduce(r, a)["verdict"] == "CLEAR")
    for name, (bound, higher) in BOUNDS.items():
        same = dict.fromkeys(TAGS, 100.)
        check(f"{name} equal", metric(same, bound, higher)["verdict"] == "WITHIN_NOISE")
        worse = 100. / (1 + bound + 0.01) if higher else 100. * (1 + bound + 0.01)
        vals = {**same, "new_a": worse, "new_b": worse}
        check(f"{name} both pairs worse", metric(vals, bound, higher)["verdict"] == "REGRESSION")
        better = 100. * (1 + bound + 0.01) if higher else 100. / (1 + bound + 0.01)
        check(f"{name} both pairs better", metric({**same, "new_a": better, "new_b": better}, bound, higher)["verdict"] == "IMPROVED")
        check(f"{name} self noise precedes ratio", metric({**vals, "old_b": 200.}, bound, higher)["verdict"] == "NOISY")
    check("strict ratio boundary", metric(dict(old_a=1., old_b=1., new_a=1.05, new_b=1.05), .05, False)["verdict"] == "WITHIN_NOISE")
    check("strict self-spread boundary", metric(dict(old_a=1., old_b=1.05, new_a=1., new_b=1.05), .05, False)["verdict"] == "WITHIN_NOISE")
    for vals, want in [((0, 0, 0, 0), "WITHIN_NOISE"), ((1, 0, 0, 1), "REGRESSION"),
                       ((0, 1, 1, 0), "IMPROVED"), ((1, 0, 1, 1), "NOISY")]:
        check(f"zero goodput {want}", metric(dict(zip(TAGS, vals)), .1, True, zero=True)["verdict"] == want)
    mutate("missing arm", lambda r, a: a.pop("new_b"))
    mutate("wrong ABBA order", lambda r, a: r["order"].reverse())
    mutate("dependency wheel mismatch", lambda r, a: a["new_a"]["identity"].update(wheel_lock_sha256="changed"))
    mutate("input digest mismatch", lambda r, a: a["new_a"]["identity"].update(checkpoint_sha256="changed"))
    mutate("wrong release", lambda r, a: a["new_b"].update(release=r["old"]))
    mutate("inherited flag", lambda r, a: a["new_b"]["feature_env"].update(E4B_PAGED_GRAPHS="1"))
    mutate("import contamination", lambda r, a: a["new_b"]["imports"]["gnf4"].update(venv="old"))
    mutate("stand-in quality", lambda r, a: a["new_b"]["quality_1"]["rehearsal"].update(stand_in_attention=True))
    mutate("missing quality window", lambda r, a: a["new_a"]["quality_12"]["per_window"]["wikitext"]["R"].pop())
    mutate("attention not engaged", lambda r, a: a["new_a"]["quality_1"]["engagement"]["wikitext"]["R"][0].update(decode_calls=0))
    mutate("graph not captured", lambda r, a: a["new_a"]["decode"]["graph_status"].update({"16": "eager"}))
    mutate("widest bucket unused", lambda r, a: a["new_a"]["capacity"]["burst"].update(widest_bucket_replays=0))
    mutate("bad request", lambda r, a: a["new_a"]["capacity"]["requests"][0].update(valid=False))
    mutate("missing fused training", lambda r, a: a["new_a"]["training"].update(kernel_calls_per_step=[0] * 20))
    mutate("profiler contaminates wall", lambda r, a: a["new_a"]["training"].update(profile_steps=10))
    mutate("NaN fails evidence", lambda r, a: a["new_a"]["training"]["step_ms"].__setitem__(0, float("nan")))
    mutate("C1 corruption FUNCTION_FAIL", lambda r, a: a["new_a"]["training"].update(C1_bit_exact=False), "FUNCTION_FAIL")

    def shift_quality(r, a, tags):
        for tag in tags:
            for x in a[tag]["quality_1"]["per_window"]["wikitext"]["R"]:
                x["nll"] += 0.1

    mutate("quality mutant fails both pairs", lambda r, a: shift_quality(r, a, ("new_a", "new_b")), "REGRESSION")
    mutate("quality noise precedes quality verdict", lambda r, a: shift_quality(r, a, ("new_a",)), "NOISY")

    def regression_and_noise(r, a):
        for tag in ("new_a", "new_b"):
            a[tag]["training"]["step_ms"] = [1200.] * 20
        a["old_b"]["training_profile"]["profile"]["device_ms"] = 2000.

    mutate("one noisy metric cannot hide another regression", regression_and_noise, "REGRESSION")
    # A same-version drift below the floor can put just one cross-pair beyond SANE.
    def unsettled(r, a):
        for tag, delta in (("new_a", .019), ("new_b", .021)):
            for x in a[tag]["quality_1"]["per_window"]["wikitext"]["R"]:
                x["nll"] += delta

    mutate("one quality pair cannot clear", unsettled, "QUALITY_UNSETTLED")
    print(f"RA self-test: {sum(tests)}/{len(tests)} passed (synthetic, CPU only)")
    return 0 if all(tests) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.dir or not args.out:
        ap.error("--dir and --out required")
    try:
        verified_files(args.dir)
        run = json.loads((args.dir / "run.json").read_text())
        arms = {t: json.loads((args.dir / f"arm_{t}.json").read_text()) for t in TAGS}
        result = reduce(run, arms)
    except (OSError, ValueError) as exc:
        result = {"schema": 1, "lane": "RA", "verdict": "VOID", "reasons": [str(exc)],
                  "release_clearance": False, "regression_blocks_next_release": False}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"RA {result['verdict']} clearance={result['release_clearance']}")
    return 0 if result["verdict"] == "CLEAR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
