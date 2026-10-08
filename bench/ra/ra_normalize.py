#!/usr/bin/env python3
"""Bind RA envelopes to retained native records, then apply the frozen reducer.

This checks normalization, not the truth of GPU/provenance telemetry. The
executor and on-card premise must supply that independent evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import types
from pathlib import Path

import ra_reduce as reducer
import ra_stage

FILES = ("metadata", "training", "training_profile", "decode", "capacity", "capacity_warm",
         "capacity_burst", "capacity_extra", "capacity_health_warm", "capacity_health_burst",
         "capacity_health_end", "quality_1", "quality_12")
META = ("tag", "status", "identity", "release", "feature_env", "matched_init_sha", "venv", "imports",
        "arena_sha256", "resolved_defaults")


def require(ok, message):
    reducer.require(ok, "normalization: " + message)


def requests(native):
    """SC2 request identity/order and HTTP status are checked before projection.

    SC2's finish() derived validity from SSE events. A projected 'valid' field
    cannot override HTTP errors, request-plan mismatches or missing usage.
    """
    plan, rows = native["plan"], native["requests"]
    require(len(plan) == len(rows) == native["n"], "request count")
    out = []
    for i, (item, row) in enumerate(zip(plan, rows)):
        require(row["status"] == 200 and not row.get("error"), f"request {i} HTTP/error")
        require(row["i"] == i and row["prompt_index"] == item[1] and row["planned_s"] == item[0] and
                row["max_tokens"] == item[2], f"request {i} plan/order")
        require(row["valid"] is True and row["finish_reason"] == "length" and
                row["completion_tokens"] == item[2] and row["prompt_tokens"] == row["prompt_len"] == 512,
                f"request {i} token usage/validity")
        require(not row.get("request_id_conflict"), f"request {i} stream identity changed")
        out.append({k: copy.deepcopy(row[k]) for k in ("valid", "prompt_tokens", "completion_tokens",
                                                       "finish_reason", "ttft_s", "tpot_s")})
    require(native["summary"]["valid"] == len(out) and native["summary"]["invalid"] == 0,
            "SC2 summary/request disagreement")
    return out


def capacity_health(raw):
    """Derive reducer counts from actual /health snapshots; never typed totals."""
    buckets = None
    for label, admitted in (("warm", 4), ("burst", 68), ("end", 188)):
        health = raw[f"capacity_health_{label}"]
        require(health["status"] == "ready" and health["error"] is None and health["queue_depth"] == 0,
                f"{label} server not drained/ready")
        engine, pg, kv = health["engine"], health["prefill_graph"], health["kv_bookkeeping"]
        if buckets is None:
            buckets = engine["buckets"]
        require(engine["buckets"] == buckets and bool(buckets) and len(set(buckets)) == len(buckets),
                "capacity resolved buckets changed")
        require(engine["graphs"] is True and set(engine["graph_status"]) == {str(b) for b in buckets} and
                all(v == "graph" for v in engine["graph_status"].values()), "capacity graphs not captured")
        require(engine["chunk_tokens"] == 512 and engine["max_tokens_limit"] == 2048, "capacity context/chunk")
        require(pg["status"] == "on" and pg["T"] == 512 and pg["replays"] == admitted and pg["eager_chunks"] == 0,
                f"{label} prefill counts/fallback")
        require(kv["requested"] is True and kv["bulk"] is True and kv["flush_layers"] == kv["ready_layers"] ==
                kv["ready_bulk"] == 0 and kv["flush_bulk"] == kv["ready_at_flush"] == admitted,
                f"{label} KV counters/fallback")
        require(bool(engine["graph_stats"]) and all(s["eager_steps"] == 0 for s in engine["graph_stats"].values()),
                f"{label} decode fallback")
    key = str(max(buckets))
    warm = raw["capacity_health_warm"]["engine"]["graph_stats"]
    burst = raw["capacity_health_burst"]["engine"]["graph_stats"]
    replays = burst[key]["replays"] - warm[key]["replays"]
    require(replays > 0, "burst did not exercise widest bucket")
    end = raw["capacity_health_end"]
    return replays, {"admitted": 188, "prefill_replays": end["prefill_graph"]["replays"],
                     "prefill_eager_chunks": end["prefill_graph"]["eager_chunks"],
                     "kv_bulk_flush": end["kv_bookkeeping"]["flush_bulk"],
                     "decode_eager_steps": sum(s["eager_steps"] for s in end["engine"]["graph_stats"].values())}


def normalize(raw, driver):
    require(set(raw) == set(FILES), "raw component set")
    meta = raw["metadata"]
    require(set(meta) == set(META), "metadata field set")
    out = copy.deepcopy(meta)
    for name in ("training", "training_profile", "decode"):
        out[name] = copy.deepcopy(raw[name])
    c, extra = raw["capacity"], raw["capacity_extra"]
    require(set(extra) == {"slo", "features"}, "capacity extra field set")
    require((c["mode"], c["rate"], c["seed"], c["n"], c["max_tokens_range"]) ==
            ("poisson", 12, 112, 120, [64, 256]), "capacity fixture")
    measured = requests(c)
    warm, burst = raw["capacity_warm"], raw["capacity_burst"]
    require((warm["mode"], warm["rate"], warm["seed"], warm["n"]) == ("serial", 0, 999, 4), "warm fixture")
    require((burst["mode"], burst["rate"], burst["seed"], burst["n"]) == ("poisson", 1000, 998, 64), "burst fixture")
    for native in (c, warm, burst):
        require(native["profile"] == "e4b" and native["max_tokens_range"] == [64, 256], "SC2 profile/range")
        expected = driver.plan(native["mode"], native["rate"], native["n"], native["seed"], 16, 64, 256)
        require(native["plan"] == [list(item) for item in expected], "SC2 seeded plan differs from frozen helper")
    requests(warm)
    requests(burst)
    require(c["prompts_sha256"] == warm["prompts_sha256"] == burst["prompts_sha256"] ==
            meta["identity"]["prompts_sha256"], "capacity prompt identity")
    burst_replays, end = capacity_health(raw)
    out["capacity"] = {"status": "ok", "offered_rate": c["rate"], "plan": copy.deepcopy(c["plan"]),
                       "prompt_tokens": 512, "invalid": c["summary"]["invalid"],
                       "slo": copy.deepcopy(extra["slo"]), "requests": measured,
                       "burst": {"valid": burst["summary"]["valid"],
                                 "widest_bucket_replays": burst_replays},
                       "health_end": end, "features": copy.deepcopy(extra["features"])}
    for group in (1, 12):
        name = f"quality_{group}"
        q = copy.deepcopy(raw[name])
        require(q["phase"] == "off" and q["arms"] == ["R"] and q["windows"] == {"wikitext": 12},
                f"{name} must score its own R only")
        require(set(q["windows_sha256"]) == {"wikitext"}, f"{name} window hash set")
        q["windows_sha256"] = q["windows_sha256"]["wikitext"]
        out[name] = q
    return out


def read_raw(root, tag):
    raw, hashes = {}, {}
    for name in FILES:
        path = root / "raw" / tag / f"{name}.json"
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "raw path escapes run")
        data = path.read_bytes()
        raw[name] = json.loads(data)
        hashes[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
    return raw, hashes


def bind(root, *, check=False):
    stage = root / "instruments"
    staged = ra_stage.verify(stage)
    driver = types.ModuleType("ra_frozen_sc2_driver")
    path = stage / "sc2_driver.py"
    driver.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), driver.__dict__)
    run = json.loads((root / "run.json").read_bytes())
    require(run["identity"]["source_pins_sha256"] == staged["source_pins_sha256"], "source pins identity")
    arms, hashes = {}, {}
    for tag in reducer.TAGS:
        raw, digests = read_raw(root, tag)
        arm = normalize(raw, driver)
        require(arm["tag"] == tag, "arm tag mismatch")
        path = root / f"arm_{tag}.json"
        if check:
            require(json.loads(path.read_bytes()) == arm, f"{tag} envelope differs from raw")
        else:
            path.write_text(json.dumps(arm, sort_keys=True, indent=2, allow_nan=False) + "\n")
        arms[tag] = arm
        hashes.update(digests)
    verdict = reducer.reduce(run, arms)
    binding = {"schema": 1, "checks": "native-to-envelope equality",
               "proves_gpu_engagement": False, "raw_sha256": hashes}
    if check:
        require(json.loads((root / "normalization.json").read_bytes()) == binding, "raw binding changed")
        reducer.verified_files(root)
    else:
        (root / "normalization.json").write_text(json.dumps(binding, sort_keys=True, indent=2) + "\n")
        records = {}
        for path in root.rglob("*"):
            require(not path.is_symlink(), "symlink in receipt")
            if path.is_file() and path not in (root / "SHA256SUMS", root / "verdict.json"):
                records[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(records.items())))
        reducer.verified_files(root)
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    result = bind(args.dir, check=args.check)
    print(f"RA native binding verified; envelope verdict {result['verdict']}; GPU truth requires executor/premise")
    return 0 if result["verdict"] == "CLEAR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
