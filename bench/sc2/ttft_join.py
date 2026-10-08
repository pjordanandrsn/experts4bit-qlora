#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""Join SC2 client requests to E4B_PAGED_TRACE by completion id, never row order.

Reports queue wait, admission-to-token availability and client TTFT. The residual
client TTFT - server TTFT includes HTTP intake, post-prefill decode/dispatch,
detokenization and transport; it is not a measurement of network time alone.
Durations are compared, never absolute timestamps across the two clocks.
Quantiles of components do not add to the quantile of total TTFT.

Historical client files without ids cannot be joined and are refused.
This is descriptive analysis; it does not change SC2's registered rules.

    python bench/sc2/ttft_join.py CLIENT.json SERVER_TRACE.jsonl
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def _index(rows, source):
    out = {}
    for row in rows:
        rid = row.get("request_id")
        if not isinstance(rid, str) or not rid or row.get("request_id_conflict"):
            raise ValueError(f"{source}: missing or conflicting request_id")
        if rid in out:
            raise ValueError(f"{source}: duplicate request_id {rid}")
        out[rid] = row
    return out


def _number(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{row['request_id']}: {key} must be finite")
    return value


def _summary(values):
    values = sorted(values)
    def pct(p):
        k = (len(values) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 6)
    return {"mean_s": round(statistics.fmean(values), 6), "p50_s": pct(.5), "p99_s": pct(.99)}


def analyse(client, trace):
    """Require an unambiguous server match for each VALID client request.

    Extra trace rows (warmup and other workloads) are counted, not paired.
    Negative residuals are retained and counted for investigation.
    """
    requests = client["requests"]
    valid = [r for r in requests if r.get("valid")]
    if not valid:
        raise ValueError("client: no valid requests")
    clients, servers = _index(valid, "client"), _index(trace, "server")
    missing = clients.keys() - servers.keys()
    if missing:
        raise ValueError(f"server: no match for {len(missing)} client request(s)")
    pairs = []
    for rid, c in clients.items():
        s = servers[rid]
        if c.get("prompt_len") != s.get("prompt_len") or c.get("completion_tokens") != s.get("out_len"):
            raise ValueError(f"{rid}: prompt/output length mismatch")
        arrival, admitted, first = (_number(s, k) for k in ("arrival", "admitted_at", "first_token_at"))
        if not arrival <= admitted <= first:
            raise ValueError(f"{rid}: server timestamps out of order")
        queue, prefill, total = admitted - arrival, first - admitted, first - arrival
        for key, value in (("ttft", total), ("queue_wait", queue)):
            if not math.isclose(_number(s, key), value, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"{rid}: inconsistent server {key}")
        client_ttft = _number(c, "ttft_s")
        if client_ttft < 0:
            raise ValueError(f"{rid}: negative client TTFT")
        pairs.append({"request_id": rid, "client_ttft_s": client_ttft,
                      "server_ttft_s": total, "queue_wait_s": queue,
                      "admission_to_first_token_s": prefill,
                      "client_minus_server_s": client_ttft - total})
    keys = ("client_ttft_s", "server_ttft_s", "queue_wait_s",
            "admission_to_first_token_s", "client_minus_server_s")
    return {"matched": len(pairs), "invalid_clients": len(requests) - len(valid),
            "unmatched_server_rows": len(servers) - len(pairs),
            "negative_residuals": sum(p["client_minus_server_s"] < 0 for p in pairs),
            "summary": {key: _summary([p[key] for p in pairs]) for key in keys}, "requests": pairs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args(argv)
    try:
        client = json.loads(args.client.read_text())
        trace = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
        print(json.dumps(analyse(client, trace), indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"REFUSED: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
