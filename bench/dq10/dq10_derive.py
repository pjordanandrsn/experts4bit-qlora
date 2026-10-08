"""Freeze placement hypotheses from known DQ9 diagnostic rows, never license an import."""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

SHAPES = (("llama31_8b", "device", 2048), ("llama31_8b", "stream", 2048),
          ("llama31_8b", "device", 4096), ("llama31_8b", "stream", 4096),
          ("qwen3_32b", "device", 2048), ("qwen3_32b", "stream", 2048),
          ("qwen3_14b", "stream", 512), ("qwen3_14b", "stream", 4096))
EXPECTED = {(*shape, mode) for shape in SHAPES for mode in ("baseline", "empty")}


def ceil_fraction(value, fraction):
    return (value * fraction.numerator + fraction.denominator - 1) // fraction.denominator


def derive(directory):
    root = Path(directory)
    registered = json.loads(Path(__file__).with_name("dq9-source-sha256.json").read_text())
    sources, rows, seen = {}, [], set()
    for path in sorted(root.glob("read-*.json")):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if registered.get(path.name) != digest:
            raise ValueError("source receipt differs from the frozen DQ9 diagnostic")
        receipt = json.loads(raw)
        tag = receipt["dq9"]
        key = tuple(tag[k] for k in ("subject", "placement", "seq", "cache_mode"))
        if key not in EXPECTED or key in seen or receipt.get("status") != "OK":
            raise ValueError("unregistered, duplicate or failed source row")
        seen.add(key)
        comparison = receipt["comparison"]
        e = comparison["device_allocator"]["estimated"]
        a = comparison["device_allocator"]["measured"]
        r = receipt["measured"]["device_reserved_peak_bytes"]
        d = comparison["device_driver"]["measured"]
        if any(type(x) is not int or x <= 0 for x in (e, a, r, d)) or r < a or d < r:
            raise ValueError("invalid source peak bytes")
        sources[path.name] = digest
        rows.append({"subject": key[0], "placement": key[1], "seq": key[2], "cache_mode": key[3],
                     "allocator_estimate_bytes": e, "allocated_peak_bytes": a,
                     "reserved_peak_bytes": r, "driver_peak_bytes": d,
                     "driver_minus_reserved_bytes": d - r, "training_peak_slack_bytes": r - a})
    if seen != EXPECTED:
        raise ValueError("missing source row")
    hypotheses = {}
    for placement in ("device", "stream"):
        subset = [row for row in rows if row["placement"] == placement]
        fraction = max([Fraction(1, 5)] + [Fraction(row["training_peak_slack_bytes"], row["allocated_peak_bytes"])
                                         for row in subset])
        context = max(512 * 2**20, max(row["driver_minus_reserved_bytes"] for row in subset))
        hypotheses[placement] = {"reserve_numerator": fraction.numerator,
                                  "reserve_denominator": fraction.denominator, "context_bytes": context}
        for row in subset:
            e, a, r, d = (row[k] for k in ("allocator_estimate_bytes", "allocated_peak_bytes",
                                          "reserved_peak_bytes", "driver_peak_bytes"))
            charge = ceil_fraction(e, fraction)
            total = e + charge + context
            row.update(reserve_charge_bytes=charge, context_charge_bytes=context,
                       plan_total_bytes=total, allocator_residual_bytes=a-e,
                       reserve_term_residual_bytes=r-a-charge, reserved_total_residual_bytes=r-e-charge,
                       context_residual_bytes=d-r-context, driver_residual_bytes=d-total,
                       residual_charge_bytes=0)
    closes = all(row[k] <= 0 for row in rows for k in ("allocator_residual_bytes", "reserve_term_residual_bytes",
                 "reserved_total_residual_bytes", "context_residual_bytes", "driver_residual_bytes"))
    return {"schema": "dq10-frozen-hypothesis/1", "status": "IN_SAMPLE_CLOSURE" if closes else "IN_SAMPLE_FAIL",
            "source_class": "DQ9 known-subject diagnostic; these are fitting rows, never holdouts",
            "capacity_licensed": False, "calibration_replacement_licensed": False,
            "reserve_metric": "training reserved peak minus allocated peak; peaks need not be contemporaneous",
            "context_metric": "sampled driver peak minus reserved peak; not a contemporaneous context census",
            "hypotheses": hypotheses, "source_sha256": sources, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = derive(args.directory)
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(result["status"])
    raise SystemExit(0 if result["status"] == "IN_SAMPLE_CLOSURE" else 12)
