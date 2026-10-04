#!/usr/bin/env python
"""Tabulate every ladder receipt in a directory: per family, per rung, wall and device time, the engagement census, and the
fused rung's device-time breakdown by kernel family.

    python report.py bench/moegen/receipts
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import summarize  # noqa: E402


def delta(after, before, key):
    a, b = (after or {}).get(key), (before or {}).get(key)
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: (v - b.get(k, 0)) if isinstance(v, (int, float)) else v for k, v in a.items()}
    return a


def main():
    d = sys.argv[1]
    for p in sorted(glob.glob(os.path.join(d, "*_ladder.json"))):
        lad = json.load(open(p))
        recs = lad.get("rungs", [])
        if not recs:
            continue
        s = summarize(recs)
        print(f"\n## {os.path.basename(p)}  {lad.get('model')}  ({lad.get('model_type')}, offload={lad.get('offload')}, "
              f"seq {lad.get('seq')}, {lad.get('n_expert_modules')} expert modules, attn4 {lad.get('n_attn4')}/"
              f"{lad.get('n_attn_expected')}, attn LoRA {lad.get('n_attn_lora')}, {lad.get('gpu')})")
        print(f"{'rung':10} {'wall s/step':>12} {'wall xF':>8} {'device s':>9} {'dev xF':>7} {'peak GB':>8} {'syncs':>6} {'busy':>12}")
        for k, v in s.items():
            busy = ",".join(f"{x:.2f}" for x in v["device_busy_frac"] if x is not None)
            print(f"{k:10} {v['s_per_step_mean']:>12} {str(v.get('ratio_to_fused')):>8} {str(v.get('device_s_mean')):>9} "
                  f"{str(v.get('device_ratio_to_fused')):>7} {v['peak_alloc_gb']:>8} {str(v['syncs_one_step']):>6} {busy:>12}")
        for r in recs:
            if r["rung"].split("#")[0] != "fused" or "#" in r["rung"]:
                continue
            c, c0 = r.get("counters", {}), r.get("counters_before", {})
            print(f"  engagement (fused): patched {r.get('n_patched')} skipped {(c.get('fast_train') or {}).get('skipped')} recurrent-fallback {(c.get('fast_train') or {}).get('recurrent_fallbacks')} | "
                  f"dgrad {delta(c, c0, 'dgrad')} | lora {delta(c, c0, 'lora_path')} | ring {delta(c, c0, 'ring')} | "
                  f"rmsnorm patched {(c.get('rmsnorm') or {}).get('patched', 0) - (c0.get('rmsnorm') or {}).get('patched', 0)} "
                  f"fallback {delta(c, c0, 'rmsnorm').get('fallback_calls') if isinstance(c.get('rmsnorm'), dict) else '-'} | "
                  f"rope modules {(c.get('rope') or {}).get('patched_modules')} refused {c.get('rope_refused')} | "
                  f"route {(c.get('route') or {}).get('train_gemm')}")
            prof = r.get("profile") or {}
            fam = prof.get("family_ms", {})
            tot = sum(fam.values()) or 1
            print("  device ms (fused): " + ", ".join(f"{k} {v:.0f} ({100 * v / tot:.0f}%)" for k, v in fam.items()))
        for r in recs:
            if r["rung"] == "keep":
                c = r.get("counters", {})
                print(f"  keep: {c.get('moe_keep')}")


if __name__ == "__main__":
    main()
