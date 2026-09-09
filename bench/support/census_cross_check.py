#!/usr/bin/env python3
"""Join e4b's architecture claims to grouped-nf4-gemm's shape census.

grouped-nf4-gemm#353: the census covers four models while e4b claims nine
architectures, and a `(N, K)` absent from the census does not get refused -- it
takes the universal-constant decode plan. So a family can hold PASS receipts on
real weights while its fused-expert shapes were never censused. The issue's own
words for why nobody notices: *"The failure mode is not a wrong answer, it is an
unremarkable one. A refusal would have been a row; a generic config is a number
that looks like every other number."* And: *"there is no artifact where the two
registers meet."*

This is that artifact, and it lives HERE rather than in the kernel package on
purpose. gnf4's `ci.yml` already states the principle for the mirror case -- a
consumer's floor is "the consumer's metadata, asserted in its CI, never in this
wheel". e4b makes the architecture claim, so e4b checks whether the kernel has
seen its shapes.

**It reports; it does not refuse.** Whether the kernel should warn-then-refuse
is open in #353 and is the kernel owner's call, not this script's. What was
missing was not a policy but a measurement.

Shapes come from `bench/support/rows/*.json` -- geometry read off checkpoints
that actually loaded, not from configs someone typed. A fused-expert layer is
two per-expert GEMMs:

    gate_up : N = 2 * moe_intermediate,  K = hidden
    down    : N = hidden,                K = moe_intermediate

Usage:
    python bench/support/census_cross_check.py [--census PATH] [--check]

`--check` exits non-zero if a family that WAS covered stops being covered, using
census-coverage-baseline.json -- the same regression shape as the support gate,
and for the same reason: an absolute gate would fail on arrival and be disabled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = Path(__file__).resolve().parent / "rows"
BASELINE = Path(__file__).resolve().parent / "census-coverage-baseline.json"

# e4b clones gnf4 as a sibling; fall back to the usual checkout location.
CENSUS_CANDIDATES = [
    ROOT.parent / "grouped-nf4-gemm" / "census" / "shape_census.json",
    ROOT / "_sibling" / "grouped-nf4-gemm" / "census" / "shape_census.json",
]

# The prediction grouped-nf4-gemm#353 makes about the guard it proposes, quoted
# so it can be checked rather than remembered: "run it today and it refuses
# granitemoe and mixtral while admitting olmoe, qwen3_moe, gemma4_text and
# gpt_oss unchanged. That is checkable before anyone commits to the guard's
# semantics, and if it refuses something else the guard is wrong."
PREDICTED_REFUSED = {"granitemoe", "mixtral"}
PREDICTED_ADMITTED = {"olmoe", "qwen3_moe", "gemma4_text", "gpt_oss"}


def load_census(path: Path) -> tuple[set[tuple[int, int]], list[str]]:
    d = json.loads(path.read_text())
    shapes: set[tuple[int, int]] = set()
    names = []
    for m in d.get("models", []):
        names.append(m.get("model", "?"))
        for proj in (m.get("per_expert_gemms") or {}).values():
            if "N" in proj and "K" in proj:
                shapes.add((int(proj["N"]), int(proj["K"])))
    return shapes, names


def family_shapes() -> dict[str, dict]:
    """model_type -> the strongest row's geometry and derived GEMM shapes.

    Only rows that actually loaded count. A refused or blocked row's geometry is
    real but the family was never exercised at those shapes here, and a dense
    checkpoint sharing a claimed model_type (gemma-4-31B) has no experts at all
    -- feeding its 43008x5376 "gate_up" into a census comparison would be
    inventing a shape the kernel will never see.
    """
    out: dict[str, dict] = {}
    for p in sorted(ROWS.glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("exit_code") != 0:
            continue
        c = d.get("config") or {}
        h, i = c.get("hidden_size"), c.get("moe_intermediate_size")
        mt = d.get("model_type")
        # isinstance(int), not truthiness: hunyuan_v1_moe ships
        # moe_intermediate_size as a LIST of 32 per-layer values, so a family can
        # have a different expert shape per layer and there is no single (N, K)
        # to census. Skipping is the honest answer -- deriving one from element 0
        # would invent a shape and quietly claim coverage for 31 others.
        if not (mt and isinstance(h, int) and isinstance(i, int)):
            continue
        cand = {
            "model": d.get("model"),
            "tier": d.get("tier"),
            "hidden": h,
            "moe_intermediate": i,
            "experts": c.get("num_experts"),
            "gate_up": (2 * i, h),
            "down": (h, i),
        }
        # Prefer a reference-tier row: a toy's shapes are not the family's.
        prev = out.get(mt)
        if prev is None or (cand["tier"] == "reference" and prev["tier"] != "reference"):
            out[mt] = cand
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    census_path = Path(args.census) if args.census else next(
        (p for p in CENSUS_CANDIDATES if p.exists()), None)
    if census_path is None or not census_path.exists():
        print("no shape_census.json found; pass --census PATH "
              f"(looked in: {', '.join(str(p) for p in CENSUS_CANDIDATES)})", file=sys.stderr)
        return 2

    censused, census_models = load_census(census_path)
    fams = family_shapes()
    sys.path.insert(0, str(ROOT))
    from experts4bit_qlora.loader import SUPPORTED_ARCHITECTURES as CLAIMED

    print("# Claimed architectures vs the grouped-nf4-gemm shape census\n")
    print(f"Census: `{census_path}` — {len(census_models)} models, "
          f"{len(censused)} distinct per-expert `(N, K)`:\n")
    for m in census_models:
        print(f"- `{m}`")
    print("\nShapes below are derived from rows that LOADED, so they are the geometry of "
          "real checkpoints:\n")
    print("| model_type | claimed | checkpoint | gate_up (N,K) | down (N,K) | censused |")
    print("|---|---|---|---|---|---|")

    covered: dict[str, bool] = {}
    for mt in sorted(fams):
        f = fams[mt]
        hits = [f["gate_up"] in censused, f["down"] in censused]
        ok = all(hits)
        covered[mt] = ok
        mark = "**yes**" if ok else ("partial" if any(hits) else "**NO**")
        print(f"| `{mt}` | {'yes' if mt in CLAIMED else 'convention'} | `{f['model']}` "
              f"| {f['gate_up']} | {f['down']} | {mark} |")

    claimed_uncovered = [mt for mt in sorted(covered)
                         if mt in CLAIMED and not covered[mt]]
    print(f"\n**{sum(1 for mt in covered if mt in CLAIMED and covered[mt])} of "
          f"{sum(1 for mt in covered if mt in CLAIMED)}** probed claimed families have BOTH "
          f"fused-expert shapes in the census.")
    if claimed_uncovered:
        print("\nClaimed and uncensused — these take the universal-constant plan today, and "
              "would be REFUSED by the guard #353 proposes: "
              + ", ".join(f"`{m}`" for m in claimed_uncovered))

    # ── the issue's own prediction, checked ───────────────────────────────────
    print("\n## grouped-nf4-gemm#353's prediction, checked against measured geometry\n")
    print("> run it today and it refuses `granitemoe` and `mixtral` while admitting `olmoe`,")
    print("> `qwen3_moe`, `gemma4_text` and `gpt_oss` unchanged … if it refuses something")
    print("> else the guard is wrong.\n")
    verdicts = []
    for mt in sorted(PREDICTED_REFUSED | PREDICTED_ADMITTED):
        if mt not in covered:
            verdicts.append((mt, "not probed here", "—"))
            continue
        actual = "admitted" if covered[mt] else "refused"
        want = "refused" if mt in PREDICTED_REFUSED else "admitted"
        verdicts.append((mt, actual, "matches" if actual == want else f"**DIFFERS (predicted {want})**"))
    for mt, actual, note in verdicts:
        print(f"- `{mt}`: {actual} — {note}")

    unpredicted = [mt for mt in claimed_uncovered
                   if mt not in PREDICTED_REFUSED and mt not in PREDICTED_ADMITTED]
    if unpredicted:
        print("\n**The prediction is incomplete, not wrong.** It also refuses "
              + ", ".join(f"`{m}`" for m in unpredicted)
              + " — claimed families the memo did not name. That matters for the issue's own "
                "open question 1 (\"Refusing today stops five-plus families that currently "
                "run\"): the count is now measurable rather than estimated.")

    if args.check:
        cur = {mt: covered[mt] for mt in covered if mt in CLAIMED}
        if not BASELINE.exists():
            BASELINE.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n")
            print(f"\nwrote initial {BASELINE.name}")
            return 0
        base = json.loads(BASELINE.read_text())
        lost = [mt for mt, was in base.items() if was and not cur.get(mt, False)]
        if lost:
            print(f"\ncensus coverage REGRESSED for: {', '.join(lost)}", file=sys.stderr)
            return 1
        gained = [mt for mt, now in cur.items() if now and not base.get(mt, False)]
        if gained:
            print(f"\ncoverage improved for {', '.join(gained)} — update {BASELINE.name}")
        print("\nOK: no claimed family lost census coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
