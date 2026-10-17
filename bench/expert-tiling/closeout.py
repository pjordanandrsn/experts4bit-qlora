"""Expert tiling close-out: does any recorded lane sit in section 3's regime? CPU, no GPU, no run.

Reproduces the close-out table in README.md ("8. Close-out"). Every peak and budget is read from the record named beside it;
the per-layer and per-expert bytes are arithmetic from each family's committed shapes. Nothing here is measured.

    python closeout.py

A sub-expert weight tile earns its place only where the unit it replaces does not fit: either a second layer slot (for
overlap) or the single layer slot itself (for fit). It is NEEDED only where even one whole routed expert does not fit in
the room left, because a whole expert is the unit routed staging and per-expert pipelining already use.
"""

GIB = 2 ** 30
GB = 1e9


def nf4_expert_bytes(hidden, inter):
    """One routed expert under offload: NF4 gate_up [2I, H] + down [H, I], packed two values per byte, plus one fp32 absmax
    per 64 values (offloaded stacks keep fp32 absmax: double quantization is refused under offload)."""
    n = 3 * hidden * inter
    return n // 2 + n // 64 * 4


# Family shapes: (hidden, expert intermediate, experts per layer), from the record named.
SHAPES = {
    "Qwen3-235B-A22B": (4096, 1536, 128),      # bench/q235/PREREG-q235.md
    "Mixtral-8x22B": (6144, 16384, 8),         # bench/fam235/PREREG-r2-mixtral.md
    "Qwen3-30B-A3B": (2048, 768, 128),         # bench/h2h-2026-10-02/tc3 (the TC1 campaign's subject)
    "Mixtral-8x7B": (4096, 14336, 8),          # bench/fam235/PREREG-r2-mixtral.md (calibration column)
    "gemma-4-26B-A4B": (2816, 704, 128),       # bench/flagship-matrix-model2
    "OLMoE-1B-7B": (2048, 1024, 64),           # docs/METHODOLOGY.md section 11b
}


def layer_bytes(family):
    h, i, e = SHAPES[family]
    return e * nf4_expert_bytes(h, i)


def expert_bytes(family):
    h, i, _ = SHAPES[family]
    return nf4_expert_bytes(h, i)


# Rows: (lane, family, card, budget bytes, single-slot peak bytes, kind, record).
# budget = the room the record states for the card; peak = the recorded peak with ONE layer slot (training's policy).
# kind "run" = a recorded run; "plan" = a registered capacity row, never run.
ROWS = [
    ("Q235 Shape B, q235-pro6000-3", "Qwen3-235B-A22B", "RTX PRO 6000 96 GB", 90.81 * GIB, 58.91 * GIB, "run",
     "PREREG-q235.md capacity (budget); Amendment 4 (58.9 GiB device peak, an upper bound: it held four fp32 check copies)"),
    ("Q235 Shape A bf16 (stretch class)", "Qwen3-235B-A22B", "RTX 5090 32 GB", 30.25 * GIB, 29.24 * GIB, "plan",
     "PREREG-q235.md capacity table: fits, 29.24 of 30.25 GiB (1.0 GiB margin); every other 5090 row refused"),
    ("R2 Mixtral-8x22B, shape B fp32", "Mixtral-8x22B", "RTX 5090 32 GB", 30.25 * GIB, 26.5 * GIB, "plan",
     "PREREG-r2-mixtral.md capacity: 26.5 GiB (its largest row); 5090 budget as PREREG-q235.md"),
    ("TC3 fused offload, tc3-4090-8", "Qwen3-30B-A3B", "RTX 4090 24 GB", 24_564 * 2 ** 20, 11.881 * GB, "run",
     "tc3/README.md: peak 11.881 GB"),
    ("TC2 Mixtral-8x7B offload, tc1-5090-26", "Mixtral-8x7B", "RTX 5090 32 GB", 30.25 * GIB, 7.151 * GB, "run",
     "tc2/README.md: peak 7.151 GB"),
    ("METHODOLOGY 11c, Gemma-4 offload", "gemma-4-26B-A4B", "RTX A2000 12 GB (shared)", 10.3 * GB, 8.47 * GB, "run",
     "docs/METHODOLOGY.md 11c: peak 8.47 GB; budget = tc3/README.md's 10.3 GB free on this shared card"),
    ("METHODOLOGY 11c, Qwen3-30B offload", "Qwen3-30B-A3B", "RTX A2000 12 GB (shared)", 10.3 * GB, 7.16 * GB, "run",
     "docs/METHODOLOGY.md 11c: peak 7.16 GB"),
    ("METHODOLOGY 11a, OLMoE offload", "OLMoE-1B-7B", "RTX A2000 12 GB (shared)", 10.3 * GB, 2.57 * GB, "run",
     "docs/METHODOLOGY.md 11a: peak 2.57 GB"),
]


def table():
    print(f"{'lane':40} {'card':26} {'spare GiB':>9} {'layer GiB':>9} {'expert MiB':>10}  2nd slot  1 expert  kind")
    worst_ratio = None
    for lane, fam, card, budget, peak, kind, _ in ROWS:
        spare = budget - peak
        lb, eb = layer_bytes(fam), expert_bytes(fam)
        two, one_e = spare >= lb, spare >= eb
        print(f"{lane:40} {card:26} {spare / GIB:9.2f} {lb / GIB:9.3f} {eb / 2 ** 20:10.1f}  "
              f"{'fits' if two else 'NO':8}  {'fits' if one_e else 'NO':8}  {kind}")
        r = spare / eb
        worst_ratio = r if worst_ratio is None else min(worst_ratio, r)
    print(f"\nsmallest spare / one whole expert, over the rows: {worst_ratio:.0f}x")
    print("\nrecords:")
    for lane, *_, rec in ROWS:
        print(f"  {lane}: {rec}")


def fit_cases():
    """The two recorded cases where the single slot, not a second one, is the question."""
    # TC3 on the shared A2000: the field recipe (micro-batch 2) OOMs; it peaks at 11.88 GB on the 4090 (tc3/README.md),
    # against 10.3 GB free here. Removing the whole layer slot would not fit it.
    excess = 11.881 * GB - 10.3 * GB
    slot = layer_bytes("Qwen3-30B-A3B")
    print(f"\nTC3 A2000 micro-batch 2: excess {excess / GB:.2f} GB vs the whole layer slot {slot / GB:.2f} GB -> "
          f"{'tiles could close it' if excess < slot else 'no tile size can close it'} (micro-batch 1 fits: 10.46 GB)")
    # mxfp4-arena-train 284B on a 23.5 GiB A5000 (RESULTS-284b-ladder-fp8.md): the binding allocation is the staged stack's
    # full-[E] destination, 1.00 GiB of a 2.12 GiB layer; the record's named next step is a compacted stack of the ROUTED
    # experts (whole experts, about 8.3 MiB each), not sub-expert tiles.
    print("mxfp4-arena-train 284B A5000: binding = the full-[E] staged stack (2.12 GiB/layer); the named fix is a compacted "
          "routed stack of whole experts (~8.3 MiB each)")


if __name__ == "__main__":
    table()
    fit_cases()
