"""Every file a committed receipt's SHA256SUMS lists is in the tree.

``.gitignore`` has ``*.log``, so ``git add`` of a receipt directory silently drops its logs. That happened on the P99
read and again on P115 Phase C (#1342). Force-add them: ``git add -f bench/pNN/receipts/<run>``.

Presence only. Every stored hash matched its committed file when this check landed, but a Windows checkout with
``core.autocrlf`` rewrites text files, so hashes are not compared here.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = re.compile(r"^([0-9a-f]{64})\s+\*?(?:\./)?(.+?)\s*$")

#: receipts committed before this check whose SHA256SUMS list files the tree does not hold (launcher markers, gzipped
#: copies, large references, and logs lost to the ignore rule), with how many. Frozen: a count may shrink (backfill
#: from the receipt store), never grow, and no directory joins.
LEGACY_GAPS = {
    "bench/h2h-2026-10-02/sc1g/receipts/sc1g-r5-2/ref": 5,
    "bench/p100/receipts/p100-5090-2": 10,
    "bench/p101/receipts/p101-5090-5": 17,
    "bench/p102/receipts/p102-5090-5": 8,
    "bench/p102/receipts/p102-5090-6": 8,
    "bench/p103/receipts/p103-prove-2": 20,
    "bench/p104/receipts/p104-prove-1": 23,
    "bench/p105/receipts/p105-5090-1": 23,
    "bench/p81/receipts/p81-5090-2": 5,
    "bench/p82/receipts/p82-5090-3": 5,
    "bench/p89/receipts/p89-5090-3": 1,
    "bench/p89/receipts/p89-5090-4": 1,
    "bench/p90/receipts/p90-5090-1": 4,
    "bench/p90/receipts/p90-5090-2": 2,
    "bench/p90/receipts/p90-h100-1": 4,
    "bench/p97/receipts/p97-5090-1": 13,
    "bench/p98/receipts/p98-5090-2": 16,
    "bench/p99/receipts/p99-5090-1": 18,
}


def _gaps() -> dict:
    out = {}
    for sums in sorted((ROOT / "bench").rglob("SHA256SUMS")):
        if "receipts" not in sums.relative_to(ROOT).parts:
            continue
        d = sums.parent
        entries = (ENTRY.match(ln) for ln in sums.read_text(encoding="utf-8").splitlines())
        missing = sorted({m.group(2) for m in entries if m and not (d / m.group(2)).is_file()})
        if missing:
            out[d.relative_to(ROOT).as_posix()] = missing
    return out


def test_every_file_a_receipt_lists_is_committed():
    grown = {d: m for d, m in _gaps().items() if len(m) > LEGACY_GAPS.get(d, 0)}
    assert not grown, (
        "SHA256SUMS lists files the tree does not hold; receipt logs need `git add -f` (.gitignore has *.log): "
        + "; ".join(f"{d}: {', '.join(m[:5])}{' ...' if len(m) > 5 else ''}" for d, m in grown.items()))


def test_the_check_reads_the_receipts_it_guards():
    sums = [p for p in (ROOT / "bench").rglob("SHA256SUMS") if "receipts" in p.relative_to(ROOT).parts]
    assert len(sums) >= 70                                          # 73 when this landed: the glob still finds them
    assert all((ROOT / d).is_dir() for d in LEGACY_GAPS)            # a frozen entry names a real directory
