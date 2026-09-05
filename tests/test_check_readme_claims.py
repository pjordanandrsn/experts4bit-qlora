"""README = current main, mechanically.

Two things the README used to leave to hand-editing: which release its
"current position" links pointed at (a tag pin that went stale between
releases), and whether a headline number was still the register's current
value (a retracted Granite row sat in the table for a day as a licensed best).
`scripts/check_readme_claims.py` makes both a CI failure. These tests pin the
parts that decide a pass: the number tokeniser (what is a result and what is a
name), the rounding rule, the status rule, and the generated release block.
The last test runs the real check on the repository's own README.
"""
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_readme_claims as crc  # noqa: E402
import check_readme_links as clc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def D(*xs):
    return [Decimal(x) for x in xs]


# ------------------------------------------------------------- the tokeniser --

def test_result_numbers_reads_results_and_skips_names():
    cell = ("Qwen3-30B-A3B ×2.067 at B=1 (238.1 tok/s; anchor-class projection 159.2 × 2.067 ≈ 329 tok/s) "
            "and ×2.602 at B=16 (1327.5 tok/s); Gemma-4-26B ×1.281 (103.6); round-1/2 folds; fp8 KV; 64k tokens; "
            "K8 gate; ([#359](https://github.com/o/r/issues/359)); 2026-09-05; RTX 5090")
    got = crc.result_numbers(cell)
    assert D("2.067", "238.1", "159.2", "2.067", "329", "2.602", "1327.5", "1.281", "103.6")[:9] == got[:9]
    # names, keys, issues, dates and URLs contribute nothing
    for absent in ("3", "30", "1", "16", "4", "26", "8", "64", "359", "2026", "9", "5"):
        assert Decimal(absent) not in got, f"{absent} was read as a result"
    # the one thing left is the card, which is a number in prose and is the author's to keep out of a result cell
    assert got[9:] == D("5090")


def test_result_numbers_signs_ranges_and_thousands():
    got = crc.result_numbers("−0.0528 ppl on wikitext and −0.0662 on C4; 1.52–1.81× per step at 0.75–0.81× VRAM; "
                             "+37.1% over by-index; ≈1,238 tok/s; 2.56× / 3.80× / 6.40× less; 1.4813 → 1.0290")
    assert got == D("-0.0528", "-0.0662", "1.52", "1.81", "0.75", "0.81", "+37.1", "1238", "2.56", "3.80", "6.40",
                    "1.4813", "1.0290")


def test_claim_numbers_come_from_value_unit_and_claim_never_notes():
    c = {"value": 304.9, "unit": "tok/s", "claim": "the licensed stack: 304.9 tok/s, x1.341 over NF4 (227.3)",
         "notes": "measured, NOT licensed: int4_r12epi 487.8 (x2.146)"}
    got = crc.claim_numbers(c)
    assert Decimal("1.341") in got and Decimal("227.3") in got and Decimal("304.9") in got
    assert Decimal("487.8") not in got and Decimal("2.146") not in got, "an unlicensed arm must never match a headline"


def test_value_numbers_handles_ranges_strings_and_null():
    assert crc.value_numbers({"value": "1.52-1.81"}) == D("1.52", "1.81")
    assert crc.value_numbers({"value": "+37.1"}) == D("37.1")
    assert crc.value_numbers({"value": 1238}) == D("1238")
    assert crc.value_numbers({"value": None}) == []


# ----------------------------------------------------------- the rounding rule --

@pytest.mark.parametrize("readme, register, ok", [
    ("155", "154.9", True),          # the README's own precision
    ("0.046", "0.04645", True),
    ("1.0290", "1.029", True),
    ("4.70", "4.7", True),
    ("-0.053", "-0.0528", True),
    ("100", "98.3", False),          # rounding only: an approximation is not the value
    ("204", "204.6", False),         # truncation is not rounding
    ("238.1", "240.3", False),       # drift
])
def test_number_matches_at_the_readme_precision(readme, register, ok):
    assert crc.number_matches(Decimal(readme), [Decimal(register)]) is ok


# ------------------------------------------------------------ the row checks --

CLAIMS = {
    "e4b.x.a": {"status": "measured", "value": 154.9, "unit": "tok/s", "claim": "154.9 tok/s, x1.59 over 97"},
    "e4b.x.b": {"status": "measured-private", "value": 204.6, "unit": "tok/s", "claim": "204.6 tok/s"},
    "e4b.x.old": {"status": "superseded", "value": 6.31, "superseded_by": "e4b.x.a", "claim": "6.31x"},
    "e4b.x.gone": {"status": "retired", "value": 0.047, "claim": "+0.047 ppl"},
    "e4b.x.q": {"status": "measured", "value": None, "claim": "no reference at this resolution"},
    "e4b.g.a": {"status": "measured", "value": 154.9, "unit": "tok/s", "claim": "154.9 tok/s"},
    "e4b.g.b": {"status": "measured-private", "value": 204.6, "unit": "tok/s", "claim": "204.6 tok/s"},
}


def _row(desc, result, status):
    return crc.check_row(7, [desc, result, status], 2, CLAIMS)


def test_a_clean_row_passes():
    assert _row("speed (`e4b.x.a`)", "97 → 155 tok/s, ×1.59", "measured") == []


def test_a_glob_resolves_and_every_matched_value_must_be_quoted():
    """A ``*`` glob names every matching claim, and each one's value must be in the row."""
    assert _row("speed (`e4b.g.*`)", "155 and 204.6 tok/s", "measured-private") == []
    f = _row("speed (`e4b.g.*`)", "155 tok/s", "measured-private")
    assert any("`e4b.g.b` has value 204.6" in x for x in f)


def test_a_number_that_is_not_the_claims_value_is_drift():
    f = _row("speed (`e4b.x.a`)", "160 tok/s, ×1.59", "measured")
    assert any("160 is not a current value" in x for x in f)
    assert any("`e4b.x.a` has value 154.9 and the row does not quote it" in x for x in f)


def test_superseded_and_retired_ids_fail_and_name_the_replacement():
    f = _row("old (`e4b.x.old`)", "6.31×", "measured")
    assert any("`e4b.x.old` is superseded; quote `e4b.x.a` instead" in x for x in f)
    f = _row("gone (`e4b.x.gone`)", "+0.047 ppl", "measured")
    assert any("`e4b.x.gone` is retired" in x for x in f)


def test_a_private_receipt_is_never_presented_as_measured():
    f = _row("speed (`e4b.x.a`, `e4b.x.b`)", "155 and 204.6 tok/s", "measured")
    assert any("'measured-private'" in x and "weakest" in x for x in f)
    assert _row("speed (`e4b.x.a`, `e4b.x.b`)", "155 and 204.6 tok/s", "measured-private") == []


def test_a_row_without_an_id_or_with_an_unknown_id_fails():
    assert any("names no claim id" in x for x in _row("speed", "155 tok/s", "measured"))
    assert any("`e4b.x.nope` is not in" in x for x in _row("speed (`e4b.x.nope`)", "155", "measured"))
    assert any("`e4b.x.*.zzz` is not in" in x for x in _row("s (`e4b.x.*.zzz`)", "155", "measured"))


def test_a_qualitative_claim_needs_no_number():
    assert _row("no reference (`e4b.x.q`)", "no reference at this resolution", "measured") == []


def test_tables_are_found_by_their_status_column_and_escaped_pipes_survive():
    text = ("# t\n\n| | result | status |\n|---|---|---|\n"
            "| a (`e4b.x.a`) | \\|Δ\\| 155 tok/s | measured |\n\n| x | y |\n|---|---|\n| 1 | 2 |\n")
    tables = crc.claim_tables(crc.parse_tables(text))
    assert len(tables) == 1 and tables[0][1] == 2
    (_, cells), = tables[0][0]["rows"]
    assert cells[1] == "|Δ| 155 tok/s"
    assert crc.check_tables(text, CLAIMS) == []


def test_ids_outside_tables_must_exist_and_inactive_ones_must_say_so():
    text = "see `e4b.x.a` and `e4b.x.old` (superseded)\nbut `e4b.x.gone` is quoted as current\n`e4b.x.nope`\n"
    f = crc.check_ids_outside_tables(text, CLAIMS)
    assert any("README.md:2: `e4b.x.gone` is retired" in x for x in f)
    assert any("README.md:3: `e4b.x.nope` is not in" in x for x in f)
    assert not any("e4b.x.old" in x for x in f)


# ------------------------------------------------------- the release block --

def _fake_repo(tmp_path, changelog, version="0.35.0"):
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "experts4bit-qlora"\nversion = "{version}"\n'
        '[project.urls]\nSource = "https://github.com/o/r"\n')
    (tmp_path / "CHANGELOG.md").write_text(changelog)
    return tmp_path


def test_released_version_is_the_top_release_heading_skipping_unreleased(tmp_path):
    root = _fake_repo(tmp_path, "# Changelog\n\n## Unreleased\n\n- x\n\n## 0.35.0 — 2026-09-04\n\n## 0.34.0 — 2026-09-04\n")
    assert clc.released_version(str(root)) == "0.35.0"


def test_released_version_accepts_a_titled_heading_and_rejects_non_release_headings(tmp_path):
    root = _fake_repo(tmp_path, "## 0.35.1 — 2026-09-05 — documentation: tp1\n\n## 0.35.0 — 2026-09-04\n", "0.35.1")
    assert clc.released_version(str(root)) == "0.35.1"
    root = _fake_repo(tmp_path, "## Corrections to 0.30.0 — 2026-09-03 (same day)\n", "0.30.0")
    with pytest.raises(clc.ReleaseVersionError):
        clc.released_version(str(root))


def test_released_version_refuses_a_changelog_pyproject_mismatch(tmp_path):
    root = _fake_repo(tmp_path, "## 0.35.0 — 2026-09-04\n", version="0.35.1")
    with pytest.raises(clc.ReleaseVersionError, match="0.35.0 but pyproject.toml says 0.35.1"):
        clc.released_version(str(root))


def test_release_block_is_generated_checked_and_rewritten(tmp_path):
    block = crc.render_release_block("0.35.0", "o/r", "experts4bit-qlora")
    assert block.startswith(crc.BLOCK_START) and block.endswith(crc.BLOCK_END)
    assert "https://pypi.org/project/experts4bit-qlora/0.35.0/" in block
    assert "https://github.com/o/r/blob/main/docs/STATUS.md" in block
    assert "https://github.com/o/r/tree/v0.35.0/docs" in block
    text = "# x\n\n" + crc.BLOCK_START + "\nstale, typed by hand\n" + crc.BLOCK_END + "\n\nbody\n"
    assert crc.check_release_block(text, block) == [
        "README.md: the release block differs from the generated one -- run "
        "`python scripts/check_readme_claims.py --write-release-block` (never edit it by hand)"]
    new = crc.replace_block(text, block)
    assert crc.check_release_block(new, block) == []
    assert new.endswith("\n\nbody\n") and new.startswith("# x\n\n")
    assert crc.check_release_block("no markers", block)[0].startswith("README.md: no ")
    with pytest.raises(crc.ContractError):
        crc.replace_block("no markers", block)


# ------------------------------------------------------- the repository --

def test_the_repository_readme_is_current_main():
    """The CI gate itself: README numbers are docs/claims.json's current values
    and the release block is CHANGELOG.md's latest release."""
    assert crc.check(ROOT) == []


def test_value_numbers_keep_minus_signs_in_string_values():
    """A register value stored as text ("-0.0528 wikitext / -0.0662 c4val1") must yield signed numbers,
    so a README row quoting −0.0528 matches instead of reporting drift (Bugbot on #401)."""
    import importlib
    m = importlib.import_module("check_readme_claims")
    got = m.value_numbers({"value": "-0.0528 wikitext / -0.0662 c4val1"})
    assert [str(x) for x in got] == ["-0.0528", "-0.0662"], got
    assert m.number_matches(m.Decimal("-0.0528"), got)
    assert not m.number_matches(m.Decimal("0.0528"), got)
