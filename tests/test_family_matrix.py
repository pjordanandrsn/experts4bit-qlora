"""tools/family_matrix.py: docs/FAMILIES.md is generated from docs/capabilities.json and docs/claims.json, never
hand-kept (map 3, 2026-10-22). These pin the page against the registers, the repository -> family mapping, and what
counts as current evidence."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("family_matrix", ROOT / "tools" / "family_matrix.py")
fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm)


def test_the_committed_page_is_what_the_registers_say():
    assert fm.main(["--check"]) == 0, "run `python tools/family_matrix.py` and commit docs/FAMILIES.md"


def test_a_stale_page_fails_the_check(tmp_path, monkeypatch, capsys):
    stale = tmp_path / "FAMILIES.md"
    stale.write_text("# hand-edited\n", encoding="utf-8")
    monkeypatch.setattr(fm, "OUT", stale)
    monkeypatch.setattr(fm, "ROOT", tmp_path)
    assert fm.main(["--check"]) == 1 and "is not what the registers say" in capsys.readouterr().err


def test_a_crlf_page_fails_the_check_and_a_rewrite_is_lf(tmp_path, monkeypatch):
    out = tmp_path / "FAMILIES.md"
    monkeypatch.setattr(fm, "OUT", out)
    monkeypatch.setattr(fm, "ROOT", tmp_path)
    assert fm.main([]) == 0 and b"\r" not in out.read_bytes()
    assert fm.main(["--check"]) == 0
    out.write_bytes(out.read_bytes().replace(b"\n", b"\r\n"))       # what a Windows checkout or editor leaves
    assert fm.main(["--check"]) == 1


def test_a_claims_model_maps_to_every_family_it_names_and_nothing_else():
    assert fm.families_of("Qwen/Qwen3-30B-A3B @ ad44e777bcd1") == ["qwen3_moe"]
    assert fm.families_of("Qwen/Qwen3-30B-A3B (GPTQ-Int4 on the vLLM side)") == ["qwen3_moe"]
    assert fm.families_of("Qwen/Qwen3.6-35B-A3B @995ad96e") == ["qwen3_5_moe"]
    assert fm.families_of("ibm-granite/granite-3.1-3b-a800m-instruct; mistralai/Mixtral-8x7B-Instruct-v0.1") == [
        "granitemoe", "mixtral"]
    assert fm.families_of("ibm-granite/granite-4.0-h-tiny @791e0d3d") == ["granitemoehybrid"]
    assert fm.families_of("openai/gpt-oss-20b; Qwen/Qwen3-30B-A3B") == ["gpt_oss", "qwen3_moe"]       # as named
    for unmapped in (None, "", "moonshotai/Kimi-K3", "synthetic Qwen3-14B, Llama-3.1-8B", "VRAM/DRAM tiers",
                     "mistralai/Mistral-7B-v0.1"):
        assert fm.families_of(unmapped) == [], unmapped


def _registers(claims):
    caps = {"capabilities": [
        {"id": "qlora", "status": "supported", "model_families": ["olmoe", "mixtral"], "claim_ids": ["1", "2", "4"],
         "training_support": {"headline_path": "fast_train", "by_model_type": {
             "olmoe": {"fast_train": {"status": "supported", "claim_ids": ["a", "b"]}, "nvme_train": {"status": "not_tested"}},
             "mixtral": {"fast_train": {"status": "refused", "claim_ids": ["c"]},
                         "nvme_train": {"status": "experimental", "claim_ids": []}}}}},
        {"id": "serve", "status": "experimental", "model_families": ["mixtral"]}]}
    return caps, {"claims": claims}


def test_only_current_claims_count_and_a_private_receipt_is_counted_apart():
    claims = [
        {"id": "1", "area": "train", "model": "allenai/OLMoE-1B-7B", "status": "measured", "measured_on": "2026-10-01"},
        {"id": "2", "area": "train", "model": "allenai/OLMoE-1B-7B", "status": "verified", "measured_on": "2026-10-03"},
        {"id": "3", "area": "train", "model": "allenai/OLMoE-1B-7B", "status": "measured-private", "measured_on": "2026-10-09"},
        {"id": "4", "area": "train", "model": "allenai/OLMoE-1B-7B", "status": "superseded", "measured_on": "2026-10-20"},
        {"id": "5", "area": "serve", "model": "allenai/OLMoE-1B-7B", "status": "retired", "measured_on": "2026-10-21"},
        {"id": "6", "area": "serve", "model": "allenai/OLMoE-1B-7B", "status": "open"},
        {"id": "7", "area": "serve", "model": "moonshotai/Kimi-K3", "status": "measured"},
        {"id": "8", "area": "serve", "model": None, "status": "measured"},
    ]
    page = fm.render(*_registers(claims))
    assert "| `olmoe` | - | 2 +1 private | 2026-10-09 |" in page          # serve: none current; train: 2 + 1 private
    assert "| moonshotai/Kimi-K3 | 1 |" in page and "| (no `model`) | 1 |" in page
    assert "| `olmoe` | - (0)" not in page


def test_training_cells_and_capability_listing_come_from_the_capabilities_register():
    page = fm.render(*_registers([]))
    assert "| `olmoe` | supported (2) | not_tested |" in page
    assert "| `mixtral` | refused (1) | experimental, no claim id |" in page     # credits with nothing behind it
    assert "`serve` (experimental)" in page
    assert "| `olmoe` | listed, no claim id | - |" in page                      # no claims on record at all
    assert "| `mixtral` | listed, no claim id | listed, no claim id |" in page
    assert "## Claims not mapped to a family" in page and "None." in page


def test_a_listing_counts_only_the_capabilitys_own_current_claims_that_name_the_family():
    claims = [
        {"id": "1", "area": "train", "model": "allenai/OLMoE-1B-7B", "status": "measured"},
        {"id": "2", "area": "train", "model": "allenai/OLMoE-1B-7B; mistralai/Mixtral-8x7B", "status": "measured-private"},
        {"id": "3", "area": "train", "model": "mistralai/Mixtral-8x7B", "status": "measured"},     # not qlora's id
        {"id": "4", "area": "train", "model": "mistralai/Mixtral-8x7B", "status": "superseded"},  # qlora's, not current
    ]
    page = fm.render(*_registers(claims))
    assert "| `olmoe` | listed (2) | - |" in page
    assert "| `mixtral` | listed (1) | listed, no claim id |" in page


def test_the_page_is_deterministic():
    caps, claims = _registers([{"id": "1", "area": "train", "model": "allenai/OLMoE-1B-7B", "status": "measured"}])
    assert fm.render(caps, claims) == fm.render(caps, claims)


def _table(page, heading):
    """The data rows of the table under ``heading``, as lists of cells (family first)."""
    body = page.split(heading, 1)[1].split("\n## ", 1)[0]
    return [[c.strip() for c in ln.strip("|").split(" | ")] for ln in body.splitlines() if ln.startswith("| `")]


def test_on_the_real_registers_every_crediting_cell_shows_its_evidence_or_says_it_has_none():
    caps = json.loads(fm.CAPABILITIES.read_text(encoding="utf-8"))
    page = fm.render(caps, json.loads(fm.CLAIMS.read_text(encoding="utf-8")))
    listing = [cell for row in _table(page, "## Capabilities that list each family") for cell in row[1:]]
    assert listing and all(re.fullmatch(r"-|listed \(\d+\)|listed, no claim id", c) for c in listing), listing
    unbacked = sum(c == "listed, no claim id" for c in listing)
    assert unbacked == sum(c != "-" for c in listing) - sum(c.startswith("listed (") for c in listing) and unbacked > 0
    training = []
    for c in caps["capabilities"]:
        if c.get("training_support"):
            training += [cell for row in _table(page, f"## Training paths: `{c['id']}`") for cell in row[1:]]
    assert training
    for cell in training:
        status = re.split(r" \(|, ", cell, maxsplit=1)[0]
        if cell == "-" or status in fm.NO_CREDIT:
            continue
        assert re.fullmatch(r"[\w-]+ \(\d+\)|[\w-]+, no claim id", cell), cell
