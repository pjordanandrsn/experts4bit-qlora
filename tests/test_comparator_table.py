"""tools/comparator_table.py: docs/COMPARATORS.md lists the comparator versions the current claims name, generated from
docs/claims.json (map 5D, 2026-10-22). These pin the page, the version shapes, and what counts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("comparator_table", ROOT / "tools" / "comparator_table.py")
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)


def _reg(*rows):
    return {"claims": [dict({"status": "measured", "area": "serve", "measured_on": "2026-10-01"}, **r) for r in rows]}


def test_the_committed_page_is_what_the_register_says():
    assert ct.main(["--check"]) == 0, "run `python tools/comparator_table.py` and commit docs/COMPARATORS.md"


def test_a_stale_page_fails_the_check(tmp_path, monkeypatch, capsys):
    stale = tmp_path / "COMPARATORS.md"
    stale.write_text("# hand-edited\n", encoding="utf-8")
    monkeypatch.setattr(ct, "OUT", stale)
    monkeypatch.setattr(ct, "ROOT", tmp_path)
    assert ct.main(["--check"]) == 1 and "is not what the register says" in capsys.readouterr().err


def test_a_version_is_read_only_in_the_comparators_release_shape():
    got = ct.read(_reg(
        {"id": "a", "conditions": "vLLM 0.30.0, SGLang 0.5.21 on one RTX 5090"},
        {"id": "b", "conditions": "vllm==0.28.0"},
        {"id": "e4b.serve.h2h.vllm-0.29.0.x", "claim": "beats the comparator"},          # the version in the id
        {"id": "c", "claim": "Unsloth 2026.9.14 trains", "area": "train"},
        {"id": "d", "conditions": "axolotl 0.20.0"},
        {"id": "e", "claim": "Unsloth 0.7566 nats, vLLM 1.0043x"},                       # a loss and a ratio: no version
        {"id": "f", "conditions": "vLLM 1.2.0 and axolotl 1.0.1"},                       # a 1.x release is a version
    ))
    assert sorted(got["vLLM"]["versions"]) == ["0.28.0", "0.29.0", "0.30.0", "1.2.0"]
    assert list(got["SGLang"]["versions"]) == ["0.5.21"]
    assert list(got["Unsloth"]["versions"]) == ["2026.9.14"] and sorted(got["axolotl"]["versions"]) == ["0.20.0", "1.0.1"]
    assert [c["id"] for c in got["Unsloth"]["bare"]] == ["e"] and [c["id"] for c in got["vLLM"]["bare"]] == ["e"]


def test_notes_are_not_read_and_only_current_claims_count():
    got = ct.read(_reg(
        {"id": "a", "conditions": "Unsloth", "notes": "Unsloth 2026.9.2 in the unlicensed arm"},
        {"id": "b", "conditions": "vLLM 0.31.0", "status": "superseded"},
        {"id": "c", "conditions": "vLLM 0.31.0", "status": "retired"},
        {"id": "d", "conditions": "vLLM 0.31.0", "status": "open"},
        {"id": "e", "conditions": "vLLM 0.30.0", "status": "measured-private"},
    ))
    assert got["Unsloth"]["versions"] == {} and [c["id"] for c in got["Unsloth"]["bare"]] == ["a"]
    assert list(got["vLLM"]["versions"]) == ["0.30.0"]


def test_newest_on_record_compares_versions_numerically():
    reg = _reg({"id": "a", "conditions": "vLLM 0.9.0"}, {"id": "b", "conditions": "vLLM 0.30.0"},
               {"id": "c", "conditions": "Unsloth 2026.9.14"}, {"id": "d", "conditions": "Unsloth 2026.10.7"})
    assert ct.newest_on_record(reg) == {"vllm": "0.30.0", "unsloth": "2026.10.7"}


def test_the_page_lists_versions_newest_first_and_bare_claims(capsys):
    page = ct.render(_reg({"id": "a", "conditions": "vLLM 0.28.0"}, {"id": "b", "conditions": "vLLM 0.30.0"},
                          {"id": "c", "conditions": "an SGLang run", "measured_on": "2026-10-09"}))
    assert page.index("| 0.30.0 |") < page.index("| 0.28.0 |")
    assert "| SGLang | `sglang` | none on record | 0 | - | - |" in page
    assert "| SGLang | `c` | 2026-10-09 |" in page


def test_json_is_the_newest_version_per_pypi_name(capsys):
    assert ct.main(["--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert set(got) <= {"vllm", "sglang", "unsloth", "axolotl"} and all(isinstance(v, str) for v in got.values())
