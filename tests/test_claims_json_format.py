"""docs/claims.json keeps one serialisation: json.dumps(indent=2, ensure_ascii=False) plus a newline.

#810 re-wrote the register at indent 1. The content was unchanged, but every line moved, so every
open pull request that touched the register conflicted on all of it. Whatever writes the register
writes it in this form:

    path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\\n")
"""
from __future__ import annotations

import json
from pathlib import Path

REGISTER = Path(__file__).resolve().parents[1] / "docs" / "claims.json"


def test_register_is_in_its_canonical_serialisation():
    text = REGISTER.read_text(encoding="utf-8")
    canonical = json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n"
    assert text == canonical, (
        "docs/claims.json is not in its canonical form; re-write it with "
        "json.dumps(reg, indent=2, ensure_ascii=False) + '\\n'")
