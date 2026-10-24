"""docs/claims.json cites a rented Vast host by its machine id, never by an instance id.

An instance id names one rental and means nothing once it is destroyed. The machine id names the host, which is what
the host registry and a reproducer can use. Text fields read "Vast machine 114299" (or "machine 114299"). A host
whose machine id was never recorded reads "Vast box S<n>", a stable label with the same n wherever it appears.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REGISTER = Path(__file__).resolve().parents[1] / "docs" / "claims.json"
TEXT_FIELDS = ("claim", "hardware", "notes")
# "Vast[.ai] [instance|box] N", "instance N" or "box N", with a 7-9 digit N
INSTANCE_ID = re.compile(r"\b(?:Vast(?:\.ai)?\s+(?:(?:instance|box)\s*#?\s*)?|(?:instance|box)\s*#?\s*)\d{7,9}\b", re.I)


@pytest.mark.parametrize("text", [
    "one rented Vast.ai host (instance 52069847)",
    "RTX 5090, rented (Vast box 49785896)",
    "on an EPYC host, Vast.ai instance 49975016, image",
    "rented hosts (vast 51566878 / machine 59197)",
    "same box, Vast 49975016",
    "bo7 (2026-09-05, box 49916675, an RTX 5090",
])
def test_the_pattern_finds_every_instance_form(text):
    assert INSTANCE_ID.search(text)


@pytest.mark.parametrize("text", [
    "one rented Vast.ai host (machine 114299)",
    "RTX 5090, rented (Vast box S2)",
    "rented hosts (Vast machine 34261 / machine 59197)",
    "trainable 2621440 != e4b's 49807360",
    "a 4-box run on 16 instances",
])
def test_the_pattern_passes_machine_ids_labels_and_other_numbers(text):
    assert not INSTANCE_ID.search(text)


def test_no_claim_text_cites_a_vast_instance_id():
    claims = json.loads(REGISTER.read_text(encoding="utf-8"))["claims"]
    found = [(c["id"], field, m.group(0))
             for c in claims for field in TEXT_FIELDS
             for m in INSTANCE_ID.finditer(c.get(field) or "")]
    assert not found, f"cite the machine id, not the instance id: {found[:5]} ({len(found)} in all)"
