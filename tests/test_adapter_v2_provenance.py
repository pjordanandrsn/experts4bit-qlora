# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``backend-adapter/2``: the reader refuses a manifest that does not name the run that produced the adapter.

``run_id`` must be a non-empty string and ``created_at`` a real UTC time written as ``YYYY-MM-DDTHH:MM:SSZ``: a presence
check alone passed a null. These cases live apart from ``tests/test_adapter_schema.py``, which a lane's kit stages
byte-exact (``bench/p224/staged-p224.sha256``), so adding them leaves that file unchanged.
"""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from experts4bit_qlora import adapter_dir as AD  # noqa: E402
from test_adapter_schema import _v2  # noqa: E402  (the v2 fixture directory: run_id "r", created_at a UTC time)


def _edit(d, f):
    """Apply ``f`` to the written manifest. The manifest itself is not in ``files``, so no checksum moves."""
    p = d / "adapter_manifest.json"
    man = json.loads(p.read_text(encoding="utf-8"))
    f(man)
    p.write_text(json.dumps(man), encoding="utf-8")
    return d


def test_a_named_run_and_a_utc_time_verify(tmp_path):
    ad = AD.verify_adapter_dir(_v2(tmp_path))
    assert ad.manifest["run_id"] == "r" and ad.manifest["created_at"] == "2026-10-21T00:00:00Z"


@pytest.mark.parametrize("edit,needle", [
    # present but empty: a presence check alone would pass each of these
    (lambda m: m.update(run_id=None), "does not name the run"),
    (lambda m: m.update(run_id=""), "does not name the run"),
    (lambda m: m.update(run_id="  "), "does not name the run"),
    (lambda m: m.update(run_id=7), "does not name the run"),
    (lambda m: m.update(created_at=None), "is not a UTC time"),
    (lambda m: m.update(created_at="yesterday"), "is not a UTC time"),
    (lambda m: m.update(created_at="2026-10-21 00:00:00"), "is not a UTC time"),
    (lambda m: m.update(created_at="2026-10-21T00:00:00Z\n"), "is not a UTC time"),     # fullmatch
    (lambda m: m.update(created_at="2026-13-01T00:00:00Z"), "is not a UTC time"),       # parsed, too
    (lambda m: m.update(created_at="2026-1-05T00:00:00Z"), "is not a UTC time"),        # strptime takes it
])
def test_v2_refuses_a_manifest_that_does_not_name_its_run(tmp_path, edit, needle):
    with pytest.raises(AD.AdapterRefused) as e:
        AD.verify_adapter_dir(_edit(_v2(tmp_path), edit))
    assert e.value.code == 7 and needle in str(e.value), str(e.value)
