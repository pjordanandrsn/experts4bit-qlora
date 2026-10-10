# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""SC5, read: the committed receipts of sc5-5090-2 are intact, main's sc5_reduce.py turns the committed record into the
committed verdict, and bench/sc5/README.md's counts, losing cells and quality rows are the verdict's."""
import hashlib
import importlib.util
import json
import pathlib
import re

SC5 = pathlib.Path(__file__).resolve().parents[1] / "bench" / "sc5"
R = SC5 / "receipts" / "sc5-5090-2"
README = (SC5 / "README.md").read_text(encoding="utf-8")


def _reduce():
    spec = importlib.util.spec_from_file_location("sc5_reduce", SC5 / "sc5_reduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verdict():
    return json.loads((R / "verdict.json").read_text(encoding="utf-8"))


def test_every_promoted_file_matches_sha256sums():
    rows = [ln.split("  ", 1) for ln in (R / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if ln.strip()]
    listed = {name for _sha, name in rows}
    on_disk = {p.relative_to(R).as_posix() for p in R.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    assert listed == on_disk
    for sha, name in rows:
        assert hashlib.sha256((R / name).read_bytes()).hexdigest() == sha, name


def test_main_reducer_reproduces_the_committed_verdict():
    rec = json.loads((R / "sc5.json").read_text(encoding="utf-8"))
    assert _reduce().reduce_obj(rec) == _verdict()


def test_the_readme_reports_the_verdict_and_lists_every_losing_cell():
    v = _verdict()
    c = v["counts"]
    assert v["verdict"] == "READ" and not v["quality_void"]
    assert f"LEADS {c['LEADS']}, TRAILS {c['TRAILS']}, WITHIN NOISE {c['WITHIN NOISE']}" in README
    losing = README[README.index("### Cells where e4b trails"):README.index("### Cells where e4b leads")]
    listed = sum(len(ln.split(": ", 1)[1].split(", ")) for ln in losing.splitlines() if ln.startswith("- "))
    assert listed == len(v["losing_cells"]) == c["TRAILS"]
    for key, row in v["quality"].items():
        assert f"{row['nll_delta']:+.4f}" in README and f"{row['argmax_agreement']:.4f}" in README, key
    assert "not a ranking of correctness" in README and "not diagnosed" in README
    assert not re.search(r"\b(fastest|best|winner|wins)\b", README, re.I)      # no ranking sentence
