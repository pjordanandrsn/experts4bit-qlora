# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The support doc must not drift from the probe rows, and rows must be well-formed.

``docs/ARCHITECTURE_SUPPORT.md`` went stale for a month for a structural reason,
not a careless one: it was hand-maintained, and nothing in the tree failed when
it stopped matching reality. A generated block fixes that only if something
notices when the block and the rows disagree — otherwise regenerating is just
another step someone can forget, exactly like re-running the sweep was.

These tests need no GPU, no checkpoint and no network: the rows are committed
JSON, so this runs in ordinary CPU CI. That is deliberate — a guard that only
runs where the hardware is would not run at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "support"))
ROWS = ROOT / "bench" / "support" / "rows"


@pytest.fixture(scope="module")
def reducer():
    import reduce_support
    return reduce_support


def test_every_row_parses_and_carries_what_the_table_needs(reducer):
    """A row missing model_type silently vanishes from the claimed-vs-evidenced
    table -- it would be counted as no evidence for a family it actually tested."""
    files = sorted(ROWS.glob("*.json"))
    assert files, f"no probe rows under {ROWS}"
    for p in files:
        d = json.loads(p.read_text())
        for key in ("model", "model_type", "device", "versions", "stages", "exit_code",
                    "source", "provenance"):
            assert key in d, f"{p.name} has no {key!r}"
        assert d["stages"], f"{p.name} records no stages"
        # A graded row must be gradeable: the grader keys off exit_code and the
        # load/verify/forward statuses, so those must exist whenever the run got
        # past its preconditions.
        if d["exit_code"] == 0:
            for stage in ("load", "verify", "forward"):
                assert stage in d["stages"], f"{p.name} exited 0 without a {stage} stage"


def test_the_generated_doc_block_is_what_the_rows_produce(reducer):
    """Regenerating must be a no-op. If it is not, someone added a row (or edited
    SUPPORTED_ARCHITECTURES) and did not re-run the reducer, so the doc is once
    again describing a state that no longer holds."""
    doc = reducer.DOC.read_text()
    assert reducer.BEGIN in doc and reducer.END in doc, (
        "the generated block markers are missing from docs/ARCHITECTURE_SUPPORT.md -- "
        "run `python bench/support/reduce_support.py --write-doc`")
    start = doc.index(reducer.BEGIN)
    end = doc.index(reducer.END) + len(reducer.END)
    on_disk = doc[start:end]
    expected = reducer._generated_block(reducer._load_rows())
    assert on_disk == expected, (
        "docs/ARCHITECTURE_SUPPORT.md's generated block is not what the rows produce -- "
        "run `python bench/support/reduce_support.py --write-doc` and commit the result")


def test_the_hand_written_evidence_scope_survives(reducer):
    """The generated block is additive. The doc's own taxonomy and its earned
    warning -- an earlier sweep called four families broken and three of the four
    loaded fine -- are the parts a reader needs to interpret any row at all, and
    a writer that clobbered them would be worse than the staleness it fixes."""
    doc = reducer.DOC.read_text()
    assert "A fixture is not a checkpoint" in doc
    assert "Evidence scope" in doc
    head = doc[:doc.index(reducer.BEGIN)]
    assert "A fixture is not a checkpoint" in head, (
        "the earned warning must stay ABOVE the generated block, where it is read first")


def test_the_grader_ranks_evidence_the_way_the_doc_claims(reducer):
    """The ranking is an opinion, so pin it: a reference-tier pass must outrank a
    toy pass, and 'none' must be the weakest. A ranking that put `blocked` above
    `toy-ok` would let a family with a too-small fixture look better evidenced
    than one that actually loaded."""
    r = reducer.RANK
    assert r.index("reference-ok") > r.index("toy-ok") > r.index("blocked")
    assert r.index("blocked") > r.index("refused") > r.index("error") > r.index("none")
    assert r[0] == "none"


def test_a_blocked_row_is_not_graded_as_a_support_failure(reducer):
    """Exit 4 (checkpoint too small) must not grade as `refused` or `error`.
    Collapsing them is the specific misreading this probe exists to prevent."""
    assert reducer._grade({"exit_code": 4, "stages": {}}) == "blocked"
    assert reducer._grade({"exit_code": 3, "stages": {}}) == "refused"
    assert reducer._grade({"exit_code": 6, "stages": {}}) == "error"


def test_a_broken_copy_is_not_graded_as_a_family_fault(reducer):
    """Exit 5 means a shard was short of its own declared length -- the transfer
    failed. Grading that as `error` would file a network problem as a support
    problem, and it must not outrank a real fault or a real pass."""
    assert reducer._grade({"exit_code": 5, "stages": {}}) == "copy-broken"
    r = reducer.RANK
    assert r.index("copy-broken") > r.index("none")
    assert r.index("copy-broken") < r.index("error") < r.index("reference-ok")


def test_baseline_if_present_covers_every_claimed_family(reducer):
    """A claimed family absent from the baseline is how a new unevidenced claim
    would slip in -- `--check` treats it as a failure, so the committed baseline
    must stay in step with the shipped list."""
    if not reducer.BASELINE.exists():
        pytest.skip("no committed baseline yet")
    baseline = json.loads(reducer.BASELINE.read_text())
    claimed = reducer._claimed()
    missing = sorted(set(claimed) - set(baseline))
    assert not missing, (
        f"claimed but absent from {reducer.BASELINE.name}: {missing} -- run "
        f"`python bench/support/reduce_support.py --check` and commit the baseline")


def test_the_integrity_check_catches_a_truncated_shard(tmp_path):
    """Calibrate it in the direction that matters.

    This check exists because a size-stability heuristic let a probe start
    against a checkpoint that was still being copied. A check that only ever
    passes would have been no better, so prove it fails on a shard that is short
    of the length its own header declares -- and passes on one that is not.
    """
    import struct
    sys.path.insert(0, str(ROOT / "bench" / "support"))
    from support_probe import _safetensors_integrity

    header = {"t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
    blob = json.dumps(header).encode()
    good = tmp_path / "model-00001-of-00001.safetensors"
    good.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 16)
    assert _safetensors_integrity(tmp_path)["status"] == "ok"

    # Drop four bytes of tensor data: the file is still a valid header with a
    # parseable JSON body, which is exactly why a listing cannot tell.
    good.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 12)
    bad = _safetensors_integrity(tmp_path)
    assert bad["status"] == "error"
    assert "short by 4" in bad["bad"][0]["why"], bad


def test_the_integrity_check_says_not_tested_with_no_shards(tmp_path):
    """An empty directory must not read as verified."""
    sys.path.insert(0, str(ROOT / "bench" / "support"))
    from support_probe import _safetensors_integrity
    assert _safetensors_integrity(tmp_path)["status"] == "not_tested"


def test_a_deliberate_loader_refusal_is_not_classified_as_a_fault():
    """The loader raises bare RuntimeError for both its most deliberate refusal
    and for real faults, so the probe cannot classify by type alone.

    `loader.py:1152` refuses a checkpoint with no fused expert stacks -- "Refusing
    to return a model with zero quantized expert layers -- silently skipping the
    experts is the exact failure this loader exists to prevent". That guard
    firing on a dense checkpoint is the loader WORKING, and grading it as a crash
    would be exactly backwards. Meanwhile "unmaterialized meta tensors remain" is
    the same exception type and is a genuine fault.

    Pinned here because the classification is message-anchored and therefore
    brittle: if e4b grows a dedicated refusal exception, this test is where the
    simplification lands.
    """
    src = (ROOT / "bench" / "support" / "support_probe.py").read_text()
    assert 'REFUSAL_MARKERS = ("Refusing", "no fused expert stacks found")' in src

    # The loader's own wording, quoted from loader.py so the two cannot drift
    # apart silently without this failing.
    loader = (ROOT / "experts4bit_qlora" / "loader.py").read_text()
    assert "no fused expert stacks found in" in loader, (
        "the zero-expert-stacks guard no longer uses the wording the probe keys on -- "
        "re-check REFUSAL_MARKERS in bench/support/support_probe.py")
    assert "Refusing to return a model with zero quantized expert" in loader


def test_a_host_that_could_not_finish_is_not_a_family_verdict(reducer):
    """Exit 9 is the stub lan_queue.sh writes when the probe produced no row at
    all -- the OS killed it, most likely an OOM on one of the large checkpoints.

    That must not grade as `error` or `refused`, which are statements about the
    code, and must not grade as `none`, which would make an attempt that was
    actually made look like one that never happened. The whole harness exists to
    stop absence reading as untested; the largest models are exactly where that
    would matter most.
    """
    assert reducer._grade({"exit_code": 9, "stages": {"load": {"status": "error"}}}) == "host-limited"
    r = reducer.RANK
    assert r.index("none") < r.index("host-limited") < r.index("error")
    assert r.index("host-limited") < r.index("reference-ok")
