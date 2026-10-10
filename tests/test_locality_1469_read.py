# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""#1469 item 1, read: the committed census receipts (loc-a4000-1) are in the agreed format, and main's locality_summary.py
reproduces both committed summaries from the committed npz, so every number in the README's results reproduces."""
import gzip
import hashlib
import importlib.util
import json
import pathlib

import pytest

np = pytest.importorskip("numpy")

LANE = pathlib.Path(__file__).resolve().parents[1] / "bench" / "locality-1469"
R = LANE / "receipts" / "loc-a4000-1"
SHAS = {"decode.npz": "f59d7517b7aed61d21f7fc2737ef1db6c6dbbb084122cc5fcf9cc0da801f9148",
        "prefill.npz": "17db4e562ff0c7f0303f578d87f90d1dba0c024ebd312de904c8601eb54e8d10",
        "summary_decode.json": "f9f0febc519b4e3633e0f3b8648d8769f26954abfaf8c6f6f2e675eea482bd11",
        "summary_prefill.json": "11d46d9a296d3b8d79318db09775ba19045aab202085a22311f513f164decd12",
        "manifest.json": "16e604262bd6717feb75c77999f8d543468629ed930ffb3758df2f4490859e48"}


def _summary_module():
    spec = importlib.util.spec_from_file_location("locality_summary", LANE / "locality_summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_committed_receipts_are_the_ones_the_readme_hashes():
    readme = (LANE / "README.md").read_text(encoding="utf-8")
    for name, sha in SHAS.items():
        assert hashlib.sha256((R / name).read_bytes()).hexdigest() == sha, name
        assert sha in readme, name


@pytest.mark.parametrize("phase,tokens,seqs", [("decode", 2048, 4), ("prefill", 10256, 16)])
def test_the_npz_are_in_the_agreed_format(phase, tokens, seqs):
    z = np.load(R / f"{phase}.npz")
    assert set(z.files) == {"expert_ids", "seq_offsets", "labels", "n_experts"}
    ids, off = z["expert_ids"], z["seq_offsets"]
    assert ids.dtype == np.int16 and ids.shape == (tokens, 48, 8) and off.dtype == np.int32
    assert off[0] == 0 and off[-1] == tokens and len(off) == seqs + 1 and len(z["labels"]) == seqs
    assert int(z["n_experts"]) == 128 and ids.min() >= 0 and ids.max() < 128
    srt = np.sort(ids, axis=-1)
    assert not (srt[..., 1:] == srt[..., :-1]).any()            # no expert routed twice in one token's top-8


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_main_summary_reproduces_the_committed_summary(phase, tmp_path):
    out = tmp_path / f"summary_{phase}.json"
    args = ["--npz", str(R / f"{phase}.npz"), "--phase", phase, "--out", str(out)]
    if phase == "decode":
        args += ["--manifest", str(R / "manifest.json")]
    _summary_module().main(args)
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads((R / f"summary_{phase}.json").read_text(encoding="utf-8"))


def test_the_decode_traces_cover_four_kinds_of_512_steps():
    for kind in ("code", "dialogue", "math", "prose"):
        with gzip.open(R / f"decode_{kind}.jsonl.gz", "rt", encoding="utf-8") as f:
            assert sum(1 for _ in f) == 513, kind                  # capture_routing's header line plus 512 steps
