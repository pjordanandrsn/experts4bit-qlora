"""Guards for two defects found by running the stack on a rented RTX 5090.

1. __version__ had drifted to 0.6.3 while the distribution shipped 0.6.4.
   Nothing tested it, so anything logging __version__ into a receipt recorded
   the wrong version -- a provenance bug, not a cosmetic one.

2. `python -m experts4bit_qlora.train --help` had no argparse at all, so argv
   was ignored and it fell through into a real run: model load, CUDA init and
   32 inductor compile workers. ~10 min and 6.2 GB of VRAM to learn there is
   no such flag. --help must answer without LOADING A MODEL or initialising
   CUDA. (It still imports torch/bitsandbytes: the package __init__ is eager.
   That costs seconds, not minutes, and is out of scope here.)
"""
import subprocess
import sys


def test_version_matches_distribution_metadata():
    import importlib.metadata as md
    import experts4bit_qlora
    assert experts4bit_qlora.__version__ == md.version("experts4bit-qlora")


def test_help_exits_without_loading_a_model():
    for mod in ("experts4bit_qlora.train", "experts4bit_qlora.infer"):
        import os
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable, "-m", mod, "--help"],
                           capture_output=True, text=True, timeout=300, env=env)
        assert r.returncode == 0, f"{mod} --help rc={r.returncode}: {r.stderr[-400:]}"
        assert "ENVIRONMENT VARIABLE" in r.stdout, f"{mod} --help printed no usage"
        # the tell-tale of the old behaviour: a real run logs the load line
        assert "streaming 4-bit loader" not in r.stdout, f"{mod} --help still loads a model"


def test_changelog_documents_the_released_version():
    """The shipped `__version__` must have its own CHANGELOG section.

    0.14.0 and 0.15.0 both shipped with no entry — the changelog stopped at 0.13.0
    and nothing complained, so the release notes silently described a two-versions-old
    package. Unreleased work belongs under `## Unreleased`; bumping `__version__`
    without giving it a section fails here instead of at publish time.
    """
    import pathlib
    import experts4bit_qlora

    v = experts4bit_qlora.__version__
    text = (pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    heads = [ln.strip() for ln in text.splitlines() if ln.startswith("## ")]
    # Compare the version TOKEN, not a prefix: `startswith("## 0.15.0")` also accepts
    # "## 0.15.0-notaversion", and a version of "0.1" would match "## 0.15.0".
    versions = [h.split()[1] for h in heads if len(h.split()) > 1]
    assert v in versions, (
        f"__version__ is {v} but CHANGELOG.md has no '## {v}' section. Sections "
        f"present: {versions[:5]}. Add the release notes, or keep in-flight work under "
        f"'## Unreleased' until the version is bumped."
    )
