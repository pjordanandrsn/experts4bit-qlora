### `scripts/changelog_fragments.py` joins the shared tooling (tooling only)

- grouped-nf4-gemm moved its own `## Unreleased` to `changelog.d/` fragments and took `scripts/changelog_fragments.py` as
  shared tooling, with itself as upstream. This repository's copy was already byte-identical. `scripts/check_shared_tooling.py`
  gains the entry, so CI now holds the two copies equal (15 shared files).
