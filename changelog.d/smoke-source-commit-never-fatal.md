### GPU serve smoke: the source commit is reporting-only and never aborts a cell

`bench/smoke/gpu_serve_smoke.py`'s `source_identity()` ran `git rev-parse HEAD` unguarded. On an image without git (the
`pytorch/pytorch` CUDA images) every cell raised `FileNotFoundError`, and the run said nothing about the code under test.
That happened on #1625's A2000 smoke.

The commit now comes from git, else `SMOKE_SOURCE_SHA` from the caller's environment (not an `E4B_*` name, which the suite
strips from its workers), else the installed package's PEP 610 `direct_url.json`. `git_head_source` records which one, or
`none`. The installed-version lookups are guarded the same way. Five CPU tests cover each source and the never-raise
contract.
