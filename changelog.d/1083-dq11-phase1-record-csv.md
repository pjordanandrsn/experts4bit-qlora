### DQ11 MLP Phase 1 runner: wheel RECORDs parsed as CSV (#1083)

`dq11_mlp_phase1_run.py` compares each locked wheel's RECORD with the installed one. It split RECORD lines by hand, but
RECORD is CSV, and torch's wheel quotes every path (`"functorch/__init__.py",sha256=...`) while pip writes the
installed copy unquoted. So torch's wheel side read as empty, and the check refused a correct install on the A2000. It
failed closed: no read ran. The runner now parses RECORD with `csv`. A test covers a quoted RECORD against its unquoted
install.
