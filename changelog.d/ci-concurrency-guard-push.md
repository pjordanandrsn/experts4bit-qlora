### CI: one concurrency group per lane; the conflict-marker guard runs on main pushes and pull requests

`ci.yml` now collapses pending runs on `main` (a started run is never cancelled) and cancels a pull request's
superseded run. The conflict-marker guard no longer runs on pushes to branches other than `main`; every pull
request head is still scanned. `tests/test_ci_workflow_shape.py` pins both, and fails if a required CI job regains
a job-level label `if:` (a skipped required check counts as passing). The release guide adds a CI run at the tag.
