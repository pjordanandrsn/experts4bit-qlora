### Release: publishing needs a successful CI run at the tag's commit

The release workflow's publish job now checks, before the PyPI upload, that `ci.yml` has a successful push or
dispatch run on exactly the tagged commit (a pull_request run tested the merge ref, so it does not count), and stops with the command to run if it has not. With pending runs on `main`
collapsed, a tagged commit can lack its own run; the release guide's step (dispatch CI at the tag) is now enforced.
