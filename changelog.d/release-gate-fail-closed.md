### Release: the CI-green gate fails closed

`release.yml`'s gate used to publish when the API count wasn't an integer (empty or `null`): the old `-lt 1` test
errored inside its `if`, and the job went on. It now refuses, and so does a failed `gh` call. It queries `ci.yml`'s
successful runs at the checked-out commit (`git rev-parse HEAD`), counting only push and dispatch runs, and its refusal
says whether to wait for a running CI or dispatch one at the tag. It now runs first after checkout, before setup, the
build and the smoke's installs, since the job holds `id-token: write` throughout. `tests/test_ci_workflow_shape.py` runs the step
against a fake `gh` over a tagged repo: push or dispatch success passes; a failed push run, pull-request runs only, a
green run at another commit, no runs, a `gh` failure, and an unreadable count all refuse.
