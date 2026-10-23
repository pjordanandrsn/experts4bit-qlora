### SC2f Amendment 4, SC2h Amendment 2, SC2i Amendment 1: the launch commit is a `main` commit at or after `1a25b209`

- The three lanes named their launch commit as the first `main` commit containing both `v0.57.0` and e4b#2011. Read
  literally, that commit predates the lanes' own kit (box N's and box O's kernels pin, the default check, box P). Before
  any run, each lane now names a `main` commit at or after `1a25b209`, the merge that holds all of it. The launch-time
  default check against `v0.57.0` is recorded with each run, as before. No constant, threshold or cost changes.
- The lanes' kit is held at `1a25b209`: at the launch commit, every `bench/sc1/staged.sha256` entry a lane's box stages,
  the shared `sc1_run.sh` included, must be byte-identical to that entry at `1a25b209`. Any difference is its own pre-run
  amendment.
