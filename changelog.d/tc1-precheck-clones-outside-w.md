### TC1: source checkouts sit beside the run's work dir, never in it

- `bench/tc1/tc1_run.sh`: amendment 76's pre-check clones grouped-nf4-gemm beside the work dir (`gnf4-pos76`) instead of into it, and
  removes the clone and its pytest dir after the pre-check. Amendment 46's decoded-route gate checks out beside the work dir too
  (`gnf4-decgate`); it was already removed after the gate and excluded from the fetch. The fetch copies the work dir, and a checkout
  inside it can carry a `.git` or a nested receipt into the receipt store.
- `tests/test_tc1_clone_targets.py`: every `git clone` target, every `git -C DIR checkout` and every `git worktree add` path in
  `tc1_run.sh` must resolve outside the work dir. Quotes are removed; variables expand recursively from their nearest assignment;
  `${NAME:-X}` is judged as X and as NAME's own assignment; a relative target counts as inside (the runner runs from the work dir);
  an unresolvable target fails the test.
