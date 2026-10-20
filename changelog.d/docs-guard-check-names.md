### Docs: AGENTS.md names the required guard checks by their new names

AGENTS.md now names the two required guard checks, `private-marker-guard` and `conflict-marker-guard` (from #1903), in
place of the old `guard` job. Each runs on every push and pull-request event, not on a label, and push-scan runs
("(push scan)") are not required. It also says that a label event on a pull request that does not carry
`ready-to-merge` now fails the required CI jobs instead of skipping them, while on one that does, any label re-runs CI
on its head.
