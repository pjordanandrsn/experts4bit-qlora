### CI: a `maintainer-hold` required check -- a PR labelled `hold` cannot merge by any route

- `.github/workflows/maintainer-hold.yml` fails while the PR carries the `hold` label and passes otherwise; it re-runs on
  every label change and push. It is a required check on `main`, whose protection has `enforce_admins`, so a held PR
  cannot merge by auto-merge, a direct merge or an admin merge until the label comes off.
- Why: removing `ready-to-merge` and disabling auto-merge did not hold a PR once its CI was green, and four PRs were merged
  directly over open change requests on 2026-10-06. Every actor is one GitHub account, so GitHub's review requirement cannot
  express "not until reviewed". Added at Jordan's direction.
