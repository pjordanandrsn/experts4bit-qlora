### Tooling: `check_change_impact.py` no longer passes when it checked nothing

Synced byte for byte from grouped-nf4-gemm. Without `--base` it used to print SKIP and exit 0, so a local review that
ran it bare read as clean while CI, which passes the pull request's base, failed the same diff. It now diffs against
`origin/main` by default and says so. When that ref does not resolve, or `--base` is given empty, it FAILs with exit 2
and says what to pass. CI is unchanged: it always passes the pull request's base sha.
