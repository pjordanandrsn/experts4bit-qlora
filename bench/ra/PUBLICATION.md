# Published release binding

Run `ra_publication.py` with `python -I -S -B`, `--manifest` pointing to the
source-metadata pair manifest, and `--out` a fresh absolute receipt directory.
The gate contacts only fixed GitHub/PyPI HTTPS authorities, refuses redirects
and proxies, bounds responses, and retains raw response bytes and clock reads.
It never downloads or executes a release from the network.

For each source-verified project it requires a published final GitHub release,
an annotated signed tag targeting the exact commit, GitHub's reported valid
verification matching local tag bytes, and one unyanked PyPI wheel with the
retained filename, SHA-256 and size. Runtime and dependency-source binding runs
between two publication observations; changed identities or local inputs refuse.
Failed attempts retain collected observations and a failure record.

This binds published indexes to retained artifacts and source. GitHub reports
signature validity; local key trust/signature verification, reproducible builds,
launch authorization, installation, imports and GPU engagement remain separate.
Synthetic authority responses are CPU controls, not live publication evidence.
