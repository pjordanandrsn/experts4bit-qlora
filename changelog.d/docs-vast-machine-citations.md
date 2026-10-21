### Docs: claims cite a rented host by its machine id

Claim text in `docs/claims.json` named rented Vast hosts by instance id, which identifies one rental and stops
meaning anything once the rental ends. It now gives the host's machine id, taken from each run's own receipt. The
11 September rentals with no recorded machine id read "Vast box S1" to "S11", one label per rental (their host identity
was never recorded, so two labels may name the same host). Text only: no value, id or evidence changes. `tests/test_claims_citation_style.py` keeps
instance ids out of claim text.
