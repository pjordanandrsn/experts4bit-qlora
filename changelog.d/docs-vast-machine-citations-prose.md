### Docs: the bench READMEs and RESULTS files cite a rented host by its machine id

The claims register's citation style now covers the live prose too. 184 mentions across 86 READMEs and RESULTS files
name a rented Vast host by its machine id, read from each run's own receipt, instead of an instance id. Seven
August rentals that predate the receipt store, and so have no recorded machine id, read "Vast box S12" to "S18",
one label per rental, after the claims register's S1 to S11. Receipts, PREREG and AMENDMENT files, STATUS-RECORD.md,
released changelogs, anchored files and code blocks (which quote receipts verbatim) are left as written.
`tests/test_claims_citation_style.py` now checks the prose as well.
