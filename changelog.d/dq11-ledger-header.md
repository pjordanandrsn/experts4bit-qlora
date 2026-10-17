### Accept canonical census ledger comments

The census attempt guard now follows the controller ledger parser: it strips lines, skips blank lines and comments, and rejects malformed non-comment JSON. Regression tests use a minimal ledger excerpt with the real header and census-4 refusal row.
