# Proposed WikiText text decoder

`ra_wikitext_text.py` is a pure, bounded decoder proposal. It supports one
optional UTF8 `text` column with no null values, Snappy compression, a PLAIN
dictionary page followed by one V1 RLE_DICTIONARY data page per row group.
Unsupported layouts raise `ValueError`; it does not download or import a
dataset, tokenizer, release package, or runtime dependency.

The decoder preserves CompactProtocol field and collection types and rejects
unknown or missing fields, duplicate field IDs, integer overflow, invalid
compression offsets, excessive padding, nulls, invalid dictionary indexes,
and invalid UTF8. It joins footer, page and repeated column metadata offsets,
row counts, encoding census, compressed and uncompressed sizes, and complete
min/max text statistics. Page streams and repeated column metadata must cover
every byte from the initial magic to the footer. Metadata keys are limited to
`huggingface` and `ARROW:schema`; their opaque values confer no authority.

The limits are 16 MiB for an archive, 8 MiB for a decompressed page, 10,000
rows, and 64 row groups. These are parser limits for the proposed component,
not changes to the registered benchmark battery or its thresholds.

`project(rows)` returns complete ordered UTF8 JSON rows and the text produced
by filtering on `t.strip()` and joining the retained rows with two newlines.
These outputs are inputs to a future source-bound tokenizer gate; their
existence does not show that a tokenizer or measured runtime consumed them.

A standalone selected-Python source/guard/receipt wrapper belongs to a
separate reviewed component. It must bind the raw archive through the reviewed WikiText authority and
retained complete Hub records, independently check the registered P109/P97
join recipe, verify helper/input bytes before and after execution, protect
receipt paths, and retain failed results. Parent composition must independently
reparse all raw records and regenerated text, join PID/manifest/guard/reap/group
absence, and use an explicitly reviewed inventory expansion. This helper is
an unreviewed extra in every existing fixed worker inventory.

The focused synthetic parser controls run with repository conftest active.
The retained CPU diagnostic compares the actual archive's complete rows and
joined bytes on the host and extracted selected Linux interpreter against
public-service records reporting the reviewed revision. Public-service
corroboration is not a pinned independent local decoder or publisher trust.
The extracted interpreter is not execution in the full proof image.

No tokenizer execution, native loader revision binding, calibration authority,
actual runtime consumption, GPU engagement, release or launch authority is
established by this proposal. Existing source pins and baseline remain unchanged.
