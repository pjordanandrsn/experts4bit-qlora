# Complete tokenizer call transcript evaluator

`ra_tokenizer_transcript.py` is a pure, bounded UTF-8 JSON evaluator. It validates
all 1,248 TC1 train/eval calls followed by one full Wiki call. Every record retains
complete text, fixed kwargs, full nonnegative integer IDs and before/after state
references. Complete backend JSON objects are SHA256 addressed; every referenced
state must be present, every listed state referenced, and the state chain must
join from the initial state through the final call. No window or retained eval
prefix substitutes for a complete call stream.

`parse(raw)` requires exact schema-1 fields and COMPLETE status. TC1 ID streams
may be empty or short because dropping happens after the original call; they are
bounded at 2,048. The Wiki stream spans at least the final registered decode
window and at most two million IDs. JSON is bounded at 96 MiB, each state at
32 MiB, each text at 16 MiB, five million value nodes and depth 64. Duplicate keys,
nonfinite numbers, invalid UTF-8, scalar type coercions in typed fields, unknown
fields, gaps and unreferenced states refuse with ValueError.

`compare(expected_raw, observed_raw)` parses both and compares their complete
canonical UTF-8 JSON bytes. Object key order and insignificant whitespace are
ignored; JSON scalar types, complete state values, all texts/kwargs and every ID
are compared. This includes dropped short rows, all 48 eval calls beyond the
retained eight, and Wiki tails outside measured windows.

These are supplied transcripts. The helper does not call a tokenizer, import
release libraries, read files, launch processes or authenticate provenance. A
caller can forge both arguments. A future separately reviewed guarded caller
must establish independent origins, original methods/objects, source/site/native
extension identity, file-read and wrapper-state evidence, and actual consumer
binding. CPU synthetic transcript controls do not establish those premises.
There is no worker inventory expansion, ABBA wiring or launch authority here.
