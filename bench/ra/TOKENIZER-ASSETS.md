# Proposed tokenizer asset/config gate

`ra_tokenizer_assets.py` binds fixed tokenizer assets in the reviewed common
checkpoint tree to repeated complete Hub model-info/tree indexes at the
registered model revision. Ordinary Git blobs and materialized LFS bytes are
checked independently. The gate parses the declared family/class, EOS/pad
IDs, typed modes and complete added-token declarations, and joins every
configured added token to the tokenizer JSON asset. Remote-code and asset
resolution overrides refuse.

The CLI requires selected Linux Python `-I -S -B` and an inherited SIGKILL
parent guard. The manifest supplies only the common spec, stage, reviewed
lock/digest and expected parent. Four sequential bounded raw records and the
result form the exact five-file success receipt. Failure prefixes remain.
The parent evaluator requires its own verified process/guard/reap/group-absence
record, reparses all indexes, rehashes assets and compares the complete result.
It cannot authenticate invented caller process records.

This is indexed asset/config equality. The declared tokenizer class is not an
observed runtime class. The gate imports no Transformers, Tokenizers, Torch or
release package, downloads no assets, and does not establish native file reads,
backend final state, token regeneration, consumer binding, model weight
consumption, calibration, publisher signatures, GPU engagement or launch
clearance. It is not wired into the fixed worker inventory or ABBA. A guarded
runtime caller and any inventory expansion need separate review.
