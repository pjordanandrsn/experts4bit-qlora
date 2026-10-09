# Fixed supplied-pin asset views

`CapturedAssets(root, model, pins)` captures the fixed five Qwen or seven Granite
tokenizer asset filenames into sealed copies. It does not interpret configuration
or choose a runtime class. Pins are exact `{size, sha256}` records; the complete
collection is bounded to 32 MiB. The helper loads only the two sibling file-capture
and sealed-view helpers. There is no CLI, constructor, guard, receipt wrapper,
worker inventory expansion or ABBA wiring.

Every source file receives a full no-follow capture and known-owned descriptor
cleanup. After all copies are sealed, a second complete capture checks the pins
and repeated directory/leaf identities before paths can return. `paths` maps all
fixed filenames to distinct owned copy descriptor paths. Later `check()` verifies
the complete sealed copies. It does not reread the original paths or claim those
paths stayed unchanged after construction.

Close and refusal clean every known-owned view in reverse order. A failed prefix,
source recheck, readback or close retains all records. Unknown ownership or missing
cleanup receipts remain unproven; failure is terminal and never retried. Context
cleanup preserves the original caller exception and records cleanup failures.

The caller must serialize descriptor operations. Expected pins and model labels
may be forged; helpers and arbitrary same-process tampering/syscall races are not
authenticated. All source, runtime, native-read and consumer authority fields
remain false. The copies are not evidence of original Tokenizer/Rust asset reads,
actual token IDs, a consumer, image execution, GPU engagement or proof clearance.
