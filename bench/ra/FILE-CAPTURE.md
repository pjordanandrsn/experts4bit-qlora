# Supplied-pin regular file capture

`ra_file_capture.capture_file(path, expected_size, expected_sha256)` returns the
complete captured bytes and a cleanup record. It requires Linux x86_64 with
`-I -S -B`, a canonical absolute path, a nonempty size up to 32 MiB and a full
lowercase SHA-256 supplied independently by its caller.

Every component opens relative to a retained directory descriptor with
`O_NOFOLLOW` and close-on-exec. The leaf opens nonblocking to avoid waiting on a
FIFO and must be a regular single-link file. Bounded complete positional reads,
EOF, full hash, leaf descriptor/path metadata and each directory identity must
agree. All known-owned descriptors close with EBADF readback before returning.
Unknown or replaced descriptors are preserved; their cleanup is unproven.
Failures retain partial read counts, original errors and each cleanup outcome.

The caller must serialize descriptor operations in its cooperating owned process.
Identity checks cannot eliminate syscall races or authenticate arbitrary
same-process mutation. A supplied size/hash can be forged. This helper checks
supplied-pin byte equality; it does not establish indexed source provenance.

A future reviewed caller can pass these exact bytes to the separately merged
`ra_sealed_view.CapturedView`. This helper creates no sealed view, performs no
constructor/tokenizer call and proves no original native asset-path consumption.
It has no CLI, receipt wrapper, inherited guard, role map, worker expansion or
ABBA wiring. Source/runtime/native-read/consumer authority remains false.
