# Sealed captured byte view (proposal)

`ra_sealed_view.CapturedView` captures one supplied nonempty `bytes` payload,
at most 32 MiB, after checking its complete lowercase SHA256. In Linux x86_64
under `-I -S -B`, it creates a close-on-exec memfd, writes the entire payload,
and applies WRITE, GROW, SHRINK and SEAL seals. Every check independently
reopens its own `/proc/self/fd/N`, compares device/inode/size/seals, and reads
and hashes the entire copy. This proc path names the copy, not an original
asset path. The helper does not authenticate supplied bytes or SHA labels.

Construction retains a refusal snapshot in `ViewError.record`. Partial write
or seal failures attempt owned-descriptor cleanup. Close requires matching
identity and EBADF readback. A replaced descriptor is never closed by the
helper: cleanup remains unproven, and close/refusal is terminal. Reader cleanup
failures also remain visible. Context exit preserves a caller exception and
records cleanup failure; callers must inspect `view.record` after exception
handling. A prior failed check cannot become a successful status after close.

This mechanism assumes serialized descriptor operations in a cooperating
owned process. It cannot prevent arbitrary same-process code from replacing
methods, fields or descriptors, or eliminate races between identity checks
and syscalls. Unknown descriptor identities require separately verified owned
process teardown; a failed cleanup receipt is never complete cleanup.

The explicit ABI constants are Linux v6.6 UAPI `fcntl.h`, `memfd.h` and x86_64
`syscall_64.tbl`: commands 1033/1034, seals 1|2|4|8, syscall 319 and flags 3.
No interpreter, `os` or `fcntl` module patch is needed. Unsupported platforms,
ABI or flags refuse before allocation. Interpreter binary identity, inherited
parent-death guard and full site/library provenance remain caller gates.

This is an unwired component proposal for issue #1363. It has no CLI, source
file acquisition, tokenizer constructor, asset role map, native extension
activation, worker inventory expansion or ABBA wiring. All source/runtime/
native-read/consumer authority fields remain false. Real tokenizer calls,
complete final wrapper state and IDs, actual tensor consumption, image/GPU
and launch authority require separate reviewed gates. It grants no proof,
rental, release or launch clearance.
