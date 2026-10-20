### P159 Amendment 2: a rerun after an instrument void, a THP pre-flight refusal and an early bytes gate

`p159-1` read VOID_BYTES. On its host the NF4 bytes differed in every block while the fused bytes matched; the same stack
is deterministic on three other hosts, and the cause is not established. Amendment 2 (`bench/p159/PREREG-p159.md`):
- `p159-2` may run after any VOID.
- A host is refused before the install unless torch's allocations can get huge pages (THP `always`, or `madvise` with
  `THP_MEM_ALLOC_ENABLE=1`).
- The box stops after its first block when the bytes are not the registered ones.
- The box records the GPU's driver, VBIOS and ECC state.
