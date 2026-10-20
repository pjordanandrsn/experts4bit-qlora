### P159 registered (design): what makes the loader's host-side expert fuse run 17x slower in some host windows

`bench/p159/PREREG-p159.md` registers a timing premise, never a ruling. On one host the registered fuse block ran 17x
slower outside one wall-clock window that spanned four processes; another host ran it fast throughout. The primary
hypothesis is the fault cost of fresh anonymous memory, switched by huge-page availability. Two arms manipulate it
directly: THP disabled for the process (predicted slow even in a fast window) and a pre-faulted heap (predicted fast
even in a slow window). A slab-residency arm, an identical-arm null, and a headroom arm at the 1.3 x VmHWM RAM floor
(a separate question, from Q235's offload) run beside them, interleaved. The kit (`bench/p159/`, `tests/test_p159.py`)
reuses the SMT fuse block and Q235's pinned install unchanged.
