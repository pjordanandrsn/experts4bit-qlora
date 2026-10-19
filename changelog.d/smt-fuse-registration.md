### Added

- The SMT fuse measurement registered (`bench/smt-fuse/PREREG-smt-fuse.md`): one short rental of a server-class SMT host times the streaming loader's own host-side expert fuse plus nf4 quantize at one Qwen3-235B layer, with torch's pool at 16 and 32 threads interleaved (16, 32, 32, 16 x 4), reporting per-layer seconds and each pair's 32/16 ratio with its range, and checking the fused and nf4 bytes are identical across pools. A timing premise for Q235's simplification amendment, never a ruling. Worst $0.64. CI runs its contract.
