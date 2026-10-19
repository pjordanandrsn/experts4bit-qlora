### Added

- P146 read: the CLI trainer at its own default recipe steps 1.455x faster with `OFFLOAD_EXPERTS=1 LAYER_MAJOR=1` than with `OFFLOAD_EXPERTS=1 FAST_TRAIN=1` on Qwen3-30B-A3B under expert offload on an RTX 4090 (claim `e4b.train.p146.layer-major-cli.qwen3.4090.2026-10-18`); `docs/STATUS.md` no longer calls that recipe unread.
