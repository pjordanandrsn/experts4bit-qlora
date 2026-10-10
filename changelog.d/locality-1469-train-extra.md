### #1469 census runner: install e4b's [train] extra with transformers 5.16.1 (bench and tests only)

- `loc-a2000-4` refused at its import tripwire (rc 9, $0.014, before the model fetch). The e4b wheel's base
  dependencies do not include transformers; it lives in the `[train]` extra.
- The runner now installs `experts4bit-qlora[train]` at the launch commit, with transformers pinned to 5.16.1, the
  Qwen3 version SC1's runner pins. `staged.sha256` is regenerated, and the shape test asserts the install line.
