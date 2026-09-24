# P67 draw receipts — `p67-gemma4-floor-1` (RTX 5090, 2026-09-24)

The box's own outputs from the registered draw (`bench/p67/P67-PREREG.md`), copied out of the fetched `tp4/` tree
unchanged except that the logs carry a `.txt` suffix (`*.log` is gitignored). The reducer's read
(`session.md`, `session.json`) was produced on the controller with `--consistency` against the two same-fixture
sessions in the private store, as the registration names.

| path | what |
|---|---|
| `gemma4_e4b_<tag>.json` | tp4_arm.py's receipt per arm: the 20 per-step train losses, `init_sha`, tokens, trainable count, C1 verify, `n_attn4`, `n_patched`, `batched_stats`, `reference_order`; `not_run` stubs for the three arms `TP4_SKIP` names |
| `session.md`, `session.json` | `p67_reduce.py --session … --consistency …` (the registered read) |
| `summary.txt`, `versions.txt`, `box.json`, `forensics.txt` | the runner's own record |
| `p67_guard.txt`, `logs_p67_pins.txt` | `p67_run.sh`'s pin check (8 files), knob check and host-RAM check on the box |
| `RESULTS-tp4-boxC.md` | tp4_reduce.py's box read (status / validity / engagement per arm) |
| `vram_<arm>.txt` | per-arm VRAM traces |
| `logs/`, `outer.log.txt` | fetch, dataset, pip, tripwire and per-arm logs |
| `teardown-proof.json` | the launcher's destroy record |

Not here: the staged scripts (identical to this tree, `logs_p67_pins.txt`), `tokens_gemma4.json` (2.3 MB; its sha
`4ea779a5…` is in `summary.txt` and every arm receipt), the alpaca `data/` dir, the Unsloth compile cache, the
`TP4_*` marker files. Kept in `adertha-receipts` (`receipts/experts4bit-qlora/2026-09-24/p67-gemma4-floor-1/`, commit
`f44c594`): `receipt.json`, the guard log and heartbeat, the full tree.

## Reproduce the read

```sh
python bench/p67/p67_reduce.py --session bench/p67/receipts --md /tmp/p67-session.md --json /tmp/p67-session.json
```

Over this directory the reducer writes the same floor and judged tables as `session.md`, byte for byte; only the four
`- consistency …` lines are absent, because they need the two private sessions
(`receipts/experts4bit-qlora/2026-09-19/tp4-c-parity-2`, `2026-09-22/p56-gemma4-ladder-3`).
