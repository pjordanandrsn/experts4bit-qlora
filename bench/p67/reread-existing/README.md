# P67 — the existing training-parity receipts, re-read under the floor band (CPU, before any P67 draw)

Disclosed in `../P67-PREREG.md`, §"What the existing receipts show". This is **not the registered reading** and
moves no status: no existing session holds an admissible floor draw, so every row is either a constant-band PASS
(which the rule makes a floor-band PASS without a floor) or NO-FLOOR.

| file | what it is |
|---|---|
| `reread.md` | `p67_reduce.py --reread` rendered: every same-session judged pair, then every cross-session same-arm pair (disclosed, never admitted) |
| `reread.json` | the same, machine-readable, plus `inputs`: the sha256 of every receipt read (72 files, 14 sessions incl. tp1) |
| `sessions.txt` | the 13 tp4-shaped sessions, as paths inside the adertha receipt store (`receipts/experts4bit-qlora/`) |

The tp4-shaped receipts are private (the adertha receipt store, as P56's are); tp1's bundle is committed at
`bench/train-parity-20260905/tp1/`. Nothing here copies a receipt; the table is the reducer's output.

Reproduce, from a checkout of this repository and a checkout of the receipt store at `$STORE`
(`.../adertha-receipts/receipts/experts4bit-qlora`):

```sh
python3 bench/p67/p67_reduce.py --reread $(sed "s|^|$STORE/|" bench/p67/reread-existing/sessions.txt) \
    --tp1 bench/train-parity-20260905/tp1 --md /tmp/reread.md --json /tmp/reread.json
```

and compare `inputs[].sha256` in the JSON against the store's files. Run 2026-09-23 against the store's branch
`cdo/state-2026-09-06-1720` at `2978a44`.
