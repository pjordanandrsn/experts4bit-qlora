### P130 Amendment 1 (#846): the box records the wikitext commit it read, beside the fetch gate's (bench and tests only)

- The windows' corpus is not pinned by revision in P117's box. The fetch gate records the commit `main` names; the box
  now records the commit or commits it actually read, from `datasets`' arrow cache and the hub snapshot, which are both
  keyed by commit. The driver passes the gate's commit to the box (`P130_GATE_CORPUS`), and the reducer reports both
  side by side in `verdict.json`. It is reported, never gated: the windows digest already gates the windows themselves.
- No arm, gate, bar, prediction, size, budget or sequencing rule changes. `staged.sha256` is re-pinned, and the reducer's
  self-test grows to 42 cases.
