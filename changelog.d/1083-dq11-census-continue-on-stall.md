### DQ11 L-census: a pre-command host stall continues the envelope (#1083)

`dq11_l_census_launch.py` stopped an envelope at any created instance. A created instance that stalls at the
controller's pre-flight and never runs its command (proven from the controller's own receipt and the ledger) now lets
the envelope continue to its next declared ID, under three journal-derived bounds: at most one instance that runs the
command, a cumulative all-in across every attempt, and every stalled machine excluded from discovery, binding and the
quote. `--run-id next` takes the first unused declared ID. Any other created instance still stops for the maintainer.
