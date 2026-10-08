### Close the DQ9 checksum staging contract

Derive the staged file list from the registered checksum manifest, including
the controller script, and refuse absent, changed or colliding subjects.
The first DQ9 run stopped at this tripwire before proofs or readings. Add an
offline test of the real controller's complete stage plan against every
registered checksum, including basename uniqueness. The diagnostic rules,
executor, budget and failed receipt remain unchanged.
