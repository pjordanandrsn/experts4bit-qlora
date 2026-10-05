### DQ3 Amendment 3 registered (#1083): S and S0 run on #1183's late-bound backward; the rehearsal gate asserts the memory direction (bench only; #1185)

- **Three 5090 attempts ($0.329 in all):**
  - a host that refused large allocations (fixed in #1171);
  - a host with ~33 KB/s GitHub egress (fixed in #1173);
  - `dq3-5090-3`, where streaming freed nothing because bnb 0.50.2 keeps the frozen weight on ctx (worked around in
    #1183; bnb behaviour, not reported upstream).
- **R's run-3 numbers** were seen and are not reused.
- **Unchanged:** the rule, gates, bands, arms, predictions and guard.
- **`bench/dq3/dq3_rehearsal_check.py`** exits 1 unless parity is bitwise and each streamed arm peaks (L−2) layers
  below resident. It fails every streamed arm of the pre-fix rehearsal that had been read as green.
