### Correlate client TTFT with the server trace

SC2 retains the server's completion id. `bench/sc2/ttft_join.py` joins each valid
client request to its server trace and reports queue wait, admission-to-first-token
time and the client/server residual. Missing, duplicate or conflicting ids are
refused. Existing timing and validity rules are unchanged.
