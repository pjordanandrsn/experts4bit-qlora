### Fix the DQ11 census host observation window

Measure the five-minute read-only offer history from its first completed
controller search. Search latency can no longer make a full observation run
fail the same first-to-last duration check it uses for launch admission.
