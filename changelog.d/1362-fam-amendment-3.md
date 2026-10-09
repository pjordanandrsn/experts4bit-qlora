### FAM Amendment 3 (#1362): the hybrid's warm-up forward is counted once per process

`fam-qw36-2` ($2.345) VOIDed on an engagement count. The registration expected Qwen3.6's linear-state warm-up forward on
every padded pass, but e4b's `PagedModelRunner._warm_linear_state` runs it once per process, on its first padded pass.
The records show exactly that. The reducer now expects the warm-up on each process's first pass only (`FIRST_CELL`) and
VOIDs on anything else. Its self-test has five new cases (50). The gates, cells and predictions are unchanged, and
`fam-qw36-2`'s records are re-reduced under the amended rule, with no new rental.
