Real pre-command host stall: `dq11-l-census-7`, from canonical receipt commit
fbc88cea0807ab432e520bd151f706d7588200c6. Original receipt SHA256
78ff24f44dfd3e1431007480a531d9195092954e965fa9066bfc3832ffe39fd2. The local command, executor identity, the lane
record and private environment fields (paths, seat coordination) are omitted, and the approval's agent and role read
`fixture`. Everything `preflight_stall` and admission read is unchanged: outcome, cost, notes, the teardown proof,
`environment.vast_preflight` and `vast_machine_id`, the approval URL, and the experiment and work IDs.

`ledger.jsonl` is census-7's row of the receipt store's ledger at 450b7c97a40f1348d2893d4ef1f8cd471c969c79, verbatim.
The tests use both unchanged; the hand-edited negatives start from `dq11-census-refused`.
