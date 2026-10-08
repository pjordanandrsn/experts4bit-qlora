# RA training process component

`ra_training.py --spec /absolute/spec.json --instruments /absolute/stage
--out /absolute/fresh-component` runs the frozen TC1 field arm. A separate
invocation with `--profile` runs its 10-step steady profile. Both keep 20
steps and the field recipe, including autocast off. No speed flag is forced.

Spec fields: `battery` (`proof` or `reading`), `venv`, `cache`, `threads`,
`allocator`, `expect_trainable`, `deadline_epoch_s`, `timeout_s`, and `data`,
`tokens`, `prereg` records containing absolute `path` and `sha256`. The model
revision comes from the registered source pins. There is no extra-argument
escape hatch. The supervisor must derive the family's trainable count.

The runner verifies staged instruments and input bytes before and after the
process. It retains TC1's native JSON, adapters, log and process receipt,
checks the emitted fixture and schedules, and never overwrites an attempt.
The whole token file hash binds the input; TC1's separate compact train/eval
payload hash is recomputed for `--tokens-sha` and checked against its receipt.
A timeout kills the process group created by that invocation. Every phase
must leave ten minutes before its deadline for retrieval and teardown.

`component.json` records native evidence pending engagement checks. It does
not create a reducer-ready feature dictionary or prove GPU engagement.
Independent wheel/import/default/fallback telemetry, capacity orchestration,
ABBA supervision, dependency lock, on-card premises and launch/ledger/
retrieval/teardown gates remain. A supplied deadline grants no rental authority.
