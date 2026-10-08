# RA capacity process component

`ra_capacity.py --spec /absolute/spec.json --instruments /absolute/stage
--out /absolute/fresh-component` starts one release-default server and invokes
the frozen SC2 driver in fresh processes: four serial warm requests, the
64-request burst, then 120 requests at 12/s. Plans and prompt hashes are
checked against native receipts. No speed/slot/bucket/fusion switch is forced.

The supervisor holds a new loopback listener and passes its descriptor to
the server. It never contacts a pre-existing server. The Linux child has a
parent-death guard; cleanup kills only its newly created process group.
The listener uses [Uvicorn's explicit sockets interface](https://github.com/encode/uvicorn/blob/master/uvicorn/server.py).
Hooks collect fusion census, kernel calls and QKV counts before capture.
Native health and counter snapshots follow each drained phase; the existing
normalizer checks prefill/KV reconciliation and widest-bucket engagement.
Both native traces must exist and be nonempty. Their destinations are derived
inside this fresh component; a caller cannot point them at another resource.

Spec fields: `battery`, `venv`, `cache`, `threads`, `allocator`, `fixture`,
`prompts` (`path`/whole-file `sha256`), `deadline_epoch_s`, `startup_s`,
`driver_s`. `fixture` uses the capacity allowlist, names the pinned model and
revision and prepared arena/calibration paths, and fixes context 2048/chunk 512.
CUDA/all-vram and torch threads follow the registered common fixture. Other
speed settings remain unset.
Startup plus all driver timeouts must leave ten minutes for retrieval and
teardown. `driver_s` covers a driver's completion and snapshot retrieval.

All native outputs, traces, logs and failure receipts remain in the fresh
component directory. CPU composition tests establish process wiring only.
The local test environment lacks Uvicorn; actual HTTP serving is not rehearsed.
Output explicitly remains pending engagement: independent wheel/input/default/
fallback verification, full trace/request joins, GPU premises, ABBA, lock and
reviewed launch/ledger/retrieval/teardown gates still precede proof clearance.

The listener is the only host/port config exception. Shared assembly compares
other fields to the native factory and resolves fusion modes by the loaded
model family; evidence retains both raw and resolved modes and their sources.
