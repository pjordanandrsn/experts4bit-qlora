# DQ11 census wheel transfer recovery

The first paid census stopped during bootstrap after the bitsandbytes wheel
connection was reset. This operational change retries only connection resets and
timeouts, including a reset while reading the response. It makes at most five
attempts at the identical locked URL with 5, 10, 20 and 40 second backoffs and a
180 second per-wheel budget. Socket operations time out after at most 20 seconds;
a process timer also interrupts blocked IO or a peer that keeps trickling bytes
past the overall budget. The existing 900 second bootstrap phase and five-minute
teardown reserve remain the outer bounds.

`dq11_wheel_retry.py` prefetches wheels and verifies each complete SHA256 before
atomically promoting the partial file into the cache. Partial failed transfers
are discarded. HTTP failures, wrong hashes and invalid cache entries refuse
immediately. There is no alternate source. The original `dq11_bootstrap.install`
then rechecks all cached wheel hashes and uses the existing hash-required pip
installation and source-authority path.

The helper is in the manifest's operational hash closure and both transport
hash lists. The frozen `dq11_bootstrap.py`, `science.sha256`, locked URLs/hashes,
instrument and science source remain unchanged. The box saves start and outcome
records for every attempt in `receipts/wheel-fetch-attempts.jsonl`, including URL,
UTC, expected and observed hashes, partial byte count, error and backoff. A killed
attempt retains its start record. Cache hits also record their verified hashes.

The previous envelope's single draw is spent. A retry of the census requires a
reviewed merge and a fresh maintainer envelope bound to the new operational
closure; this change itself grants no compute authority.
