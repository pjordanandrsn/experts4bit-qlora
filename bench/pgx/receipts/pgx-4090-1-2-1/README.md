# pgx-4090-1-2-1: PGX's first reading box, ended at the fetch (never read)

The box ran the reading kit cut from e4b#2175's merge (519a46d9). It STARTED at 16:27:05Z on 2026-11-06. The installs, the
tripwires and the box-side self-tests passed, and the bf16 and FP8 checkpoints fetched in full. The GPTQ-Int4 fetch then
failed. Its log's first errors are TLS handshake timeouts to the hub, then `[Errno 101] Network is unreachable` on every
retry, and it ends `httpx.ConnectError: [Errno 101] Network is unreachable` (`logs/fetch_int4.log`). The runner wrote
`DL FAIL (int4)` and exited 11. That is `PGX_EXIT_CODE.d910131296e70565`.

Under PGX's registered stop this box is a strike: `bench/pgx/pgx_box_end.py` classes it `STRIKE` ("rc 11: only 0 and 27
can end a reading box without a strike"). With `pgx-shake-1` as strike one, it ended the PGX lane. That verdict stands.
PGX2 (`bench/pgx/PREREG-pgx2.md`) cites this record as the source of its host-fault rule and tests that rule on it. The
classification of this box is not changed.

These are the files the driver fetched (`pgx_drive.sh`), minus the staged copies of this repo's own code, hooks and prompt
sets. Those are the launch commit's files, named in `summary.txt`'s `KNOBS` line. The provider's instance and machine
records stay in the private receipt store. `SHA256SUMS` lists every file here, and each one is byte-identical to the box's.
