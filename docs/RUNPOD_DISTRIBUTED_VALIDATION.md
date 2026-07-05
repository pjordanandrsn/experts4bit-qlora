# Distributed validation over a shared RunPod volume

How multiple pods run independent validation-grid jobs against one persistent volume without
stepping on each other. Provenance first, then speed: every rule below exists to keep each
result attributable to exactly one pod, one commit, one command.

## Topology: one controller, N workers

**The controller** (a laptop/CPU host is fine — no GPU needed) owns everything shared: the git
branch and all code edits, job-manifest generation (`scripts/make_olmoe_repeat_manifest.py`),
docs and README, aggregation (`scripts/summarize_runpod_jobs.py`), the aggregate files under
`runs/results/`, and the final report.

**Workers** (GPU pods) do exactly four things: claim jobs, run them, write job-local outputs,
exit. Workers never edit docs, never write aggregate JSONL/CSV/Markdown, never overwrite shared
adapter directories, and never `git pull` mid-campaign (all workers run the same pinned commit,
shipped by the controller).

## Layout (all on the shared volume)

```
runs/job_manifest/<name>.jsonl      controller-written manifest, one job per line
runs/locks/<job_id>.lock/           claim locks (atomic mkdir; owner.json inside)
runs/jobs/<job_id>/                 the job's ENTIRE write surface:
    command.sh  run.log  env.json  status.json  result.json
    result_rows.jsonl               (train/query payloads; runner lifts the last row)
    <mode-label>/adapter_best.pt + expertsnbit_adapter_metadata.json   (train jobs)
runs/results/                       controller-only aggregates (summarizer output)
```

No two pods ever write to the same job directory: the lock guarantees single ownership before
the directory is created.

## Atomic claiming

A worker claims a job with one `os.mkdir(runs/locks/<job_id>.lock)` — atomic on the shared
filesystem, succeeds for exactly one claimant, and there is no check-then-write anywhere.
`owner.json` inside the lock records pod_id, hostname, pid, commit, timestamp, and the command.
Every result records the same identity plus started_at/finished_at.

## Worker launch checklist

1. Pod created with the shared volume mounted (`runpod-qlora.sh up` handles the DC pinning).
2. Controller ships the pinned tree (`git archive <commit> | ssh ... tar -x`); worker verifies
   the commit matches the campaign's (`env.json` records it either way).
3. Verify the model cache is fully staged on the volume BEFORE launching many workers — one pod
   pre-downloads; N pods must never download simultaneously. Set `HF_HOME` (and
   `HF_HUB_CACHE`/`TRANSFORMERS_CACHE` if used) identically on every worker.
4. `export POD_ID="${POD_ID:-$(hostname)}"`.
5. Environment smoke check (`python -c "import torch, bitsandbytes, experts4bit_qlora; ..."`).
6. Start the claim loop:
   ```bash
   python scripts/runpod_claim_and_run.py \
     --manifest runs/job_manifest/olmoe_repeat_jobs.jsonl \
     --jobs-root /workspace/runs/jobs --locks-root /workspace/runs/locks \
     --pod-id "$POD_ID"
   ```
   One claim loop per GPU — a worker never runs more than one heavy job at a time.
7. Confirm job-local logs appear under `runs/jobs/<job_id>/` before walking away.

## Failure recovery and stale locks

A dead pod leaves its lock in place — that is by design; locks are never auto-stolen. The
CONTROLLER may declare a lock stale only after checking all of: lock timestamp, pod status (the
RunPod API), no process running, incomplete job outputs, and the logs. A rerun is a NEW job id
with a `_rerun1`/`_rerun2` suffix added to the manifest — failed job directories are never
overwritten and their logs are preserved.

## Aggregation

Only the controller aggregates, only by reading `runs/jobs/*/result.json`:

```bash
python scripts/summarize_runpod_jobs.py \
  --jobs-root /workspace/runs/jobs --results-root /workspace/runs/results
```

Jobs without a `result.json` (still running) are ignored; failed/skipped jobs appear in the
failure table. Safe to rerun at any moment for a mid-campaign snapshot.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T09:22:24Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `82f3d6941e7ba504578ab475a5fda64367e0560868b11c369d8fb45f36d917a3` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[*+$~!0#o:?=@%O.o]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|        . *++o.  |
|       . *oB++ + |
|        +.=..+= *|
|     .   +  oE.*=|
|    o . S . ..+oB|
|     o = + o o.= |
|      o + o   o  |
|     .   .     . |
|                 |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info RUNPOD_DISTRIBUTED_VALIDATION.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify RUNPOD_DISTRIBUTED_VALIDATION.md.ots RUNPOD_DISTRIBUTED_VALIDATION.md` succeeds against the on-disk bytes.
- Anchor file: `RUNPOD_DISTRIBUTED_VALIDATION.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
