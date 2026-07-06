# Certificate trio rev1 — FAILED (harness bug, pre-fix record)

Per Q4 (POST_AUDIT_WORK_QUEUE.md): the red is committed before the fix, the fix before the
rerun, each citing the previous. This file is the red.

- Jobs: `cert_olmoe_{bf16_dropoutOFF_default, bf16_dropoutOFF_deterministic,
  bf16_dropoutON_default, int8_dropoutOFF_default, int8_dropoutOFF_deterministic}` at commit
  `a46bb61`, pod `41a04ccbc77d` (RTX A5000, on-demand, 2026-07-05).
- Failure: every job crashed in leg (a) at `scripts/one_step_certificate.py:39` (`_sha`):
  `TypeError: Got unsupported ScalarType BFloat16` — `Tensor.numpy()` cannot convert bf16;
  the hash helper needed a byte-reinterpreting view before `numpy()`.
- Classification: **harness bug in the brand-new certificate script** — the crash precedes
  any leg comparison, so no scientific object (logits/grads/weights/optimizer state) was
  measured, and nothing about the D3 question is learned from rev1. The 13 null_eval jobs in
  the same manifest are unaffected (different payload, all pass, all claim_usable).
- Rerun: same five configs as `*_rev2` job ids in
  `runs/job_manifest/post_audit_cert_rev2_jobs.jsonl` (new lock namespace entries; rev1 locks
  and artifacts remain untouched on the volume as the pre-fix record).
