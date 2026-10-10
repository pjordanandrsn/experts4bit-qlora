### Pack manifests take a rank-0 bf16 payload

`pack_manifest.tensor_payload_bytes` and `tensor_from_payload_bytes` byte-viewed a bf16 tensor directly. A byte view of
a rank-0 tensor refuses (`self.dim() cannot be 0 to view BFloat16 as Byte`), so a scalar bf16 payload could not be
written or read. Both now flatten before the view. A ranked tensor's bytes are the same, so every existing payload and
fingerprint is unchanged: `tests/test_pack_manifest_rank0.py` pins today's payload hashes for ranked bf16, fp32, fp16,
int8, uint8 and int32, round-trips each, and writes and reads a rank-0 tensor of each dtype bit for bit.
