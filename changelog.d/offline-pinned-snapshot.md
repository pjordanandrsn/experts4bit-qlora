### A pinned, complete local cache loads offline again

`load_moe_4bit_streaming(model_id, ..., revision=<commit sha>)` with `HF_HUB_OFFLINE=1` failed with
`OfflineModeIsEnabled` even when the pinned snapshot was complete in the Hub cache. huggingface_hub 1.26.0 through
2.1.1 list the repo tree online inside `snapshot_download(..., revision=<sha>)` when offline; 1.25.0 and 2.2.0 do not.
- **Offline with a full 40-hex commit whose snapshot is cached:** the loader now resolves it locally, with no Hub call.
  It first checks that `config.json`, the safetensors index and every shard it names (or a single `model.safetensors`)
  and a tokenizer file are present.
- **Missing files:** the loader refuses, naming them. A pinned commit with no cached snapshot is refused offline.
- **Unchanged:** online loads, branch-name revisions (resolved by huggingface_hub through `refs/<branch>`) and local
  directories behave as before.

Users who pin a revision and run offline (for example on air-gapped hosts) were affected. It had been hidden while
callers passed no revision, because offline `revision=None` resolves through `refs/main`.
