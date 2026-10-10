### A pinned, complete local cache loads offline again

`load_moe_4bit_streaming(model_id, ..., revision=<commit sha>)` with `HF_HUB_OFFLINE=1` failed with
`OfflineModeIsEnabled` even when the pinned snapshot was complete in the Hub cache. Some huggingface_hub releases list
the repo tree online inside `snapshot_download(..., revision=<sha>)` when offline. This was observed on 1.26.0, 1.27.0,
1.28.0, 1.29.0, 1.30.0, 1.31.0, 1.32.0, 1.33.0, 2.0.0, 2.1.0 and 2.1.1. 0.36.0, 1.0.0, 1.20.0, 1.25.0 and 2.2.0 resolve
locally; releases not listed were not tested.
- **Offline with a full 40-hex commit whose snapshot is cached:** the loader now resolves it locally, with no Hub call.
  It first checks that `config.json`, the safetensors index and every shard it names (or a single `model.safetensors`)
  and a tokenizer file are present.
- **Missing files:** the loader refuses, naming them. A pinned commit with no cached snapshot is refused offline.
- **Unchanged:** online loads, branch-name revisions (resolved by huggingface_hub through `refs/<branch>`) and local
  directories behave as before.

Users who pin a revision and run offline (for example on air-gapped hosts) were affected. It had been hidden while
callers passed no revision, because offline `revision=None` resolves through `refs/main`.
