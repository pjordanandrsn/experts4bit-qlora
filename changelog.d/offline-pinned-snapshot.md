### A pinned, complete local cache loads offline without its tree listing

`load_moe_4bit_streaming(model_id, ..., revision=<commit sha>)` with `HF_HUB_OFFLINE=1` failed with
`OfflineModeIsEnabled` on some caches that held the complete pinned snapshot. Offline, some huggingface_hub releases
resolve a pinned commit from the `trees/<sha>.json` listing that a download writes next to the snapshot. When a cache
has the snapshot but no listing, they list the repo tree online instead. Such caches come from an older release, or
are assembled by hand, for example by linking weights already on disk. This was observed on 1.26.0, 1.27.0, 1.28.0,
1.29.0, 1.30.0, 1.31.0, 1.32.0, 1.33.0, 2.0.0, 2.1.0 and 2.1.1. 0.36.0, 1.0.0, 1.20.0, 1.25.0 and 2.2.0 resolve such a
cache locally; releases not listed were not tested. A cache written by a download on an affected release carries the
listing and already loaded.
- **Offline with a full 40-hex commit whose snapshot is cached:** the loader now resolves it locally, with no Hub call.
  It first checks that `config.json` and the safetensors index plus every shard it names (or a single
  `model.safetensors`) are present: the files the loader reads.
- **Missing files:** the loader refuses, naming them. A pinned commit with no cached snapshot is refused offline. Both
  refusals are listed in the test suite's loader-refusal guard, so a test hitting one fails rather than skipping.
- **Unchanged:** online loads, branch-name revisions (resolved by huggingface_hub through `refs/<branch>`) and local
  directories behave as before.
