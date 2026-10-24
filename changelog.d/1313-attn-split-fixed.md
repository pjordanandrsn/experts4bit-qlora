### Added

- `E4B_ATTN_SPLIT_FIXED=S` (`1` to `32`; opt-in, not yet read): the paged FP8 attention decode runs a fixed split
  count S at every graph bucket and hands every other per-call choice of `fp8_paged_decode_attention` explicitly
  (`engines/fp8_paged_kv.SPLIT_FIXED_PINS`), so a call differs across buckets only in its row count. It is the
  follow-up lane P223 registered for a cheaper schedule-invariant decode at one request. It replaces any split cap (a
  per-family default included), is refused beside an explicit `E4B_ATTN_SPLIT_CAP`, never runs on speculative verify
  or the f32 path, and is reported per captured bucket in `attn_split_fixed`. Default unset: nothing changes.
