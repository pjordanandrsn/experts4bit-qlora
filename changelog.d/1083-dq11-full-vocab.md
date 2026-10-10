### Fixed

- DQ11's tiny rehearsal keeps the full 32000-entry vocabulary and original token
  IDs so the original full-logits scorer can run unchanged. A successful tiny
  streaming preparation saves its engagement witness before scoring or proof,
  preserving real handles and streamed bytes even if a later phase refuses.
