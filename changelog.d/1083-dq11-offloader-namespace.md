### Fixed

- DQ11's tiny A2000 rehearsal now scopes its streaming threshold to the defining
  module imported by pinned Loggetta during preparation. It refuses a missing
  adapter invocation or zero streamed bytes, records the engagement witness,
  and restores the original offloader on success and failure.
