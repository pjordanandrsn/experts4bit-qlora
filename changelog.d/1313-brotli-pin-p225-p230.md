### Changed

- P225 Amendment 3 and P230 Amendment 1 (pre-data): each runner's install line adds `brotli>=1.2.0` beside the unpinned
  `huggingface_hub>=0.23`, whose newest releases decode brotli with an API older brotli builds lack, so a model fetch
  could die with a TypeError on a br-encoded edge response. The runners' staged pins move with it; no need, alarm,
  floor, rule, cap or id changes.
