### Docs: STATUS names the statistic behind each quoted result

All 36 quoted results in `docs/STATUS.md` now say which statistic their claim is decided by, in the sentence or table
row that quotes them: medians of two draws for the training head-to-heads and step ratios, the smaller of two (or the
smallest of four) ABBA pairs for the decode speed-ups, a median or mean over replays or processes for the prefill reads,
and the interleaved-block medians for the fold-stack reads. No value changes; only the touched paragraphs are re-wrapped.
