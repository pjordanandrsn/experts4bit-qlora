### Docs: REPRODUCE names the statistic behind its last four quoted values

The packed-4k device ratio (1.160) and memory gap (1.04 GB) now say they are medians of two draws, and the
bandwidth GEMV ratios (1.2417, 1.5145, 2.4075) say each is the smaller of two ABBA pairs, as their claims state.
No value changes; `statistic_naming_lint` reports no quote in REPRODUCE without its statistic.
