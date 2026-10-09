### P125 Amendment 3 (#1313): every gate's windows fit its corpus, checked before any arm

- **What happened.** The first reading, `p125-5090-1`, VOIDed: wikitext-2-raw test holds 298,938 tokens, only 73 windows
  at the borrowed loader's 4096-token stride, and the gates registered 108 and 112.
- **The fix.** P125's own loader keeps the same corpus, join and tokenisation, and starts a window every 2048 tokens
  (capacity 146).
- **The tests.**
  - At 4096, the loader reproduces the borrowed one token for token, verified on the real corpus with the pinned
    tokenizer.
  - A windows preflight refuses before any arm if a gate does not fit.
- **The prior and the predictions** are unchanged. The first attempt's speed arms are recorded as history only.
