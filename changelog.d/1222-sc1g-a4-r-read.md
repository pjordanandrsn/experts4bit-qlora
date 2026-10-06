### SC1g A4 box R read (#846): R_NOT_OK, so the KL65 grading is UNREAD; the reference's own NF4 fake-quant reads below the true model where its KL is largest, supporting A3's flattery hypothesis (bench and tests only)

- **The run.** `sc1g-r-8` (H100 NVL, $0.5556) ran to its verdict, and the registered gate refused the bucketed estimator:
  - coverage 0.77–0.98, bar 0.99;
  - KL65 / KL_full 0.75–0.91, bar 0.90;
  - the floor F reaches 2.2e-2 on wikitext and 6–7e-3 on conv1/conv2, bar 1e-2.
- **The descriptive reading.** The true model reads 0.884 on conv1, where e4b MXFP4 served reads 0.905 and NF4 served 0.736.
  The reference's own NF4 fake-quant drops NLL by 0.118 and 0.139 on conv1 and conv2, at full KL 0.108 and 0.139. MXFP4's −0.074
  on conv2 is unexplained.
- **Receipts.** Box R's receipts are committed with every attempt, r-1 to r-8 (box R total $0.6904).
  - `sc1g_ref.py --reverdict` re-derives R's verdict from them.
  - A test pins R's whole rule (digest `ee122b74…`) to its registration.
- **Next.** A5 moves to A4's registered full-vocabulary fallback, with a per-window floor gradability rule.
