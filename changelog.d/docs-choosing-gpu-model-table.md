### Docs: CHOOSING has a "which tool for my GPU and model" table

`docs/CHOOSING.md` ends with a lookup table for the cards and models already measured: an RTX A2000, RTX 4090,
RTX 5090 and H100 NVL, with Qwen3-30B-A3B, Mixtral-8x7B and Qwen3.6-35B-A3B. Each of its 11 rows names the mode to use for
training or serving, one number from a measured run with the statistic it is, and the claim in `docs/claims.json`
that holds it. A card or model not listed is not measured, and the constraint sections above the table still decide it.
The page's introduction points to the table. No claim or value changes.
