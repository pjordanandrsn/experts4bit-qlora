### Read: SV6 (#1267): the planner's 24 GB tier plan for Qwen3-30B-A3B served 8,000-token prompts inside its plan, and #1247's bulk flush priced the prompt-length delta to 0.3% (bench and receipts only)

- `sv6-4090-3`: one RTX 4090, **$0.320**, teardown proven. It launched from amendment 1's merge (`af0d95df`) after both
  registrations were reviewed. Lane total $0.372 (`bench/sv6/RESULTS-sv6.md`).
- Read through `bench/sv6/sv6_reduce.py`, verdict `READ`, integrity clean:
  - Y1 HELD: no out-of-memory at 8 × 8,000-token prompts.
  - Y2 HELD: driver peak 21.357 GiB, against a 22.343 GiB plan.
  - Y3 HELD at −1.6%.
  - Y4 MISSED below at −6.4%, as registered likely: the short prompts leave the flush and staging ceilings unused.
  - Y5 HELD exactly: tier rows 5,109 / 1,035 / 0.
  - Y6 HELD: the long-minus-short allocator delta was 1,001.0 MiB against the flush and staging items' 1,003.9 MiB,
    the first measurement of #1247's item.
- **Consequences, as registered:**
  - the tier plan, the reserve and context, and the estimate's long-prompt total, tier pricing, flush and staging
    items stand;
  - the receipts become same-setup evidence for the planner's reserve and context on this class;
  - Y4's below-band miss changes nothing.
