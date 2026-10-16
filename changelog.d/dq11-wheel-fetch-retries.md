### Retry transient DQ11 census wheel transfers without changing locked bytes

The census bootstrap prefetches exact locked wheels with at most five attempts
over a three-minute budget per wheel. Only connection resets and timeouts retry,
using the same URL; SHA256 mismatches and HTTP refusals stop immediately. A
separately hashed operational helper records each attempt under receipts before
the unchanged science installer rechecks the cache. The existing bootstrap phase,
paid wall clock and teardown reserve remain unchanged.
