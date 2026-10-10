### Rental drivers verify the HF token again: `bench/common/token_scope.py` needs no third-party package (#1313)

- **The defect.** The controllers' token-scope check imported `huggingface_hub`, which the Mac mini controller's python
  lacks. Every check returned "unverified", and every lane launched from there (TC1, P127, FAM, SD1) staged no token and
  downloaded unauthenticated. That is safe by default, but slower and rate-limited.
- **The fix.** The check is now one standard-library GET of the Hub's `whoami-v2` endpoint, with the same 20-second
  deadline, the same read-only rules and the same refuse / unverified outcomes. It never echoes the token or a response
  body.
- **The tests.** They fake the network at `urllib`. One proves the check verifies with `huggingface_hub` unimportable. A
  network failure, a 401 or a non-JSON answer stays "unverified", and stages nothing.
