### DQ11 L-census: the envelope binds its one executor (#1083)

`dq11_l_census_launch.py` required one fixed `POD_LAUNCH_AGENT` identity before discovery. The envelope now binds the
executing seat as `executor_sha256` (the SHA-256 of its `CTO/...` identity), and the launch refuses before any search
unless `POD_LAUNCH_AGENT` is exactly that seat; the bound manifest's authorization basis and the receipt record it. No
value, cap, gate or rule changes.
