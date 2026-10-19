### DQ11 L-census manifest under the strict launcher (#1083)

The census template and `dq11_l_census_launch.py` keep the census's own record (`status`, the `l_census` closure and the
bound `authorization_basis`) in `lane_record`, because the launcher now refuses unknown top-level manifest keys.
No value, cap, gate or rule changes.
