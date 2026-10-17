# R2-Mixtral: the time-left fit (Amendment 3)

The fit behind `r2_run.sh`'s sealed caps and pull floor. It is `r2_fit.py` run on R2's A2000 rehearsal, which is published as an
immutable artifact (sha256 `5470cfae6cc4128466c28c57e685c6f0d0a1248ff7fdc3529a05c9d97b3d8d84`). That artifact is
`r2_run.sh` in `a2000` mode on a 4-layer slice of the subject, at e4b `a3e73429`. Every receipt in it is byte-identical to the leg's.
Its `warm-fetch.json` is the leg's own cold fetch (`receipts/fetch.json`: 25.25 GB, hashed in 19.3 s).

    python bench/fam235/r2_fit.py --rehearsal r2-a2000-rehearsal --warm-install-s 147 --bandwidth-mbps 500 --usd-per-hour 0.85
    python bench/fam235/r2_fit.py --rehearsal r2-a2000-rehearsal --warm-install-s 147 --bandwidth-mbps 500 --usd-per-hour 2.00

- `fit-500-0.85.json` and `fit-500-2.00.json`: both exit 0 at this commit. Every phase fits, the sealed floors match, and the
  worst-case wallclock is 4.57 h (3.88 and 9.13 USD).
- The install figure, 147 s, is the cold-cache install from the first leg at `14dc45ba`, which shared the workspace and VOIDed at its
  fetch. The leg itself installed on the warm cache in 40 s.
- `listing-20261017T0010Z.json`: how many current RTX 5090 offers meet all of R2's floors, from a read-only offer search. It records
  counts only.
