### Changed

- R2-Mixtral reads Q235 at the commit it cites, not Q235's working tree (#1630). `r2_drift.cited_tree` extracts Q235's files at
  `CITED` from git, and R2's wrappers (stage, fetch gate, verdict) run Q235's tools from it. `r2.sha256` seals R2's own files from
  the tree and Q235's at `CITED`. A Q235 change therefore no longer breaks R2's tests or needs a re-seal in Q235's PR. Launch-time
  staging still hashes the working tree, so once Q235 moves, R2 refuses to stage until it re-cites: it never ships uncited Q235 bytes.
