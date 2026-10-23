### P228 Amendment 7 and P229 Amendment 6: the fetch never imports a link

- Each drive fetches the box tree with `rsync -az --no-links`, so a symbolic link left on a rented box never
  reaches the controller (bench/common rule 8). Only the fetch line moved; no staged pin and no runner byte changes.
