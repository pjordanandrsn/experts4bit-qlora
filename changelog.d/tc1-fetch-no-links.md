### tc1_drive.sh: the fetch never imports a link

- Both fetches (the incremental `tc1_partial_fetch` and the final fetch) pass `--no-links`, so a symbolic link left
  on a rented box never reaches the controller (bench/common rule 8). No other line of the driver changes.
