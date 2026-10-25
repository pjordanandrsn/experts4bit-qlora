### SC2f Amendment 8 (pre-data): the qualify wrapper requires the lane driver's success marker

- `bench/sc2/sc2_qualify_drive.sh` now treats a `sc1_drive.sh` exit 0 as clean only with this run's fetched
  `SC1_SUCCESS.<nonce>` and `SC1_EXIT_CODE.<nonce>` reading 0; otherwise it exits 24 (a malformed record).
- The held `sc1_drive.sh` exits the box's rc even when it is 0 without the marker. SC2h and SC2i take this with
  Amendment 7.
