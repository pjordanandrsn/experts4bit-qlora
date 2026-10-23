### P231 Amendment 1 (pre-data): the fetch imports no links

P231's driver now fetches the box's records with `rsync --no-links`, so a symbolic link left on a rented host is never
copied to the controller, where the receipt writers would follow it (bench/common rule 8). Nothing else in the lane
changes; the launch commit becomes this amendment's merge. `tests/test_p231.py` checks every rsync in the driver.
