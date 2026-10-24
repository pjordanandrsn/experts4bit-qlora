### P236 Amendment 1: the driver stages again (a literal backslash-n broke its staging loop)

The registration's `bench/p236/p236_drive.sh` carried a literal backslash-n where line 22's continuation belongs, so
bash staged a file named `n` and the driver refused every launch before staging. No box was created. Line 22 is now a
real continuation, and the launch commit becomes this amendment's merge. Nothing else in the lane changes.
`tests/test_p236.py` now runs the driver's dry run for every registered id, refuses unregistered ids, parses every
lane shell script with `bash -n`, and refuses a mangled continuation; each new test fails on the registration's driver.
