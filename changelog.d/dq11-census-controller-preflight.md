### Validate census controller admission before the fresh CPU gate

The DQ11 L-census launcher checks its executor identity before discovery, runs
the controller's write-free `--validate` on the exact bound manifest, and keeps
validation and controller stdout, stderr and exit status in its private packet.
A validation refusal stops before the CPU gate, quote or paid controller call.
