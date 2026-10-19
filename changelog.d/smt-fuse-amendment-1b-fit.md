### SMT fuse Amendment 1b: the fit from the max over hosts; host counters reported only

The smt-sib time fit takes each term's max over the measured hosts plus 25 % headroom (12474 s; guard 4.0 h; worst
$2.43 per box). Every timed block now records fault, NUMA/THP and page-cache counters, reported only. The SMT lane
closes on smt-sib-2's reading; the page-cache observation goes to its own registration.
