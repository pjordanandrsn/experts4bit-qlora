### Read: TC1 amendment 45 — the threads A/B reads UNTESTED on the busiest host; that container was held to 31 CPUs under 128 threads

- `tc1-5090-98` ($3.02, EPYC 7B13, machine 145701): cgroup `cpu.max` 3071999 / 100000 (31 CPUs) under `OMP_NUM_THREADS=128`, the
  first box to record it. The host's load (4–75) left e4b's `_om1` draws 13.6 % apart, Unsloth's `_om0` draws 5.2 %, and e4b's last `_om0`
  draw unrun at the 4 h guard: P104–P106 UNTESTED. Unquotable readings show no large thread effect on either framework. A re-ask is
  allowed; the harness keeps the physical cores.
