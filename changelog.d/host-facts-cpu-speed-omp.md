### Host facts: per-core speed and the OpenMP thread count

`bench/common/host_facts.py` adds three facts. Each is a number, and each is left out when it cannot be read.
- **`host_cpu_max_mhz`:** the CPU's rated maximum clock, read from cpufreq. It is a label only.
- **`cpu_loop_mops`** (with `cpu_loop_python`): a fixed single-thread loop, best of three, well under a second. It
  measures what the run actually got from one core, after clock, co-tenants and quota.
- **`omp_num_threads`:** the run's own setting.

Why: in the TC1 receipts, a framework's speed ratio at fixed code moved 10-17% between host CPU models. It moved with
per-core speed and the thread count, not with the core count, and no receipt recorded either.
