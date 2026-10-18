### Changed

- P146 Amendment A1: a host-quiet guard after `p146-4090-5` VOIDed on its cross-process self-pairs while the host got noisy mid-box. Each arm records its cgroup throttled time and `/proc/stat` steal over its timed window, and a pinned H2D probe runs before the first arm and after every arm. Above 5 % (CPU) or 15 % below the first probe (link), or a signal not recorded, is VOID_BY_HOST: no reading and not a strike. Every other term is unchanged.
