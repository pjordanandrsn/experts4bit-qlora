### P237 amendment 1: the host floors as the market and a container see them

- `bench/p237/p237_run.sh`: rc 16 reads the RAM the process can use, which is min(MemTotal, every cgroup memory limit on its path),
  through `p237_host.py ram_gib`. A container's MemTotal is the whole host's, so the registered line would have passed a 64 GiB
  slice of a 1 TB host. An unread value stops the box (9).
- `bench/p237/p237_host.py`:
  - `vast_cpu_class` maps a Vast offer's `cpu_name` (`Core™ i9-14900K`) onto the registered CPU class, which is unchanged;
  - `SCREEN_MIN_LISTED_RAM_GIB` is the screen's listed-RAM floor;
  - `wall_s` is the shake's launch wall: the fit plus 1,500 s of slack, 18,000 s (5.0 h), worst USD 5.35, with no override;
  - the 4090 price cap rises from USD 0.60 to 0.85/h (`USD_PER_H`; `SCREEN_MAX_USD_PER_H` is bound to it). The cap turns
    USD-bound at 16.35 h, the reading's worst is USD 13.85, and the lane total USD 19.70.
- The reason is a $0 census of the RTX 4090 market (111 offers): none had both a fast CPU class and 128 GiB listed; a later
  census found one (a Core Ultra 9 285K at 188 GiB) at USD 0.80/h, over the registered USD 0.60/h. The CPU class
  stays as registered. The shake loop waits up to 48 h for a host that passes both floors, then the lane reports.
