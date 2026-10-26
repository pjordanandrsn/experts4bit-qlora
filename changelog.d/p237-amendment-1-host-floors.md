### P237 amendment 1: the host floors as the market and a container see them

- `bench/p237/p237_run.sh`: rc 16 reads the RAM the process can use, which is min(MemTotal, every cgroup memory limit on its path),
  through `p237_host.py ram_gib`. A container's MemTotal is the whole host's, so the registered line would have passed a 64 GiB
  slice of a 1 TB host. An unread value stops the box (9).
- `bench/p237/p237_host.py`:
  - `vast_cpu_class` maps a Vast offer's `cpu_name` (`Core™ i9-14900K`) onto the registered CPU class, which is unchanged;
  - `SCREEN_MIN_LISTED_RAM_GIB` is the screen's listed-RAM floor;
  - `wall_s` is the shake's launch wall: the fit plus 1,500 s of slack, 18,000 s (5.0 h), worst USD 4.10, with no override.
- The reason is a $0 census of the RTX 4090 market (111 offers): none had both a fast CPU class and 128 GiB listed. The CPU class
  stays as registered. The shake loop waits up to 48 h for a host that passes both floors, then the lane reports.
