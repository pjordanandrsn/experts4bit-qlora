### Host facts: THP_MEM_ALLOC_ENABLE, a stdout mode, and one producer

`bench/common/host_facts.py` is now a byte-identical copy of adertha-agents' `adertha/compute/host_probe.py`. That file
is the source of record, kept beside the host registry that reads its keys, and drift is checked there.
- **`thp_mem_alloc_enable`:** the setting from the helper's own environment, recorded only as an integer.
- **`-` mode:** prints one `HOST_FACTS_JSON` line instead of writing a file.
- **Every rented box:** the launcher now runs the same probe after its pre-flight, so a box's transparent-hugepage
  mode reaches the registry whichever lane rented it.
