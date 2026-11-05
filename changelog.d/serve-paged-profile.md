### `serve_paged`: a measured routing profile for the placement solver (`E4B_PAGED_PROFILE`)

- `E4B_PAGED_PROFILE=<path>` hands an `engines/expert_profile` JSONL to the placement solver. The solver already took
  a measured routing mass, but `serve_paged` always passed none, so its VRAM and DRAM tiers were chosen under
  uniform-assumed mass. Unset, nothing changes.
- The profile is checked at startup, before any weight is read, and refused by name when:
  - the file is missing;
  - it holds no routed tokens;
  - any of its (layer, expert) pairs is outside the served model's MoE layers x experts.
- The startup log and `/health` (`levers.placement_profile`) record the path and the profile's kind and sha256.
