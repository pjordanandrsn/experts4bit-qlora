### TC1 harness: an opt-in census of every grouped-LoRA delta call's single padded block (`TC1_PAD_CENSUS=1`)

- `tc1_arm.py` wraps grouped-nf4-gemm's `lora_delta_grouped` when `TC1_PAD_CENSUS=1` and records, from host facts only, each call's
  single padded block. The receipt's `pad_census` gives, per projection input width, quantiles of the block's rows (G × widest), of the
  routed rows and of its bytes in the allocation dtype, plus the share of calls at or above 0.25-4 GiB. `tc1_drive.sh` forwards the
  variable. Off, nothing is wrapped.
- Amendment 49's re-ask runs it on every arm; its sizes place a size gate for bucketed padding if the field recipe pays for buckets.
