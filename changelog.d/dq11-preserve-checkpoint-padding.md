### DQ11 preserves the checkpoint padding row

The matched DQ11 Unsloth arms restore the original checkpoint padding embedding
row after loading and record that deviation and scored token-0 counts, preserving
strict frozen-value equality after a real-model refusal exposed loader zeroing.
