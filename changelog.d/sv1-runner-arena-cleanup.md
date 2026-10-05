### SV1's runner deletes its arenas before it finishes (bench only)

- `bench/sv1/sv1_run.sh` baked its arenas under `/root/tc1`, the directory `tc1_drive.sh`'s final rsync copies back
  whole. `sv1-5090-1` left the 16 GB Qwen3-30B-A3B arena there, and the fetch would have taken hours of billed box
  time. The lane owner stopped it after saving the receipts.
- The runner now removes `arenas/` after the last arm and on every early exit that baked one.
