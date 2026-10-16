# #1526 confirmation: the held logits, measured

`run_drop.sh` runs `probe_drop.py` (the census probe with one switch, `--drop-output`) twice on one RTX A2000 12 GB, in one
container, back to back: the T = 1,024 native point as recorded (`out = model(...); (out.loss / accum).backward()`), then the
same point keeping only the loss (`loss = model(...).loss`). Model, tokens, switches and versions are the recorded row's.

| run | step 0 | step 1 | step 2 | steady peak |
|---|---:|---:|---:|---:|
| held (control) | 4,976,652,288 | 5,086,730,240 | 5,089,433,600 | 4,853.66 MiB |
| drop | 4,665,487,360 | 4,775,565,312 | 4,777,625,600 | 4,556.30 MiB |
| held - drop | 311,164,928 | 311,164,928 | 311,808,000 | 297.36 MiB |

`T x V x 2` = 1,024 x 151,936 x 2 = **311,164,928 B (296.75 MiB)**. Steps 0 and 1 differ by exactly that. The steady peak
differs by 0.61 MiB more. The control reproduces the recorded row (`L4-T1024-native.json`, 4,852.72 MiB) within 0.94 MiB.

sha256: `L4-T1024-native-held.json` 2c4d5e8f6c4a3cf803478da8343f0190a9b42103346dda9a9f3b17ee20ca4d42,
`L4-T1024-native-drop.json` 55b3c372bbe003756bd3860c4d16c7b9568a49b5eb45c50989d2bdfe5502d32d.
