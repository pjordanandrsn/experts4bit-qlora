# Q235 Amendment 6: rule 8's calibrations

Rule 8's three forms and their calibrations, each run with `q235_box.py --micro-batch 2` on the RTX A2000 ($0) on 2-, 4- and 8-layer cuts
of `ac9c66cc` and on Qwen3-30B-A3B at its 48 layers, with the box's own tokens. `am6_fit.py` reads a calibration's box records and
applies the registered BAR rule. Each JSON here is its output, the per-depth summaries without host paths.

| file | rule 8's quantity | outcome |
|---|---|---|
| `calibration-1-a-vs-b.json` | Shape A against Shape B | STOPPED: two SOFT triggers (exponent 0.676, 30B 3.13x off the fit) |
| `calibration-2-same-shapes.json` | the device accumulation against the same-shapes host sum | one SOFT trigger (30B 3.93x off the fit) |
| `calibration-3-deterministic.json` | the same, in check passes under deterministic algorithms | no trigger: BAR = 0.0231, both plants registered |

The PREREG's Amendment 6 states each as run.
