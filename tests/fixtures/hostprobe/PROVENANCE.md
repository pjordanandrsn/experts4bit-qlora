# nvidia-smi captures for `tests/test_hostprobe.py`

Verbatim output, LF line endings, captured 2026-10-20T15:17:30Z on an NVIDIA RTX A2000 12GB (GPU 0), driver
575.64.05, NVML 575.64, CUDA 12.9.

| file | command | sha256 |
|---|---|---|
| `nvidia-smi-help-query-gpu-575.64.05.excerpt.txt` | `nvidia-smi --help-query-gpu` | `c15ce933eaebf627d924aa5caa739ed2668aa4670588b79f117416061e8d5f2a` |
| `nvidia-smi-version-575.64.05.txt` | `nvidia-smi --version` | `65422cc56db1df2224a9d40b87a2792319201563bd5d417fe8f07f6276d1278e` |

The help file is an excerpt. It holds the 11 field blocks `HEALTH_QUERY` names, each unchanged (the name line with its
`or` alias, then the description), in their original order, joined by one blank line as in the original. The full
output was 659 lines, sha256 `457a0add7bc2e7bb8addaac1f38f3a723541caa969de2037fefa20d901aad23c`. The version file is
the full output.
