### SC1g A4 box R: the egress pre-flight probes in Python and records why it refused (bench and tests only)

- **What happened.** Box R's first rented attempt, `sc1g-r-6` (RunPod H100 NVL, $0.1348), refused itself at the egress
  pre-flight: "HF CDN 0.0 MB/s".
- **Why the cause is unknown.** The curl probe discarded its stderr, so whether curl was missing from the image or the host
  could not reach Hugging Face's CDN cannot be told from the receipt.
- **The fix.** The probe now runs in Python (`urllib`): the same URL, the same 50 MB range, the same 20 s cap and the same
  20 MB/s floor.
  - It logs the HTTP status, the final CDN host and any error to `logs/egress.log`.
  - It records curl's presence in `forensics.txt`.
  - A refusal line carries the probe's own message.
- **Pin.** R's staging pin is regenerated. The registered rule (`sc1g_ref.py`, `sc1g_kl.py`) is untouched.
