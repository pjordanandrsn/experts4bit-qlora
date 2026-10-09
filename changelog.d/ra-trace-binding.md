### Bind RA capacity completions and buffered traces

The release-anchor capacity wrapper now closes its drained native engine before
joining HTTP completions to request traces and reconciling aggregate step and
graph counters. CPU checks do not grant GPU engagement or release clearance.

- Preserve the owned capacity server receipt and original failure when its post-kill wait times out.
