### Docs

- "Identical tokens" is scoped to identical schedules. The serving docs' token-identity claims for the K25 decode and
  plan (P158, P161), the KV step selection (P111), PDL (P113) and the launch-bound glue (P127) compared arms on the same
  prompts, lengths and arrival order, and now say so. SERVING.md explains why. The paged FP8 attention decode plans
  its split count per graph bucket, and the split-KV combine agrees across split counts within tolerance, not bit for
  bit, so a step can pick a different near-tie token when its bucket changes. A second, much smaller difference shows
  at bucket 1 only and is not attributed (P149's follow-up, runs `p149mnt-5090-6` and `p149mnt-5090-7`, in the private
  receipt store). P158's and P161's RESULTS carry a dated note; their findings are unchanged. Wording only: no value
  changes.
