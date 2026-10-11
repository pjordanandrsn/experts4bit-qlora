# DQ11 L-only device census before a fusion BUILD registration

Issue #1083. Prospective draft, 2026-10-11. **DRAFT_NOT_LAUNCHABLE.**
This PR registers a reporting census and its decision; merging it costs $0
and authorizes no quote, rental, implementation merge or timing claim.
A reviewed instrument closure, CPU gates and fresh maintainer launch ACK
must precede the single paid draw. No census data have been collected.

## Question and subject

Measure how much of L's own update belongs to each adapter class before
choosing a dense fusion BUILD target. DQ1's standalone BF16/Qwen adapter
costs are not a ceiling on Mistral FP32 L, and DQ11's U0/U headroom is on a
different backbone. Neither substitutes for this census.

Use exactly DQ11's L science path at e4b
`13b55990433f357d63232f75926ce19f5bda5b42`, Loggetta
`34ecb6cec6f43a6f8607ff9f192749fdc7b587e9`, and Mistral-7B-v0.1 revision
`27d67f1b5f57dc0953326b2601d68371d40ea8da`. The sealed model, tokenizer,
canonical FP32 adapter initializer and token files are unchanged. Retain
the DQ11 draw4 science closure, resolved wheel hashes and native SDPA path;
xformers and FlashAttention packages remain absent. GNF4, if installed,
is `724ccc454f006c1a46836e434e997f31f293747f`; dense kernels are not imported.
No model, source, dependency, attention or arithmetic fallback.

Keep all seven targets, rank16/alpha32/dropout0, NF4 frozen bases with BF16
base compute, FP32 adapters, micro-batch1/accumulation1/sequence2048,
non-reentrant checkpointing, eager SDPA/full logits, TF32 off, no autocast,
cache off, seed3407 and the exact forty-update AdamW recipe. Keep streamed
admission, margin, pinned homes, late-bound frozen-weight backward,
`train_prefetch=True` and resident trainable adapters. Admission refusal
ends the draw. No allocator/reserve or capacity policy changes.

## Instrument and pre-run gates

The census instrument is a separately sealed reporting wrapper around that
unchanged L path. Its future source pin is **UNKNOWN** until review/merge;
installed L callables and source authority must still match the 13b closure.
The instrument amendment must fix one profiler/exporter, exact versions,
class mapping, raw schema, capture bounds and reducers before any rental.
Use correlated CUDA kernel/node capture with CPU/autograd attribution
(`torch.profiler` or an nsys graph/node exporter), not kernel-name guesses
or inclusive module wall times. Do not select the tool or classifier after
seeing the deciding data. DQ1's isolated event timings cannot do this job.

CPU fixtures must falsify duplicate parent/child counting, two operator/kernel
views, clipped-name collisions, overlapping streams, unbound kernels,
backward attribution, checkpoint recomputation/early stop, wrong source or
input hashes, dropped steps and nonclosing totals. Reuse the coverage lessons
in `bench/hybrid-g9/f1/step_budget.py`; its generic name classifier alone
cannot distinguish dense adapter classes. Unbound adapter events or ambiguous
base/adapter attribution REFUSE rather than entering a favorable class.

In-box canary precedes the census: exact host/driver/environment, admitted L,
actual native SDPA forward/backward, all 32 layers and 448 FP32 adapter
parameters, finite loss/gradients, frozen bases unchanged and actual source
bindings. Use the existing configuration-C same-arm observer-removal bitwise
proof and unchanged DQ11 initial-quality/input checks. Remove canary observers
before the baseline. Failure ends the draw, retains evidence and tears down.

## Fixed collection and attribution

One RTX5090, one physical host, one seed, fresh isolated processes:

1. **Unprofiled reference:** exact L forty updates; discard only the five
   registered trained warmups. Reference time T is the median of updates6..40.
2. **Reporting capture:** reset canonical initializer and empty optimizer;
   run the same forty updates. Capture exactly updates6,7,8 after the five
   trained warmups; preserve every captured update. Profiling ends after8.
   Record source/bindings, losses and initial/final quality for both processes.

No second arm or fusion treatment. No best-window selection, rerun, new seed,
shape change or profiler-enabled speed comparison. Profiler overhead is
reported; profiled wall times are never substituted for T. Existing DQ11
attention proof remains untimed and is removed before collection.

Attribute every device event once by its launching operation and correlated
forward/backward node to exactly one class: **MLP-adapter, qkv-adapter,
o-adapter, base NF4 GEMM, attention, optimizer, other**. Base class includes
its necessary dequantization; adapter classes include their rank GEMMs,
adapter casts/scales/adds and gradients, including checkpoint recomputation.
Attention excludes qkv/o projections. Streaming copies, norms, RoPE, logits,
loss and unrelated operations belong to named subrows of other. Optimizer
includes its gradient clipping/update work. Report ambiguous events separately
and refuse a certified adapter ceiling until resolved by the sealed map.

Report per-update kernel-duration sums, interval unions, counts, per-class
forward/backward detail and cross-stream overlap. Count the kernel view once,
never sum a region with its children or both the op and kernel views. For
wall closure, partition the device timeline into disjoint class-only intervals,
explicit mixed-class overlap and no-device/host gaps. Show the seven classes
plus those explicit residual rows summing exactly to each captured update's
synchronized wall interval. Keep raw durations alongside this partition;
summing overlapping kernel durations is not wall closure. Aggregate means
close additively; separate class medians are not claimed to sum to a median.

Define candidate C_j for each captured update as the union duration of ALL
certified device intervals belonging to class j, including its overlap with
other classes. This optimistic counterfactual erases the entire measured class, including
necessary arithmetic/output work that a real fusion cannot generally erase. Report C_j in ms and C_j/T.
Overlapping work can only reduce realized wall savings. Host dispatch/gaps
remain residual and cannot be credited to an adapter without new evidence.
Thus C_j is the certified **addressable device-time ceiling**, not a promise
of end-to-end speedup or an upper bound on all possible host-side savings.
Take C_j = minimum of the three captured per-update ceilings for the decision;
retain median/range too. If attribution cannot certify all three, no decision.

## Decision fixed before data

The prospective BUILD continuation bar remains 5% of an independently matched
unfused update, subject to unchanged K8. This census admits MLP BUILD only if
**C_MLP >= 2 * (0.05*T) = 0.10*T**. The factor2 requires room for a candidate
that removes only part of its addressable device time. DQ11's observed L/L
median spread is context, not a threshold to tune against the new data.

If MLP fails this line, rank all three candidates by their certified C_j.
Only a candidate clearing the same 0.10*T line may be proposed for a separately
reviewed BUILD registration. A tie goes to no selection pending review. If
none clears it, **close the BUILD lane with the census as its record**. A
finite small ceiling is a negative finding, not a reason to lower the bar.
Missing identity, attribution, captured updates or closure yields
**CENSUS_REFUSAL**, no BUILD decision. Preserve failures and perform teardown;
no automatic paid retry. Any retry requires prospective review and fresh ACK.

## Budget, launch and records

One draw, proposed total ceiling **$0.90**, including GPU, storage, bandwidth
and teardown. Proposed hard wall cap **30 minutes from create**, last5minutes
reserved for receipt transfer and destruction. No second draw is implied.
Exact controller/instrument pins, capture/transport size caps, physical
machine/offer, compatible native-SM12.0 driver, RAM/disk, all-in quote and
phase caps are **UNKNOWN until the reviewed launch amendment** and block rent.
Worst-case bytes and every URL must clear the sealed CPU gate before quote;
quote price alone is not a total-cost gate. A fresh stable-host machine/offer
pin and maintainer ACK are mandatory. No cheapest-host substitution or use of
the unmerged offer-set draft. Do not displace an A2000 lease or spend another
lane's budget. This DQ-owned small census may precede SC2f.

Receipt closure: unmodified raw controller receipt, pinned manifest and source
closure, canary, raw reference/capture steps, profiler/node data, attribution
and overlap tables, residuals, CPU gate reports, decision JSON, all-in actual
spend and authenticated destroy/list-after proof. Publish hashes and immutable
review artifact references through the bus; preserve raw evidence privately
when too large for the broker's 32 MiB cap, without splitting to bypass it.
Require 100% event coverage and interval closure within 1 microsecond per update
(rounding only) before calling a ceiling certified. Preserve exact timestamps;
unknown events, a missing adapter gradient or a nonfinite loss/gradient refuse.
The final reference/capture quality comparison must retain unchanged K8
absolute |delta PPL|<=0.05 on both Alpaca heldout and Wikitext test; quality
failure refuses a BUILD recommendation. No quality threshold is retuned.

Reporting is descriptive and private until a separate RESULTS review. No
public speed magnitude, fusion implementation, default change, capacity claim
or BUILD launch follows this preregistration or a passing census.
