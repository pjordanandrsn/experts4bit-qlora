# DQ11 L-only census launch amendment

Prospective amendment to [the L-only preregistration](DQ11-L-CENSUS-PREREG.md)
and [the merged instrument](DQ11-L-CENSUS-INSTRUMENT.md), issue #1083.
**PROSPECTIVE; merge alone does not authorize a paid draw.** This amendment
fixes the operational limits and prelaunch evidence. It changes no model,
input, arithmetic, capture window, attribution rule, reducer, quality gate or
BUILD decision. A maintainer envelope ACK, issued after this amendment merges,
is required before one paid draw. The prior read-only host search qualified a
five-minute window but the host disappeared before the quote; no rental or
5090 census occurred.

## Source and evidence boundary

The science path remains e4b `13b55990433f357d63232f75926ce19f5bda5b42`,
Loggetta `34ecb6cec6f43a6f8607ff9f192749fdc7b587e9`, and the existing
`science.sha256` closure `4691e7329f12ad2f4b7518b75c4d3e351207fc4e5220aee514fdd42c429fef01`.
The instrument was reviewed and merged in e4b `6e197a3f42599c8d10667a4c133627496e40f28f`.
The paid controller must bind its own clean, merged e4b amendment source and
the merged `adertha-agents` controller source in the manifest; those final
source SHA-256 values are **UNKNOWN before this amendment merges**. It must
stage and hash the three instrument Python files separately from the sealed
13b science files and verify both closures on the box. A science source change
is a refusal, not an implicit update to the instrument head.

Two $0 evidence lines are distinct. The real-Mistral A2000 rehearsal was
sealed at instrument head `7babe2f0` in the immutable review packet
with SHA-256 `e51978ca51685fa65cca39e4170e238a3203241a15cc9c51ab48c169171d6255`.
It had 39,333/39,333 captured device events classified and 0 ns interval
closure residual, with `science_eligible=false`. The $0 re-normalization of
that same raw A2000 trace at head `5405211c` again assigned 39,333/39,333
events after the prospective native-SDPA creator additions. That second line
does not claim a new full A2000 model run or a seal at the final lint-only head.
Neither line measures a 5090 ceiling or proves sm_120/cu130 profiler support.

## One-draw resource and transport ceiling

One normal 32 GB RTX 5090 on one physical verified-secure Vast host, driver
at least 580, 320 GB ordered storage, at least 98 GB host RAM. The box must
prove exact card, driver, CUDA/Torch and CUPTI correlation before collection.
The hard paid wall clock is 30 minutes **from instance creation**. At least
the last five minutes remain for evidence transfer and authenticated
destruction; phase alarms take the lesser of their registered cap and time
remaining before that reserve. Timeout or missing proof ends in
`CENSUS_REFUSAL`, evidence retention and teardown, with no automatic retry.

This wall clock is derived from the same 5090/L draw4f receipt rather than a
phase-cap sum. Instance creation was `01:52:50Z`; `tc1/summary.txt` began
`proof-L` at `02:01:06Z`, so provisioning, SSH, the 17.6 GB input transfer,
100-wheel bootstrap and model preparation consumed **496 s** together.
The draw4f process receipts give `proof-L` **20.346 s**, `read-1-L`
**50.506 s** and `read-2-L` **50.501 s**. From the `read-2-L` start at
`02:06:58Z` to the provisional reduction line at `02:09:10Z` was **132 s**;
using that full tail as the census reduction/closure allowance deliberately
counts the last L read twice. The conservative measured subtotal is
`496 + 20.346 + 50.506 + 50.501 + 132 = 749.353 s`.
At **1.5×**, that is **1124.030 s**; adding the **300 s** reserved for packet
transfer and teardown gives **1424.030 s = 23.734 min**, leaving **375.970 s**
within the registered 1800 s wall. The separate 40-update profiler capture is
budgeted using the second measured L process plus the 1.5× margin; its actual
5090 overhead is unknown until this draw, and an overrun is a refusal. This
arithmetic does not turn a timeout into a retry or a BUILD result.

The all-in ceiling is **$0.90 for one draw**, including GPU, ordered storage,
download, upload and teardown. The manifest's GPU-plus-storage ceiling is
`$0.85/hour`, so 30 minutes reserves `$0.425`. The exact registered input is
17,609,017,716 bytes (model, 100 locked cu130 wheels, tokens and initializer).
The prospective **download allowance is 27 decimal GB**, which exceeds its
1.5-fold 26.413526574 GB margin and leaves room for the small sealed source
and controller files. The maximum download tariff is `$0.011/GB`, reserving
`$0.297`. Raw proof and capture traces each retain the instrument's 512 MiB
and 2,000,000-event refusal caps. The **entire transferable evidence packet**,
including both traces, normalized events, receipts, logs and hashes, is capped
at **2 GiB uncompressed**; excluded model/wheel caches and venv cannot be
smuggled into it. Reserve **3 decimal GB upload** and require a quoted upload
tariff no greater than `$0.05/GB`, reserving `$0.150`. The maximum sum is
`$0.425 + $0.297 + $0.150 = $0.872`, below `$0.90`. A missing tariff, a larger
packet, or any quote whose actual all-in calculation exceeds `$0.90` refuses
before create or transfer. Do not split a large trace across the 32 MiB bus
broker limit; preserve it privately by SHA-256 or record a size refusal.

The driver still enforces its own policy cap, ledger, ownership fence,
download tariff, one-host and teardown rules. The launch admission is
additional: its quote must check the actual ordered-storage hourly rate,
download and upload tariffs, machine/offer identity and these reserves. An
offer with absent upload pricing is not treated as free. Actual invoice lines
for GPU, storage, download and upload and a post-destroy empty live list close
the receipt; the reservation is not represented as actual spend.

The committed [manifest template](dq11_l_census_manifest_template.json) sets
`usd_per_hour=0.85`, `wallclock_h=0.5`, `est_usd=0.425`,
`download_gb=27`, `max_download_usd_per_gb=0.011`, `storage=320`, the
previously used devel image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`,
and a command that binds the validated canonical asset path and exact driver to
`bench/dq11/dq11_l_census_drive.sh`. Its `lane_record.l_census` object
binds the 13b science head and manifest hash, per-file SHA-256 for the three
merged instrument files, three operational files and the shared transport
controller, `raw_trace_max_bytes`
536,870,912, `packet_max_bytes` 2,147,483,648, `upload_gb=3`,
`max_upload_usd_per_gb=0.05`, and `all_in_cap_usd=0.90`. The final manifest
also binds both clean merged repository heads, exact driver, `vast_pin_machine`
and the full `vast_pin_offers` list. Those late-bound values are read from
current evidence, never guessed in this amendment. The final box worker
records instrument/operational hashes separately from the unchanged science
closure. The shared driver transfers only after its remote 2 GiB file/sha
gate; it excludes bulk trace/normalized files from partial polls and checks
the final fetched files against the box-side SHA-256 list.

## Envelope ACK and final sequence

After this amendment merges, the maintainer may issue one citable GitHub
comment on issue #1083 as the **envelope ACK**. The comment body is the exact
JSON envelope fields without `approval_url`; the local record adds that URL.
The launcher fetches the comment before discovery and again before paid
invocation, verifies the `pjordanandrsn` author and exact JSON, and refuses on
an unreadable or changed comment. The JSON `issued_at` must be within 120
seconds of GitHub's `created_at`, `expires_at` must be exactly 24 hours after
`issued_at`, and GitHub's `updated_at` must still equal `created_at`; an edited
comment refuses. Its local record has schema
`dq11-l-census-envelope-ack/1`, that comment URL, issue and expiry UTC times,
the exact clean merged e4b and controller heads, the committed manifest
template SHA-256, `maximum_draws=1`, `maximum_attempts` equal to its explicit
`run_ids` list length (one through ten unique IDs matching
`dq11-l-census-[0-9]+`), and the exact criteria in
`ENVELOPE_CRITERIA` in the launcher. It lapses at the earlier of 24 hours after
issue or any change to those source/template bytes. An absent, future,
expired, longer or broader record refuses before a provider search; it is
checked again immediately before invoking the paid controller. The controller
records the same GitHub approval URL. The template uses
`__DQ11_CENSUS_RUN_ID__`; substituting `--run-id` changes only that field.
A later envelope may declare fresh IDs without changing the template. Any ID
with a preexisting receipt, run directory or ledger row outside this envelope
journal refuses. Each invocation performs one attempt with a new packet directory and all fresh
gates. Before discovery, a durable journal in
`~/.local/state/dq11-census-envelopes/` burns the selected ID under an exclusive
lock held through controller completion. The journal key is the ACK comment ID.
The initial `admitted` phase becomes `controller-invoked` durably just before
the paid controller call. A caught pre-controller exception becomes
`not-invoked`; another ID may proceed only if that attempt has no receipt,
run directory or ledger row. Process death in `admitted` or an uncertain
controller outcome fails closed. A reused ID refuses. Before another
ID is admitted, every used ID must have a complete canonical `REFUSED` receipt
with `instance_id=none`, actual cost zero, `not-launched` teardown proof and the
same approval URL. Counted receipts and the ledger must be tracked and clean
at the receipt repository HEAD, including the index. An absent receipt for
an admitted/controller-invoked attempt, orphan ledger row, unfinished attempt,
created instance or changed envelope stops for the maintainer. The journal
survives process death; canonical receipts also prevent resetting it to gain
another draw. At most ten IDs can be attempted; at most one instance can be
created. Commit each finished receipt before requesting the next attempt; no
controller or science retry is automatic. An envelope is not a result or a license to relax any
registered limit.

Under that ACK, one GET-only standing poll searches all eligible offers with
the same merged controller adapter, every 20 seconds for at most three hours.
The first machine to remain in **every poll for at least 300 seconds**, with
gaps no longer than 30 seconds, qualifies. Every returned ID must be eligible:
one verified rentable, nonexternal normal RTX 5090, reliability at least
0.98, exact three-component driver at least 580, at least 98 GB host RAM and
320 GB disk, GPU plus storage at most $0.85/hour, download at most
$0.011/GB, upload at most $0.05/GB, and the 45-minute all-in reserve at most
$1.10. The controller's existing GPU memory floor also remains binding. A
failed search, absent machine, ineligible or outside-class ID, driver change
or gap resets that machine's clock independently. Offer IDs may rotate; the
complete observed **union** for the first qualified machine becomes its exact
inclusion set. Each poll and the selected five-minute window are saved and
SHA-256 bound. The observer uses only an auth probe and offer-search GETs.

Before discovery, the launcher requires `POD_LAUNCH_AGENT` to identify
`CTO/codex-desktop/dense`. After binding the manifest, it runs the merged
controller's write-free `tools/pod-launch.sh --validate` on that exact file and
approval URL before the CPU gate and quote. Any validation refusal stops there.
The packet retains validation and controller stdout, stderr, exit codes and
output hashes, including a refusal before the controller creates a receipt.
The existing 60-second final-search clock still includes this validation time.

Immediately after qualification, the launcher fills the committed manifest
template with that machine, exact driver and observed offer union. It pins by
`vast_pin_machine` **and** `vast_pin_offers`; neither an unpinned fallback nor
an offer outside the union is permitted. The source and template hashes, all
sealed inputs and 100 locked wheel URLs, and the CPU-only policy probe are
checked against that bound manifest. A fresh read-only quote needs at least
one currently listed approved ID, no outside or ineligible ID, and every
tariff and all-in cap. The launcher then calls the guarded controller **once**
only if the last successful search to actual invocation is at most 60 seconds
and gate completion to invocation is at most 60 seconds. The controller's own
pinned search must agree or refuse without substitution. A timeout, changed
source, changed template, expired envelope or market churn refuses before
create. The receipt records the envelope ACK URL and SHA, every poll SHA,
qualifying-window SHA, machine and complete offer union, bound manifest,
gate/quote SHA and freshness clocks. The previous qualified-but-vanished host
has no reusable paid authority.

The in-box order is source/input/admission, native-SDPA/CUPTI and same-arm
bitwise canary, fresh unprofiled 40-update reference, fresh 40-update capture
of updates 6–8, reducer, capped private evidence transfer, authenticated
teardown and empty-list proof. A refusal at any stage licenses no BUILD choice.
The only decision is the preregistered `C_j` rule; a passing MLP ceiling is
still only a BUILD proposal for separate review. No public speed, default,
capacity or shipping claim follows this census.

## Manifest layout under the strict launcher (2026-10-18)

The launcher refuses any top-level manifest key it does not read (adertha-agents#243). So the census's own record sits
in `lane_record`, which the launcher copies verbatim into `receipt.configuration.lane_record`:
- the template's `status`;
- the `l_census` closure;
- once bound, the `authorization_basis`.

Top level carries only launcher terms: the template's fields plus the bound `heads`, `command`, `vast_pin_machine`,
`vast_pin_offers` and `vast_required_driver`. This moves fields and changes no value, cap, gate or rule.

