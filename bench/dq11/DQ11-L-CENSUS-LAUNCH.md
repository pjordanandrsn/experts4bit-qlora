# DQ11 L-only census launch amendment

Prospective amendment to [the L-only preregistration](DQ11-L-CENSUS-PREREG.md)
and [the merged instrument](DQ11-L-CENSUS-INSTRUMENT.md), issue #1083.
**DRAFT_NOT_LAUNCHABLE.** This amendment fixes the operational limits and
prelaunch evidence. It changes no model, input, arithmetic, capture window,
attribution rule, reducer, quality gate or BUILD decision. Merge and a fresh
maintainer ACK naming the exact machine and enumerated offer IDs are required
before one paid draw. No quote, rental or 5090 census has occurred under it.

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

The manifest sets `usd_per_hour=0.85`, `wallclock_h=0.5`, `est_usd=0.425`,
`download_gb=27`, `max_download_usd_per_gb=0.011`, `storage=320`, the
previously used devel image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`,
and a command that binds the validated canonical asset path and exact driver to
`bench/dq11/dq11_l_census_drive.sh`. Its `l_census` object
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

## Stable-host packet and final sequence

The **same guarded controller's Vast offer search** must observe the proposed
machine in every successful search for at least five continuous minutes,
with no gap longer than 30 seconds. Its offer IDs may rotate between searches.
Every ID returned for that machine must belong to the enumerated approved set
and pass the verified/rentable, normal RTX 5090, driver, disk/RAM, hourly and
tariff gates. The eligible set is the **union** of IDs seen across the interval;
after the ACK that union may be a subset of the full approved set. An outside
ID, ineligible offer, absent machine, missed search or gap resets the clock.
The last quote needs at
least one currently listed approved offer, with no outside or ineligible IDs.
Before the paid ACK, a read-only discovery interval proposes its observed
union; after the ACK, the repeated interval enforces that exact approved set.
The record includes each observation time, the exact query/filter,
all relevant offer fields and prices, selected machine/IDs, and its SHA-256.
The `adertha-agents` offer-set pin must be on merged main; the final manifest
uses `vast_pin_machine` and `vast_pin_offers`, never an unpinned or single-ID
fallback. A stale ask can move only among the approved IDs on that machine,
under every original provider and all-in gate.
The read-only observer uses the controller's live authentication gate
(`E4B_RENT_LIVE=1`) and permits only the auth probe and offer-search GETs;
it has no create method. This environment arming alone does not authorize a
paid launch.

First prepare the immutable review packet: merged source and instrument
hashes, canonical assets, fresh CPU 100-URL fetch/policy gate, five-minute
controller search record, pinned manifest, read-only quote and `$0.90` ceiling
walk. The maintainer reviews that packet and issues a **fresh paid ACK for the
exact machine and complete offer set**. After the ACK, repeat the five-minute
search on that same set. Then execute a fresh bound CPU gate, read-only quote
and guarded launch within **60 seconds of the last successful search** and
within **60 seconds from gate completion to launch invocation**. If either
clock expires, an offer changes, a source/manifest hash moves or the ACK names
anything else, refuse before create and return for review. The controller's
final search and `pod-launch.sh`'s own pinned search must both agree; the latter
still refuses an absent ask without substitution. Record all UTC and monotonic
times, gate/quote/manifest hashes and selected offer in the run receipt.

The in-box order is source/input/admission, native-SDPA/CUPTI and same-arm
bitwise canary, fresh unprofiled 40-update reference, fresh 40-update capture
of updates 6–8, reducer, capped private evidence transfer, authenticated
teardown and empty-list proof. A refusal at any stage licenses no BUILD choice.
The only decision is the preregistered `C_j` rule; a passing MLP ceiling is
still only a BUILD proposal for separate review. No public speed, default,
capacity or shipping claim follows this census.
