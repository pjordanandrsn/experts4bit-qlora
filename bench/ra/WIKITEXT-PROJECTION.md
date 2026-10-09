# Proposed source-bound WikiText text projection

`ra_wikitext_projection.py` derives the raw archive and fixed Hub URLs from
reviewed common inputs and the staged registered source. The manifest contains
only schema, input spec, stage, reviewed input lock and its SHA256, and the
expected parent PID. No caller command, revision or extra arguments are accepted.

The CLI requires Linux selected Python `-I -S -B` and the inherited SIGKILL
parent-death guard. It checks the registered P109/P97 assignments for the
complete `t.strip()` filter, two-newline join and full-text tokenizer argument.
It does not call that tokenizer or change the original unrevisioned loaders.

Four bounded raw Hub responses bracket the decode. The component reconciles
complete indexes, binds the archive SHA256/size, decodes every ordered row and
writes ordered UTF8 JSON and complete joined UTF8 text. Helper, stage, common
input, lock and manifest bytes are rechecked before return, including after
output creation. A fresh receipt directory cannot overlap protected inputs.
Failures retain the error, PID, manifest hash and completed raw-record prefix;
there are no retries. A complete successful output has exactly seven files.

`retained()` supports parent re-audit using its own verified process record.
It checks PID/manifest/guard/cleanup, four sequential bounded raw records with
fixed URLs and UTC clocks, exact regular-file inventory, complete regenerated
rows/text and the entire summary. The caller must supply actual process evidence;
this function does not authenticate invented process records or acquire a lane.
Existing `ra_process.run` establishes reaping and owned-group absence.

Focused CPU tests explicitly name synthetic host guard and replay substitutions.
The selected extracted Linux control uses a real guarded process, actual retained
archive, four replayed prior live Hub records and synthetic ancillary inputs with
an owned narrowed source-pin fixture. It compares every row and joined byte and
refuses resealed summaries/archive mutations. It is not a fresh live retrieval,
a pinned independent local decoder, full-image installation or GPU evidence.

This standalone proposal does not expand any fixed worker inventory or compose
ABBA. Both new helpers remain unreviewed extras there. Tokenizer regeneration,
native loader revision binding, calibration authority, actual consumption,
publisher trust and launch/release gates remain open. No release package imports,
dependency installation, rental or spending authority is introduced.
