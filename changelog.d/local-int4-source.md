### Local checkpoints work with expert-int4 paged serving

The expert-int4 source reader now uses an existing local model directory directly,
for both RTN and calibrated repacking. Hub model IDs retain their revision and
download-pattern handling; missing sources and zero patched layers still refuse.
