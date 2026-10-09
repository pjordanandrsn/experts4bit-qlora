### Check RA installed payload before startup

A separate no-site verifier binds the selected interpreter and installed wheel
payload before importing verified installer templates. Proposed startup bytes
are inspected without execution; release imports and launch authority remain
separate gates.

Copied and symlinked venv interpreters bind their selected and base executable
bytes independently to the proof image and retain both checks in the receipt.
