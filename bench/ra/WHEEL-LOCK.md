# RA proof wheel inputs

`proof-image.json` pins the official Linux/amd64 image and CPython 3.11.13.
`proof-wheel-lock.json` fixes 87 retained wheel archives (85 common dependencies
and the two baseline releases), including pip and Uvicorn. Successor wheels are
separately reviewed; the common dependency bytes stay identical in both venvs.

Run `ra_wheel_lock.py --lock /absolute/proof-wheel-lock.json --image
/absolute/proof-image.json --wheels /absolute/wheels` to check the complete
archive inventory and derive its common identity. The handoff rechecks bytes
before creating a manifest for the isolated installed-payload provenance probe.

Archive checks do not establish dependency closure, Linux image installation,
release/source binding or GPU engagement. Reported startup hooks require a
reviewed installation adapter; the provenance probe still refuses them. Those
gates, input bindings, ABBA supervision and launch controls remain before rental.
