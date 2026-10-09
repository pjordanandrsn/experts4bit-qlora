### Check RA retained dependency closure

A no-site check derives Linux/Python dependency closure from the hash-verified
wheel metadata and frozen roots. It expands extras, checks constraints and
rejects missing or surplus pins without importing releases or enabling startup
hooks. Archived metadata closure grants no installation or GPU proof.
