### Refuse CR line endings in changed text files

The lint job checks added and changed text files against the PR merge base and
fails on CRLF or bare CR line endings. CSV files are exempt, preserving the
existing generated results. The guard reports affected paths without changing
files or repository attributes.
