### SC2g amendment A1: box G's proof died in the harness; the box sources cleanly, and a dead lane is now seen (bench and tests only)

- **`sc2g-prove-1`** ($0.848) died at box G's install: `sc2g_box_g.sh: line 20: FOLDS: unbound variable`. `sc1_run.sh` sources the
  box scripts under `set -u` before it defines `FOLDS`. `$FOLDS` is now appended where the e4b server starts, so the server's
  environment is unchanged. A test sources every SC2 box script under `set -u` with only `W` set; it reproduces the failure on the
  old line.
- **The controller's dead-lane check never fired.** It counted `pgrep -f 'bash sc1_run.sh'` inside a shell whose own command line
  holds the pattern; procps counts that shell, so `live` read 2 for 66 polls after the box died, and the run waited out its
  deadline. It now counts `[b]ash sc1_run.sh`.
- **`sc1_run.sh` gains an EXIT trap**: an exit that skips `finish` still writes its rc and TP_DONE; rc 0 there is recorded as 79.
- No change to SC2g's design, rule or guards. Next: `sc2g-prove-2`.
