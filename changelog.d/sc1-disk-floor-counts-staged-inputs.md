### SC1 box (#846): the disk floor counts the lane's own staged inputs (bench and tests only)

- **What went wrong.** Box I's A5 proof was refused twice on a healthy host (`sc1g-prove-a5-1`, `-2`: Vast 145701, 317 GB
  free). The launcher orders a fixed 320 GB disk, and the controller stages SC1g's ~3.8 GB of reference rows before the
  box's 320 GB free-disk check. Every box was refused on the lane's account, not the host's, and asking for more storage
  in the manifest cannot fix it.
- **The fix.** `sc1_run.sh` now counts `sc1g_ref_full/` (`du -BG`) toward `MIN_DISK_GB`, and notes it in the summary. The
  floor itself is unchanged, so this is not a knob change.
- **The test.** It runs the check block as the script has it, with `df`/`du` faked. The 317 + 4 case passes; 317 with
  nothing staged, and 300 + 4, are still refused (rc 13). It fails with the old check.
