### HO3 Amendment 3: the fetch never imports links

- `bench/ho3/ho3_drive.sh` fetches with `rsync --no-links`, and refuses (exit 22) if a fetched tree still holds a
  symbolic link, so nothing a rented box leaves behind can point a store commit at a file on the controller.
